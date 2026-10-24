"""Tests for the 3GPP chunker, using synthetic hand-written Markdown (no 3GPP text)."""

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rflens.config import ChunkSettings
from rflens.ingest import spec_fetch
from rflens.ingest.spec_chunker import (
    TOKEN_SCALE,
    Chunk,
    Stats,
    chunk_dir,
    chunk_parts,
    estimate_tokens,
)
from rflens.retrieval.embedder import MAX_SAFE_TOKENS

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "spec" / "synthetic"
PREFIX = "99.101-1@19.0.0"


def fixture_parts() -> list[tuple[str, str]]:
    return [(p.name, p.read_text("utf-8")) for p in sorted(FIXTURE_DIR.glob("*.md"))]


def run(
    soft: int = 3000, hard: int = 6000, extra: tuple[tuple[str, str], ...] = ()
) -> tuple[list[Chunk], Stats]:
    return chunk_parts(fixture_parts() + list(extra), soft, hard)


def for_clause(chunks: list[Chunk], clause: str) -> list[Chunk]:
    return [c for c in chunks if c.clause == clause]


def big_table_part(rows: int = 120) -> tuple[str, str]:
    body = "\n".join(
        f"<tr>\n<td>row-{i:03d}</td>\n<td>{i * 3}<sup>3</sup></td>\n</tr>" for i in range(rows)
    )
    text = (
        "# 8 Big table clause\n\nTable 8-1: Big table\n\n<table>\n<thead>\n<tr>\n<th>Name</th>\n"
        f"<th>Value</th>\n</tr>\n</thead>\n<tbody>\n{body}\n</tbody>\n</table>\n\n"
        "NOTE 1: first note text.\n\nNOTE 2: second note text.\n"
    )
    return "03_big.md", text


# --- metadata and invariants -----------------------------------------------------------------


def test_cover_metadata() -> None:
    chunks, _ = run()
    assert {(c.spec, c.version, c.release) for c in chunks} == {("99.101-1", "19.0.0", 19)}


def test_every_chunk_has_a_clause_number() -> None:
    chunks, _ = run()
    assert chunks
    for c in chunks:
        assert c.clause, c.chunk_id
        assert f"clause {c.clause}:" in c.text.split("\n", 1)[0]


def test_chunk_ids_stable_across_runs() -> None:
    first, _ = run(extra=(big_table_part(),))
    second, _ = run(extra=(big_table_part(),))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [asdict(c) for c in first] == [asdict(c) for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_duplicate_clause_numbers_get_distinct_ids() -> None:
    chunks, _ = run()
    ids = [c.chunk_id for c in for_clause(chunks, "7.1")]
    assert ids == [f"{PREFIX}/7.1/0", f"{PREFIX}/7.1~2/0"]


# --- excluded parts ---------------------------------------------------------------------------


def test_cover_foreword_and_change_history_are_excluded() -> None:
    chunks, stats = run()
    everything = " ".join(c.text + c.raw_text for c in chunks)
    assert "never appear" not in everything
    assert "foreword is boilerplate" not in everything
    assert stats.excluded["cover"] == 1
    assert stats.excluded["foreword"] == 1
    assert stats.excluded["change_history"] == 1
    assert not for_clause(chunks, "Annex C")


def test_void_and_reserved_clauses_produce_no_chunks() -> None:
    chunks, stats = run()
    assert not for_clause(chunks, "6.1G")
    assert not for_clause(chunks, "6.1H")
    assert stats.excluded["void_reserved"] == 2


def test_empty_body_clauses_produce_no_chunks() -> None:
    chunks, stats = run()
    for clause in ("6", "6.2", "6.5", "6.6", "7", "Annex B"):
        assert not for_clause(chunks, clause), clause
    assert stats.excluded["empty_body"] >= 6


def test_unnumbered_heading_text_is_excluded() -> None:
    chunks, stats = run()
    assert not any("excluded" in c.text and "unnumbered" in c.text for c in chunks)
    assert stats.excluded["unnumbered_heading"] == 1


# --- parser rules -----------------------------------------------------------------------------


def test_clause_number_comes_from_heading_text_not_depth() -> None:
    chunks, _ = run()
    deep = for_clause(chunks, "6.5.1")  # level-4 heading directly under a level-2 heading
    assert deep
    assert deep[0].parent_path == ["6", "6.5"]
    assert for_clause(chunks, "6.2.1")[0].parent_path == ["6", "6.2"]
    assert for_clause(chunks, "1")[0].parent_path == []


def test_empty_heading_is_skipped_not_a_clause() -> None:
    chunks, stats = run()
    assert stats.excluded["empty_heading"] == 1
    assert any("follows an empty heading" in c.text for c in for_clause(chunks, "6.5.1"))
    assert all(c.clause for c in chunks)


def test_annex_heading_and_title_paragraph_form_the_annex() -> None:
    chunks, _ = run()
    annex_a = for_clause(chunks, "Annex A")[0]
    assert annex_a.clause_title == "Measurement channels"
    assert "Intro text of the annex" in annex_a.text
    assert annex_a.parent_path == []
    # annex form with an empty level-8 heading and two continuation lines
    assert for_clause(chunks, "B.1")[0].parent_path == ["Annex B"]
    assert for_clause(chunks, "B.1")[0].parent_titles == ["Environmental notes"]


def test_clause_numbers_with_a_trailing_dot_or_a_missing_annex_dot() -> None:
    part = (
        "03_odd.md",
        "# 8 Odd numbering\n\nBody eight.\n\n## 8.1. Dotted heading\n\nBody dotted.\n\n"
        "######## Annex F (normative):\\\nOdd annex\n\n# F10.1 General\n\nBody F ten.\n",
    )
    chunks, stats = run(extra=(part,))
    assert for_clause(chunks, "8.1")[0].parent_path == ["8"]
    assert for_clause(chunks, "8.1")[0].clause_title == "Dotted heading"
    odd = for_clause(chunks, "F10.1")[0]
    assert odd.parent_path == ["Annex F"] and odd.clause_title == "General"
    assert stats.excluded["unnumbered_heading"] == 1  # only the fixture's own unnumbered heading


def test_annex_clauses_have_annex_in_parent_path() -> None:
    chunks, _ = run()
    assert for_clause(chunks, "A.1")[0].parent_path == ["Annex A"]
    a21 = for_clause(chunks, "A.2.1")[0]
    assert a21.parent_path == ["Annex A", "A.2"]
    assert a21.parent_titles == ["Measurement channels", "Reference channels"]


def test_table_caption_headings_attach_to_next_table_not_clauses() -> None:
    chunks, stats = run()
    assert not any(c.clause.startswith("Table") for c in chunks)
    assert stats.excluded["caption_heading_attached"] == 2
    tables = {c.table_number: c for c in chunks if c.kind == "table" and c.clause == "6.5.1"}
    assert set(tables) == {"6.5.1-1a", "6.5.1-1b", "6.5.1-2"}
    assert tables["6.5.1-1a"].table_caption == "Table 6.5.1-1a: Emission limits"
    assert tables["6.5.1-2"].table_caption == "Table 6.5.1-2"  # caption only in the heading
    assert "caption only in the heading" in tables["6.5.1-2"].text


def test_pipe_table_becomes_a_table_chunk() -> None:
    chunks, _ = run()
    pipe = next(c for c in chunks if c.table_number == "6.5.1-1a")
    assert pipe.kind == "table"
    assert "9 kHz - 150 kHz | -36" in pipe.text
    assert pipe.raw_text.startswith("| Frequency range")


def test_html_table_rowspan_colspan_sup_and_notes() -> None:
    chunks, _ = run()
    clause = for_clause(chunks, "6.2.1")
    assert [c.kind for c in clause] == ["text", "table", "text"]  # position order preserved
    table = clause[1]
    assert table.table_number == "6.2.1-1"
    lines = table.text.split("\n")
    assert "Band | Class A | Class B" in lines  # colspan cells written once
    assert "Band | Power (dBm) | Tolerance (dB) | Power (dBm) | Tolerance (dB)" in lines
    assert "x1 |  |  | 23 | ±2" in lines
    assert "x2 | 26^1 | +2/-3 | 23 | ±2" in lines  # <sup> marker kept with the table
    assert "x2 | 29^2 | +2/-3 | 20 | ±1" in lines  # rowspan value repeated
    assert "NOTE 3: This row note lives inside the table." in lines
    assert any(ln.startswith("NOTE 1: Class A applies") for ln in lines)
    assert "<sup>1</sup>" in table.raw_text and 'rowspan="2"' in table.raw_text
    # the NOTE row stays inside the raw table; the NOTE paragraphs follow it
    assert table.raw_text.index("NOTE 3: This row note") < table.raw_text.index("</table>")
    assert table.raw_text.index("</table>") < table.raw_text.index("NOTE 1: Class A applies")
    assert "closing sentence" in clause[2].text


def test_note_row_with_several_paragraphs_gives_one_note_per_line() -> None:
    part = (
        "03_notes.md",
        "# 8 Notes row\n\nTable 8-1: With notes\n\n<table>\n<thead>\n<tr>\n<th>A</th>\n<th>B</th>\n"
        "</tr>\n</thead>\n<tbody>\n<tr>\n<td>1</td>\n<td>2</td>\n</tr>\n<tr>\n"
        '<td colspan="2"><p>NOTE 1: first.</p>\n<p>NOTE 2: second.</p></td>\n</tr>\n'
        "</tbody>\n</table>\n",
    )
    chunks, _ = run(extra=(part,))
    table = for_clause(chunks, "8")[0]
    lines = table.text.split("\n")
    assert lines[-2:] == ["NOTE 1: first.", "NOTE 2: second."]
    assert "1 | 2" in lines
    assert table.raw_text.index("NOTE 2: second.") < table.raw_text.index("</table>")


def test_oversized_notes_get_their_own_chunk_and_pieces_point_to_it() -> None:
    notes = "\n\n".join(f"NOTE {n}: " + "explanatory words " * 14 for n in range(1, 4))
    big = big_table_part()
    part = (big[0], big[1].replace("NOTE 1: first note text.\n\nNOTE 2: second note text.", notes))
    chunks, stats = run(soft=400, hard=800, extra=(part,))
    pieces = [c for c in chunks if c.clause == "8"]
    notes_chunks = [c for c in pieces if c.table_part == "notes"]
    rows = [c for c in pieces if c.table_part != "notes"]
    assert len(notes_chunks) >= 1 and len(rows) > 1
    assert stats.separate_notes_tables == 1 and stats.forced_splits == 0
    assert all(c.char_count <= 800 for c in pieces)
    assert "NOTE 3:" in "\n".join(c.text for c in notes_chunks)
    for piece in rows:
        assert f"Notes for this table are in chunk {notes_chunks[0].chunk_id}." in piece.text
        assert "NOTE 1:" not in piece.text
        assert "Name | Value" in piece.text


def test_html_sub_and_math_and_image_placeholder() -> None:
    chunks, stats = run()
    c = for_clause(chunks, "6.2.2")[0]
    assert "$`P_{max} \\leq 23\\ dBm`$" in c.text
    assert "[image omitted]" in c.text
    assert c.media == ["synthetic_media/media/image1.wmf"]
    assert "![](synthetic_media/media/image1.wmf)" in c.raw_text
    assert c.image_placeholders == 1
    assert "N_RB is the number of blocks" in for_clause(chunks, "6.6.1")[0].text
    img_table = next(c for c in chunks if c.table_number == "6.5.1-1b")
    assert img_table.media == ["synthetic_media/media/image2.wmf"]
    assert stats.image_placeholders == 2


# --- size limits ------------------------------------------------------------------------------


def test_hard_cap_holds_with_small_limits() -> None:
    long_para = " ".join(f"This is sentence number {i}." for i in range(150))
    extra = (("04_long.md", f"# 9 Long text\n\n{long_para}\n"), big_table_part())
    chunks, stats = run(soft=300, hard=600, extra=extra)
    assert max(c.char_count for c in chunks) <= 600
    assert all(c.char_count == len(c.text) for c in chunks)
    assert len(for_clause(chunks, "9")) > 1
    assert stats.forced_splits == 0


def test_split_table_repeats_header_and_notes_in_every_piece() -> None:
    chunks, stats = run(soft=400, hard=800, extra=(big_table_part(),))
    pieces = [c for c in chunks if c.clause == "8"]
    assert len(pieces) > 1
    assert stats.split_tables == 1
    for n, piece in enumerate(pieces, start=1):
        assert piece.table_part == f"{n}/{len(pieces)}"
        assert piece.char_count <= 800
        assert "Name | Value" in piece.text  # header row repeated
        assert "NOTE 1: first note text." in piece.text
        assert "NOTE 2: second note text." in piece.text
        assert "NOTE 1: first note text." in piece.raw_text
        assert f"(part {n}/{len(pieces)})" in piece.text
    all_text = "\n".join(p.text for p in pieces)
    for i in range(120):  # every row appears in exactly one piece, with its <sup> marker
        assert all_text.count(f"row-{i:03d} | {i * 3}^3") == 1


def test_unsplittable_row_is_force_split_under_the_cap() -> None:
    huge = " ".join(f"word{i}" for i in range(800))  # one cell far larger than the cap
    part = (
        "03_huge.md",
        "# 8 Huge cell\n\n<table>\n<thead>\n<tr>\n<th>A</th>\n</tr>\n</thead>\n<tbody>\n"
        f"<tr>\n<td>{huge}</td>\n</tr>\n</tbody>\n</table>\n",
    )
    chunks, stats = run(soft=300, hard=600, extra=(part,))
    pieces = [c for c in chunks if c.clause == "8"]
    assert len(pieces) > 1
    assert all(p.forced_split and p.char_count <= 600 for p in pieces)
    assert stats.forced_splits == len(pieces)


def test_default_sizes_come_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ChunkSettings(_env_file=None)
    assert (settings.chunk_soft_chars, settings.chunk_hard_chars) == (3000, 6000)
    assert settings.chunk_table_hard_chars == 3600
    monkeypatch.setenv("CHUNK_SOFT_CHARS", "1234")
    monkeypatch.setenv("CHUNK_HARD_CHARS", "2345")
    monkeypatch.setenv("CHUNK_TABLE_HARD_CHARS", "2000")
    settings = ChunkSettings(_env_file=None)
    assert settings.chunk_soft_chars == 1234
    assert settings.chunk_hard_chars == 2345
    assert settings.chunk_table_hard_chars == 2000
    monkeypatch.setenv("CHUNK_TABLE_HARD_CHARS", "9000")
    with pytest.raises(ValueError, match="CHUNK_TABLE_HARD_CHARS"):
        ChunkSettings(_env_file=None)
    monkeypatch.setenv("CHUNK_TABLE_HARD_CHARS", "2000")
    monkeypatch.setenv("CHUNK_SOFT_CHARS", "9000")
    with pytest.raises(ValueError, match="CHUNK_SOFT_CHARS"):
        ChunkSettings(_env_file=None)


def test_table_chunks_obey_the_lower_table_cap() -> None:
    long_para = " ".join(f"This is sentence number {i}." for i in range(150))
    extra = (("04_long.md", f"# 9 Long text\n\n{long_para}\n"), big_table_part())
    chunks, _ = chunk_parts(fixture_parts() + list(extra), 400, 800, 500)
    tables = [c for c in chunks if c.kind == "table"]
    assert tables and max(c.char_count for c in tables) <= 500
    assert max(c.char_count for c in chunks) <= 800
    wide, _ = chunk_parts(fixture_parts() + list(extra), 400, 800, 800)
    assert len([c for c in chunks if c.clause == "8"]) > len([c for c in wide if c.clause == "8"])


def dense_table_part(rows: int = 60) -> tuple[str, str]:
    """A table of band-combination-like codes: few characters per token."""
    body = "\n".join(
        f"<tr>\n<td>CA_n{i}A-n{i + 1}A-n{i + 2}B</td>\n<td>{i}</td>\n<td>5 MHz</td>\n</tr>"
        for i in range(10, 10 + rows)
    )
    return "05_dense.md", (
        "# 9 Dense table\n\nTable 9-1: Combinations\n\n<table>\n<thead>\n<tr>\n<th>Combo</th>\n"
        f"<th>N</th>\n<th>BW</th>\n</tr>\n</thead>\n<tbody>\n{body}\n</tbody>\n</table>\n"
    )


def test_estimate_tokens_is_higher_for_dense_codes_than_for_prose() -> None:
    prose = "the transmitter shall be able to operate in the bands described here " * 4
    codes = "CA_n12A-n13A-n14B, CA_n7A-n28A, n41A_n78A " * 6
    assert len(prose) > len(codes)
    assert estimate_tokens(codes) / len(codes) > 2 * estimate_tokens(prose) / len(prose)
    assert estimate_tokens("") > 0  # the document prefix itself costs tokens
    assert estimate_tokens(prose + prose) > estimate_tokens(prose)
    assert TOKEN_SCALE >= 1.145  # the smallest scale that never under-estimated 170 measured chunks


def test_token_limit_splits_a_dense_table_that_the_character_cap_would_keep_whole() -> None:
    part = dense_table_part()
    plain, _ = chunk_parts(fixture_parts() + [part], 3000, 6000, 6000)
    whole = [c for c in plain if c.clause == "9"]
    assert len(whole) == 1 and estimate_tokens(whole[0].text) > 400  # chars alone let it through
    limited, stats = chunk_parts(fixture_parts() + [part], 3000, 6000, 6000, max_tokens=400)
    pieces = [c for c in limited if c.clause == "9"]
    assert len(pieces) > 2 and stats.forced_splits == 0
    assert all(estimate_tokens(p.text) <= 400 for p in pieces)
    assert all("Combo | N | BW" in p.text for p in pieces)  # header repeated in every piece
    expected_parts = [f"{n}/{len(pieces)}" for n in range(1, len(pieces) + 1)]
    assert [p.table_part for p in pieces] == expected_parts
    joined = "\n".join(p.text for p in pieces)
    for i in range(10, 70):  # every row appears exactly once
        assert joined.count(f"CA_n{i}A-n{i + 1}A-n{i + 2}B | {i} | 5 MHz") == 1


def test_token_limit_also_applies_to_dense_text() -> None:
    dense = ", ".join(f"CA_n{i}A-n{i + 1}A" for i in range(10, 200))
    part = ("04_dense_text.md", f"# 9 Dense text\n\n{dense}\n")
    chunks, stats = chunk_parts(fixture_parts() + [part], 3000, 6000, max_tokens=300)
    pieces = [c for c in chunks if c.clause == "9"]
    assert len(pieces) > 1 and all(estimate_tokens(p.text) <= 300 for p in pieces)
    assert stats.max_est_tokens <= 1000  # the rest of the fixture is unaffected
    assert any(p.forced_split for p in pieces)  # a single over-long paragraph is a forced split


def test_no_token_limit_by_default_and_a_tiny_one_is_rejected() -> None:
    chunks, stats = run()
    assert stats.max_est_tokens > 0 and chunks
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_parts(fixture_parts(), 300, 600, max_tokens=10)


def test_token_limit_default_matches_the_embedder_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ChunkSettings(_env_file=None).chunk_max_tokens == MAX_SAFE_TOKENS == 1840
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "1000")
    assert ChunkSettings(_env_file=None).chunk_max_tokens == 1000
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "10")
    with pytest.raises(ValueError, match="CHUNK_MAX_TOKENS"):
        ChunkSettings(_env_file=None)


def test_table_cap_cannot_exceed_the_hard_cap() -> None:
    with pytest.raises(ValueError, match="table_hard"):
        chunk_parts(fixture_parts(), 300, 600, 700)


def test_chunk_dir_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "chunks.jsonl"
    stats = chunk_dir(FIXTURE_DIR, out, 3000, 6000)
    rows = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert len(rows) == stats.chunks
    assert {"chunk_id", "text", "raw_text", "clause", "parent_path", "table_part"} <= set(rows[0])


def test_missing_cover_raises() -> None:
    with pytest.raises(ValueError, match="no cover part"):
        chunk_parts([("01_main.md", "# 1 Scope\n\nText.\n")], 3000, 6000)


# --- fetch cache (no network, no pandoc) -----------------------------------------------------


def make_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sha: str) -> spec_fetch.SpecSource:
    monkeypatch.setattr(spec_fetch, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(spec_fetch, "INTERIM_DIR", tmp_path / "interim")
    return spec_fetch.SpecSource("99.101-1", "99_series", "99101-1-j00.zip", "19.0.0", 19, sha)


def no_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("must not download")


def test_zip_with_pinned_sha_is_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"not really a zip"
    source = make_source(tmp_path, monkeypatch, hashlib.sha256(data).hexdigest())
    source.zip_path.parent.mkdir(parents=True)
    source.zip_path.write_bytes(data)
    monkeypatch.setattr(spec_fetch.urllib.request, "urlopen", no_network)
    assert spec_fetch.ensure_zip(source) is False


def test_download_with_wrong_sha_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, monkeypatch, "0" * 64)

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(spec_fetch.urllib.request, "urlopen", lambda *a, **k: Response(b"x"))
    with pytest.raises(ValueError, match="SHA-256"):
        spec_fetch.ensure_zip(source)
    assert not source.zip_path.exists()


def test_conversion_skipped_when_cache_is_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path, monkeypatch, "a" * 64)
    source.md_dir.mkdir(parents=True)
    (source.md_dir / "part.md").write_text("# 1 Scope\n", encoding="utf-8")
    marker = {"zip_sha256": "a" * 64, "pandoc": "pandoc 9.9", "files": ["part.md"]}
    (source.md_dir / spec_fetch.MARKER).write_text(json.dumps(marker), encoding="utf-8")
    monkeypatch.setattr(spec_fetch, "pandoc_version", lambda: "pandoc 9.9")
    monkeypatch.setattr(spec_fetch, "extract_docx", no_network)
    assert spec_fetch.convert(source) is False
