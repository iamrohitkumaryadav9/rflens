"""Turn converted 3GPP spec Markdown (pandoc gfm) into clause-aware chunks, written as JSONL.

3GPP specs are copyrighted: chunks live only under the gitignored data/ directory.

Parser rules
- The clause number always comes from the heading text, never from the Markdown depth.
- An Annex heading is a level-8 heading "Annex X (normative|informative):" plus the title that
  follows it (as the next paragraph(s), joined by pandoc's trailing-backslash line breaks).
  Clauses numbered X.n belong to that annex: parent path "Annex X" -> "X.n".
- Headings that are table captions ("Table N-M ...") are attached to the next table, not clauses.
- Empty headings are skipped.
- Table cells keep their <sup> footnote markers; a table's NOTE paragraphs (and trailing NOTE rows)
  are repeated in every row-group piece of a split table, with the header rows.
- Excluded from chunks: the cover, Foreword, the Change history annex, Void/Reserved clauses,
  clauses with an empty body, and text under unnumbered headings.

Size limits apply to the embedding `text` (a chunk's `raw_text` is the original Markdown or HTML).
"""

import html
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rflens.config import get_chunk_settings
from rflens.retrieval.embedder import DOCUMENT_PREFIX

# Token estimate for the embedding model (nomic-embed-text, a WordPiece tokenizer behind Ollama).
# Pre-tokens are runs of letters, runs of digits, and single punctuation marks; long runs split
# into several word pieces. Calibrated on 170 chunks measured with Ollama (85 densest by this
# estimate, 85 random): real tokens / raw estimate had min 0.87, median 1.00, max 1.145. The
# scale 1.20 makes the estimate >= the real count for every measured chunk (needs >= 1.145) and
# leaves a buffer for the unmeasured ones.
PRETOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
TOKEN_SCALE = 1.20
PLACEHOLDER = "[image omitted]"
# "6.2.1", "6.1A", "5.5A.3.2", "A.3M", and the spec's own typo "F10.1" (annex letter + digits)
CLAUSE_RE = re.compile(r"^(?:\d+[A-Z]?|[A-Z]\d*(?=\.))(?:\.[0-9A-Za-z]+)*$")
ANNEX_RE = re.compile(r"^Annex ([A-Z])\s*\((normative|informative)\):?\s*(.*)$", re.S)
HEADING_RE = re.compile(r"^(#{1,8})(?: (.*))?$")
CAPTION_PARA_RE = re.compile(r"^\**Table\s+[\dA-Z][\w.\-]*\s*[:~]")
CAPTION_HEADING_RE = re.compile(r"^(Table|Figure)\s")
TABLE_NUMBER_RE = re.compile(r"Table\s+([\dA-Z][\w.\-]*)")
NOTE_RE = re.compile(r"^\**NOTE\b")
VOID_RE = re.compile(r"^\(?(void|reserved)\)?\.?$", re.I)
IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)")
IMG_HTML_RE = re.compile(r"<img\b[^>]*?\bsrc=\"([^\"]+)\"[^>]*>", re.I)
MATH_RE = re.compile(r"(\$`[^`]*`\$)")
TABLE_TAG_RE = re.compile(r"<(/?)(table|thead|tbody|tr|td|th)\b([^>]*)>", re.I)
ANY_TAG_RE = re.compile(r"<[^>]+>")
PART_MARKER_EXTRA = len(" (part 99/99)")


def estimate_tokens(text: str) -> int:
    """Upper-biased estimate of the embedding tokens for DOCUMENT_PREFIX + text."""
    pieces = 0
    for pre in PRETOKEN_RE.findall(DOCUMENT_PREFIX + text):
        if pre.isalpha():
            pieces += max(1, -(-len(pre) // 5))
        elif pre.isdigit():
            pieces += max(1, -(-len(pre) // 3))
        else:
            pieces += 1
    return math.ceil(pieces * TOKEN_SCALE)


# --- text cleaning ----------------------------------------------------------------------------


def _unescape_md(text: str) -> str:
    """Remove Markdown backslash escapes outside math spans."""
    pieces = MATH_RE.split(text)
    for i in range(0, len(pieces), 2):
        pieces[i] = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|<>~$])", r"\1", pieces[i])
    return "".join(pieces)


def clean_inline(raw: str) -> tuple[str, list[str]]:
    """Markdown/HTML fragment -> plain text for embedding. Returns (text, image paths)."""
    media: list[str] = []

    def image(m: re.Match) -> str:
        media.append(m.group(1))
        return PLACEHOLDER

    s = IMG_MD_RE.sub(image, raw)
    s = IMG_HTML_RE.sub(image, s)
    s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s, flags=re.S)
    s = re.sub(r"<sub>(.*?)</sub>", r"_\1", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = ANY_TAG_RE.sub("", s)
    s = html.unescape(s)
    s = re.sub(r"\\\n", "\n", s)
    s = re.sub(r"\\$", "", s)
    s = _unescape_md(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip(), media


def clean_cell(raw_html: str) -> tuple[str, list[str]]:
    s = re.sub(r"</p>\s*<p[^>]*>", " ", raw_html)
    s = re.sub(r"<br\s*/?>", " ", s)
    text, media = clean_inline(s)
    return re.sub(r"\s+", " ", text), media


def clean_note_cell(raw_html: str) -> tuple[str, list[str]]:
    """Like clean_cell, but each <p> becomes its own line (one NOTE per line)."""
    text, media = clean_inline(re.sub(r"</p>\s*<p[^>]*>", "\n", raw_html))
    return re.sub(r"[ \t]+", " ", text), media


def clean_heading(raw: str) -> str:
    s = ANY_TAG_RE.sub("", re.sub(r"\\\n", " ", raw))
    s = html.unescape(s).replace("**", "")
    s = re.sub(r"\\$", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


# --- tables -----------------------------------------------------------------------------------


@dataclass
class Cell:
    html: str
    colspan: int = 1
    rowspan: int = 1


@dataclass
class Row:
    cells: list[Cell]
    raw: str
    header: bool = False


@dataclass
class Table:
    kind: str  # "html" or "pipe"
    rows: list[Row]
    source_file: str
    open_tag: str = "<table>"
    colgroup: str = ""
    separator: str = ""  # pipe tables: the |---| line
    caption: str | None = None
    # (text, raw paragraph or "" for a NOTE row, media); NOTE rows are also kept in note_rows
    notes: list[tuple[str, str, list[str]]] = field(default_factory=list)
    note_rows: list[Row] = field(default_factory=list)  # trailing NOTE rows of the table

    def raw_for(self, body: list[Row]) -> str:
        header = [r for r in self.rows if r.header]
        if self.kind == "pipe":
            return "\n".join([header[0].raw, self.separator, *[r.raw for r in body]])
        parts = [self.open_tag]
        if self.colgroup:
            parts.append(self.colgroup)
        if header:
            parts += ["<thead>", *[r.raw for r in header], "</thead>"]
        parts += ["<tbody>", *[r.raw for r in body], "</tbody>", "</table>"]
        return "\n".join(parts)


def _span(attrs: str, name: str) -> int:
    m = re.search(rf'{name}="?(\d+)', attrs, re.I)
    return max(1, int(m.group(1))) if m else 1


def parse_html_table(html_text: str, source_file: str) -> Table:
    """Parse the outer table's rows and cells; nested tables stay inside their cell."""
    depth = 0
    section: str | None = None
    rows: list[Row] = []
    cur_cells: list[Cell] | None = None
    row_start = 0
    cell_start = 0
    cell_attrs = ""
    open_tag = "<table>"
    for m in TABLE_TAG_RE.finditer(html_text):
        closing, name, attrs = bool(m.group(1)), m.group(2).lower(), m.group(3)
        if name == "table":
            if not closing and depth == 0:
                open_tag = m.group(0)
            depth += -1 if closing else 1
        elif depth == 1:
            if name in ("thead", "tbody"):
                section = None if closing else name
            elif name == "tr":
                if not closing:
                    cur_cells, row_start = [], m.start()
                elif cur_cells is not None:
                    rows.append(
                        Row(cur_cells, html_text[row_start : m.end()], header=section == "thead")
                    )
                    cur_cells = None
            elif name in ("td", "th") and cur_cells is not None:
                if not closing:
                    cell_start, cell_attrs = m.end(), attrs
                else:
                    cur_cells.append(
                        Cell(
                            html_text[cell_start : m.start()],
                            _span(cell_attrs, "colspan"),
                            _span(cell_attrs, "rowspan"),
                        )
                    )
    cg = re.search(r"<colgroup>.*?</colgroup>", html_text, re.S)
    table = Table("html", rows, source_file, open_tag, cg.group(0) if cg else "")
    if rows and not any(r.header for r in rows) and "<th" in rows[0].raw:
        rows[0].header = True
    return table


def parse_pipe_table(lines: list[str], source_file: str) -> Table | None:
    if len(lines) < 2 or not re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[1].strip()):
        return None

    def cells(line: str) -> list[Cell]:
        inner = line.strip().removeprefix("|").removesuffix("|")
        return [Cell(c.strip()) for c in re.split(r"(?<!\\)\|", inner)]

    rows = [Row(cells(lines[0]), lines[0], header=True)]
    rows += [Row(cells(line), line) for line in lines[2:]]
    return Table("pipe", rows, source_file, separator=lines[1])


def render_rows(rows: list[Row]) -> tuple[list[str], list[list[str]]]:
    """One text line per row, rowspan values repeated into the rows they span.

    A colspan cell is written once. Returns (lines, image paths per row).
    """
    carry: dict[int, tuple[int, str, int]] = {}  # column -> (cell id, text, rows left)
    lines: list[str] = []
    media: list[list[str]] = []
    next_id = 0

    def fill_carried(out: list[str], col: int) -> int:
        """Write values carried down by rowspans starting at `col`; returns the next column."""
        last_id = None
        while col in carry:
            cid, text, left = carry[col]
            if cid != last_id:
                out.append(text)
                last_id = cid
            if left <= 1:
                del carry[col]
            else:
                carry[col] = (cid, text, left - 1)
            col += 1
        return col

    for row in rows:
        out: list[str] = []
        row_media: list[str] = []
        col = 0
        for cell in row.cells:
            col = fill_carried(out, col)
            text, cell_media = clean_cell(cell.html)
            row_media += cell_media
            out.append(text)
            if cell.rowspan > 1:
                for c in range(col, col + cell.colspan):
                    carry[c] = (next_id, text, cell.rowspan - 1)
                next_id += 1
            col += cell.colspan
        fill_carried(out, col)
        lines.append(" | ".join(out))
        media.append(row_media)
    return lines, media


# --- Markdown block parsing ------------------------------------------------------------------


@dataclass
class Block:
    kind: str  # "heading", "para", "table"
    file: str
    text: str = ""
    level: int = 0
    table: Table | None = None


def parse_blocks(file: str, text: str) -> list[Block]:
    blocks: list[Block] = []
    lines = text.split("\n")
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(Block("para", file, "\n".join(para)))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
        elif line.startswith("<table"):
            flush()
            depth, j, chunk = 0, i, []
            while j < len(lines):
                chunk.append(lines[j])
                depth += len(re.findall(r"<table\b", lines[j])) - lines[j].count("</table>")
                j += 1
                if depth <= 0:
                    break
            blocks.append(
                Block("table", file, table=parse_html_table("\n".join(chunk), file))
            )
            i = j
        elif line.startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            pipe = parse_pipe_table(lines[i:j], file)
            if pipe is not None:
                blocks.append(Block("table", file, table=pipe))
            else:
                blocks.append(Block("para", file, "\n".join(lines[i:j])))
            i = j
        elif (m := HEADING_RE.match(line)) and not para:
            blocks.append(Block("heading", file, (m.group(2) or ""), level=len(m.group(1))))
            i += 1
        else:
            para.append(line)
            i += 1
    flush()
    return blocks


# --- chunks ----------------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    spec: str
    version: str
    release: int
    clause: str
    clause_title: str
    parent_path: list[str]
    parent_titles: list[str]
    kind: str  # "text" or "table"
    position: int
    table_caption: str | None
    table_number: str | None
    table_part: str | None  # "2/3" for a split table
    source_file: str
    text: str
    raw_text: str
    media: list[str]
    image_placeholders: int
    forced_split: bool
    char_count: int


@dataclass
class Stats:
    spec: str
    chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    split_tables: int = 0
    separate_notes_tables: int = 0
    image_placeholders: int = 0
    forced_splits: int = 0
    max_est_tokens: int = 0
    sizes: list[int] = field(default_factory=list)
    excluded: Counter = field(default_factory=Counter)

    def report(self) -> str:
        median = int(statistics.median(self.sizes)) if self.sizes else 0
        excl = ", ".join(f"{k}={v}" for k, v in sorted(self.excluded.items())) or "none"
        return (
            f"{self.spec}: chunks={self.chunks} (text={self.text_chunks}, "
            f"table={self.table_chunks}) median_chars={median} max_chars="
            f"{max(self.sizes, default=0)} max_est_tokens={self.max_est_tokens} "
            f"split_tables={self.split_tables} "
            f"(notes in own chunk: {self.separate_notes_tables}) "
            f"image_placeholders={self.image_placeholders} forced_splits={self.forced_splits}\n"
            f"  excluded: {excl}"
        )


@dataclass
class Clause:
    number: str
    title: str
    parent_path: list[str]
    parent_titles: list[str]
    occurrence: int
    items: list = field(default_factory=list)  # (file, raw paragraph) or Table
    excluded: str | None = None
    accepting_notes: bool = False


def parent_numbers(clause: str, annex: str | None) -> list[str]:
    """Ancestor clause numbers, outermost first. Annex clauses start at 'Annex X'."""
    if clause.startswith("Annex "):
        return []
    parts = clause.split(".")
    ancestors = [".".join(parts[:k]) for k in range(1, len(parts))]
    if annex and clause[:1] == annex and len(parts) > 1:
        return [f"Annex {annex}", *ancestors[1:]]
    return ancestors


def split_long(raw: str, budget: int) -> list[str]:
    """Split text longer than `budget` at sentence/line breaks, then at whitespace."""
    if len(raw) <= budget:
        return [raw]
    pieces: list[str] = []
    units = re.split(r"(?<=[.;:])\s+|\n", raw)
    cur = ""
    for unit in units:
        while len(unit) > budget:
            cut = unit.rfind(" ", 0, budget)
            cut = cut if cut > 0 else budget
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(unit[:cut])
            unit = unit[cut:].lstrip()
        if cur and len(cur) + 1 + len(unit) > budget:
            pieces.append(cur)
            cur = unit
        else:
            cur = f"{cur} {unit}".strip() if cur else unit
    if cur:
        pieces.append(cur)
    return pieces


class SpecChunker:
    def __init__(
        self, soft: int, hard: int, table_hard: int | None = None, max_tokens: int | None = None
    ) -> None:
        """Size limits in characters (`soft`, `hard`, and `table_hard` for tables, default
        `hard`) and, optionally, `max_tokens`: no chunk may have an estimated embedding token
        count above it (see estimate_tokens). Tables are number-dense and can need under 1.5
        chars/token, so characters alone cannot keep them inside the model's context."""
        table_hard = hard if table_hard is None else table_hard
        if not 0 < soft <= hard or not 0 < table_hard <= hard:
            raise ValueError("need 0 < soft <= hard and 0 < table_hard <= hard")
        if max_tokens is not None and max_tokens < 50:
            raise ValueError("max_tokens must be at least 50")
        self.soft, self.hard, self.table_hard = soft, hard, table_hard
        self.max_tokens = max_tokens

    def _fits(self, text: str, cap: int) -> bool:
        return len(text) <= cap and (
            self.max_tokens is None or estimate_tokens(text) <= self.max_tokens
        )

    def chunk(self, parts: list[tuple[str, str]]) -> tuple[list[Chunk], Stats]:
        cover = next((p for p in parts if "cover" in p[0].lower()), None)
        if cover is None:
            raise ValueError("no cover part found (file name must contain 'cover')")
        self.spec, self.version, self.release = self._cover_metadata(cover[1])
        self.stats = Stats(self.spec)
        self.stats.excluded["cover"] += 1
        self.chunks: list[Chunk] = []
        self.titles: dict[str, str] = {}
        self.seen: Counter = Counter()
        self.annex: str | None = None
        self.caption_heading: str | None = None

        blocks: list[Block] = []
        for name, text in parts:
            if name != cover[0]:
                blocks += parse_blocks(name, text)
        self._walk(blocks)
        ids = [c.chunk_id for c in self.chunks]
        if len(ids) != len(set(ids)):
            dup = [i for i, n in Counter(ids).items() if n > 1][:3]
            raise ValueError(f"duplicate chunk ids, e.g. {dup}")
        self.stats.chunks = len(self.chunks)
        self.stats.sizes = [c.char_count for c in self.chunks]
        self.stats.max_est_tokens = max((estimate_tokens(c.text) for c in self.chunks), default=0)
        return self.chunks, self.stats

    @staticmethod
    def _cover_metadata(text: str) -> tuple[str, str, int]:
        plain = html.unescape(ANY_TAG_RE.sub(" ", text))
        m = re.search(r"TS\s+([\d.]+(?:-\d+)?)\s+V(\d+\.\d+\.\d+)", plain)
        r = re.search(r"\(Release\s+(\d+)\)", plain)
        if not m or not r:
            raise ValueError("cover has no 'TS <number> V<x.y.z>' / '(Release N)'")
        return m.group(1), m.group(2), int(r.group(1))

    # -- block walk ---------------------------------------------------------------------------

    def _walk(self, blocks: list[Block]) -> None:
        ctx: Clause | None = None
        i = 0
        while i < len(blocks):
            b = blocks[i]
            i += 1
            if b.kind == "heading":
                raw_text = b.text
                text = clean_heading(raw_text)
                if b.level == 8 or text.startswith("Annex "):
                    parts = [raw_text]
                    while parts[-1].rstrip().endswith("\\") and i < len(blocks):
                        if blocks[i].kind != "para":
                            break
                        parts.append(blocks[i].text)
                        i += 1
                    full = " ".join(clean_heading(p) for p in parts).strip()
                    self._close(ctx)
                    ctx = self._open_annex(full)
                    continue
                if not text:
                    self.stats.excluded["empty_heading"] += 1
                    continue
                if CAPTION_HEADING_RE.match(text):
                    self.caption_heading = text
                    self.stats.excluded["caption_heading_attached"] += 1
                    continue
                self._close(ctx)
                ctx = self._open_clause(text)
            elif ctx is None:
                self.stats.excluded["before_first_heading"] += 1
            elif b.kind == "table":
                ctx.items.append(self._attach_caption(ctx, b.table))
                ctx.accepting_notes = True
            else:  # para
                last = ctx.items[-1] if ctx.items else None
                if ctx.accepting_notes and isinstance(last, Table) and NOTE_RE.match(b.text):
                    text, media = clean_inline(b.text)
                    last.notes.append((text, b.text, media))
                    continue
                ctx.accepting_notes = False
                ctx.items.append((b.file, b.text))
        self._close(ctx)

    def _attach_caption(self, ctx: Clause, table: Table) -> Table:
        last = ctx.items[-1] if ctx.items else None
        if (
            isinstance(last, tuple)
            and "\n" not in last[1]
            and len(last[1]) < 400
            and CAPTION_PARA_RE.match(last[1])
        ):
            table.caption = clean_heading(last[1])
            ctx.items.pop()
        elif self.caption_heading:
            table.caption = self.caption_heading
        self.caption_heading = None
        body = table.rows
        while body and not body[-1].header:  # trailing NOTE rows travel as notes
            text, media = clean_note_cell(" ".join(c.html for c in body[-1].cells))
            if not NOTE_RE.match(text):
                break
            table.notes.insert(0, (text, "", media))
            table.note_rows.insert(0, body.pop())
        return table

    def _open_annex(self, full: str) -> Clause | None:
        m = ANNEX_RE.match(full)
        if not m:
            return Clause("", full, [], [], 0, excluded="unnumbered_heading")
        letter, title = m.group(1), m.group(3).strip()
        self.annex = letter
        number = f"Annex {letter}"
        self.titles[number] = title
        ctx = self._new(number, title, [], [])
        if title.lower() == "change history":
            ctx.excluded = "change_history"
        return ctx

    def _open_clause(self, text: str) -> Clause:
        token, _, title = text.partition(" ")
        token = token.rstrip(".")  # the spec writes a few numbers as "6.3.3.1."
        if text == "Foreword":
            return Clause("", text, [], [], 0, excluded="foreword")
        if not CLAUSE_RE.match(token):
            return Clause("", text, [], [], 0, excluded="unnumbered_heading")
        if token[0].isdigit():
            self.annex = None
        self.titles[token] = title
        parents = parent_numbers(token, self.annex)
        ctx = self._new(token, title, parents, [self.titles.get(p, "") for p in parents])
        if VOID_RE.match(title.strip()):
            ctx.excluded = "void_reserved"
        return ctx

    def _new(self, number: str, title: str, parents: list[str], titles: list[str]) -> Clause:
        self.seen[number] += 1
        return Clause(number, title, parents, titles, self.seen[number])

    # -- chunk building -----------------------------------------------------------------------

    def _close(self, ctx: Clause | None) -> None:
        if ctx is None:
            return
        if ctx.excluded:
            self.stats.excluded[ctx.excluded] += 1
            return
        made_before = len(self.chunks)
        prefix = f"TS {self.spec} V{self.version} clause {ctx.number}: {ctx.title}"
        seq = 0
        buf: list[tuple[str, str]] = []

        def flush_text() -> None:
            nonlocal seq
            seq = self._emit_text(ctx, prefix, buf, seq)
            buf.clear()

        for item in ctx.items:
            if isinstance(item, Table):
                flush_text()
                seq = self._emit_table(ctx, prefix, item, seq)
            else:
                file, raw = item
                if VOID_RE.match(raw.strip()):
                    continue
                buf.append((file, raw))
        flush_text()
        if len(self.chunks) == made_before:
            self.stats.excluded["empty_body"] += 1

    def _make(
        self, ctx: Clause, seq: int, kind: str, text: str, raw: str, media: list[str], file: str,
        caption: str | None = None, number: str | None = None, part: str | None = None,
        forced: bool = False,
    ) -> Chunk:
        chunk = Chunk(
            chunk_id=self._chunk_id(ctx, seq),
            spec=self.spec, version=self.version, release=self.release, clause=ctx.number,
            clause_title=ctx.title, parent_path=ctx.parent_path,
            parent_titles=ctx.parent_titles, kind=kind, position=seq, table_caption=caption,
            table_number=number, table_part=part, source_file=file, text=text, raw_text=raw,
            media=media, image_placeholders=text.count(PLACEHOLDER), forced_split=forced,
            char_count=len(text),
        )
        self.stats.text_chunks += kind == "text"
        self.stats.table_chunks += kind == "table"
        self.stats.forced_splits += forced
        return chunk

    def _chunk_id(self, ctx: Clause, seq: int) -> str:
        key = ctx.number if ctx.occurrence == 1 else f"{ctx.number}~{ctx.occurrence}"
        return f"{self.spec}@{self.version}/{key}/{seq}"

    def _enforce_cap(self, prefix: str, text: str, cap: int) -> list[tuple[str, bool]]:
        """Last-resort cap: split text that still exceeds `cap` characters or the token limit.
        Returns (text, forced)."""
        if self._fits(text, cap):
            return [(text, False)]
        pieces: list[tuple[str, bool]] = []
        remaining = text
        while remaining:
            budget = cap if not pieces else cap - len(prefix) - 2
            cut = remaining[:budget]
            if len(remaining) > budget and len(cut.split()) > 1:
                cut = cut.rsplit(None, 1)[0]  # break at whitespace
            while self.max_tokens is not None and len(cut.split()) > 1:
                piece = cut if not pieces else f"{prefix}\n\n{cut}"
                if estimate_tokens(piece) <= self.max_tokens:
                    break
                cut = cut[: int(len(cut) * 0.9)].rsplit(None, 1)[0]  # too many tokens: shrink
            pieces.append((cut if not pieces else f"{prefix}\n\n{cut}", True))
            remaining = remaining[len(cut) :].lstrip()
        return pieces

    def _emit_text(
        self, ctx: Clause, prefix: str, paras: list[tuple[str, str]], seq: int, *,
        kind: str = "text", caption: str | None = None, number: str | None = None,
        part: str | None = None,
    ) -> int:
        if not paras:
            return seq
        cap = self.table_hard if kind == "table" else self.hard  # table notes use the table cap
        soft = min(self.soft, cap)
        budget = cap - len(prefix) - 2
        units: list[tuple[str, str]] = []  # (file, raw piece)
        for file, raw in paras:
            units += [(file, piece) for piece in split_long(raw, budget)]
        group: list[tuple[str, str]] = []
        group_len = len(prefix)
        prefix_tokens = estimate_tokens(prefix)
        group_tokens = prefix_tokens

        def flush() -> None:
            nonlocal seq, group, group_len, group_tokens
            if not group:
                return
            cleaned = [clean_inline(raw) for _, raw in group]
            media = [m for _, ms in cleaned for m in ms]
            text = prefix + "\n\n" + "\n\n".join(t for t, _ in cleaned)
            raw_text = "\n\n".join(raw for _, raw in group)
            for piece, forced in self._enforce_cap(prefix, text, cap):
                self.chunks.append(
                    self._make(
                        ctx, seq, kind, piece, raw_text, media, group[0][0], caption=caption,
                        number=number, part=part, forced=forced,
                    )
                )
                if kind == "text":
                    self.stats.image_placeholders += piece.count(PLACEHOLDER)
                seq += 1
            group, group_len, group_tokens = [], len(prefix), prefix_tokens

        for file, raw in units:
            clean = clean_inline(raw)[0]
            size = len(clean) + 2
            tokens = estimate_tokens(clean) if self.max_tokens is not None else 0
            over_tokens = self.max_tokens is not None and group_tokens + tokens > self.max_tokens
            if group and (group_len + size > soft or over_tokens):
                flush()
            group.append((file, raw))
            group_len += size
            group_tokens += tokens
        flush()
        return seq

    def _emit_table(self, ctx: Clause, prefix: str, table: Table, seq: int) -> int:
        lines, row_media = render_rows(table.rows)
        header_idx = [i for i, r in enumerate(table.rows) if r.header]
        body_idx = [i for i, r in enumerate(table.rows) if not r.header]
        header_text = "\n".join(lines[i] for i in header_idx)
        header_media = [m for i in header_idx for m in row_media[i]]
        notes_text = "\n".join(t for t, _, _ in table.notes)
        notes_raw = "\n\n".join(r for _, r, _ in table.notes if r)  # NOTE paragraphs only
        notes_media = [m for _, _, ms in table.notes for m in ms]
        number_match = TABLE_NUMBER_RE.search(table.caption or "")
        number = number_match.group(1) if number_match else None
        caption = table.caption
        notes_line = notes_text  # replaced by a pointer when the notes get their own chunk
        inline_notes = True

        def assemble(rows: list[int], part: str | None) -> str:
            cap = f"{caption} (part {part})" if caption and part else (caption or "")
            grid = "\n".join(s for s in (cap, header_text, *(lines[i] for i in rows)) if s)
            return "\n\n".join(s for s in (prefix, grid, notes_line) if s)

        def raw_for(rows: list[int]) -> str:
            selected = [table.rows[i] for i in rows]
            if inline_notes:  # NOTE rows stay inside the table, NOTE paragraphs follow it
                raw = table.raw_for(selected + table.note_rows)
                return f"{raw}\n\n{notes_raw}" if notes_raw else raw
            return table.raw_for(selected)

        def media_for(rows: list[int]) -> list[str]:
            own = [m for i in rows for m in row_media[i]]
            return header_media + own + (notes_media if inline_notes else [])

        groups: list[list[int]] = []
        if self._fits(assemble(body_idx, None), self.table_hard):
            groups = [body_idx]
        else:
            notes_too_big = len(notes_text) > self.table_hard // 3 or (
                self.max_tokens is not None and estimate_tokens(notes_text) > self.max_tokens // 3
            )
            if notes_too_big:
                # Notes too large to repeat in every piece: give them their own chunk(s) and
                # point to them from each row-group piece.
                notes_paras = [(table.source_file, f"{caption or 'Table'} (notes)")]
                notes_paras += [(table.source_file, raw) for _, raw, _ in table.notes]
                first_id = self._chunk_id(ctx, seq)
                seq = self._emit_text(
                    ctx, prefix, notes_paras, seq, kind="table", caption=caption, number=number,
                    part="notes",
                )
                notes_line = f"Notes for this table are in chunk {first_id}."
                inline_notes = False
                self.stats.separate_notes_tables += 1
            cur: list[int] = []
            for r in body_idx:
                trial = assemble(cur + [r], "99/99") if cur else ""
                if cur and not self._fits(trial, self.table_hard):
                    groups.append(cur)
                    cur = [r]
                else:
                    cur.append(r)
            if cur:
                groups.append(cur)
            self.stats.split_tables += 1
        for n, rows in enumerate(groups, start=1):
            part = f"{n}/{len(groups)}" if len(groups) > 1 else None
            text = assemble(rows, part)
            for piece, forced in self._enforce_cap(prefix, text, self.table_hard):
                self.chunks.append(
                    self._make(
                        ctx, seq, "table", piece, raw_for(rows), media_for(rows),
                        table.source_file, caption=caption, number=number, part=part,
                        forced=forced,
                    )
                )
                seq += 1
        # count each image once, not once per piece that repeats the header or notes
        self.stats.image_placeholders += (
            sum(lines[i].count(PLACEHOLDER) for i in header_idx + body_idx)
            + notes_text.count(PLACEHOLDER)
        )
        return seq


def chunk_parts(
    parts: list[tuple[str, str]], soft: int, hard: int, table_hard: int | None = None,
    max_tokens: int | None = None,
) -> tuple[list[Chunk], Stats]:
    return SpecChunker(soft, hard, table_hard, max_tokens).chunk(parts)


def chunk_dir(
    md_dir: Path, out_path: Path, soft: int, hard: int, table_hard: int | None = None,
    max_tokens: int | None = None,
) -> Stats:
    files = sorted(p for p in md_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no Markdown files in {md_dir}")
    chunks, stats = chunk_parts(
        [(p.name, p.read_text("utf-8")) for p in files], soft, hard, table_hard, max_tokens
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    from rflens.ingest.spec_fetch import SPECS

    settings = get_chunk_settings()
    for source in SPECS:
        stats = chunk_dir(
            source.md_dir, source.chunks_path, settings.chunk_soft_chars,
            settings.chunk_hard_chars, settings.chunk_table_hard_chars,
            settings.chunk_max_tokens,
        )
        print(stats.report())
        print(f"  wrote {source.chunks_path}")


if __name__ == "__main__":
    main()
