"""Download 3GPP spec zips, verify them, and convert the Word files to Markdown with pandoc.

3GPP specs are copyrighted. Everything this module writes goes under the gitignored data/
directory; nothing from 3GPP belongs in the repository.

Downloads use 3GPP's own FTP host (www.3gpp.org answers 403 to automated requests). A zip is
verified against the pinned SHA-256 below. Converted Markdown is cached next to a marker that
holds the zip's SHA-256 and the pandoc version, so a rerun with unchanged input does nothing.
"""

import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_ROOT / "raw" / "3gpp"
INTERIM_DIR = DATA_ROOT / "interim" / "3gpp"
FTP_ROOT = "ftp://ftp.3gpp.org/Specs/archive"
MARKER = ".conversion.json"


@dataclass(frozen=True)
class SpecSource:
    spec: str  # e.g. 38.101-1
    series_dir: str  # e.g. 38_series
    zip_name: str
    version: str
    release: int
    sha256: str  # of the zip, recorded when the file was first inspected

    @property
    def url(self) -> str:
        return f"{FTP_ROOT}/{self.series_dir}/{self.spec}/{self.zip_name}"

    @property
    def zip_path(self) -> Path:
        return RAW_DIR / self.spec / self.zip_name

    @property
    def docx_dir(self) -> Path:
        return RAW_DIR / self.spec / "docx"

    @property
    def md_dir(self) -> Path:
        return INTERIM_DIR / self.spec / "md"

    @property
    def chunks_path(self) -> Path:
        return INTERIM_DIR / self.spec / "chunks.jsonl"


# Release 19, the latest of one release for both specs (decided in Week 2 step 1).
SPECS = (
    SpecSource(
        spec="36.101",
        series_dir="36_series",
        zip_name="36101-j60.zip",
        version="19.6.0",
        release=19,
        sha256="068363f0f3c3447103ec108bbd15863269429d2df1883cec6eea68c561daa551",
    ),
    SpecSource(
        spec="38.101-1",
        series_dir="38_series",
        zip_name="38101-1-j70.zip",
        version="19.7.0",
        release=19,
        sha256="d52caea300a803effe375f7a50927025f15cea565c7e3478f01819b5a1864e4a",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_zip(source: SpecSource) -> bool:
    """Make sure the zip is present with the pinned SHA-256. Returns True if it downloaded."""
    path = source.zip_path
    if path.exists():
        if sha256_file(path) == source.sha256:
            return False
        print(f"{path.name} does not match the pinned SHA-256; downloading again")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(source.url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    actual = sha256_file(partial)
    if actual != source.sha256:
        partial.unlink()
        raise ValueError(f"{source.url}: SHA-256 {actual} != pinned {source.sha256}")
    partial.replace(path)
    return True


def pandoc_version() -> str:
    exe = shutil.which("pandoc")
    if exe is None:
        raise RuntimeError("pandoc is not installed or not on PATH")
    out = subprocess.run([exe, "--version"], check=True, capture_output=True, text=True)
    return out.stdout.splitlines()[0]


def extract_docx(source: SpecSource) -> list[Path]:
    """Extract the Word files from the zip (names are validated) into data/raw."""
    source.docx_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(source.zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if "/" in name or "\\" in name or ".." in name or not name.endswith(".docx"):
                raise ValueError(f"unexpected member in {source.zip_name}: {name!r}")
            target = source.docx_dir / name
            target.write_bytes(z.read(info))
            paths.append(target)
    return sorted(paths)


def _convert_one(docx: Path, md_dir: Path) -> None:
    out = md_dir / f"{docx.stem}.md"
    subprocess.run(
        [
            "pandoc",
            str(docx),
            "-f",
            "docx",
            "-t",
            "gfm",
            "--wrap=none",
            f"--extract-media={docx.stem}_media",
            "-o",
            str(out),
        ],
        cwd=md_dir,  # relative media paths keep the Markdown portable inside md_dir
        check=True,
        capture_output=True,
    )


def convert(source: SpecSource) -> bool:
    """Convert the spec's Word files to Markdown unless the cache is current.

    Returns True if it converted.
    """
    pandoc = pandoc_version()
    marker = source.md_dir / MARKER
    if marker.exists():
        cached = json.loads(marker.read_text("utf-8"))
        if (
            cached.get("zip_sha256") == source.sha256
            and cached.get("pandoc") == pandoc
            and all((source.md_dir / f).exists() for f in cached.get("files", []))
        ):
            return False
    if source.md_dir.exists():
        shutil.rmtree(source.md_dir)
    source.md_dir.mkdir(parents=True)
    docx_files = extract_docx(source)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda d: _convert_one(d, source.md_dir), docx_files))
    files = sorted(f"{d.stem}.md" for d in docx_files)
    marker.write_text(
        json.dumps({"zip_sha256": source.sha256, "pandoc": pandoc, "files": files}, indent=1),
        encoding="utf-8",
    )
    return True


def main() -> None:
    for source in SPECS:
        start = time.perf_counter()
        downloaded = ensure_zip(source)
        converted = convert(source)
        print(
            f"{source.spec} {source.version}: "
            f"{'downloaded' if downloaded else 'download skipped (SHA-256 verified)'}, "
            f"{'converted' if converted else 'conversion skipped (cache current)'}, "
            f"{time.perf_counter() - start:.1f}s"
        )


if __name__ == "__main__":
    main()
