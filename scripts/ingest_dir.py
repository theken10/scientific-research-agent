import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest.pipeline import ingest_pdf  # noqa: E402


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pdfs")
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"no pdfs found in {folder}")
        return
    for p in pdfs:
        try:
            print(ingest_pdf(p))
        except Exception as e:
            print(f"failed on {p.name}: {e}")


if __name__ == "__main__":
    main()
