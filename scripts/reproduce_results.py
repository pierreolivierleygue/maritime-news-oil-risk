"""Verify frozen inputs, recompute results/exhibits, and check the delivered deck."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_research_package import load_inputs, verify_hashes


def verify_presentation(root=ROOT):
    folder = root / "output/presentation"
    manifest = json.loads((folder / "manifest.json").read_text())
    verify_hashes(root, manifest["source_sha256"])
    verify_hashes(folder, manifest["artifact_sha256"])
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
          "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
          "c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
    with zipfile.ZipFile(folder / "maritime_research.pptx") as z:
        slides = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
        notes = [n for n in z.namelist() if n.startswith("ppt/notesSlides/notesSlide") and n.endswith(".xml")]
        if len(slides) != 6 or len(notes) != 6:
            raise ValueError("Expected six slides and six speaker-note parts")
        for number in (2, 3, 5):
            if not ET.fromstring(z.read(f"ppt/slides/slide{number}.xml")).findall(".//c:chart", ns):
                raise ValueError(f"Missing native chart on slide {number}")
        if not ET.fromstring(z.read("ppt/slides/slide4.xml")).findall(".//a:tbl", ns):
            raise ValueError("Missing native model table")
    content = json.loads((root / "config/presentation_content_v1.json").read_text())
    if len(content["slides"]) != 6 or sum(s["seconds"] for s in content["slides"]) != 420:
        raise ValueError("Presentation must total six slides and seven minutes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-nlp", action="store_true", help="Also refit/recompute the frozen NLP evaluation (requires cached encoder)")
    args = parser.parse_args()
    load_inputs()  # Verify source manifests, upstream inputs and saved outputs.
    steps = []
    if args.full_nlp:
        steps.append((".venv/bin/python", "scripts/evaluate_nlp_holdout.py"))
    steps += [(".venv/bin/python", "scripts/evaluate_finance.py"),
              (".venv-reporting/bin/python", "scripts/build_research_package.py")]
    for python, script in steps:
        if not (ROOT / python).is_file():
            raise SystemExit(f"Missing {python}; follow RESEARCH_PACKAGE.md setup")
        subprocess.run([str(ROOT / python), str(ROOT / script), "--validate"], cwd=ROOT, check=True)
    verify_presentation()
    print("Verified frozen inputs, exact financial/reporting recomputation, and presentation integrity.")
    if not args.full_nlp:
        print("NLP artifacts were hash-verified; use --full-nlp to recompute them as well.")


if __name__ == "__main__":
    main()
