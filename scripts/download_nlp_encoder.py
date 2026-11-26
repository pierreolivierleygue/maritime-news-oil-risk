"""Download a public frozen encoder and bind its exact revision and file hashes."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from .prepare_news_review import ROOT, digest
    from .reserve_news_evaluation import save_json
except ImportError:
    from prepare_news_review import ROOT, digest
    from reserve_news_evaluation import save_json


def verify_model(directory):
    lock = json.loads((directory / "encoder_manifest.json").read_text())
    if len(lock["revision"]) != 40:
        raise ValueError("Encoder revision must be a full commit hash")
    for name, expected in lock["file_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Encoder file mismatch: {name}")
    return lock


def download(root=ROOT):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import HfApi, snapshot_download
    config = json.loads((root / "config/nlp_comparison_v1.json").read_text())["embedding"]
    directory = root / config["model_directory"]
    if (directory / "encoder_manifest.json").exists():
        return verify_model(directory)
    # No corpus text, reference labels or credentials are sent to the service.
    revision = HfApi(token=False).model_info(config["model_id"], timeout=10).sha
    snapshot_download(repo_id=config["model_id"], revision=revision, token=False,
                      local_dir=directory, cache_dir=root / "data/models/hub_cache",
                      allow_patterns=["*.json", "*.txt", "model.safetensors", "README.md"],
                      max_workers=4)
    files = [p for p in directory.rglob("*") if p.is_file() and ".cache" not in p.parts]
    lock = {"model_id": config["model_id"], "revision": revision,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "file_sha256": {str(p.relative_to(directory)): digest(p) for p in sorted(files)},
            "use": "Frozen retrospective representation; original pretraining history is not reconstructed."}
    save_json(directory / "encoder_manifest.json", lock)
    return verify_model(directory)


if __name__ == "__main__":
    manifest = download()
    print(json.dumps({k: v for k, v in manifest.items() if k != "file_sha256"}, indent=2))
    print("Verified files:", len(manifest["file_sha256"]))
