"""Launch the original model-backed service on an HTTP hosting platform."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from urllib.request import urlopen


def prepare_model() -> Path:
    destination = Path(os.getenv("CLIP_MODEL_PATH", "./models/best_retrieval_clip.pth"))
    if not destination.is_file():
        url = os.getenv("MODEL_DOWNLOAD_URL", "")
        if not url.startswith("https://"):
            raise RuntimeError("Set MODEL_DOWNLOAD_URL to a trusted HTTPS model URL, or mount CLIP_MODEL_PATH.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(url, timeout=120) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise RuntimeError("Model download failed; check the server-side URL and network access.") from None
    expected = os.getenv("MODEL_SHA256", "").strip().lower()
    if expected:
        with destination.open("rb") as model_file:
            actual = hashlib.file_digest(model_file, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError("Model SHA-256 does not match MODEL_SHA256.")
    return destination


def main() -> None:
    model = prepare_model()
    os.environ["MODEL_FINETUNED_MODEL_PATH"] = str(model)
    os.environ["APP_HOST"] = "0.0.0.0"
    os.environ["APP_PORT"] = os.getenv("PORT", "5000")
    os.environ["GALLERY_CAD_DRAWING_DIR"] = os.getenv("GALLERY_DIR", "./demo_gallery")
    from main import main as run_original_service
    run_original_service()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
