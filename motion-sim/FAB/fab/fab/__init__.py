"""FAB: Fast and Accurate Body estimation.

    from fab.predictor import FABPredictor
    fab = FABPredictor("fab-l", fast=True)
    people = fab.predict(image_rgb)
"""
import os
import warnings

__version__ = "1.0.0"

# Known, harmless warnings raised from FAB's MHR code (docs/troubleshooting.md, "Messages in the log"): the optional
# pymomentum package is not used (fab/models/heads/mhr_fast.py evaluates the rig), and the rig's correctives are sparse
# tensors on purpose.
warnings.filterwarnings("ignore", message="Momentum is not enabled", module=r"fab\.models\.heads\.mhr_head")
warnings.filterwarnings("ignore", message="Sparse (invariant checks are implicitly disabled|CSR tensor support is in beta)",
                        module=r"fab\.models\.heads\.mhr_fast")

SAM3D_REPO = "facebook/sam-3d-body-dinov3"
SAM3D_CONFIG = "model_config.yaml"
MHR_ASSETS = "https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"
MHR_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"


def __getattr__(name):  # heavy imports on first use
    if name == "FABPredictor":
        from .predictor import FABPredictor

        return FABPredictor
    if name in ("FABImage", "load_fab_image"):
        from . import model

        return getattr(model, name)
    if name in ("load_sam_3d_body", "load_sam_3d_body_hf"):
        from . import build_models

        return getattr(build_models, name)
    if name == "SAM3DBodyEstimator":
        from .sam_3d_body_estimator import SAM3DBodyEstimator

        return SAM3DBodyEstimator
    raise AttributeError(name)


def download_sam3d(dest=None):
    """The two SAM 3D Body files FAB needs: the model config from the Hugging Face hub (the repository is gated under
    the SAM License: accept it on huggingface.co/facebook/sam-3d-body-dinov3 and log in with `hf auth login` first)
    and the MHR model from Meta's MHR release (Apache-2.0; fetch_mhr_model). SAM 3D Body's network weights (model.ckpt)
    are not needed for inference: every FAB model.pt holds all of its weights."""
    import shutil

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    from . import paths

    dest = str(dest or paths.sam3d_dir())
    existed = os.path.isdir(dest)
    try:
        hf_hub_download(SAM3D_REPO, SAM3D_CONFIG, local_dir=dest)
    except (GatedRepoError, RepositoryNotFoundError) as e:  # licence not accepted on the hub, or not logged in
        if not existed:  # nothing but the hub's partial download in it
            shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"could not download {SAM3D_CONFIG} from the gated repository "
                           f"https://huggingface.co/{SAM3D_REPO} ({type(e).__name__}): accept the SAM licence on that "
                           "page, run `hf auth login`, then run fab.download_sam3d() again") from None
    fetch_mhr_model(os.path.join(dest, "assets", "mhr_model.pt"))
    return dest


def fetch_mhr_model(path, url=MHR_ASSETS, sha256=MHR_SHA256):
    """assets/mhr_model.pt from the MHR release archive (only this member is kept), checked by sha256; an existing file
    with the right hash is kept."""
    import hashlib
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    def digest(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        return h.hexdigest()

    if os.path.exists(path) and digest(path) == sha256:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path) or ".") as tmp:
        archive = os.path.join(tmp, "assets.zip")
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z, z.open("assets/mhr_model.pt") as src:
            with open(os.path.join(tmp, "mhr_model.pt"), "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 24)
        got = digest(os.path.join(tmp, "mhr_model.pt"))
        if got != sha256:
            raise RuntimeError(f"{url}: mhr_model.pt sha256 {got} != {sha256}")
        os.replace(os.path.join(tmp, "mhr_model.pt"), path)
    return path
