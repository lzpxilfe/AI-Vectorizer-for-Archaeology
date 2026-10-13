"""Fixed RGB dual-head U-Net, with all optional dependencies loaded lazily."""

import hashlib
from pathlib import Path
from urllib.parse import urlparse


ARCHITECTURE = "resnet18-smp-unet-small-dual-head-v1"
INPUT_SIZE = 512
HEADS = ("contour_support", "other_ink_risk")
DECODER_CHANNELS = (128, 64, 32, 16, 16)
IMAGENET_RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def build_model(*, initialize_pretrained=False, encoder_weights_path=None,
                encoder_cache_directory=None):
    """Build locally; only explicit ImageNet initialization may fetch weights.

    Inputs are float32 RGB/255. ImageNet normalization is part of forward and
    therefore part of the exported graph. Both heads return full-size logits.
    A local encoder checkpoint can replace the explicit torchvision download.
    """
    import torch
    import segmentation_models_pytorch as smp

    if encoder_weights_path is not None and not initialize_pretrained:
        raise ValueError("local ImageNet weights require explicit initialization")

    class ContourUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = smp.Unet(
                encoder_name="resnet18", encoder_weights=None, encoder_depth=5,
                decoder_channels=DECODER_CHANNELS, in_channels=3, classes=2,
                activation=None,
            )
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.initialization_provenance = {"kind": "random", "pretrained": False}

        def forward(self, rgb):
            if not torch.jit.is_tracing() and (
                rgb.ndim != 4 or rgb.shape[1:] != (3, INPUT_SIZE, INPUT_SIZE)
            ):
                raise ValueError("model input must be N x 3 x 512 x 512 RGB/255")
            return self.network((rgb - self.imagenet_mean) / self.imagenet_std)

    model = ContourUNet()
    if initialize_pretrained:
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1
        if encoder_weights_path is None:
            cache = Path(encoder_cache_directory) if encoder_cache_directory else Path.cwd() / "work" / "learning-weights"
            torch.hub.set_dir(str(cache.absolute()))
            state = weights.get_state_dict(progress=True, check_hash=True)
            source = weights.url
            path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(source).path).name
        else:
            path = Path(encoder_weights_path).resolve(strict=True)
            if not path.is_file():
                raise ValueError("encoder weights must be a regular local file")
            state = torch.load(path, map_location="cpu", weights_only=True)
            source = "local:" + path.name
        weight_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        official_prefix = Path(urlparse(weights.url).path).stem.rsplit("-", 1)[-1]
        if weight_digest != IMAGENET_RESNET18_SHA256:
            raise ValueError("encoder checkpoint does not match official torchvision ImageNet weight hash")
        model.network.encoder.load_state_dict(state, strict=True)
        model.initialization_provenance = {
            "kind": "torchvision-resnet18-imagenet1k-v1", "pretrained": True,
            "source": source, "sha256": weight_digest,
            "official_weight_hash_prefix": official_prefix,
        }
    return model

