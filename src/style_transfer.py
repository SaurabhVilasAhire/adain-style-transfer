"""Shared inference path used by both the Flask app and the CLI script:
load the frozen encoder + trained decoder once, then stylize any
content/style pair."""
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .models import Decoder, VGGEncoder
from .utils import adaptive_instance_normalization


def load_models(vgg_path: str, decoder_path: str, device: torch.device):
    """Load the frozen VGG encoder and the trained decoder, in eval mode."""
    encoder = VGGEncoder(vgg_path).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def stylize(
    content_image: Image.Image,
    style_image: Image.Image,
    encoder: VGGEncoder,
    decoder: Decoder,
    device: torch.device,
    alpha: float = 1.0,
    size: int = 512,
) -> torch.Tensor:
    """Run one content/style pair through encode -> AdaIN -> decode.

    alpha interpolates between the original content features (alpha=0) and
    the fully style-normalized features (alpha=1), giving a style-strength
    control knob without retraining anything.
    """
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    content = transform(content_image).unsqueeze(0).to(device)
    style = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feat = encoder(content, is_test=True)
        style_feat = encoder(style, is_test=True)
        stylized_feat = adaptive_instance_normalization(content_feat, style_feat)
        stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat
        output = decoder(stylized_feat)

    return output.clamp(0, 1)


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image = transforms.ToPILImage()(tensor.cpu().squeeze(0))
    image.save(path)
