"""CLI: apply a style image to a content image using the trained AdaIN
decoder. Runs fine on CPU -- a single pair at 256-512px takes well under a
second once the models are loaded.

Usage:
    python infer.py --content data/content/lenna.jpg --style data/style/mondrian.jpg \
        --output examples/output.png --alpha 1.0
"""
import argparse

import torch
from PIL import Image

from src.style_transfer import load_models, save_tensor_image, stylize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", required=True, help="Path to the content image")
    parser.add_argument("--style", required=True, help="Path to the style image")
    parser.add_argument("--output", default="examples/output.png", help="Where to save the result")
    parser.add_argument("--alpha", type=float, default=1.0, help="Style strength, 0 (none) to 1 (full)")
    parser.add_argument("--size", type=int, default=512, help="Square size images are resized to")
    parser.add_argument("--vgg", default="weights/vgg_normalised.pth")
    parser.add_argument("--decoder", default="weights/decoder_final.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = load_models(args.vgg, args.decoder, device)

    content_image = Image.open(args.content).convert("RGB")
    style_image = Image.open(args.style).convert("RGB")

    output = stylize(content_image, style_image, encoder, decoder, device, alpha=args.alpha, size=args.size)
    save_tensor_image(output, args.output)
    print(f"Saved stylized image to {args.output}")


if __name__ == "__main__":
    main()
