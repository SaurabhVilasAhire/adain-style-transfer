# AdaIN-Based Encoder-Decoder Network for Real-Time Style Transfer

PyTorch implementation of **Adaptive Instance Normalization (AdaIN)** style
transfer (Huang & Belongie, 2017), with a Flask web demo. Unlike optimization-based
neural style transfer, AdaIN needs a single forward pass through a frozen
VGG encoder + a small trained decoder — fast enough to run on CPU.

## How it works

1. **Encode** — a VGG-19 stack (frozen, pretrained, truncated at `relu4_1`)
   extracts features from both the content and style image (`src/models.py::VGGEncoder`).
2. **AdaIN** — the content feature map's per-channel mean/std are replaced
   with the style feature map's mean/std. This single, parameter-free
   operation transfers the "style statistics" without any extra network
   (`src/utils.py::adaptive_instance_normalization`). An `alpha` parameter
   interpolates between the original content (`alpha=0`) and full style
   transfer (`alpha=1`).
3. **Decode** — a small decoder (mirroring the encoder, trained from
   scratch) maps the AdaIN-normalized features back into an RGB image
   (`src/models.py::Decoder`). Only the decoder is trained; the encoder stays
   frozen throughout.
4. **Training loss** — a content loss (does the regenerated image's
   `relu4_1` features match the AdaIN target?) plus a style loss (do the
   regenerated image's features match the style image's mean/std at every
   encoder stage?) (`train.py`).

## What I actually ran

Unlike a lot of portfolio repos, this one doesn't need a GPU, so I ran it
myself rather than just reading the code:

- **Inference**, using the included pretrained decoder (`weights/decoder_final.pth`)
  — two real content/style pairs stylized end-to-end on CPU in under a
  second each. Results are in `examples/` and shown live in the Flask demo's
  "Examples" section.
- **Training**, as a smoke test on the handful of sample images in `data/`
  (`python train.py --final_size 64 --epochs 1 --batch_size 2`) — confirmed
  the full loop (dataloaders, AdaIN target, content/style loss, backward
  pass, checkpoint saving) runs without error and the loss decreases.
  A real training run (the one that produced `decoder_final.pth`) needs the
  full MS-COCO (content) + WikiArt (style) datasets from the original paper
  and meaningfully more epochs — that wasn't practical to redo here, so I'm
  using the pretrained decoder as shipped.
- **Unit tests** on the AdaIN math itself (`tests/test_utils.py`) — checks
  that the output's mean/std actually match the style image's, that AdaIN is
  a no-op when content equals style, and the mean/std computation itself
  against a manual calculation. All pass:

```bash
python -m pytest tests/test_utils.py -v
```

## Running it

```bash
pip install -r requirements.txt

# Stylize one image pair:
python infer.py --content data/content/lenna.jpg --style data/style/mondrian.jpg \
    --output examples/output.png --alpha 1.0

# Train (smoke-test scale by default; point at real datasets for a real run):
python train.py --content_dir data/content --style_dir data/style --epochs 1

# Web demo:
python app.py
# or, for production-style serving:
gunicorn --bind :5000 app:app
```

## Repo layout

```
app.py                  # Flask demo (upload content+style, get stylized result)
infer.py                # CLI: stylize one content/style pair
train.py                # CLI: train the decoder
src/
  models.py              # VGGEncoder, Decoder
  utils.py                # AdaIN math, dataset/transform helpers
  style_transfer.py       # shared load_models()/stylize() used by app.py and infer.py
templates/index.html     # Flask UI
tests/test_utils.py      # unit tests for the AdaIN math
weights/
  vgg_normalised.pth      # frozen VGG encoder weights
  decoder_final.pth        # pretrained decoder (included so inference works out of the box)
data/content, data/style  # a few sample images for the CLI/training smoke test
examples/                 # real outputs this repo produced, shown in the web demo
```

## Credits
Underlying method: Huang & Belongie,
["Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"](https://arxiv.org/abs/1703.06868) (2017).
