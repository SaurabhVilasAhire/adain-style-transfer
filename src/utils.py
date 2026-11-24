"""AdaIN math (the core trick of the whole method) and a plain
image-folder dataset/transform pipeline for training."""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def calc_mean_std(feat, eps: float = 1e-5):
    """Per-channel mean and std of a [N, C, H, W] feature map, computed
    over the spatial dimensions. eps avoids a div-by-zero on flat regions."""
    size = feat.size()
    assert len(size) == 4, f"expected a 4D [N,C,H,W] tensor, got shape {tuple(size)}"
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """AdaIN: normalize the content feature map to zero mean/unit std, then
    rescale it to the style feature map's mean/std. This single operation
    transfers style statistics without any learned parameters -- the
    decoder's job is just to invert the result back into a plausible image.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


class ImageFolderDataset(Dataset):
    """Flat folder of images -> tensors, used for both the content and
    style sides of training."""

    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.files = [p for p in os.listdir(root) if p.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        image = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def get_transform(size: int, crop: bool, final_size: int):
    """Resize (and optionally random-crop, for training-time augmentation)
    to a fixed square size, then convert to a tensor."""
    steps = []
    if size > 0:
        steps.append(transforms.Resize(size))
    steps.append(transforms.RandomCrop(final_size) if crop else transforms.Resize(final_size))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)
