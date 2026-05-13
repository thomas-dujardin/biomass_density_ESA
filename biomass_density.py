import os
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path

import ee
import geemap

from tqdm import tqdm
import numpy as np
import torch
torch.manual_seed(42)
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import torchvision.models as models

# =========================================================
# Optional label (re)computation
# =========================================================
if not Path("tile_labels.pt").exists():
    try:
        from tiles_download import precompute_stratification_labels  # noqa: F401
    except Exception:
        pass

# =========================================================
# Earth Engine init (should contain 504 tiles with GEDI (LiDAR) and ESA data (Sentinel-1 SAR, Sentinel-2 RGB-MS))
# =========================================================

# /!\ WARNING: Copernicus-FM v1 does NOT SUPPORT LiDAR data.
# Instructions for a possible (rudimentary) LiDAR data integration are provided in the README.md file

ee.Authenticate()
ee.Initialize(project="project5324-448512")

# =========================================================
# Training args (experimentation, losses coefficients, GPU, freeze/unfreeze the encoder, ablate components, ...)
# =========================================================

parser = argparse.ArgumentParser(
    prog="biomass_density.py",
    description="Trains a model made of 3 main components: Copernicus-FM encoder, Conv2D decoder, optional U-Net refiner",
)

# Random seed for reproducibility; can be modified for inference time benchmarking I suppose:
parser.add_argument("--random_seed", type=int, default=42)

# Whether or not to use GPU (does not support multi-GPU)
parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)

# Whether or not to use the optional U-Net refiner (decreases the RMSE, but slows down training and inference)
parser.add_argument("--refiner_on", action=argparse.BooleanOptionalAction, default=True)

# Whether the refiner, a UResNet34 backbone, should be pretrained on ImageNet-21k (1k?). Random init otherwise.
parser.add_argument("--refiner_random_init", action=argparse.BooleanOptionalAction, default=True)

# Coefficients in front of each loss. /!\ It is recommended to keep these as they are.

# Given the nature of ESA's ground truth data (1-channel 264x264 images with ~8x8 "pixels", 100m resolution),
# A differentiable interpolation is computed on the output of the model to obtain
# a "pixelated" version (to match pixel size) that can match a biomass density map  and three losses are proposed:

# - L1 loss: the pixelwise sum of the differences between the predicted and the ground-truth biomass density maps;
# Beats Huber loss, performs better than the 2 other losses each taken alone;

# GT data is originally a few pixels wider than 264x264 in both directions, and the excessary pixels are erroneous, thus creating artifacts on the edges;
# L1 loss largely fixes these artifacts, and also encourages sparsification of the output map.
parser.add_argument("--L1_loss_coeff", type=float, default=1.0)

# - Total Variation loss: tries to impose the GT edges to the predicted output through exponential weighting.
# Does not produce the desired effects; can be discarded.
parser.add_argument("--edge_aware_TV_loss_coeff", type=float, default=0.0)

# - Laplacian loss: tries to fit the "pixels" from the output and the GT maps through L1 loss;
# Ablation study shows that its combination with the L1 loss is both quantitatively (better RMSE) and qualitatively (biomass density map emulation) better

# Discrete Laplacian filter (convolution kernel [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) approximates the 2D Laplacian operator in the pixel spaces;
# Applied to both the pixelized prediction and the GT, then the L1 distance between the two is returned;
# The point is to emphasize the edges of the "pixels" of the pixelized maps and to match them with those of the GT (hence the L1 distance)
parser.add_argument("--laplacian_loss_coeff", type=float, default=1.0)

# Training batch size
parser.add_argument("--train_batch_size", type=int, default=4)

# Total number of epochs
parser.add_argument("--nr_epochs", type=int, default=20)

# When to unfreeze the refiner (the UResNet34 is pretrained, but letting the upper Conv2d layer learn on its own for a bit can be beneficial for performance)
parser.add_argument("--freeze_refiner_before_epoch_nr", type=int, default=5)

# Experiment 1: unfreezes every component of the model, with both the Copernicus-FM v1 and refiner-UResNet34 pretrained
parser.add_argument("--train_everything", action=argparse.BooleanOptionalAction, default=False)

# Experiment 2: randomly initialize the weights of Copernicus-FM v1 (i.e. train Copernicus-FM v1 on biomass data from scratch)
parser.add_argument("--random_init_copernicus", action=argparse.BooleanOptionalAction, default=False)

# Experiment 3: freeze the pretrained refiner. Train the rest of the model (pretrained Copernicus-FM v1)
parser.add_argument("--train_everything_except_refiner", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

# GPU number 0 by default. Does not support multi-GPU training
device = torch.device("cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu")

# =========================================================
# Utilities
# =========================================================

def get_lr(optimizer):
    """
    Writes the current LR corresponding to the current optimizer (AdamW with 1e-4 weight decay) for Tensorboard;
    (LR is scheduled using cosine annealing, starting from 3e-4 because of the pretrained component).
    """

    return optimizer.param_groups[0]["lr"]

def save_tensor_as_image(tensor, path):
    """
    Debug mode: in case a quick illustration of the GT data issue is required.
    tensor: a 2D (H, W) or 3D (C, H, W) tensor;
    path: where to save the resulting image.
    """

    tensor = tensor.clone().detach().cpu()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 3:
        raise ValueError(f"This function requires (C, H, W) or (H, W) tensors")

    if tensor.shape[0] > 3:
        # Retains RGB only
        tensor = tensor[:3]

    # Normalization
    tmin = tensor.min()
    tmax = tensor.max()
    tensor = (tensor - tmin) / (tmax - tmin + 1e-8)

    save_image(tensor, path)


def crop_center(img, cropx=264, cropy=264):
    """
    Supports:
      - [C,H,W]
      - [Batch,C,H,W]
    img: 3 or 4-dimensional torch tensor. GT has excessory pixels in both directions, cropping is necessary.
    """
    if img.ndim == 3:
        _, H, W = img.shape
        startx = max(0, W // 2 - cropx // 2)
        starty = max(0, H // 2 - cropy // 2)
        endx = min(W, startx + cropx)
        endy = min(H, starty + cropy)
        return img[:, starty:endy, startx:endx]

    if img.ndim == 4:
        _, _, H, W = img.shape
        startx = max(0, W // 2 - cropx // 2)
        starty = max(0, H // 2 - cropy // 2)
        endx = min(W, startx + cropx)
        endy = min(H, starty + cropy)
        return img[:, :, starty:endy, startx:endx]


# =========================================================
# GT border artifact removal
# =========================================================
def to_2d(x: torch.Tensor):
    """
    Work on a 2D tensor. Helper function for GT border artifact removal.
    """
    if x.ndim == 2:
        return x, False
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0], True


def texture_map(x2d: torch.Tensor, k: int = 9) -> torch.Tensor:
    """
    Texture score = local std + local gradient magnitude.
    The point is to detect local activity in the image (local autosimilarity in the pixel space) to detect flat zones, which the GT edges happen to possess.
    """
    x = x2d[None, None]  # [1,1,H,W]

    mean = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    mean2 = F.avg_pool2d(x * x, kernel_size=k, stride=1, padding=k // 2)
    var = (mean2 - mean * mean).clamp_min(0.0)
    std = var.sqrt()

    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[:, :, :, 1:] = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy[:, :, 1:, :] = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

    grad = dx + dy
    grad = F.avg_pool2d(grad, kernel_size=k, stride=1, padding=k // 2)

    tex = std + grad
    return tex[0, 0]


def find_content_bounds(
    tex: torch.Tensor,
    min_activity: float = 0.06,
    center_frac: float = 0.4,
    smooth_1d: int = 11,
):
    H, W = tex.shape

    ch0 = int(H * (0.5 - center_frac / 2))
    ch1 = int(H * (0.5 + center_frac / 2))
    cw0 = int(W * (0.5 - center_frac / 2))
    cw1 = int(W * (0.5 + center_frac / 2))
    center = tex[ch0:ch1, cw0:cw1]

    thr = max(float(center.mean()) * 0.35, float(center.median()) * 0.75, 1e-6)
    active = (tex > thr).float()

    row_score = active.mean(dim=1)
    col_score = active.mean(dim=0)

    def smooth_1d_signal(v):
        """
        The point is to detect a very specific artifacts patterns, hence the value for smooth_1d AND the 1D pooling.
        """
        vv = v[None, None, :]
        out = F.avg_pool1d(vv, kernel_size=smooth_1d, stride=1, padding=smooth_1d // 2)
        return out[0, 0]

    row_score = smooth_1d_signal(row_score)
    col_score = smooth_1d_signal(col_score)

    row_idx = torch.where(row_score >= min_activity)[0]
    col_idx = torch.where(col_score >= min_activity)[0]

    if len(row_idx) == 0 or len(col_idx) == 0:
        return 0, H, 0, W

    top = int(row_idx[0].item())
    bottom = int(row_idx[-1].item()) + 1
    left = int(col_idx[0].item())
    right = int(col_idx[-1].item()) + 1

    # All 3 types of GT artifacts can be isolated that way.

    top = max(0, top - 1)
    left = max(0, left - 1)
    bottom = min(H, bottom + 1)
    right = min(W, right + 1)

    return top, bottom, left, right


def pad_to_target_same_side_bias(
    x: torch.Tensor,
    target_h: int,
    target_w: int,
    trim_top: int,
    trim_bottom: int,
    trim_left: int,
    trim_right: int,
    mode: str = "replicate",
):
    """
    x: [1,H,W] or [H,W]
    """
    had_channel = (x.ndim == 3)
    if x.ndim == 2:
        x = x.unsqueeze(0)

    _, H, W = x.shape

    if H > target_h or W > target_w:
        start_y = max(0, (H - target_h) // 2)
        start_x = max(0, (W - target_w) // 2)
        x = x[:, start_y:start_y + min(H, target_h), start_x:start_x + min(W, target_w)]
        _, H, W = x.shape

    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)

    if trim_top + trim_bottom > 0:
        pad_top = int(round(pad_h * trim_top / (trim_top + trim_bottom)))
    else:
        pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    if trim_left + trim_right > 0:
        pad_left = int(round(pad_w * trim_left / (trim_left + trim_right)))
    else:
        pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)

    if had_channel:
        return x
    return x[0]


def remove_edge_artifacts_and_pad(
    mask_tensor: torch.Tensor,
    target_hw=(264, 264),
    texture_kernel: int = 3,
    min_activity: float = 0.06,
    center_frac: float = 0.4,
    pad_mode: str = "replicate",
    return_bbox: bool = False,
):
    """
    mask_tensor: [H,W] or [1,H,W]
    Attempts to remove the faulty edges from the GT maps. They exhibit a pattern that should make it easy, but I did not manage to remove these artifacts yet;
    /!\ High priority fix to avoid the RMSE to plateau.
    """
    x2d, had_channel = to_2d(mask_tensor.float())

    tex = texture_map(x2d, k=texture_kernel)
    top, bottom, left, right = find_content_bounds(
        tex,
        min_activity=min_activity,
        center_frac=center_frac,
    )

    cropped = x2d[top:bottom, left:right]

    # Is supposed to return an "aligned" GT map. Doesn't work yet.
    # /!\ HIGH priority fix.
    cleaned = pad_to_target_same_side_bias(
        cropped.unsqueeze(0),
        target_h=target_hw[0],
        target_w=target_hw[1],
        trim_top=top,
        trim_bottom=x2d.shape[0] - bottom,
        trim_left=left,
        trim_right=x2d.shape[1] - right,
        mode=pad_mode,
    )

    if not had_channel:
        cleaned = cleaned[0]

    if return_bbox:
        return cleaned, (top, bottom, left, right)
    return cleaned

# =========================================================
# Losses
# =========================================================

def laplacian_loss(pred, target):
    """
    pred, target: (B,1,H,W)
    Old Computer Vision: second-order differentials (discrete appx) in both x, y directions;
    Very sensitive to vertical and horizontal edges. We want our pixelized infered map to possess the GT's edges.
    """
    laplace_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, 3, 3)

    pred_lap = F.conv2d(pred, laplace_kernel, padding=1)
    tgt_lap = F.conv2d(target, laplace_kernel, padding=1)

    return F.l1_loss(pred_lap, tgt_lap)


def edge_aware_tv_loss(pred, target):
    """
    pred, target: (B,1,H,W)
    Discarded: significantly worsens the RMSE. Was meant to eliminate artifacts on the border of the image.
    """
    gx = target[:, :, :, 1:] - target[:, :, :, :-1]
    gy = target[:, :, 1:, :] - target[:, :, :-1, :]

    wx = torch.exp(-torch.abs(gx))
    wy = torch.exp(-torch.abs(gy))

    px = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    py = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    return (wx * px).mean() + (wy * py).mean()


def pixelize(x, scale=10):
    """
    x: (B, C, H, W)
    Important function: fits the GT map format, thus allowing actual biomass density maps inference.
    Differentiable!
    """
    B, C, H, W = x.shape
    h2 = max(1, H // scale)
    w2 = max(1, W // scale)
    small = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    pix = F.interpolate(small, size=(H, W), mode="nearest")
    return pix


# =========================================================
# EE tile access
# =========================================================
def get_tile_dict(index):
    # GEDI and ESA tiles
    image_asset_id = f"projects/project5324-448512/assets/FM_Patches/04022026v2/img_tile_{index}"
    mask_asset_id = f"projects/project5324-448512/assets/FM_Patches/04022026v2/esa_tile_{index}"

    image = ee.Image(image_asset_id)
    mask = ee.Image(mask_asset_id)

    geom = image.geometry().centroid()
    lon, lat = geom.coordinates().getInfo()

    scale = image.projection().nominalScale().getInfo()

    region = image.geometry()
    bounds = region.bounds()
    region = ee.Geometry.Polygon(bounds.getInfo()["coordinates"][0])

    # Input for the Copernicus-FM v1 Foundation Model. ESA does not have time intervals.

    return {
        "image": image,
        "mask": mask,
        "lat": lat,
        "lon": lon,
        "time": None,
        "scale": float(scale),
        "region": region,
    }


def convert_to_tensor_dict(tile_dict, debug_save=False, debug_prefix="debug"):
    """
    Annoying GEE format to cool torch.Tensor converter
    """
    region = tile_dict["region"]
    scale = tile_dict["scale"]

    image_arr = geemap.ee_to_numpy(tile_dict["image"], region=region, scale=scale)
    mask_arr = geemap.ee_to_numpy(tile_dict["mask"], region=region, scale=scale)

    image_arr = np.asarray(image_arr)
    mask_arr = np.asarray(mask_arr)

    image_arr = np.where(np.isneginf(image_arr), 0, image_arr)
    image_arr = np.nan_to_num(image_arr, nan=0.0, posinf=0.0, neginf=0.0)

    mask_arr = np.where(np.isneginf(mask_arr), 0, mask_arr)
    mask_arr = np.nan_to_num(mask_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # image: [H,W,C] -> [C,H,W]
    img_processed = np.moveaxis(image_arr, -1, 0).astype(np.float32)

    # mask may be [H,W] or [H,W,1]
    if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
        mask_processed = mask_arr[..., 0]
    elif mask_arr.ndim == 2:
        mask_processed = mask_arr
    else:
        mask_processed = np.squeeze(mask_arr)

    image_tensor = torch.tensor(img_processed, dtype=torch.float32)
    mask_tensor = torch.tensor(mask_processed, dtype=torch.float32).unsqueeze(0)

    if debug_save:
        save_tensor_as_image(mask_tensor, f"{debug_prefix}_mask_raw.png")

    mask_tensor = remove_edge_artifacts_and_pad(
        mask_tensor,
        target_hw=(264, 264),
        texture_kernel=3,
        min_activity=0.06,
        center_frac=0.4,
        pad_mode="replicate",
    )

    if debug_save:
        save_tensor_as_image(mask_tensor, f"{debug_prefix}_mask_clean.png")

    # normalize image per channel over H,W
    mean = image_tensor.mean(dim=(1, 2), keepdim=True)
    std = image_tensor.std(dim=(1, 2), keepdim=True)
    image_tensor = (image_tensor - mean) / (std + 1e-6)

    image_tensor = crop_center(image_tensor, 264, 264)

    # if image crop is smaller than 264 in any dimension, pad (never happens in practice)
    _, h, w = image_tensor.shape
    if h != 264 or w != 264:
        pad_h = max(0, 264 - h)
        pad_w = max(0, 264 - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image_tensor = F.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    lat = torch.tensor(tile_dict["lat"], dtype=torch.float32)
    lon = torch.tensor(tile_dict["lon"], dtype=torch.float32)
    time = torch.tensor([0.0], dtype=torch.float32)
    scale_tensor = torch.tensor(tile_dict["scale"], dtype=torch.float32)

    # Input dictionary, ready for Copernicus-FM v1.

    return {
        "image": image_tensor,
        "mask": mask_tensor,
        "lat": lat,
        "lon": lon,
        "time": time,
        "scale": scale_tensor,
    }


# =========================================================
# Dataset
# =========================================================

class GEDIBiomassDataset(Dataset):
    # Named "GEDIBiomassDataset" but does not consider GEDI data, only ESA RGB-MS-SAR data.
    def __init__(self, tile_indices, debug_first_n=0):
        self.tile_indices = tile_indices
        self.debug_first_n = debug_first_n

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, idx):
        tile_id = self.tile_indices[idx]
        tile_dict = get_tile_dict(tile_id)
        tensor_dict = convert_to_tensor_dict(
            tile_dict,
            debug_save=(idx < self.debug_first_n),
            debug_prefix=f"tile_{tile_id}"
        )
        return tensor_dict


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate(model, refiner, loader, device, max_batches=None):
    model.eval()
    if refiner is not None:
        refiner.eval()

    running = 0.0
    n = 0

    for b, batch in enumerate(loader):
        if (max_batches is not None) and (b >= max_batches):
            break

        x = batch["image"].to(device, non_blocking=True)
        meta = torch.stack(
            [batch["lon"], batch["lat"], batch["time"][:, 0], batch["scale"]],
            dim=1
        ).to(device)
        y = batch["mask"].to(device, non_blocking=True)

        wavelengths = [50000000, 50000000, 490, 560, 665, 705, 740, 783, 842, 860]
        bandwidths = [1e9, 1e9, 65, 35, 30, 15, 15, 20, 115, 20]

        blurry = model(x, meta, wavelengths, bandwidths, None, "spectral", 16)

        if refiner is not None:
            refined = pixelize(refiner(blurry))
            loss = args.L1_loss_coeff * F.l1_loss(refined, y) + args.edge_aware_TV_loss_coeff * edge_aware_tv_loss(refined, y) + args.laplacian_loss_coeff * laplacian_loss(refined, y)
        else:
            pred = pixelize(blurry)
            loss = args.L1_loss_coeff * F.l1_loss(pred, y) + args.edge_aware_TV_loss_coeff * edge_aware_tv_loss(pred, y) + args.laplacian_loss_coeff * laplacian_loss(pred, y)

        running += float(loss.item()) * x.size(0)
        n += x.size(0)

    return running / max(n, 1)


@torch.no_grad()
def evaluate_and_save_test(model, refiner, loader, device, out_dir):
    model.eval()
    if refiner is not None:
        refiner.eval()

    rows = []
    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="Test inference"):
        x = batch["image"].to(device)
        meta = torch.stack(
            [batch["lon"], batch["lat"], batch["time"][:, 0], batch["scale"]],
            dim=1
        ).to(device)
        y = batch["mask"].to(device, non_blocking=True)

        wavelengths = [50000000, 50000000, 490, 560, 665, 705, 740, 783, 842, 860]
        bandwidths = [1e9, 1e9, 65, 35, 30, 15, 15, 20, 115, 20]

        lat = batch["lat"].cpu().numpy()
        lon = batch["lon"].cpu().numpy()

        blurry = model(x, meta, wavelengths, bandwidths, None, "spectral", 16)

        if refiner is not None:
            pred = pixelize(refiner(blurry))
            loss = args.L1_loss_coeff * F.l1_loss(pred, y) + args.edge_aware_TV_loss_coeff * edge_aware_tv_loss(pred, y) + args.laplacian_loss_coeff * laplacian_loss(pred, y)
        else:
            pred = pixelize(blurry)
            loss = args.L1_loss_coeff * F.l1_loss(pred, y) + args.edge_aware_TV_loss_coeff * edge_aware_tv_loss(pred, y) + args.laplacian_loss_coeff * laplacian_loss(pred, y)

        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)

        pred_np = pred.cpu().numpy()
        y_np = y.cpu().numpy()

        for i in range(x.size(0)):
            p = pred_np[i, 0]
            t = y_np[i, 0]

            rows.append({
                "lat": float(lat[i]),
                "lon": float(lon[i]),
                "pred_mean": float(p.mean()),
                "target_mean": float(t.mean()),
                "rmse": float(np.sqrt(((p - t) ** 2).mean())),
                "mae": float(np.abs(p - t).mean()),
            })

    test_loss = total_loss / max(total_n, 1)
    rmses = [r["rmse"] for r in rows]
    maes = [r["mae"] for r in rows]

    metrics = {
        "test_loss": test_loss,
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "n_tiles": len(rows),
    }

    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(out_dir, "test_per_tile.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved test metrics -> {metrics_path}")
    print(f"Saved per-tile results -> {csv_path}")

    return metrics


# =========================================================
# Refiner
# =========================================================

# (B, C, H, W) -> Copernicus-FM v1 = (B, vit_encoded_patches_num, 768) ->
# UNet upper branch (Conv2Ds) = (B, 1, 264, 264) -> (optionally) UResNet34 refiner = (B, 1, 264, 264) ->
# pixelize = (B, 1, 264, 264) (final output map)

class UResNet34Refiner(nn.Module):
    """
    Residual U-ResNet34 refiner:
        refined = blurry + delta(blurry)

    Input:  (B,1,H,W)
    Output: (B,1,H,W)
    """
    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet34(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        def up(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GELU(),
            )

        self.up4 = up(512, 256)
        self.up3 = up(256 + 256, 128)
        self.up2 = up(128 + 128, 64)
        self.up1 = up(64 + 64, 64)

        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, b):
        b3 = b.repeat(1, 3, 1, 1)

        x0 = self.relu(self.bn1(self.conv1(b3)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.up4(x4)
        d4 = F.interpolate(d4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(torch.cat([d4, x3], dim=1))

        d3 = F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([d3, x2], dim=1))

        d2 = F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([d2, x1], dim=1))

        d1 = F.interpolate(d1, size=b.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.out(d1)

        return b + delta

# Given the small size of the batches, BatchNorm likely hurts performances;
# This blocks BatchNorm2d from acting on embeddings
def set_bn_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

def freeze_refiner_encoder(refiner: UResNet34Refiner, freeze: bool = True):
    enc_modules = [
        refiner.conv1,
        refiner.bn1,
        refiner.layer1,
        refiner.layer2,
        refiner.layer3,
        refiner.layer4,
    ]

    for m in enc_modules:
        for p in m.parameters():
            p.requires_grad = not freeze

    if freeze:
        for m in enc_modules:
            m.eval()
            for submodule in m.modules():
                if isinstance(submodule, nn.BatchNorm2d):
                    submodule.eval()

# =========================================================
# Copernicus decoder / wrapper
# =========================================================
class Decoder264(nn.Module):
    """
    Decoder that maps:
        (B, N, 768) -> (B, 1, 264, 264)
    where N = 16*16 = 256 (constrained by ViT-B's 16x16 native patches size, but upsampling is differentiable anyway)
    """
    def __init__(self):
        super().__init__()

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GELU(),
            )

        self.up1 = up_block(768, 256)
        self.up2 = up_block(256, 128)
        self.up3 = up_block(128, 64)

        self.up4 = nn.Sequential(
            nn.Upsample(size=(264, 264), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, tokens):
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)

        x = tokens.transpose(1, 2).reshape(B, C, H, W)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = x + self.refine(x)
        x = self.out(x)

        return x


class BiomassModel(nn.Module):
    # Combines all components into one BiomassModel class
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, meta, wavelengths, bandwidths, language_embed, input_mode, kernel_size):
        x = crop_center(x, cropx=264, cropy=264)
        _, feat = self.encoder(x, meta, wavelengths, bandwidths, language_embed, input_mode, kernel_size)
        out = self.decoder(feat)
        return out


# =========================================================
# Train
# =========================================================

# Helper function for BatchNorm2d

def keep_refiner_bn_eval(refiner):
    for m in refiner.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def train(
    encoder,
    decoder,
    refiner,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=50,
    lr=3e-4,
    weight_decay=1e-4,
    eta_min=1e-6,
    log_every=1,
    eval_every=1,
    out_dir="runs_biomass",
    freeze_refiner_epochs=3,
    freeze_refiner_all=False
):
    model = BiomassModel(encoder, decoder).to(device)
    refiner = refiner.to(device) if refiner is not None else None

    if refiner is not None:
        if freeze_refiner_all:
            for p in refiner.parameters():
                p.requires_grad = False
        else:
            freeze_refiner_encoder(refiner, freeze=True)

    params = []

    if any(p.requires_grad for p in model.encoder.parameters()):
        params += list(model.encoder.parameters())

    params += list(model.decoder.parameters())

    if refiner is not None:
        params += [p for p in refiner.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay
    )

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=eta_min
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(out_dir, run_name)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)

    print(f"To run tensorboard: tensorboard --logdir {out_dir}")

    global_step = 0
    best_val = float("inf")
    best_path = os.path.join(tb_dir, "best.pt")

    # Let the first 2 components warm-up before activating the refiner;
    # /!\ a few tests suggests that an optimal "freeze_refiner_epochs" value...
    # ... might be between 1/4th and 1/5th of the total number of epochs;
    # Perhaps theoretical works shed light on this specific question, and even provide higher and lower bounds...
    # ...Intuition would suggest that you need to let the first 2 components get near plateau to activate the refiner.

    for epoch in range(1, epochs + 1):
        if (
            refiner is not None
            and not freeze_refiner_all
            and epoch == (freeze_refiner_epochs + 1)
        ):
            freeze_refiner_encoder(refiner, freeze=False)

        model.train()

        if refiner is not None:
            refiner.train()
            keep_refiner_bn_eval(refiner)

            if epoch <= freeze_refiner_epochs:
                freeze_refiner_encoder(refiner, freeze=True)
                keep_refiner_bn_eval(refiner)

        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            meta = torch.stack(
                [batch["lon"], batch["lat"], batch["time"][:, 0], batch["scale"]],
                dim=1
            ).to(device)

            wavelengths = [50000000, 50000000, 490, 560, 665, 705, 740, 783, 842, 860]
            bandwidths = [1e9, 1e9, 65, 35, 30, 15, 15, 20, 115, 20]

            blurry = model(
                x,
                meta,
                wavelengths,
                bandwidths,
                language_embed=None,
                input_mode="spectral",
                kernel_size=16,
            )

            if refiner is not None:
                refined = refiner(blurry)
                pred = pixelize(refined)

                loss_main = F.l1_loss(pred, y)
                loss = (
                    float(args.L1_loss_coeff) * loss_main
                    + float(args.edge_aware_TV_loss_coeff) * edge_aware_tv_loss(pred, y)
                    + float(args.laplacian_loss_coeff) * laplacian_loss(pred, y)
                )
            else:
                pred = pixelize(blurry)

                loss_main = F.l1_loss(pred, y)
                loss = (
                    float(args.L1_loss_coeff) * loss_main
                    + float(args.edge_aware_TV_loss_coeff) * edge_aware_tv_loss(pred, y)
                    + float(args.laplacian_loss_coeff) * laplacian_loss(pred, y)
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            bs = x.size(0)
            running += loss.item() * bs
            n += bs

            if global_step % log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", get_lr(optimizer), global_step)

                with torch.no_grad():
                    gt = y[0, 0]
                    b0 = blurry[0, 0]
                    p0 = pred[0, 0]

                    def norm(v):
                        return (v - v.min()) / (v.max() - v.min() + 1e-6)

                    writer.add_image("train/gt", norm(gt).unsqueeze(0), global_step)
                    writer.add_image("train/blurry", norm(b0).unsqueeze(0), global_step)
                    writer.add_image("train/pred_pixelized", norm(p0).unsqueeze(0), global_step)

                    if refiner is not None:
                        r0 = refined[0, 0]
                        writer.add_image(
                            "train/refined",
                            pixelize(norm(r0).unsqueeze(0).unsqueeze(0)).squeeze(0),
                            global_step
                        )

            global_step += 1
            pbar.set_postfix(loss=running / max(n, 1))

        train_loss = running / max(n, 1)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)

        if val_loader is not None and epoch % eval_every == 0:
            val_loss = evaluate(model, refiner, val_loader, device)
            writer.add_scalar("val/loss", val_loss, epoch)

            print(f"[Epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss

                state = {
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "val": best_val,
                }

                if refiner is not None:
                    state["refiner"] = refiner.state_dict()

                torch.save(state, best_path)

    writer.close()
    return best_val, model, refiner, tb_dir


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    tile_indices = list(range(503)) # 504 tiles in total over the provided mask in the GEE project.
    base_dataset = GEDIBiomassDataset(tile_indices)
    print("Base dataset size:", len(base_dataset))

    tile_labels = torch.load("tile_labels.pt")
    tile_labels_np = tile_labels.numpy()
    indices = np.arange(len(base_dataset))

    # 80 / 10 / 10, train / val / test split

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        tile_labels_np,
        test_size=0.2,
        stratify=tile_labels_np,
        random_state=args.random_seed,
    )

    val_idx, test_idx, val_y, test_y = train_test_split(
        temp_idx,
        temp_y,
        test_size=0.5,
        stratify=temp_y,
        random_state=args.random_seed,
    )

    print("Split sizes:", len(train_idx), len(val_idx), len(test_idx))

    train_dataset = GEDIBiomassDataset(train_idx.tolist(), debug_first_n=0)
    val_dataset = GEDIBiomassDataset(val_idx.tolist(), debug_first_n=0)
    test_dataset = GEDIBiomassDataset(test_idx.tolist(), debug_first_n=0)

    train_labels = tile_labels[train_idx]
    class_counts = torch.bincount(train_labels)
    print("Train class counts:", class_counts.tolist())

    # Was useful when the provided data was raster data. Keeping this just in case.
    weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    # There are theoretical derivations of the optimal batch size;
    # /!\ They involve LR and number of steps, hence number of epochs /!\;
    # But in our case, [4;8] seems to work better than 16 or more.

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        # sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    from src.model_vit import vit_base_patch16

    # vit_base_patch16 is MODIFIED (in this repo!) to return the sequence of patches;
    # except for [CLS], instead of just [CLS]: this boosts performances on the task at hand.

    # Randomly initialized

    encoder = vit_base_patch16()

    # /!\ The godfather of all init ViT fct (ChatGPT-5.5 generated)

    def init_vit_random_weights_sota(
        model: nn.Module,
        token_std: float = 0.02,
        head_std: float = 1e-3,
        conv_mode: str = "kaiming",
        linear_mode: str = "xavier",
    ):
        """
        Strong modern random initialization for ViT-like models.

        Recommended defaults:
        - Patch embedding Conv2d: Kaiming normal
        - Transformer Linear layers: Xavier uniform
        - LayerNorm: weight = 1, bias = 0
        - Positional / CLS / mask / query tokens: truncated normal N(0, token_std)
        - Heads: small truncated normal N(0, head_std)

        Args:
            model:
                ViT-like nn.Module.

            token_std:
                Std for positional embeddings, cls tokens, mask tokens, etc.

            head_std:
                Std for final prediction heads.

            conv_mode:
                "kaiming" or "xavier" for Conv2d layers.

            linear_mode:
                "xavier" or "trunc_normal" for Linear layers.

        Returns:
            The same model, initialized in-place.
        """

        def is_head_name(name: str) -> bool:
            lname = name.lower()
            return any(
                k in lname
                for k in [
                    "head",
                    "classifier",
                    "fc_norm",
                    "pre_logits",
                    "prediction",
                    "lm_head",
                ]
            )

        def is_special_token_name(name: str) -> bool:
            lname = name.lower()
            return any(
                k in lname
                for k in [
                    "pos_embed",
                    "position_embedding",
                    "position_embeddings",
                    "absolute_pos_embed",
                    "cls_token",
                    "dist_token",
                    "mask_token",
                    "query_token",
                    "query_tokens",
                    "register_token",
                    "register_tokens",
                ]
            )

        with torch.no_grad():
            # ------------------------------------------------------------------
            # 1. Initialize standard modules
            # ------------------------------------------------------------------
            for module_name, module in model.named_modules():

                if isinstance(module, nn.Linear):
                    if is_head_name(module_name):
                        nn.init.trunc_normal_(
                            module.weight,
                            mean=0.0,
                            std=head_std,
                            a=-2 * head_std,
                            b=2 * head_std,
                        )
                    else:
                        if linear_mode == "xavier":
                            nn.init.xavier_uniform_(module.weight)
                        elif linear_mode == "trunc_normal":
                            nn.init.trunc_normal_(
                                module.weight,
                                mean=0.0,
                                std=token_std,
                                a=-2 * token_std,
                                b=2 * token_std,
                            )
                        else:
                            raise ValueError(f"Unknown linear_mode: {linear_mode}")

                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                elif isinstance(module, nn.Conv2d):
                    if conv_mode == "kaiming":
                        nn.init.kaiming_normal_(
                            module.weight,
                            mode="fan_out",
                            nonlinearity="relu",
                        )
                    elif conv_mode == "xavier":
                        nn.init.xavier_uniform_(module.weight)
                    else:
                        raise ValueError(f"Unknown conv_mode: {conv_mode}")

                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                elif isinstance(module, nn.LayerNorm):
                    if module.weight is not None:
                        nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                elif isinstance(module, nn.BatchNorm2d):
                    if module.weight is not None:
                        nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                elif isinstance(module, nn.GroupNorm):
                    if module.weight is not None:
                        nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            # ------------------------------------------------------------------
            # 2. Initialize standalone ViT parameters
            #    These are usually not part of Linear / Conv / LayerNorm modules.
            # ------------------------------------------------------------------
            module_param_ids = set()

            for module in model.modules():
                for param in module.parameters(recurse=False):
                    module_param_ids.add(id(param))

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                lname = name.lower()

                # Positional embeddings, CLS token, register tokens, query tokens, etc.
                if is_special_token_name(name):
                    nn.init.trunc_normal_(
                        param,
                        mean=0.0,
                        std=token_std,
                        a=-2 * token_std,
                        b=2 * token_std,
                    )

                # Final standalone head weights, if any.
                elif is_head_name(name) and param.ndim >= 2:
                    nn.init.trunc_normal_(
                        param,
                        mean=0.0,
                        std=head_std,
                        a=-2 * head_std,
                        b=2 * head_std,
                    )

                # Bias-like standalone parameters.
                elif param.ndim == 1:
                    if "norm" in lname:
                        nn.init.ones_(param)
                    else:
                        nn.init.zeros_(param)

                # Any unusual standalone matrix/tensor not covered above.
                elif id(param) not in module_param_ids:
                    nn.init.xavier_uniform_(param)

        return model

    def model_checksum(model):
        total = 0.0
        with torch.no_grad():
            for p in model.parameters():
                total += p.float().abs().sum().item()
        return total

    if args.random_init_copernicus:
        print("Before random init:", model_checksum(encoder))
        encoder = init_vit_random_weights_sota(
            encoder,
            token_std=0.02,
            head_std=1e-3,
        )
        print("After random init:", model_checksum(encoder))
    else:
        path = "CopernicusFM_ViT_base_varlang_e100.pth"
        check_point = torch.load(path, map_location="cpu")
        state_dict = check_point["model"] if "model" in check_point else check_point
        msg = encoder.load_state_dict(state_dict, strict=False)
        print("Loaded Copernicus-FM checkpoint:", msg)

    # Simple upper Conv2D network. Output tensor size: [B, C, H, W] where B = batch size (args.train_batch_size), C = 1 (grayscale biomass density map), H = W = 256
    decoder = Decoder264()

    if args.train_everything:
        for param in encoder.parameters():
            param.requires_grad = True
    else:
        for param in encoder.parameters():
            param.requires_grad = False

    encoder.to(device)
    decoder.to(device)

    refiner = None

    if args.refiner_on:
        refiner = UResNet34Refiner(pretrained=not args.refiner_random_init)

    best_val, model, refiner, run_dir = train(
        encoder=encoder,
        decoder=decoder,
        refiner=refiner,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.nr_epochs,
        lr=3e-4,
        weight_decay=1e-4,
        eta_min=1e-6,
        out_dir="runs_biomass",
        freeze_refiner_epochs=args.freeze_refiner_before_epoch_nr,
    )

    print("\nTraining complete")
    print("Best val:", best_val)

    best_ckpt_path = os.path.join(run_dir, "best.pt")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)

    model.encoder.load_state_dict(best_ckpt["encoder"])
    model.decoder.load_state_dict(best_ckpt["decoder"])

    if refiner is not None and "refiner" in best_ckpt:
        refiner.load_state_dict(best_ckpt["refiner"])

    print(
        f"Loaded best checkpoint from epoch {best_ckpt['epoch']} "
        f"with val={best_ckpt['val']:.6f}"
    )

    test_metrics = evaluate_and_save_test(
        model=model,
        refiner=refiner,
        loader=test_loader,
        device=device,
        out_dir=run_dir,
    )

    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))