"""ADNI brain MRI data loading, preprocessing, and patient-level train/validation/test splitting.

Pipeline (see README Section 5):
    1. Index every slice and attach the patient ID from the ADNI metadata JSON.
    2. Split patients 80/10/10 into train/val/test, stratified by label, with a fixed seed.
       The provided train/test folders are ignored because they leak patients across splits.
    3. Zero-pad each 256 W x 240 H slice to 256 x 256, crop to a fixed square around the
       dataset-wide brain extent (ADNI_CROP, measured by data_audit.py), then resize
       (antialiased) to the working resolution.
    4. Preload all slices into memory as uint8 (optionally cached to disk) and scale to
       [-1, 1] per item.
    5. Drop corrupted scans whose mean intensity is below half the median scan mean.
       This happens after the patient split, so it never changes any patient's split.

Smoke test (run on a Rangpur CPU node):
    python dataset.py --resolution 64
    python dataset.py --resolution 64 --oasis    # also check the OASIS verification loader

Patient IDs are only ever held in memory. Nothing identifying is written to disk; the
on-disk cache contains preprocessed pixel data only and lives outside the repository.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ADNI_ROOT = "/home/groups/comp3710/ADNI"
OASIS_GLOB = "/home/groups/comp3710/OASIS/keras_png_slices_train/*.png"  # confirm against your Lab 2 path
DEFAULT_CACHE = os.path.expanduser("~/.cache/stylegan2_adni")

# Fixed brain crop (top, left, size) in padded 256 x 256 coordinates, or None.
# Measured by data_audit.py: midline sagittal slices nearly fill the frame, so a crop
# containing every brain is effectively the full image. Cropping is therefore disabled.
ADNI_CROP = None

# Scans whose mean intensity is below this fraction of the median scan mean are
# excluded as corrupted (two scans in the current dataset; see data_audit.py).
MIN_RELATIVE_SCAN_INTENSITY = 0.5
DEFAULT_SEED = 42
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
SPLITS = ("train", "val", "test")

CLASS_NAMES = ("CN", "AD")  # model labels: 0 = CN, 1 = AD
_JSON_TO_LABEL = {0: 0, 2: 1}  # metadata JSON: 0 = CN, 1 = MCI (not in this dataset), 2 = AD
_FOLDER_TO_LABEL = {"NC": 0, "AD": 1}
_SUBJECT_RE = re.compile(r"ADNI_(\d{3}_S_\d{4})")  # e.g. ADNI_123_S_4567 -> 123_S_4567


@dataclass(frozen=True)
class SliceRecord:
    """Metadata for one 2D slice."""

    path: str
    scan_id: str  # ADNI image ID (one MRI acquisition)
    subject_id: str  # ADNI patient ID (in memory only; never written to disk)
    label: int  # 0 = CN, 1 = AD
    slice_index: int  # absolute slice index from the filename
    slice_pos: int  # position within the scan's 20-slice window (0-19)


# --------------------------------------------------------------------------- #
# Indexing and splitting
# --------------------------------------------------------------------------- #
def index_adni(root=ADNI_ROOT):
    """Index every ADNI slice and attach its patient ID and label from the metadata JSON.

    Raises ValueError if a scan is missing from the metadata, has an unexpected label,
    or its folder label disagrees with the metadata (none of these occur in the
    current dataset, per data_audit.py).
    """
    with open(os.path.join(root, "meta_data_with_label.json")) as f:
        meta = json.load(f)

    raw = []
    for path in sorted(glob.glob(os.path.join(root, "AD_NC", "*", "*", "*.jpeg"))):
        folder = os.path.basename(os.path.dirname(path))
        scan_id, slice_index = os.path.basename(path)[: -len(".jpeg")].split("_")
        entry = meta.get(scan_id)
        if entry is None or entry["label"] not in _JSON_TO_LABEL:
            raise ValueError(f"Scan {scan_id} is missing from the metadata or has an unexpected label")
        label = _JSON_TO_LABEL[entry["label"]]
        if label != _FOLDER_TO_LABEL[folder]:
            raise ValueError(f"Scan {scan_id}: folder '{folder}' disagrees with metadata label")
        match = _SUBJECT_RE.search(entry["raw"])
        if match is None:
            raise ValueError(f"Scan {scan_id}: no patient ID found in metadata path")
        raw.append((path, scan_id, match.group(1), label, int(slice_index)))

    if not raw:
        raise FileNotFoundError(f"No ADNI slices found under {root}/AD_NC")

    window_start = {}
    for _, scan_id, _, _, idx in raw:
        window_start[scan_id] = min(window_start.get(scan_id, idx), idx)
    return [
        SliceRecord(path, scan_id, subject_id, label, idx, idx - window_start[scan_id])
        for path, scan_id, subject_id, label, idx in raw
    ]


def split_subjects(records, fractions=DEFAULT_FRACTIONS, seed=DEFAULT_SEED):
    """Assign each patient to exactly one split, stratified by label.

    Patients are sorted before shuffling with a seeded RNG, so the split is fully
    deterministic and can be regenerated without storing patient IDs.

    Returns a dict mapping subject_id -> split name ('train', 'val' or 'test').
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must be three values summing to 1")

    subject_label = {}
    for r in records:
        if subject_label.setdefault(r.subject_id, r.label) != r.label:
            raise ValueError("A patient has scans with different labels; stratification is ambiguous")

    rng = random.Random(seed)
    assignment = {}
    for label in sorted(set(subject_label.values())):
        subjects = sorted(s for s, lab in subject_label.items() if lab == label)
        rng.shuffle(subjects)
        n_val = round(len(subjects) * fractions[1])
        n_test = round(len(subjects) * fractions[2])
        n_train = len(subjects) - n_val - n_test
        for i, subject in enumerate(subjects):
            assignment[subject] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


# --------------------------------------------------------------------------- #
# Preprocessing and loading
# --------------------------------------------------------------------------- #
def pad_to_square(x):
    """Zero-pad a (C, H, W) tensor symmetrically to (C, S, S) with S = max(H, W)."""
    h, w = x.shape[-2:]
    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    return F.pad(x, (left, size - w - left, top, size - h - top), value=0)


def preprocess(x, resolution, crop=None):
    """Pad a uint8 (1, H, W) slice to square, optionally crop, and resize to uint8 (1, R, R).

    crop is (top, left, size) in padded coordinates, or None for no crop.
    """
    x = pad_to_square(x)
    if crop is not None:
        top, left, size = crop
        if top < 0 or left < 0 or top + size > x.shape[-2] or left + size > x.shape[-1]:
            raise ValueError(f"crop {crop} falls outside the padded {tuple(x.shape[-2:])} image")
        x = x[..., top:top + size, left:left + size]
    if x.shape[-1] != resolution:
        x = TF.resize(x.float(), [resolution, resolution], antialias=True)
        x = x.round().clamp(0, 255).to(torch.uint8)
    return x


class _SliceFiles(Dataset):
    """Reads and preprocesses image files; used only to parallelise preloading."""

    def __init__(self, paths, resolution, crop=None):
        self.paths, self.resolution, self.crop = paths, resolution, crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = TF.pil_to_tensor(im.convert("L"))  # uint8 (1, H, W)
        return preprocess(x, self.resolution, self.crop)


def default_num_workers():
    """CPU cores available to this process (respects Slurm allocations), capped at 8."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on Windows or macOS
        cores = os.cpu_count() or 1
    return min(8, cores)


def load_images(paths, resolution, cache_dir=DEFAULT_CACHE, num_workers=None, prefix="adni", crop=None):
    """Load, preprocess, and stack images into a uint8 tensor of shape (N, 1, R, R).

    Results are cached in cache_dir, keyed by the exact file list, crop, and resolution, so
    later runs load in seconds. Pass cache_dir=None to disable caching. Approximate
    cache sizes for ADNI: 125 MB at 64 px, 500 MB at 128 px, 2 GB at 256 px.
    """
    key = hashlib.sha1(("\n".join(paths) + f"\n{resolution}\n{crop}").encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"{prefix}_{resolution}px_{key}.pt") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        images = torch.load(cache_path)
        if images.shape[0] == len(paths):
            return images

    workers = default_num_workers() if num_workers is None else num_workers
    loader = DataLoader(_SliceFiles(paths, resolution, crop), batch_size=256, num_workers=workers)
    images = torch.cat(list(loader))

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = cache_path + ".tmp"
        torch.save(images, tmp_path)
        os.replace(tmp_path, cache_path)  # atomic: no half-written cache if the job dies
    return images


def find_dark_scans(records, images, min_relative=MIN_RELATIVE_SCAN_INTENSITY, chunk=2048):
    """Return scan IDs whose mean intensity is below min_relative x the median scan mean.

    Computed on the preloaded images in chunks (to bound memory at high resolution).
    Padding and resizing scale every scan's mean by nearly the same factor, so this
    relative rule gives the same result as the full-resolution check in data_audit.py.
    """
    slice_means = torch.cat([images[i:i + chunk].float().mean(dim=(1, 2, 3))
                             for i in range(0, images.shape[0], chunk)]).tolist()
    totals = defaultdict(lambda: [0.0, 0])
    for r, m in zip(records, slice_means):
        totals[r.scan_id][0] += m
        totals[r.scan_id][1] += 1
    scan_means = {scan: total / n for scan, (total, n) in totals.items()}
    threshold = min_relative * statistics.median(scan_means.values())
    return {scan for scan, m in scan_means.items() if m < threshold}


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class ADNISlices(Dataset):
    """Preprocessed ADNI slices for one split.

    __getitem__ returns (image, label): image is float32 (1, R, R) in [-1, 1] and
    label is 0 = CN or 1 = AD.

    Per-slice metadata for analysis (memorisation audit, slice-position checks) is
    exposed as tensors aligned with dataset indices: labels, scan_idx, subject_idx
    and slice_pos. The integer indices are local to this split.
    """

    def __init__(self, records, images, transform=None, excluded_scans=0, excluded_slices=0):
        if len(records) != images.shape[0]:
            raise ValueError("records and images must have the same length")
        self.records = records
        self.excluded_scans, self.excluded_slices = excluded_scans, excluded_slices
        self.images = images  # uint8 (N, 1, R, R)
        self.transform = transform

        scan_lookup = {s: i for i, s in enumerate(sorted({r.scan_id for r in records}))}
        subject_lookup = {s: i for i, s in enumerate(sorted({r.subject_id for r in records}))}
        self.labels = torch.tensor([r.label for r in records], dtype=torch.long)
        self.scan_idx = torch.tensor([scan_lookup[r.scan_id] for r in records], dtype=torch.long)
        self.subject_idx = torch.tensor([subject_lookup[r.subject_id] for r in records], dtype=torch.long)
        self.slice_pos = torch.tensor([r.slice_pos for r in records], dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        x = self.images[i].float() / 127.5 - 1.0
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i]

    def summary(self):
        """Patients, scans, and slices per class name."""
        out = {}
        for label, name in enumerate(CLASS_NAMES):
            recs = [r for r in self.records if r.label == label]
            out[name] = dict(
                patients=len({r.subject_id for r in recs}),
                scans=len({r.scan_id for r in recs}),
                slices=len(recs),
            )
        return out


def get_adni_splits(root=ADNI_ROOT, resolution=64, seed=DEFAULT_SEED, fractions=DEFAULT_FRACTIONS,
                    cache_dir=DEFAULT_CACHE, num_workers=None, crop=ADNI_CROP,
                    min_relative_intensity=MIN_RELATIVE_SCAN_INTENSITY):
    """Return {'train', 'val', 'test'} -> ADNISlices with a leakage-free patient-level split.

    Patients are split first, using every scan, and corrupted (dark) scans are dropped
    afterwards, so the exclusion never changes any patient's split assignment. Pass
    min_relative_intensity=None to keep every scan.
    """
    records = index_adni(root)
    images = load_images([r.path for r in records], resolution, cache_dir, num_workers, prefix="adni", crop=crop)
    assignment = split_subjects(records, fractions, seed)
    excluded = set() if min_relative_intensity is None else find_dark_scans(records, images, min_relative_intensity)

    splits = {}
    for name in SPLITS:
        in_split = [i for i, r in enumerate(records) if assignment[r.subject_id] == name]
        keep = [i for i in in_split if records[i].scan_id not in excluded]
        dropped = [records[i] for i in in_split if records[i].scan_id in excluded]
        splits[name] = ADNISlices([records[i] for i in keep], images[keep],
                                  excluded_scans=len({r.scan_id for r in dropped}), excluded_slices=len(dropped))
    return splits


class OASISSlices(Dataset):
    """OASIS brain slices, used only to verify GAN implementations against known-good Lab 2 results.

    Returns (image, 0) so it can be used interchangeably with ADNISlices in training loops.
    """

    def __init__(self, pattern=OASIS_GLOB, resolution=64, cache_dir=DEFAULT_CACHE, num_workers=None):
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No OASIS images match {pattern}")
        self.images = load_images(paths, resolution, cache_dir, num_workers, prefix="oasis")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        return self.images[i].float() / 127.5 - 1.0, torch.tensor(0)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def _check_no_leakage(splits):
    """Assert that no patient or scan appears in more than one split."""
    for attr in ("subject_id", "scan_id"):
        seen = {}
        for name, ds in splits.items():
            for value in {getattr(r, attr) for r in ds.records}:
                if value in seen:
                    raise AssertionError(f"{attr} found in both {seen[value]} and {name}")
                seen[value] = name
    print("Leakage check passed: every patient and scan belongs to exactly one split")


def _print_split_table(splits):
    print(f"\n{'split':<6} {'class':<5} {'patients':>8} {'scans':>6} {'slices':>7}")
    totals = defaultdict(int)
    for name, ds in splits.items():
        summary = ds.summary()
        for cls in CLASS_NAMES:
            s = summary[cls]
            print(f"{name:<6} {cls:<5} {s['patients']:>8} {s['scans']:>6} {s['slices']:>7}")
            for k, v in s.items():
                totals[k] += v
        ad_share = summary["AD"]["slices"] / len(ds)
        print(f"{name:<6} {'':5} {'':>8} {'':>6} {'':>7}  AD share of slices: {ad_share:.1%}")
    print(f"{'total':<6} {'':5} {totals['patients']:>8} {totals['scans']:>6} {totals['slices']:>7}")


def main():
    from torchvision.utils import save_image

    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Smoke test for the ADNI data pipeline.")
    p.add_argument("--root", default=ADNI_ROOT)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    p.add_argument("--no_cache", action="store_true", help="disable the on-disk preprocessing cache")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--oasis", action="store_true", help="also check the OASIS verification loader")
    p.add_argument("--oasis_glob", default=OASIS_GLOB)
    p.add_argument("--fig_dir", default=os.path.join(here, "figures"))
    args = p.parse_args()
    cache_dir = None if args.no_cache else args.cache_dir

    start = time.time()
    splits = get_adni_splits(args.root, args.resolution, args.seed, cache_dir=cache_dir,
                             num_workers=args.num_workers)
    print(f"Brain crop (top, left, size): {ADNI_CROP}" + ("  (no crop by design: see ADNI_CROP comment)" if ADNI_CROP is None else ""))
    print(f"Loaded ADNI at {args.resolution}x{args.resolution} in {time.time() - start:.1f}s "
          f"(workers: {args.num_workers or default_num_workers()}, cache: {cache_dir})")

    _check_no_leakage(splits)
    print(f"Excluded dark scans (mean < {MIN_RELATIVE_SCAN_INTENSITY} x median scan mean): " +
          ", ".join(f"{n}: {ds.excluded_scans} scans / {ds.excluded_slices} slices" for n, ds in splits.items()))
    _print_split_table(splits)

    train = splits["train"]
    print(f"\nStored images: {tuple(train.images.shape)} {train.images.dtype}, "
          f"{train.images.numel() * train.images.element_size() / 1e6:.0f} MB for train")
    print(f"Slice positions per scan: {sorted(train.slice_pos.unique().tolist())}")

    x, y = next(iter(DataLoader(train, batch_size=64, shuffle=True)))
    print(f"Batch: {tuple(x.shape)} {x.dtype}, range [{x.min():.2f}, {x.max():.2f}], mean {x.mean():.3f}, "
          f"labels CN={int((y == 0).sum())} AD={int((y == 1).sum())}")

    # Real images through the same grid function that will be used for generated samples
    os.makedirs(args.fig_dir, exist_ok=True)
    grid_path = os.path.join(args.fig_dir, "dataset_batch_check.png")
    save_image(x, grid_path, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"Saved {grid_path}")

    if args.oasis:
        start = time.time()
        oasis = OASISSlices(args.oasis_glob, args.resolution, cache_dir, args.num_workers)
        xo, _ = next(iter(DataLoader(oasis, batch_size=64, shuffle=True)))
        print(f"\nOASIS: {len(oasis)} slices loaded in {time.time() - start:.1f}s, batch {tuple(xo.shape)}, "
              f"range [{xo.min():.2f}, {xo.max():.2f}]")
        oasis_path = os.path.join(args.fig_dir, "oasis_batch_check.png")
        save_image(xo, oasis_path, nrow=8, normalize=True, value_range=(-1, 1))
        print(f"Saved {oasis_path}")


if __name__ == "__main__":
    main()