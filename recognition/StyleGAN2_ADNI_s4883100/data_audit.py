"""Audits the ADNI AD/NC slice dataset (structure, labels, patient-level leakage) and generates README figures.

Run on Rangpur (CPU node is sufficient):
    python data_audit.py                      # figures written to ./figures
    python data_audit.py --out some/other/dir

Only aggregate statistics are printed or saved. ADNI subject IDs are held in
memory for grouping and are never written to disk, so the outputs are safe to
commit to a public repository.
"""

import argparse
import glob
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")  # headless rendering on Rangpur
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_ROOT = "/home/groups/comp3710/ADNI"
SUBJECT_RE = re.compile(r"ADNI_(\d{3}_S_\d{4})")  # e.g. ADNI_123_S_4567 -> 123_S_4567
JSON_LABELS = {0: "CN", 1: "MCI", 2: "AD"}
FOLDER_NAMES = {"NC": "CN", "AD": "AD"}  # folders say NC; ADNI and the JSON use CN
CLASS_COLOURS = {"CN": "#4C72B0", "AD": "#DD8452"}
CLASSES = ["CN", "AD"]
SPLITS = ["train", "test"]


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="ADNI directory containing AD_NC/ and the metadata JSON")
    p.add_argument("--out", default=os.path.join(here, "figures"), help="output directory for figures")
    p.add_argument("--seed", type=int, default=42, help="seed for sampling example slices")
    p.add_argument("--n_intensity", type=int, default=400, help="slices per class sampled for intensity statistics")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def build_scan_index(root, meta):
    """Group slice files by scan and attach the JSON label and subject ID.

    Returns a dict: scan_id -> {split, cls, slices, paths, label, subject},
    with slices/paths sorted by slice index. Scans missing from the JSON
    have no 'label' or 'subject' key.
    """
    scans = {}
    for path in sorted(glob.glob(os.path.join(root, "AD_NC", "*", "*", "*.jpeg"))):
        split, folder = path.split(os.sep)[-3:-1]
        scan_id, slice_idx = os.path.basename(path)[: -len(".jpeg")].split("_")
        s = scans.setdefault(scan_id, dict(split=split, cls=FOLDER_NAMES[folder], slices=[], paths=[]))
        s["slices"].append(int(slice_idx))
        s["paths"].append(path)

    for scan_id, s in scans.items():
        order = np.argsort(s["slices"])
        s["slices"] = [s["slices"][i] for i in order]
        s["paths"] = [s["paths"][i] for i in order]
        entry = meta.get(scan_id)
        if entry is None:
            continue
        s["label"] = JSON_LABELS[entry["label"]]
        match = SUBJECT_RE.search(entry["raw"])
        s["subject"] = match.group(1) if match else None
    return scans


def group_by_subject(scans):
    """Return subject_id -> list of scan records (scans without a subject are skipped)."""
    subjects = defaultdict(list)
    for s in scans.values():
        if s.get("subject"):
            subjects[s["subject"]].append(s)
    return subjects


# --------------------------------------------------------------------------- #
# Checks (printed to stdout)
# --------------------------------------------------------------------------- #
def check_files(scans):
    """Verify size and mode of every file and look for byte-identical duplicates."""
    sizes, modes = Counter(), Counter()
    by_hash = defaultdict(list)
    for scan_id, s in scans.items():
        for path in s["paths"]:
            with open(path, "rb") as f:
                data = f.read()
            by_hash[hashlib.md5(data).hexdigest()].append(scan_id)
            with Image.open(io.BytesIO(data)) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
    dup_groups = [ids for ids in by_hash.values() if len(ids) > 1]
    cross_scan = sum(1 for ids in dup_groups if len(set(ids)) > 1)
    print(f"Image sizes (width, height): {dict(sizes)}")
    print(f"Image modes: {dict(modes)}")
    print(f"Byte-identical duplicate groups: {len(dup_groups)} ({cross_scan} spanning more than one scan)")


def check_slices(scans):
    """Report slices per scan, contiguity, and the overall slice index range."""
    lengths = Counter(len(s["slices"]) for s in scans.values())
    contiguous = sum(
        s["slices"] == list(range(s["slices"][0], s["slices"][0] + len(s["slices"]))) for s in scans.values()
    )
    all_idx = [i for s in scans.values() for i in s["slices"]]
    print(f"Slices per scan: {dict(lengths)}")
    print(f"Scans with contiguous slices: {contiguous}/{len(scans)}")
    print(f"Slice index range (all scans): {min(all_idx)}-{max(all_idx)}")
    for cls in CLASSES:
        starts = [s["slices"][0] for s in scans.values() if s["cls"] == cls]
        print(f"  {cls} window start index: min {min(starts)}, median {int(np.median(starts))}, max {max(starts)}")


def check_labels_and_subjects(scans, subjects):
    """Report JSON coverage, label agreement, label conversions, visits, and split leakage."""
    missing = sum(1 for s in scans.values() if "label" not in s)
    mismatched = sum(1 for s in scans.values() if "label" in s and s["label"] != s["cls"])
    mixed = sum(1 for ss in subjects.values() if len({s["label"] for s in ss}) > 1)
    print(f"Scans missing from JSON: {missing} | folder/JSON label mismatches: {mismatched}")
    print(f"Unique subjects: {len(subjects)} | subjects with more than one label: {mixed}")

    for cls in CLASSES:
        cls_subj = [ss for ss in subjects.values() if ss[0]["label"] == cls]
        n_scans = sum(len(ss) for ss in cls_subj)
        multi = sum(1 for ss in cls_subj if len(ss) > 1)
        print(
            f"  {cls}: {len(cls_subj)} subjects, {n_scans} scans "
            f"({n_scans / len(cls_subj):.2f} per subject), {multi} with 2+ scans"
        )

    both = sum(1 for ss in subjects.values() if {s["split"] for s in ss} == set(SPLITS))
    for sp in SPLITS:
        n = sum(1 for ss in subjects.values() if any(s["split"] == sp for s in ss))
        print(f"  subjects in provided {sp}: {n}")
    print(f"Subjects in BOTH provided train and test: {both} ({100 * both / len(subjects):.1f}% of all subjects)")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout(rect=(0, 0, 1, 0.94))  # leave room for the suptitle
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def pad_to_square(arr):
    """Zero-pad a (H, W) image symmetrically to a square. Returns (padded, top, left)."""
    h, w = arr.shape
    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    padded = np.pad(arr, ((top, size - h - top), (left, size - w - left)))
    return padded, top, left


def fig_provided_split_balance(scans, subjects, out):
    """Slices and patients per class in the provided split."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x, width = np.arange(len(SPLITS)), 0.38
    for j, cls in enumerate(CLASSES):
        slice_counts = [
            sum(len(s["slices"]) for s in scans.values() if s["split"] == sp and s["cls"] == cls) for sp in SPLITS
        ]
        subj_counts = [
            sum(1 for ss in subjects.values() if ss[0]["label"] == cls and any(s["split"] == sp for s in ss))
            for sp in SPLITS
        ]
        for ax, counts in zip(axes, [slice_counts, subj_counts]):
            bars = ax.bar(x + (j - 0.5) * width, counts, width, label=cls, color=CLASS_COLOURS[cls])
            ax.bar_label(bars, fmt="%d", fontsize=8)
    for ax, title in zip(axes, ["Slices per split", "Patients per split\n(patients in both splits counted twice)"]):
        ax.set_xticks(x, SPLITS)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Provided ADNI split: class balance")
    save(fig, out, "provided_split_balance.png")


def fig_provided_split_leakage(subjects, out):
    """Patients appearing in train only, test only, or both, per class."""
    cats = [("train only", {"train"}, "#8C8C8C"), ("both (leakage)", {"train", "test"}, "#C44E52"),
            ("test only", {"test"}, "#CFCFCF")]
    fig, ax = plt.subplots(figsize=(9, 3))
    for row, cls in enumerate(CLASSES):
        split_sets = [{s["split"] for s in ss} for ss in subjects.values() if ss[0]["label"] == cls]
        left = 0
        for name, target, colour in cats:
            n = sum(1 for sp in split_sets if sp == target)
            ax.barh(row, n, left=left, color=colour, label=name if row == 0 else None)
            if n:
                text_colour = "black" if colour == "#CFCFCF" else "white"
                ax.text(left + n / 2, row, str(n), ha="center", va="center", fontsize=9, color=text_colour)
            left += n
    both = sum(1 for ss in subjects.values() if {s["split"] for s in ss} == set(SPLITS))
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Patients")
    ax.set_title(f"Provided split: {both} of {len(subjects)} patients ({100 * both / len(subjects):.0f}%) "
                 "have scans in both train and test")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    save(fig, out, "provided_split_leakage.png")


def fig_scans_per_subject(subjects, out):
    """Distribution of scans (visits) per patient, by class."""
    max_n = max(len(ss) for ss in subjects.values())
    x, width = np.arange(1, max_n + 1), 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, cls in enumerate(CLASSES):
        counts = Counter(len(ss) for ss in subjects.values() if ss[0]["label"] == cls)
        values = [counts.get(n, 0) for n in x]
        mean = np.mean([len(ss) for ss in subjects.values() if ss[0]["label"] == cls])
        bars = ax.bar(x + (j - 0.5) * width, values, width, color=CLASS_COLOURS[cls],
                      label=f"{cls} (mean {mean:.2f} scans/patient)")
        ax.bar_label(bars, fmt="%d", fontsize=7)
    ax.set_xticks(x)
    ax.set_xlabel("Scans per patient")
    ax.set_ylabel("Patients")
    ax.set_title("Repeat visits per patient")
    ax.legend()
    save(fig, out, "scans_per_subject.png")


def fig_slice_windows(scans, out):
    """Where each scan's 20-slice window starts, and slice-index coverage across scans."""
    all_idx = [i for s in scans.values() for i in s["slices"]]
    lo, hi = min(all_idx), max(all_idx)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cls in CLASSES:
        cls_scans = [s for s in scans.values() if s["cls"] == cls]
        starts = [s["slices"][0] for s in cls_scans]
        axes[0].hist(starts, bins=np.arange(lo, hi + 2) - 0.5, alpha=0.6, color=CLASS_COLOURS[cls], label=cls)
        coverage = Counter(i for s in cls_scans for i in s["slices"])
        idx = np.arange(lo, hi + 1)
        axes[1].plot(idx, [coverage.get(i, 0) / len(cls_scans) for i in idx], color=CLASS_COLOURS[cls], label=cls)
    axes[0].set(title="Start index of each scan's slice window", xlabel="Start slice index", ylabel="Scans")
    axes[1].set(title="Fraction of scans containing each slice index", xlabel="Slice index", ylabel="Fraction")
    for ax in axes:
        ax.legend()
    save(fig, out, "slice_windows.png")


def fig_sample_slices(scans, out, rng):
    """Five evenly spaced slices from two randomly chosen training scans per class."""
    positions = [0, 5, 10, 15, 19]
    rows = []
    for cls in CLASSES:
        ids = sorted(k for k, s in scans.items() if s["cls"] == cls and s["split"] == "train")
        rows += [(cls, scans[k]) for k in rng.choice(ids, size=2, replace=False)]
    fig, axes = plt.subplots(len(rows), len(positions), figsize=(2.2 * len(positions), 2.2 * len(rows)))
    for r, (cls, s) in enumerate(rows):
        for c, pos in enumerate(positions):
            ax = axes[r, c]
            ax.imshow(np.array(Image.open(s["paths"][pos])), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"slice {pos + 1}/20", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{cls} scan {r % 2 + 1}", fontsize=9)
    fig.suptitle("Example slices (random training scans)")
    save(fig, out, "sample_slices.png")


def fig_preprocessing(scans, out, rng, crop):
    """Original slice -> padded square -> brain crop -> downsampled working resolutions."""
    ids = sorted(k for k, s in scans.items() if s["split"] == "train")
    s = scans[rng.choice(ids)]
    original = np.array(Image.open(s["paths"][10]))
    padded, top, left = pad_to_square(original)
    c_top, c_left, c_size = crop
    cropped = padded[c_top:c_top + c_size, c_left:c_left + c_size]
    stages = [("Original", original), ("Padded to square", padded), ("Cropped to brain region", cropped)]
    for res in (128, 64):
        stages.append((f"Resized to {res}x{res}", np.array(Image.fromarray(cropped).resize((res, res), Image.BILINEAR))))

    fig, axes = plt.subplots(1, len(stages), figsize=(3.2 * len(stages), 3.6))
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{title}\n{img.shape[0]} H x {img.shape[1]} W", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    h, w = original.shape
    axes[1].add_patch(patches.Rectangle((left - 0.5, top - 0.5), w, h, fill=False, ls="--", ec="red", lw=1))
    axes[1].add_patch(patches.Rectangle((c_left - 0.5, c_top - 0.5), c_size, c_size, fill=False, ec="yellow", lw=1))
    fig.suptitle("Preprocessing pipeline (red dashed = original extent, yellow = brain crop)")
    save(fig, out, "preprocessing_pipeline.png")


def measure_brain_extent(scans, threshold=10, min_pixels=3):
    """Bounding box of brain tissue for every slice, in padded (square) coordinates.

    A row or column counts as brain if at least `min_pixels` pixels exceed `threshold`,
    which ignores isolated JPEG noise. Also accumulates how often each pixel is brain.

    Returns (boxes, frequency, n_empty): boxes is an (N, 4) array of
    [top, bottom, left, right] (inclusive), frequency is the fraction of slices in
    which each pixel is brain, and n_empty counts slices with no brain pixels.
    """
    boxes, n_empty, freq = [], 0, None
    for s in scans.values():
        for path in s["paths"]:
            mask = pad_to_square(np.array(Image.open(path)))[0] > threshold
            freq = mask.astype(np.float64) if freq is None else freq + mask
            rows = np.flatnonzero(mask.sum(axis=1) >= min_pixels)
            cols = np.flatnonzero(mask.sum(axis=0) >= min_pixels)
            if rows.size == 0 or cols.size == 0:
                n_empty += 1
                continue
            boxes.append((rows[0], rows[-1], cols[0], cols[-1]))
    return np.array(boxes), freq / sum(len(s["paths"]) for s in scans.values()), n_empty


def compute_crop(boxes, image_size, margin=4):
    """Smallest square containing every slice's brain box plus a margin, as (top, left, size).

    The crop is fixed for the whole dataset so that brain size and position are not
    altered per image. It is centred on the dataset-wide brain extent and clamped to
    stay inside the image.
    """
    top, bottom = int(boxes[:, 0].min()), int(boxes[:, 1].max())
    left, right = int(boxes[:, 2].min()), int(boxes[:, 3].max())
    size = min(max(bottom - top + 1, right - left + 1) + 2 * margin, image_size)
    c_top = int(round((top + bottom) / 2 - size / 2))
    c_left = int(round((left + right) / 2 - size / 2))
    c_top = min(max(c_top, 0), image_size - size)
    c_left = min(max(c_left, 0), image_size - size)
    return c_top, c_left, size


def report_brain_extent(boxes, n_empty, crop, image_size):
    """Print the brain extent, its robustness to outliers, and the crop to use."""
    names = ["top", "bottom", "left", "right"]
    print(f"Slices with no brain pixels: {n_empty}")
    print(f"{'edge':<7} {'min':>5} {'0.1%':>6} {'median':>7} {'99.9%':>6} {'max':>5}")
    for i, name in enumerate(names):
        col = boxes[:, i]
        print(f"{name:<7} {col.min():>5} {np.percentile(col, 0.1):>6.0f} {np.median(col):>7.0f} "
              f"{np.percentile(col, 99.9):>6.0f} {col.max():>5}")
    c_top, c_left, size = crop
    heights = boxes[:, 1] - boxes[:, 0] + 1
    widths = boxes[:, 3] - boxes[:, 2] + 1
    print(f"Per-slice brain size (median): {np.median(heights):.0f} H x {np.median(widths):.0f} W px")
    print(f"Crop (top, left, size) in padded {image_size}x{image_size} coordinates: {crop}")
    print(f"Brain occupies {100 * np.median(np.maximum(heights, widths)) / image_size:.0f}% of the padded "
          f"width before cropping and {100 * np.median(np.maximum(heights, widths)) / size:.0f}% after "
          f"(median slice)")
    outside = np.sum((boxes[:, 0] < c_top) | (boxes[:, 1] >= c_top + size) |
                     (boxes[:, 2] < c_left) | (boxes[:, 3] >= c_left + size))
    print(f"Slices with brain outside the crop: {outside}")
    print(f"\n>>> Paste into dataset.py:  ADNI_CROP = ({c_top}, {c_left}, {size})")


def fig_brain_extent(scans, freq, crop, out, rng):
    """Brain-pixel frequency map and example slices with the crop square overlaid."""
    c_top, c_left, size = crop
    ids = sorted(scans)
    examples = [scans[k]["paths"][i] for k, i in zip(rng.choice(ids, size=3, replace=False), (0, 10, 19))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    im = axes[0].imshow(freq, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="Fraction of slices")
    axes[0].set_title("Where brain tissue appears\n(all slices, padded coordinates)", fontsize=9)
    for ax, path in zip(axes[1:], examples):
        ax.imshow(pad_to_square(np.array(Image.open(path)))[0], cmap="gray", vmin=0, vmax=255)
        ax.set_title("Example slice", fontsize=9)
    for ax in axes:
        ax.add_patch(patches.Rectangle((c_left - 0.5, c_top - 0.5), size, size, fill=False, ec="cyan", lw=1.2))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Fixed brain crop: {size}x{size} px at (top {c_top}, left {c_left})")
    save(fig, out, "brain_extent.png")


def fig_intensity(scans, out, rng, n_per_class):
    """Pixel intensity distribution per class, and the share of near-black background."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls in CLASSES:
        paths = [p for s in scans.values() if s["cls"] == cls for p in s["paths"]]
        chosen = rng.choice(paths, size=min(n_per_class, len(paths)), replace=False)
        pixels = np.concatenate([np.array(Image.open(p)).ravel() for p in chosen])
        background = np.mean(pixels < 10)
        print(f"  {cls} intensity: min {pixels.min()}, max {pixels.max()}, "
              f"mean {pixels.mean():.1f}, near-black (<10) fraction {background:.2f}")
        ax.hist(pixels, bins=64, range=(0, 255), density=True, alpha=0.6, color=CLASS_COLOURS[cls],
                label=f"{cls} ({100 * background:.0f}% near-black)")
    ax.set_yscale("log")
    ax.set(title=f"Pixel intensity distribution ({n_per_class} random slices per class)",
           xlabel="Intensity (0-255)", ylabel="Density (log scale)")
    ax.legend()
    save(fig, out, "intensity_histogram.png")


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.root, "meta_data_with_label.json")) as f:
        meta = json.load(f)
    print(f"JSON entries: {len(meta)} | labels: "
          f"{ {JSON_LABELS[k]: v for k, v in sorted(Counter(e['label'] for e in meta.values()).items())} }")

    scans = build_scan_index(args.root, meta)
    subjects = group_by_subject(scans)
    print(f"Slices: {sum(len(s['slices']) for s in scans.values())} | scans: {len(scans)}")

    print("\n== File checks ==")
    check_files(scans)
    print("\n== Slice structure ==")
    check_slices(scans)
    print("\n== Labels and subjects ==")
    check_labels_and_subjects(scans, subjects)

    print("\n== Brain extent and crop ==")
    boxes, freq, n_empty = measure_brain_extent(scans)
    image_size = freq.shape[0]
    crop = compute_crop(boxes, image_size)
    report_brain_extent(boxes, n_empty, crop, image_size)

    print("\n== Figures ==")
    fig_provided_split_balance(scans, subjects, args.out)
    fig_provided_split_leakage(subjects, args.out)
    fig_scans_per_subject(subjects, args.out)
    fig_slice_windows(scans, args.out)
    fig_sample_slices(scans, args.out, rng)
    fig_preprocessing(scans, args.out, rng, crop)
    fig_intensity(scans, args.out, rng, args.n_intensity)
    fig_brain_extent(scans, freq, crop, args.out, rng)


if __name__ == "__main__":
    main()