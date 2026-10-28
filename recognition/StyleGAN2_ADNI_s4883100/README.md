# StyleGAN2 for Synthetic ADNI Brain MRI Generation

Generating realistic, non-memorised brain MRI slices from the ADNI dataset with StyleGAN2, benchmarked against a WGAN-GP baseline.

> **Status:** Implementation and experiments complete. StyleGAN2 (with and without augmentation and class conditioning) and the WGAN-GP baseline are trained and audited on ADNI at 64 × 64, with a sensitivity check at 128 × 128; results and recommendation are in Section 11.

## Contents

1. [Overview](#1-overview)
2. [Feasibility Review](#2-feasibility-review)
3. [Dataset](#3-dataset)
4. [Dataset Discoveries and Fixes](#4-dataset-discoveries-and-fixes)
5. [Preprocessing and Data Splits](#5-preprocessing-and-data-splits)
6. [Design Decisions](#6-design-decisions)
7. [Memorisation Audit](#7-memorisation-audit)
8. [Reproducing the Dataset Audit](#8-reproducing-the-dataset-audit)
9. [Dependencies](#9-dependencies)
10. [Usage](#10-usage)
11. [Results](#11-results)
12. [Artificial Intelligence Usage Disclosure](#12-artificial-intelligence-usage-disclosure)
13. [References](#13-references)

---

## 1. Overview

Medical imaging research is constrained by patient privacy law and limited scan availability. Synthetic cohorts produced by generative models could be shared and used for augmentation, but only if they are anatomically plausible **and** do not reproduce real patients. This project trains StyleGAN2 on cognitively normal (CN) and Alzheimer's disease (AD) brain MRI slices from ADNI, compares it against a WGAN-GP baseline, and audits whether generated brains are genuinely novel or memorised copies of training patients.

**How it works.** StyleGAN2 (Karras et al., 2020) generates an image in two stages. A mapping network of 8 fully connected layers turns a random latent vector *z* into a style vector *w*. A synthesis network then starts from a learned 4 × 4 constant and doubles the resolution up to 64 × 64; at every layer, *w* modulates the convolution weights (scaling them per input channel, then renormalising them, "demodulation"), and per-pixel noise adds fine stochastic detail. A residual discriminator, with a minibatch standard-deviation feature that exposes a lack of variety, scores images as real or generated. Training uses the non-saturating logistic loss with R1 regularisation (Mescheder et al., 2018), and samples come from an exponential moving average of the generator's weights. The baseline is my Lab 2 WGAN-GP: a DCGAN-style generator and critic trained with the Wasserstein loss and a gradient penalty. Every model is then measured three ways: **fidelity** (FID, precision, density and coverage against held-out patients), **memorisation** (the calibrated audit of Section 7, plus a matched-reference test that separates training-specific resemblance from general realism), and **class fidelity and coverage** (a joint t-SNE embedding and a dedicated AD/CN classifier).

| WGAN-GP baseline | StyleGAN2, recommended configuration (augmentation p = 0.2, 17,500 iterations, uncurated) |
|---|---|
| ![Baseline samples](figures/wgan_adni64_random_samples.png) | ![StyleGAN2 samples](figures/sg2_adni64_augp02_iter017500_uncurated_seed1.png) |

*Uncurated generated ADNI slices at 64 × 64. StyleGAN2 produces sharper cortical folds, cleaner outlines and a wider range of head positions.*

## 2. Feasibility Review

**Aim.** To find out whether StyleGAN2 generates more realistic ADNI brain slices than a WGAN-GP baseline without moving closer to its training patients, so a project manager can decide whether the synthetic cohort is safe to share.

**User need, scope and acceptance criteria.** An imaging research team wants to share a synthetic cohort of cognitively normal and Alzheimer's disease brain slices without breaching patient confidentiality. Scope: StyleGAN2 generating 64 × 64 sagittal slices (128 × 128 if the budget allows), plus an audit of whether its output is realistic, novel and safe to share.

1. **Fidelity:** lower FID against held-out test slices than the baseline at the same resolution; median nearest-neighbour SSIM to real slices above 0.6.
2. **No copies:** none of 3,100 generated slices reaches copy level (Section 7).
3. **Memorisation:** at the selected checkpoint, a flag rate and matched-reference excess no higher than the baseline's at equal or better FID.
4. **Coverage:** a t-SNE or UMAP embedding shows generated slices across both classes and the range of slice positions.
5. **Resources:** one A100 (40 GB), resumable jobs of 12 hours or less, at most 24 GPU-hours per configuration (an estimate; measured in the first run).

**Model choice and course concepts.** StyleGAN2 [Hard difficulty] on ADNI (`/home/groups/comp3710/ADNI`), unconditional first. It extends the course's GAN material (adversarial training, convolutional generators and discriminators, regularisation). A mapping network turns the latent vector into a style vector that modulates every synthesis layer, giving fine texture and a latent space that can be analysed with PCA and interpolation (Karras et al., 2019, 2020). R1 regularisation stabilises training, path-length regularisation smooths the latent space, and adaptive discriminator augmentation targets small datasets; with 542 training patients, overfitting is the central risk. Liu, Chandra et al. (2023) apply style-based models to neuroimaging. The baseline is my Lab 2 WGAN-GP (Gulrajani et al., 2017), sharing the same data pipeline and evaluation.

**Plan.** (1) FID for every baseline snapshot, giving a fidelity-versus-memorisation curve. (2) A minimal StyleGAN2, verified on OASIS 64 px. (3) StyleGAN2 on ADNI 64 px, measuring memory and time. (4) Add augmentation, then diagnosis conditioning, auditing every snapshot. (5) Embeddings, a failure autopsy and the final recommendation.

**Preliminary feasibility evidence.**

| Evidence | Result |
|---|---|
| Data audit and split | The provided split leaks 216 of 680 patients; re-split 80/10/10 by patient (542 / 68 / 68) |
| Memorisation audit | Thresholds from validation patients flag 0.3–0.9% of unseen test patients and at least 99.2% of near-duplicates |
| Baseline | Reproduces Lab 2 on OASIS; on ADNI 64 px, 15,100 iterations in 25 min using 0.43 GB |
| Baseline audit | No copies; flag rate rises from 0.03% to 4.2% over training |

**Risks, budget and fallback.** Instability or mode collapse: watched through loss curves, sample grids and per-snapshot audits, and mitigated by R1 and augmentation. Compute: measured at 64 px before scaling. Memorisation: checkpoints chosen on the fidelity-versus-memorisation curve. Class confound: slice windows differ by class (Discovery 9), so the model is unconditional first. The audit barely recognises a patient across visits, so a feature-space identity check is planned. Budget: about 30 A100-hours (an estimate). Fallback: StyleGAN2 at 64 px with a reduced network, which keeps the Hard tier; the comparison and audit stand either way.

---

## 3. Dataset

### 3.1 Source

The preprocessed ADNI subset provided on the Rangpur cluster at `/home/groups/comp3710/ADNI`. It contains 2D JPEG slices only. The metadata JSON references NIfTI volumes and tissue segmentations, but those files are not included, so the JSON is used only for labels and patient identity.

### 3.2 Directory structure

```
/home/groups/comp3710/ADNI/
├── meta_data_with_label.json      # metadata for 2,189 ADNI scans
└── AD_NC/
    ├── train/
    │   ├── AD/                     # <scanID>_<sliceIndex>.jpeg
    │   └── NC/
    └── test/
        ├── AD/
        └── NC/
```

The folders use **NC** (normal control); ADNI and the metadata use **CN** (cognitively normal). They are the same group, and this project uses CN throughout.

### 3.3 Provided split

| Split | Class | Slices | Scans | Slices per scan |
|---|---|---:|---:|---:|
| train | AD | 10,400 | 520 | 20 |
| train | CN | 11,120 | 556 | 20 |
| test | AD | 4,460 | 223 | 20 |
| test | CN | 4,540 | 227 | 20 |
| **Total** | | **30,520** | **1,526** | |

![Provided split class balance](figures/provided_split_balance.png)

### 3.4 Image properties

| Property | Value |
|---|---|
| Format | JPEG, 8-bit single-channel greyscale (PIL mode `L`) |
| Dimensions | 256 px wide × 240 px high (all 30,520 files) |
| Slices per scan | Exactly 20, always contiguous (slice index step of 1), for all 1,526 scans |
| Slice index range | 67–114 overall; each scan covers a different 20-slice window |
| Window start index | CN: median 88 (range 67–95); AD: median 78 (range 70–95) |
| Duplicates | None (no byte-identical files) |
| Background | 74% of pixels are near-black (intensity < 10) in both classes |
| File naming | `<scanID>_<sliceIndex>.jpeg`, where `scanID` is the ADNI image ID |
| View | Sagittal, skull-stripped (cerebellum, corpus callosum, and lateral ventricle visible) |
| Brain size | Median 167 H × 146 W px within the padded 256 × 256 frame; midline slices nearly fill the frame |
| Data quality | 2 corrupted scans (40 slices) excluded; 3 scans whose brains touch the top edge are kept (Discoveries 11–12) |

![Example slices](figures/sample_slices.png)

![Slice windows](figures/slice_windows.png)

![Brain extent](figures/brain_extent.png)

### 3.5 Metadata JSON

`meta_data_with_label.json` maps each ADNI image ID (the `scanID` in filenames) to file paths and a diagnostic label.

| Field | Contents |
|---|---|
| `raw` | Path to the original T1-weighted MPRAGE volume. The filename embeds the patient ID (`ADNI_XXX_S_XXXX`) |
| `masked` | Path to a masked, N4 bias-corrected volume (per the filename) |
| `c1`–`c5` | Paths to binarised SPM tissue segmentations |
| `label` | Diagnosis: 0 = CN, 1 = MCI, 2 = AD |

| Label | Diagnosis | JSON entries | In this dataset |
|---:|---|---:|---:|
| 0 | CN | 810 | 783 scans |
| 1 | MCI | 625 | 0 (excluded) |
| 2 | AD | 754 | 743 scans |

All 1,526 scans in the folders have a JSON entry, and every folder label agrees with its JSON label.

### 3.6 Patient-level structure

The 1,526 scans come from **680 patients**, many of whom were scanned at multiple visits.

| Class | Patients | Scans | Scans per patient |
|---|---:|---:|---:|
| CN | 459 | 783 | 1.71 |
| AD | 221 | 743 | 3.36 |
| **Total** | **680** | **1,526** | 2.24 |

| Scans per patient | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Patients | 282 | 214 | 67 | 40 | 37 | 18 | 14 | 8 |

No patient changes diagnosis between visits: every patient is either CN-only or AD-only.

![Scans per patient](figures/scans_per_subject.png)

---

## 4. Dataset Discoveries and Fixes

| # | Discovery | Evidence | Impact | Fix |
|---|---|---|---|---|
| 1 | Filename IDs identify **scans, not patients** | JSON keys match the `I<ID>` suffix of ADNI image paths; 1,526 scan IDs map to 680 patients | Splitting by filename ID keeps a patient's other visits on both sides of the split | Extract the patient ID from the JSON `raw` path and split by patient |
| 2 | The provided train/test split **leaks patients** | 216 of 680 patients (32%) have scans in both train and test; 216 of the 331 test patients (65%) also appear in training | Held-out comparisons (FID, memorisation audit) would be contaminated by patients the model has seen | Discard the provided split and re-split at patient level (Section 5.2) |
| 3 | MCI scans are excluded | JSON contains 625 MCI entries; none appear in the folders | None for this task; the problem is CN vs AD only | Use only labels 0 and 2 |
| 4 | No diagnostic conversions | 459 CN-only and 221 AD-only patients | Each patient has one unambiguous class | Stratify the split by patient label |
| 5 | AD patients have about twice as many visits | 3.36 vs 1.71 scans per patient | Slice-level class balance shifts after re-splitting, and AD contributes most of the same-patient pairs in the memorisation audit | Report every split statistic and every audit threshold and rate per class |
| 6 | Images are not square | 256 wide × 240 high | StyleGAN2 requires square power-of-two resolutions | Zero-pad the height to 256 (8 rows top and bottom) |
| 7 | Slice windows differ between scans | Each scan has 20 contiguous slices, but window start positions vary across scans | The same slice index shows different anatomy in different scans | Memorisation comparisons use nearest-neighbour search over slices rather than matched indices |
| 8 | Background dominates every image | 74% of pixels are near-black (< 10) in both classes | Pixel metrics such as SSIM are inflated by shared black background, making any two brains look similar | Masked SSIM is the primary audit metric: at the same false-flag rate it detects 80% of AD same-session repeat scans, against 48% for whole-image SSIM (Section 7.3) |
| 9 | Slice windows are **class-dependent** | AD windows start at a median slice index of 78; CN windows at 88 | Class is confounded with anatomical level: a class-conditional model could learn "which part of the brain" rather than disease-related anatomy, and CN vs AD comparisons partly compare different brain regions | Compare classes at matched slice positions where possible, and account for the offset when interpreting conditional generation and per-class metrics |
| 10 | The brain already fills most of the frame | Median brain box 167 H × 146 W px in the padded 256 × 256 image; between the 0.1th and 99.9th percentiles, brain boxes span rows 8–228 and columns 14–239 | A fixed crop (which must contain every brain, so that brain size and position are not altered per image) would be the full frame and gain no resolution | No crop: padding only (`ADNI_CROP = None`) |
| 11 | Two scans are corrupted | Per-scan mean intensity of 0.9 and 7.8, against a median of 29.2 and a 1st percentile of 21.2. With contrast stretched, one is blocky noise; the other mixes a non-skull-stripped head with pure-noise slices. The first also accounts for all 3 empty slices and the only left-edge contact | A GAN would learn near-black or noise images as valid brains | Exclude scans with mean intensity < 0.5 × the median scan mean (14.6), applied after the patient split so no split assignment changes. Removes 2 scans (40 slices) belonging to 2 CN training patients with no other scans |
| 12 | Three scans' brains touch the top edge | 45 slices from 3 scans reach original row 0 | Visual inspection shows complete, anatomically normal brains positioned high in the frame | Kept |
| 13 | Some patients have **same-day repeat scans** | Acquisition dates and series IDs parsed from the metadata `raw` path (1,226 of 1,226 training scans). Training contains 253 same-day scan pairs (81 CN, 172 AD). CN same-day pairs have a median masked pair SSIM of 0.992 (58% above 0.99); no pair from different days exceeds 0.99 | A "same-person" memorisation threshold is pushed to about 1.0, so only pixel-perfect copies would be flagged. Repeat scans also count a patient's anatomy twice in training | Same-day pairs become a copy reference in the audit instead of setting the threshold (Section 7) |
| 14 | Visits on different days are **not aligned** | Different-day pairs share all 20 slice indices, yet each slice's best match is a median of 4 slices away and the median masked pair SSIM is 0.29–0.32 at every gap (under 6 months to over 2 years) | The same patient's other visit looks less similar than the nearest stranger among 24,520 training slices, so pixel SSIM cannot recognise a patient across visits | Threshold calibrated on validation strangers; different-day pairs reported as an identity reference and a stated limitation (Section 7) |

No byte-identical duplicate images exist across the 30,520 files.
**Open observations.** AD same-day pairs are a mixture: a few near-identical repeats, but most have a pair SSIM around 0.75–0.8, with a thin tail down to 0.2–0.5. Different acquisition protocols on the same day are a plausible cause, to be checked against the series descriptions. Two AD scan pairs share a series ID yet have a pair SSIM of only 0.36, which is not yet explained.

![Provided split leakage](figures/provided_split_leakage.png)

![Intensity histogram](figures/intensity_histogram.png)

![Per-scan mean intensity](figures/scan_mean_intensity.png)

![Flagged scans](figures/flagged_scans.png)

---

## 5. Preprocessing and Data Splits

### 5.1 Preprocessing pipeline

1. **Index:** record each slice's scan ID, slice index, position within its 20-slice window, and class. Attach the patient ID from the metadata JSON at runtime.
2. **Split:** assign every patient to train, validation, or test using all scans (Section 5.2). All slices of a patient's scans inherit that patient's split.
3. **Exclude corrupted scans:** drop scans whose mean intensity is below 0.5 × the median scan mean (Discovery 11). This happens after splitting, so it never changes any patient's split assignment.
4. **Pad:** zero-pad each 240 × 256 slice to 256 × 256 (8 rows top and bottom). Zero matches the existing background and preserves the anatomy's aspect ratio, which resizing directly would distort.
5. **No crop:** a fixed crop containing every brain would be the full frame (Discovery 10).
6. **Resize:** downsample with antialiasing to 64 × 64, the resolution used for all experiments (Section 6). The code also supports 128 × 128 and 256 × 256, which are untested.
7. **Scale:** map intensities from [0, 255] to [−1, 1], matching the generators' output range (`tanh` in the WGAN-GP; the StyleGAN2 generator has no output activation and produces values in roughly the same range, as in the original paper).
8. **Preload:** hold all slices in memory as `uint8` and convert to float per batch, with an on-disk cache outside the repository. At 64 × 64 the training split occupies 100 MB, and the first load takes about 20 s with 8 workers.

No augmentation is applied to the training data itself. Augmentation is tested only on the images the discriminator sees, real and generated alike: adaptive discriminator augmentation (ADA) and a fixed-probability variant (Section 11.5).

![Preprocessing pipeline](figures/preprocessing_pipeline.png)

### 5.2 Split strategy

| Split | Share of patients | Purpose |
|---|---:|---|
| train | 80% (542 patients after excluding two corrupted scans) | Training StyleGAN2 and the WGAN-GP baseline |
| validation | 10% (68 patients) | Calibrating the memorisation audit's thresholds (Section 7) and the unseen reference group in the matched-reference test (Section 11.3) |
| test | 10% (68 patients) | FID, precision/recall and density/coverage; the audit's false-flag check (Section 7); the control group in the matched-reference test (Section 11.3) |

**Why re-split rather than use the provided split:** the provided split places 216 patients on both sides (Discovery 2). For a generative model, a clean held-out set is what makes the memorisation audit meaningful: if held-out patients were seen in training, similarity between generated and held-out images proves nothing.

**Why patient-level and stratified:** all visits of a patient stay in one split, and stratifying by patient label keeps the CN:AD patient ratio equal across splits. The split is generated deterministically from a fixed seed.

**Why no split file is committed:** ADNI's data use agreement restricts redistribution, so patient identifiers are kept out of this public repository. The split is regenerated from the seed at runtime.

Resulting split (seed 42), verified by the `dataset.py` smoke test, which also asserts that no patient or scan appears in more than one split:

| Split | Patients (CN / AD) | Scans (CN / AD) | Slices (CN / AD) | AD share of slices |
|---|---:|---:|---:|---:|
| train | 365 / 177 | 622 / 604 | 12,440 / 12,080 | 49.3% |
| val | 46 / 22 | 72 / 71 | 1,440 / 1,420 | 49.7% |
| test | 46 / 22 | 87 / 68 | 1,740 / 1,360 | 43.9% |
| **Total** | **678** | **1,524** | **30,480** | |

Both excluded scans (Discovery 11) belonged to CN patients in the training split who had no other scans, so train holds 365 CN patients rather than the 367 assigned. Validation and test are unaffected.

Stratification balances patients, not slices. Because AD patients have between 1 and 8 scans each, the slice-level class balance varies between splits (the test split has fewer AD scans), so all evaluation metrics are reported per class rather than pooled.

---

## 6. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | StyleGAN2, pure-PyTorch implementation in `modules.py` | Style modulation gives fine texture and a structured latent space; written from the paper rather than adapted from NVIDIA's code, which relies on custom CUDA kernels |
| Baseline | WGAN-GP | Previously implemented and validated on OASIS brain slices; retrained on the same ADNI split for a like-for-like comparison |
| Verification dataset | OASIS brain slices at 64 × 64 | A known-good WGAN-GP result exists at this setting, so a new StyleGAN2 implementation that underperforms it indicates a bug |
| Resolution | 64 × 64 for the main comparison; one 128 × 128 run as a sensitivity check | At 64 px each StyleGAN2 run takes about 67 minutes, which allowed nine ADNI configurations with every snapshot audited. 128 px costs 1.6× the time per iteration and needs its own audit calibration (Section 11.9) |
| Split | Patient-level, label-stratified, 80/10/10, fixed seed | Removes the leakage in the provided split (Section 4) |
| Privacy | No patient IDs or split files committed | ADNI data use terms |
| Memorisation audit | Nearest-neighbour masked SSIM to the training set, with per-class thresholds calibrated on validation patients and checked on test patients | Calibrated on real data instead of an arbitrary cutoff; the originally planned same-person threshold proved invalid (Section 7.1) |
| Code organisation | `modules.py`, `dataset.py`, `train.py`, `predict.py`, plus `audit.py`, `data_audit.py` and the analysis scripts `embed.py`, `autopsy.py`, `latent.py` and `classify.py` | Required structure; model code depends only on PyTorch |
| Brain crop | None (padding only) | Measured: a crop containing every brain is the full frame (Discovery 10) |
| Scan quality filter | Exclude scans with mean intensity < 0.5 × median, after the patient split | Removes the two corrupted scans with a relative rule rather than hard-coded IDs, without altering the split (Discovery 11) |
| Simplified resampling and width | Bilinear upsampling and average pooling instead of transposed convolutions with FIR blur; channels capped at 256 | Simpler code that fits the budget; the OASIS check confirmed the implementation produces sharp, varied brains |
| Training length | 15,100 iterations (966 thousand real images shown) for every model | Equal real-image budget for a like-for-like comparison, matching the verified Lab 2 baseline run |
| Checkpoint selection | Every 2,500-iteration snapshot audited; checkpoint chosen on fidelity and memorisation together | The baseline already showed memorisation rising with training, so the final snapshot is not automatically the best |
| Discriminator augmentation | Translation, isotropic scaling, rotation, brightness, contrast; no flips or colour. Applied adaptively (ADA) or with a fixed probability p = 0.2 | A mirrored sagittal slice is anatomically wrong, and the data is greyscale. The fixed-probability option tests augmentation directly, since ADA's standard trigger never activated on this dataset (Section 11.5) |
| Evaluation features | Inception-v3 (Szegedy et al., 2016; standard FID weights) for FID, precision/recall, density/coverage and t-SNE; a dedicated AD/CN CNN for class fidelity (`classify.py`) | Inception features are standard and comparable across runs but barely separate AD from CN (Section 11.6), so class fidelity is measured with a classifier trained on the real training split and checked against the slice-level confound (Discovery 9) |
| Class conditioning | Optional: one-hot class embedded in the mapping network; projection discriminator (Miyato and Koyama, 2018) | Follows StyleGAN2-ADA; tested because unconditional generation covers real AD slices less well than CN slices (Section 11.6). Off by default, so unconditional snapshots still load |
| Replication | A second training seed for the recommended configuration | Separates a real effect from run-to-run variation, the main weakness of single runs |
| Latent-space analysis | Interpolation with nearest-neighbour tracking, and an inversion-based membership test (`latent.py`) | The specification asks for latent interpolation between synthetic images and training scans; inversion tests what the generator *can* produce, not only what it samples (Section 11.8) |
---

## 7. Memorisation Audit

**Question:** is a generated brain a new brain, or a copy of a training image?

Each generated slice is compared with its nearest neighbour in the training set. The decision threshold is calibrated on real data rather than chosen arbitrarily. Validation patients were never seen in training, so their slices show how close a genuinely new brain gets to the training set by chance. A generated slice closer to a training slice than 99% of new real brains get is flagged and inspected side by side with its neighbour. Every rung except the generated one uses real data only, so the audit is calibrated before any model is trained (`audit.py`).

### 7.1 From the original design to the current one

The original design took the threshold from repeat visits of the same patient. The first calibration run showed that this fails, and a visit-pair diagnostic that dates each scan from its metadata path explained why (Discoveries 13 and 14):

- **Same-day repeat scans are near-identical** (CN median pair SSIM 0.992). They pushed the 95th-percentile same-person threshold to 0.9995 for CN, so a generator would have to copy almost pixel-perfectly to be flagged.
- **Visits on different days are not aligned.** A patient's other visit scores lower (median masked SSIM 0.30) than the nearest stranger among 24,520 training slices (median 0.59), so pixel SSIM does not recognise a patient across visits.

Same-day and different-day pairs are therefore kept as reference rungs, and the threshold now comes from strangers.

### 7.2 Similarity ladder

| Rung | Comparison | Role |
|---|---|---|
| Near-duplicate | Adjacent slices of the same training scan | Copy reference: a copy displaced by one slice should be flagged |
| Same session | Training slice → nearest slice of another scan of the same patient acquired on the same day | Copy reference: a repeat acquisition should be flagged |
| Different day | Training slice → nearest slice of the same patient's scans from other days | Identity reference: does the metric recognise a patient across visits? |
| Stranger (val) | Validation slice → nearest training slice | **Calibration:** threshold = 99th percentile, per class and metric |
| Stranger (test) | Test slice → nearest training slice | False-flag check on patients not used for calibration |
| Generated | Generated slice → nearest training slice | Under evaluation |

### 7.3 Results at 64 × 64

All 2,860 validation and 3,100 test slices were searched against all 24,520 training slices. The table shows the share of each rung at or above the class threshold:

| Rung | Role | CN whole | CN masked | AD whole | AD masked |
|---|---|---:|---:|---:|---:|
| Near-duplicate | copy | 99.6% | 99.2% | 99.7% | 99.5% |
| Same session | copy | 92.3% | 93.9% | 47.6% | 80.2% |
| Different day | identity | 3.8% | 4.0% | 9.0% | 10.7% |
| Stranger (val) | calibration | 1.0% | 1.0% | 1.1% | 1.1% |
| Stranger (test) | false flags | 0.7% | 0.3% | 0.9% | 0.6% |
| **Threshold (SSIM)** | | 0.863 | 0.716 | 0.856 | 0.694 |

![Memorisation audit ladder](figures/audit_ladder_r64.png)

*Nearest-neighbour SSIM to the training set at 64 px. The threshold (dashed) is the 99th percentile of validation strangers. Copy references (purple, red) fall above it and test strangers (green) below it; different-day visits of the same patient (orange) mostly fall below it.*

**Findings:**

1. **The calibration transfers to unseen patients.** Test strangers are flagged at 0.3–0.9%, matching the nominal 1%. A generator producing genuinely new brains should therefore see about 1% of its samples flagged; a clearly higher rate is evidence of copying.
2. **The audit catches copies.** At least 99.2% of near-duplicates are flagged, even though the anatomy shifts by one slice, as are 92–94% of CN same-day repeats.
3. **Masked SSIM is the primary metric.** At the same false-flag rate, it flags 80% of AD same-session pairs against 48% for whole-image SSIM, because shared background hides real differences (Discovery 8).
4. **Limitation: the audit barely sees identity.** Only 4% (CN) and 11% (AD) of different-day slices are flagged. That is above the 1% stranger rate, so some identity signal exists, but a generator that reproduces a training patient's anatomy in a new head position would mostly pass. A feature-space check is planned: whether Inception features find a patient's other visit more often than chance.
5. **Limitation: SSIM neighbours carry little class information.** The nearest training slice shares the test slice's class 59–61% of the time for CN and 51.5% for AD, against 50.7% CN among training slices. SSIM audits copying, not whether generated AD brains look like AD.

### 7.4 Implementation details

- **SSIM** follows Wang et al. (2004): 11 × 11 Gaussian window (σ = 1.5), K₁ = 0.01, K₂ = 0.03, data range 1, computed with valid convolution so borders never mix with padding. It agrees with scikit-image's `structural_similarity` to within 5 × 10⁻⁶ on test images.
- **Masked SSIM** averages the SSIM map over the union of both images' brain pixels (intensity > 10). Background that is black in both images is ignored, but brain present in only one image still counts against the pair.
- **Nearest-neighbour search everywhere**, because slice windows differ between scans and generated slices have no known position (Discovery 7). Each metric picks its own nearest neighbour. Per-image statistics are computed once, so each comparison only adds the cross term, run as batched GPU operations in PyTorch.
- **Per-class thresholds.** The masked thresholds for CN (0.716) and AD (0.694) are close, so an unconditional generator can be audited against a pooled threshold.
- **Resolution-specific.** These thresholds hold at 64 × 64 only and are recomputed at each training resolution.
- **Cost:** 77 queries per second against 24,520 training slices, 6.0 GB peak GPU memory, about 1.5 minutes for the full run on one A100.
- **Caveats.** The 99th percentile rests on about 14 slices per class from 68 validation patients, and slices from one patient are correlated; a patient-level bootstrap of the threshold is the next refinement if a generator's flag rate lands near 1%. Acquisition dates come from the timestamp in the metadata `raw` path, which behaves as an acquisition date: no pair from different days exceeds an SSIM of 0.99.

### 7.5 Reproducing the audit

```bash
sbatch audit.sh                      # full run, about 1.5 min on one A100
sbatch audit.sh --max_queries 500    # quick run with subsampled stranger queries
```

Results are written to `outputs/audit/` (gitignored; local split indices only, no patient IDs) and the figure to `figures/audit_ladder_r64.png`.

**Rangpur notes.** Jobs on the `comp3710` partition must be charged to the `comp3710` account (`#SBATCH --account=comp3710`), because the partition only accepts that account and a student's default account differs. The nodes expose no schedulable memory, so `--mem` must be omitted; otherwise the job waits indefinitely with reason `PartitionConfig`. The course guide's example `job.sh` includes `--mem=16G`.

---

## 8. Reproducing the Dataset Audit

All statistics and figures in Sections 3 and 4 come from `data_audit.py`. On Rangpur:

```bash
srun --partition=cpu --time=00:45:00 --pty bash
conda activate torch
python data_audit.py            # prints statistics, writes figures/ next to the script
exit
```

The script prints only aggregate statistics and never writes patient IDs to disk.

---

## 9. Dependencies

All experiments ran on UQ's Rangpur cluster: one NVIDIA A100-PCIE-40GB per job, Slurm partition `comp3710`.

| Package | Version | Used for |
|---|---|---|
| Python | 3.11.15 | |
| PyTorch | 2.13.0 (CUDA 13.0) | all models, training, SSIM audit |
| torchvision | 0.28.0 | image I/O, sample grids |
| NumPy | 2.4.6 | evaluation and plotting only (not in model code) |
| SciPy | 1.17.1 | FID matrix square root |
| scikit-learn | 1.9.0 | PCA and t-SNE (`embed.py`) |
| matplotlib | 3.11.1 | figures |
| Pillow | 12.3.0 | JPEG loading |
| pytorch-fid | 0.3.0 | standard FID Inception-v3 weights |

**Reproducibility.** The patient split is deterministic (seed 42, `random.Random`); training seeds are fixed (`--seed`, default 42); evaluation samples use a fixed latent seed (`predict.py --seed`, default 0), including StyleGAN2's per-pixel noise. GPU convolutions are not bit-for-bit deterministic, so retraining reproduces the reported trends and approximately, not exactly, the reported numbers. Most configurations were trained once; the recommended one twice (Section 11.13).

## 10. Usage

All commands run from `recognition/StyleGAN2_ADNI_s4883100/` on Rangpur. Every `.sh` script is a Slurm batch script that passes its arguments through to the Python script of the same name.

**Rangpur notes.** Jobs on the `comp3710` partition must use `#SBATCH --account=comp3710`, and must not request `--mem` (the nodes expose no schedulable memory; such jobs wait forever with reason `PartitionConfig`). The scripts here already do both.

```bash
# 1. Data: audit and preprocessing smoke test (CPU node)
srun --partition=cpu --cpus-per-task=8 --time=00:45:00 --pty bash
conda activate torch
python data_audit.py
python dataset.py --resolution 64          # builds the cached 64 px tensors, checks the split
exit

# 2. Calibrate the memorisation audit on real data (about 2 min)
sbatch audit.sh

# 3. Train (about 25 min for WGAN-GP, 67 min for StyleGAN2 at 64 px, 107 min at 128 px)
sbatch train.sh --dataset oasis --run_name wgan_oasis64                       # verify the baseline port
sbatch train.sh --dataset adni --run_name wgan_adni64                         # baseline
sbatch train.sh --model stylegan2 --dataset oasis --run_name sg2_oasis64      # verify StyleGAN2
sbatch train.sh --model stylegan2 --dataset adni --run_name sg2_adni64        # main model
sbatch train.sh --model stylegan2 --dataset adni --run_name sg2_adni64_r1g01 --r1_gamma 0.1
sbatch train.sh --model stylegan2 --dataset adni --run_name sg2_adni64_ada --ada
sbatch train.sh --model stylegan2 --dataset adni --run_name sg2_adni64_ada_t04 --ada --ada_target 0.4
sbatch train.sh --model stylegan2 --dataset adni --run_name sg2_adni64_augp02 --aug_p 0.2           # recommended
sbatch train.sh --model stylegan2 --dataset adni --run_name sg2_adni64_augp02_s2 --aug_p 0.2 --seed 7
sbatch train.sh --model stylegan2 --dataset adni --run_name sg2_adni64_cond --conditional
sbatch train.sh --model stylegan2 --dataset adni --run_name sg2_adni64_cond_augp02 --conditional --aug_p 0.2
sbatch audit.sh --resolution 128                                              # calibrate the audit at 128 px first
sbatch --time=06:00:00 train.sh --model stylegan2 --dataset adni --resolution 128 --run_name sg2_adni128_augp02 --aug_p 0.2
# add --resume (and a larger --iters) to continue a run from its last checkpoint

# 4. Evaluate every snapshot: memorisation audit, fidelity, trade-off figure, direct comparison
sbatch predict.sh --run_name wgan_adni64 --snapshots all
sbatch predict.sh --run_name sg2_adni64 --snapshots all --compare wgan_adni64 --compare_excess wgan_adni64
sbatch predict.sh --run_name sg2_adni64 --snapshot 10000                     # figures for one snapshot

# 5. Embedding, failure autopsy and latent analysis
sbatch embed.sh --runs wgan_adni64:10000 sg2_adni64:10000 sg2_adni64_ada:15100
sbatch embed.sh --runs sg2_adni64:10000 sg2_adni64_cond:10000 --tag embedding_tsne_cond
sbatch autopsy.sh --run_name sg2_adni64 --iter 10000 --copy_iter 12500
sbatch latent.sh --run_name sg2_adni64 --iter 10000
sbatch latent.sh --run_name wgan_adni64 --iter 10000
sbatch autopsy.sh --run_name sg2_adni64_augp02 --iter 17500 --copy_iter 20000   # recommended checkpoint
sbatch latent.sh --run_name sg2_adni64_augp02 --iter 17500

# 6. Class fidelity: train the AD/CN classifier (about 1 min) and classify generated cohorts
sbatch classify.sh --runs wgan_adni64:10000 sg2_adni64:10000 sg2_adni64_augp02:17500 sg2_adni64_cond:15100
# add --eval_only to reuse the saved classifier
```

**Outputs.** Training writes to `outputs/runs/<run_name>/` (config, loss history and curves, fixed-noise sample grids, generator snapshots every 2,500 iterations, a resumable checkpoint). Evaluation writes JSON results to `outputs/eval/<run_name>/` and figures to `figures/`. `outputs/` is gitignored; no weights or data are committed.
`figures/` also contains per-run diagnostics not shown here (loss curves, trade-off plots, audit histograms and nearest-neighbour grids for every run), produced by the commands above.

**Example input and output.** The input to a trained generator is a random latent vector; the output is a 64 × 64 greyscale slice in [−1, 1]. `predict.py` loads a snapshot, generates 3,100 slices and prints tables such as:

```text
 iter metric   flagged    rate            95% CI   NN p5  NN p50  NN p95
10000 masked       174   5.61% [ 4.86%,  6.48%]  0.5081  0.6229  0.7063
      set      FID  precision   recall  density  coverage
gen 10000    13.67      0.347    0.237    0.369     0.421
```

## 11. Results

All results come from 3,100 generated slices per snapshot (every 2,500 iterations), compared with the 3,100 held-out test slices (68 patients) and the 24,520 training slices. Every run uses the same data, batch size (64) and training length (15,100 iterations, 966 thousand real images) unless noted, at 64 × 64 unless noted:

| Run | Model | Change |
|---|---|---|
| `wgan_adni64` | WGAN-GP baseline | ported unchanged from Lab 2 |
| `sg2_adni64` | StyleGAN2 | default R1 weight (γ = 0.0128) |
| `sg2_adni64_r1g01` | StyleGAN2 | stronger R1 (γ = 0.1) |
| `sg2_adni64_ada` | StyleGAN2 | ADA, standard target 0.6 |
| `sg2_adni64_ada_t04` | StyleGAN2 | ADA, target 0.4 |
| `sg2_adni64_augp02` | StyleGAN2 | fixed augmentation probability 0.2 (seed 42), continued to 30,200 iterations |
| `sg2_adni64_augp02_s2` | StyleGAN2 | as above, second seed (7) |
| `sg2_adni64_cond` | StyleGAN2 | class-conditional |
| `sg2_adni64_cond_augp02` | StyleGAN2 | class-conditional + fixed augmentation 0.2 |
| `sg2_adni128_augp02` | StyleGAN2 | fixed augmentation 0.2 at 128 × 128, continued to 30,200 iterations |

### 11.1 Verifying both implementations on OASIS

Before ADNI, both models were trained on OASIS 64 × 64, where a known-good result exists (my Lab 2 WGAN-GP). The ported baseline reproduced Lab 2: its Wasserstein estimate fell to a minimum of 2.5 around epoch 20 and drifted up to 4.1 by epoch 100 (Lab 2: about 3.2 and 4.3), and its samples matched in realism and variety. StyleGAN2 at the same budget produced visibly sharper ventricles, folds and outlines with similar variety. Problems on ADNI can therefore be attributed to the data or the task, not to the implementations.

| WGAN-GP on OASIS | StyleGAN2 on OASIS |
|---|---|
| ![WGAN-GP OASIS](figures/wgan_oasis64_samples.png) | ![StyleGAN2 OASIS](figures/sg2_oasis64_samples.png) |

### 11.2 Fidelity

| Run, iteration | FID ↓ | Precision | Recall | Density | Coverage |
|---|---:|---:|---:|---:|---:|
| Baseline, 10,000 | 48.4 | 0.056 | 0.322 | 0.039 | 0.080 |
| Baseline, 15,100 | 49.2 | 0.085 | 0.288 | 0.060 | 0.107 |
| StyleGAN2, 10,000 | 13.7 | 0.347 | 0.237 | 0.369 | 0.421 |
| StyleGAN2, 15,100 | 12.6 | 0.285 | 0.341 | 0.259 | 0.387 |
| Stronger R1, 15,100 | 12.0 | 0.280 | 0.426 | 0.272 | 0.392 |
| ADA (target 0.6), 15,100 | 11.4 | 0.305 | 0.397 | 0.299 | 0.417 |
| *Real validation vs test (reference)* | *6.3* | *0.331* | *0.331* | *0.351* | *0.478* |

FID (Heusel et al., 2017) compares Inception-v3 feature distributions (lower is better); precision and recall follow Kynkäänniemi et al. (2019), density and coverage Naeem et al. (2020). Precision and density measure realism (generated slices that fall where real slices do); recall and coverage measure how much of the real variety is reproduced. The real-vs-real row is what a perfect generator would score at these sample sizes: 20 near-identical adjacent slices per scan keep all four neighbourhood metrics low even for real data, so values are judged against it.

StyleGAN2 is about 4× better than the baseline on FID, and at 10,000 iterations its precision and density match real data and its coverage reaches 88% of the real reference (baseline: 17%). The baseline's recall (0.32) looked respectable but was inflated by off-manifold samples; coverage, which only uses real slices' neighbourhoods, exposed this. Differences between the three StyleGAN2 variants (about 1 FID point) are within what single runs can resolve. The spec's SSIM target (> 0.6) is met: the median nearest-neighbour SSIM of StyleGAN2 slices to real training slices is 0.80 over the whole image and 0.62 over the brain.

### 11.3 Memorisation

| Run, iteration | Masked flag rate | Excess over control (95% CI) | Copy-level slices | Max masked SSIM |
|---|---:|---|---:|---:|
| Baseline, 10,000 | 2.29% | +0.0056 [−0.0027, +0.0134] | 0 | 0.754 |
| Baseline, 15,100 | 4.19% | +0.0067 [−0.0016, +0.0142] | 0 | 0.760 |
| StyleGAN2, 10,000 | 5.61% | −0.0000 [−0.0084, +0.0078] | 0 | 0.788 |
| StyleGAN2, 15,100 | 5.81% | +0.0040 [−0.0044, +0.0118] | 0 | 0.785 |
| Stronger R1, 15,100 | 7.00% | +0.0063 [−0.0019, +0.0139] | 0 | 0.778 |
| ADA (target 0.6), 15,100 | 5.19% | +0.0048 [−0.0035, +0.0124] | 0 | 0.784 |
| *Real unseen test patients* | *0.61%* | | | |

Three findings:

1. **The flag rate is confounded with realism.** StyleGAN2 is flagged 5–7% of the time against the baseline's 0.5–4.2%, yet its slices are about as close to unseen validation patients as to training patients. Realistic, typical slices resemble real brains in general more closely than real strangers resemble each other, so a more realistic generator is flagged more whatever it has memorised. The clearest case: at 5,000 iterations StyleGAN2 is flagged 7.0% against the baseline's 0.5%, yet its training-specific excess is no higher (difference −0.0018, 95% CI [−0.0042, +0.0005]).
2. **The matched-reference test separates the two.** It asks whether generated slices are closer to training patients than to equally many unseen patients, with real test slices as a control (Section 7). Comparing two models directly cancels the control's noise. **StyleGAN2 is pulled towards its training patients significantly less than the baseline** at every snapshot from 7,500 iterations (masked difference at 10,000: −0.0056, 95% CI [−0.0079, −0.0033]; at 15,100: −0.0027 [−0.0051, −0.0005]; whole-image intervals below zero from 7,500 to 15,100).
3. **Near-copies are rare but not eliminated.** A copy-level slice has masked SSIM ≥ 0.796, the 5th percentile of adjacent slices from the same scan. Across all 64 px StyleGAN2 runs and snapshots, at most one slice in 3,100 reached it in any snapshot: default StyleGAN2 and ADA (target 0.6) at 12,500 iterations, ADA (target 0.4) at 5,000, the second augmentation seed at 10,000 and 12,500, and the first augmentation seed at 20,000 (and once at the whole-image copy level, at 27,500). The baseline, the conditional runs and the 128 px run produced none at any snapshot. Flagged slices spread across many training slices and patients (e.g. StyleGAN2 at 15,100: 180 flags over 176 training slices from 109 patients), so the flags reflect general realism, not copies (Section 11.7).

**All 64 px StyleGAN2 variants at 15,100 iterations:**

| Run | Masked flag rate | Excess over control (95% CI) | Excess minus default StyleGAN2's (95% CI) |
|---|---:|---|---|
| Default StyleGAN2 | 5.81% | +0.0040 [−0.0044, +0.0118] | n/a |
| Stronger R1 | 7.00% | +0.0063 [−0.0019, +0.0139] | +0.0024 [−0.0002, +0.0049] |
| ADA, target 0.6 | 5.19% | +0.0048 [−0.0035, +0.0124] | no significant difference (95% CI includes zero) |
| ADA, target 0.4 | 5.94% | +0.0002 [−0.0082, +0.0081] | **−0.0037 [−0.0062, −0.0012]** |
| Fixed p = 0.2, seed 42 | 5.65% | −0.0013 [−0.0097, +0.0066] | **−0.0053 [−0.0080, −0.0026]** |
| Fixed p = 0.2, seed 7 | 5.39% | −0.0028 [−0.0109, +0.0051] | **−0.0068 [−0.0093, −0.0041]** |
| Conditional | 6.81% | +0.0019 [−0.0064, +0.0095] | −0.0021 [−0.0047, +0.0006] |

Only the runs whose augmentation actually engaged are significantly less pulled towards their training patients than default StyleGAN2 (bold). Section 11.5 explains why.

### 11.4 The fidelity-versus-memorisation trade-off

![Trade-off across snapshots](figures/sg2_adni64_ada_tradeoff.png)

*FID and coverage against the masked flag rate for every snapshot of all four runs (labels are iterations; the star is real unseen patients).*

For every model, realism improves steeply early and then flattens, while resemblance to the training data keeps rising. The baseline's FID stops improving after about 10,000 iterations while its flag rate nearly doubles. All three StyleGAN2 variants show a small, late training-specific drift: the excess is near zero up to 10,000 iterations and rises to +0.004 to +0.006 by 15,100. No individual excess is significant, but the trend is consistent across runs and subsets.

### 11.5 What causes the late drift, and what reduces it

From about 9,000 iterations, default StyleGAN2's discriminator sharpened: its loss fell from 1.0 to 0.68, its real and fake scores separated from ±0.7 to ±1.3, and its R1 gradient penalty rose from about 10 to 20–35. This coincided with the drift towards training patients (Section 11.4). Four experiments tested why:

1. **Stronger R1 (γ = 0.1): the "sharp discriminator" hypothesis is rejected.** The discriminator stayed near balance throughout (final loss 1.11, scores ±0.38, R1 about 2), yet the drift still appeared; at 15,000–15,100 iterations its excess was slightly *higher* than default StyleGAN2's (whole-image difference +0.0019, 95% CI [+0.0005, +0.0035]). Preventing the sharpening did not prevent the drift.
2. **ADA with the standard target (0.6) barely engaged.** Its augmentation probability stayed below 0.043, because its overfitting signal r_t = E[sign(D(real))] averaged about 0.5, below the target. With 24,520 training slices, the standard trigger never registered overfitting, and the run behaved like default StyleGAN2.
3. **Fixed-probability augmentation (p = 0.2) reduces the drift, and the result replicates.** In two runs with different seeds, the discriminator stayed balanced (loss about 1.2, scores ±0.3) and the excess was significantly below default StyleGAN2's at every snapshot from 7,500 iterations (at 15,100: −0.0053 [−0.0080, −0.0026] and −0.0068 [−0.0093, −0.0041]). The two seeds do not differ significantly from 7,500 iterations on (at 15,100: −0.0015 [−0.0041, +0.0012]), and neither's own excess rises above the control within that budget (on continued training, see below).
4. **ADA with a lower target (0.4) engages and also reduces the drift.** p rose to 0.21 by about 5,000 iterations, fell back to about 0.04 as the discriminator balanced, then rose to 0.12. Its excess was significantly below default StyleGAN2's from 12,500 iterations (at 15,100: −0.0037 [−0.0062, −0.0012]).

**Interpretation.** Constraining how sharply the discriminator judges (R1) does not help; preventing it from seeing exact training images (augmentation) does. This is consistent with the drift being driven by a discriminator that memorises the precise appearance of training slices and pulls the generator towards them, which is the original motivation for ADA (Karras et al., 2020b). On this dataset ADA's default target is too high to trigger; a fixed probability or a lower target is needed.

**Does augmentation make longer training safe?** Fixed p = 0.2 (seed 42) was continued to 30,200 iterations, twice the standard budget:

| Fixed p = 0.2 | FID | Precision | Recall | Density | Coverage | Excess over control (masked, 95% CI) |
|---|---:|---:|---:|---:|---:|---|
| 15,100 iterations | 13.1 | 0.555 | 0.183 | 0.668 | 0.491 | −0.0013 [−0.0097, +0.0066] |
| 17,500 | 11.9 | 0.481 | 0.183 | 0.556 | 0.480 | −0.0016 [−0.0101, +0.0060] |
| 25,000 | 12.9 | 0.409 | 0.275 | 0.429 | 0.455 | +0.0005 [−0.0078, +0.0083] |
| 30,200 | **10.8** | 0.441 | 0.289 | 0.506 | 0.485 | +0.0018 [−0.0064, +0.0094] |
| *Default StyleGAN2, 15,100* | *12.6* | *0.285* | *0.341* | *0.259* | *0.387* | *+0.0040 [−0.0044, +0.0118]* |

Longer training keeps improving it: FID reaches 10.8 (the best of any run) and recall recovers to 0.29, so the narrow, over-dense sample set seen at 15,100 iterations broadens towards real-data balance. The excess creeps up slowly, from −0.0013 to +0.0018, never significantly above zero and still below default StyleGAN2's +0.0040 at half the training. The discriminator also sharpens slowly late in training (real scores from about ±0.3 to ±0.55, R1 from about 10 to 17). **Augmentation delays and dampens the drift rather than abolishing it**, and rare near-copies still occur (one at 20,000 iterations).

![Fixed p = 0.2 to 30,200 iterations](figures/sg2_adni64_augp02_30k_tradeoff.png)

*The fixed p = 0.2 run (seed 42) across all snapshots to 30,200 iterations, with default StyleGAN2 and the baseline.*

| Default StyleGAN2 | Stronger R1 (γ = 0.1) | ADA, target 0.6 |
|---|---|---|
| ![Default](figures/sg2_adni64_losses.png) | ![Stronger R1](figures/sg2_adni64_r1g01_losses.png) | ![ADA](figures/sg2_adni64_ada_losses.png) |
| **ADA, target 0.4** | **Fixed p = 0.2 (seed 42)** | **Fixed p = 0.2 (seed 7)** |
| ![ADA 0.4](figures/sg2_adni64_ada_t04_losses.png) | ![Fixed p](figures/sg2_adni64_augp02_losses.png) | ![Fixed p seed 7](figures/sg2_adni64_augp02_s2_losses.png) |

![Trade-off with augmentation](figures/sg2_adni64_ada_t04_tradeoff.png)

*FID and coverage against the masked flag rate for the augmentation runs, default StyleGAN2 and the baseline.*

**What augmentation costs.** Both augmented runs improve fidelity and coverage over default StyleGAN2 at 15,100 iterations, but fixed p = 0.2 concentrates its samples in the densest, most typical regions of the real data:

| 15,100 iterations | FID | Precision | Recall | Density | Coverage |
|---|---:|---:|---:|---:|---:|
| Default StyleGAN2 | 12.6 | 0.285 | 0.341 | 0.259 | 0.387 |
| ADA, target 0.4 | 11.4 | 0.453 | 0.255 | 0.486 | 0.461 |
| Fixed p = 0.2, seed 42 | 13.1 | 0.555 | 0.183 | 0.668 | 0.491 |
| Fixed p = 0.2, seed 7 | 13.7 | 0.467 | 0.212 | 0.509 | 0.471 |
| *Real validation vs test* | *6.3* | *0.331* | *0.331* | *0.351* | *0.478* |

Fixed p = 0.2's density is about twice the real level and its recall about half: very realistic slices, from a narrower generated set. ADA (target 0.4) is more balanced. FID varies by up to about 3 points between the two seeds at the same iteration, so FID differences of 1–2 points between configurations are within run-to-run variation.

### 11.6 Coverage of classes and slice positions

![t-SNE embedding](figures/embedding_tsne.png)

*Joint t-SNE (van der Maaten and Hinton, 2008) of Inception features: real test slices coloured by class and by slice position, then each model's generated slices over the real slices in grey. Local neighbourhoods are meaningful; distances between clusters are not.*

The real slices are organised mostly by slice position: the lateral ends of each scan's window cluster on the left and the middle positions on the right and bottom. AD slices concentrate in the upper-right region. The baseline's slices form a separate cluster barely overlapping the real data, consistent with its low precision and coverage. Both StyleGAN2 snapshots spread across the map and cover the full range of slice positions, but leave the AD-rich upper-right region largely uncovered.

A 10-nearest-neighbour class vote in Inception space was tried first. Its features barely separate the classes (real validation slices: 56.4% balanced accuracy), and it suggested generated cohorts look AD-like far less often than real slices (StyleGAN2 13.4% against 34.0%). The dedicated classifier at the end of this section, which separates the classes far better, does not confirm this, so the class-vote shares are not used as evidence.

The failure autopsy measures the coverage gap directly: for default StyleGAN2 at 10,000 iterations, 71.9% of real AD test slices have no generated slice in their Inception neighbourhood, against 47.0% of CN slices; for the recommended configuration at 17,500 iterations, 65.7% against 41.3% (Section 11.7).

**Coverage by class.** For every run, the share of real test slices of each class that some generated slice reaches (Inception neighbourhoods, k = 5):

| Run, iteration | Real CN reached | Real AD reached | AD as share of reference |
|---|---:|---:|---:|
| *Real validation vs test (reference)* | *0.537* | *0.402* | *100%* |
| Default StyleGAN2, 10,000 | 0.530 | 0.281 | 70% |
| ADA (target 0.4), 15,100 | 0.579 | 0.310 | 77% |
| Fixed p = 0.2 (seed 42), 15,100 | 0.617 | 0.329 | 82% |
| Fixed p = 0.2 (seed 7), 15,100 | 0.590 | 0.320 | 80% |
| Conditional, 15,100 | 0.496 | 0.275 | 68% |
| Fixed p = 0.2 (seed 42), 17,500 (recommended) | 0.587 | 0.343 | 85% |
| Fixed p = 0.2 (seed 42), 30,200 | 0.598 | 0.342 | 85% |
| Conditional + fixed p = 0.2, 10,000 | 0.562 | 0.312 | 78% |
| Conditional + fixed p = 0.2, 15,100 | 0.503 | 0.274 | 68% |

**Class conditioning did not close the coverage gap.** In Inception space the label has a small effect: AD-labelled slices reach real AD slices more often than CN-labelled ones (0.199 vs 0.179 at 15,100; 0.220 vs 0.172 at 10,000). But AD-labelled slices still reach real CN slices more than real AD ones, overall AD coverage is unchanged (0.275), and in the embedding the AD-rich region stays largely empty. The dedicated classifier below shows that the label does strongly control class-visible features, so the problem is the range of AD anatomy reproduced, not whether the label is followed. Likely reasons: AD's distinguishing anatomy (atrophy, enlarged ventricles) is subtle at 64 × 64, AD comes from fewer training patients (177 vs 365), and the label partly shifts slice level (Discovery 9). **Augmentation improved AD coverage more than conditioning did.**

**Combining conditioning with augmentation does not close the gap either.** In Inception space it strengthens the label's effect (at 10,000 iterations AD-labelled slices reach 0.262 of real AD slices against 0.197 for CN-labelled ones, a wider gap than conditioning alone), though the dedicated classifier separates its two labels slightly less well than conditioning alone (below), but overall AD coverage at 15,100 iterations (0.274) is no better than conditioning alone and below augmentation alone (0.329). It does confirm the privacy effect a third time: its excess is significantly below the conditional-only run's from 10,000 iterations (at 15,100: −0.0041 [−0.0066, −0.0017]), indistinguishable from unconditional augmentation (−0.0009 [−0.0036, +0.0017]), and it produced no copy-level slices.

| Conditional | Conditional + fixed p = 0.2 |
|---|---|
| ![Conditional samples](figures/sg2_adni64_cond_samples.png) | ![Conditional + augmentation samples](figures/sg2_adni64_cond_augp02_samples.png) |

*Rows 1–4 requested CN, rows 5–8 requested AD.*

![Conditional embedding](figures/embedding_tsne_cond.png)

*Joint t-SNE with the conditional run's slices coloured by requested class.*

**Dedicated AD/CN classifier: class fidelity measured directly.** Because Inception features barely separate the classes, a small CNN (`classify.py`: four convolutional blocks, 1.17 M parameters) was trained on the real training slices with a class-weighted loss and light translation and intensity augmentation (no flips), selected on validation balanced accuracy (epoch 17 of 30; 67 s on one A100) and reported on test. Two checks guard against the slice-level confound of Discovery 9: a level-only predictor (the AD share of training slices at each absolute slice index), and the CNN's accuracy within single slice indices, where level cannot help.

| Real slices | Validation | Test |
|---|---:|---:|
| CNN balanced accuracy / AUROC | 0.687 / 0.759 | 0.783 / 0.863 |
| Level-only balanced accuracy / AUROC | 0.661 / 0.702 | 0.636 / 0.667 |
| CNN balanced accuracy within a single slice index | | median 0.783 (0.693–0.864, 30 indices) |
| Inception 10-NN class vote, balanced accuracy | 0.564 | |

The classifier reads disease-related anatomy, not only slice level: with level held fixed it still reaches 0.78 on test. It is a moderate classifier, though: on validation it beats the level-only predictor only narrowly, the gap between validation and test reflects only 68 patients per split, and its training loss fell to about 0.01, so it overfits (selection on validation limits the damage).

**Unconditional cohorts: no shortage of AD-classified slices.** A generator that reproduced the training class mix (49.3% AD) would be classified AD 44.5% of the time, given the classifier's test error rates (true-positive rate 0.732, false-positive rate 0.166); real test slices are classified AD 41.4% of the time (patient-bootstrap 95% interval 30.1–52.5%).

| Cohort (3,100 slices), iteration | Classified AD (95% CI) | Label fidelity: AD-requested → AD / CN-requested → CN |
|---|---|---|
| *Real test slices* | *0.414 [0.301, 0.525]* | |
| Baseline, 10,000 | 0.541 [0.523, 0.558] | |
| StyleGAN2, 10,000 | 0.539 [0.521, 0.556] | |
| StyleGAN2, 15,100 | 0.605 [0.588, 0.622] | |
| ADA (target 0.4), 15,100 | 0.532 [0.514, 0.549] | |
| Fixed p = 0.2, 17,500 (recommended) | 0.543 [0.525, 0.560] | |
| Fixed p = 0.2, 30,200 | 0.486 [0.469, 0.504] | |
| Fixed p = 0.2 (seed 7), 15,100 | 0.561 [0.544, 0.579] | |
| Conditional, 15,100 | 0.557 [0.540, 0.575] | 0.955 / 0.829 |
| Conditional + fixed p = 0.2, 10,000 | 0.543 [0.526, 0.561] | 0.857 / 0.762 |
| Conditional + fixed p = 0.2, 15,100 | 0.528 [0.511, 0.546] | 0.886 / 0.819 |

*Cohort intervals are Wilson intervals over slices and do not include the classifier's own uncertainty.*

![AD/CN classifier](figures/classifier_adcn.png)

*Left: the classifier's AD probability for real test slices and each generated cohort. Right: the share of each cohort classified AD, against real test slices (band) and the share expected for a faithful generator (dashed).*

Three conclusions:

1. **The Inception class vote was misleading.** Every generated cohort is classified AD at least as often as real slices (49–61% against 41%), so the earlier conclusion that unconditional generation produces too few AD-like slices does not hold. What the evidence supports is narrower: generators reach less of the *variety* of real AD anatomy (coverage, and the dropped-mode autopsy row), especially large, high-contrast AD slices.
2. **The absolute AD share does not rank generators.** The baseline, whose slices barely overlap the real data (precision 0.056), is classified AD as often as StyleGAN2 (0.541 against 0.539). Generated slices are outside the data the classifier was trained on, and its output on them reflects that shift as much as class content, so differences between cohorts (e.g. 0.605 against 0.486) are not read as differences in class fidelity.
3. **Conditioning works: the requested class is clearly visible.** Within each conditional run, which cancels most of that shared shift, 86–96% of AD-requested slices are classified AD and 76–83% of CN-requested slices CN, more separable than real AD and CN slices are to the same classifier (true-positive rate 0.73, true-negative rate 0.83). Some of this may be slice level, which the label also shifts (Discovery 9) and which the classifier partly uses; the two cannot be separated for generated slices. Adding augmentation slightly lowers the agreement (0.886 against 0.955 for AD-requested slices at 15,100).


### 11.7 Failure autopsy

![Failure autopsy](figures/sg2_adni64_augp02_iter017500_autopsy.png)

*Rule-selected failure cases from all 3,100 slices of the recommended snapshot (StyleGAN2 with fixed augmentation p = 0.2, 17,500 iterations; the near-copy row uses 20,000, where the audit found a copy-level slice). Each pair shows the case and its nearest neighbour. Selected by `autopsy.py`, not by hand. The same autopsy for default StyleGAN2 at 10,000 iterations is in `figures/sg2_adni64_iter010000_autopsy.png`.*

To avoid cherry-picking, each row shows the top three slices under a fixed rule, and every generated case appears beside its nearest training slice, so the figure is also a memorisation check.

| Rule | What the cases show | Likely cause | Memorised? |
|---|---|---|---|
| **Near-copy** (highest masked SSIM to training) | The top case (0.808) matches its training neighbour in outline, head tilt, ventricle curve and the white-matter band into the temporal lobe; only fine folds differ. The next two (0.770, 0.766) share the head position but differ in folds and outline | Slow late-training drift towards training patients (Section 11.5); isolated: one slice in 3,100, none at 17,500 iterations | Borderline for the top case, just above the copy level of 0.796; the release filter (Section 11.12) removes such slices |
| **Least like real** (lowest masked SSIM to any training slice, 0.129–0.169) | An enlarged brain filling much of the frame with mottled bright tissue; a brain with a dark, broken lower half and bright specks; a stretched, elongated brain with smeared bright patches | Malformed anatomy, not rare-but-valid modes: these are much further from any real slice than default StyleGAN2's worst cases (0.345–0.349). The enlarged and stretched shapes are consistent with the scale and rotation augmentation occasionally leaking into the generator, a known risk of discriminator augmentation (Karras et al., 2020b), though p = 0.2 is well below where leaking is usually reported; this was not tested | No |
| **Off-manifold** (farthest outside real slices' Inception neighbourhoods, 1.28–1.32 × radius) | Bright, washed-out tissue with stretched outlines; one twisted, compressed brain with smeared folds | GAN texture and shape failure, slightly further out than default StyleGAN2's (1.22–1.24 ×). Overall, fewer generated slices lie outside every real neighbourhood (30.5% against 43.1% for default StyleGAN2 at 10,000), consistent with its higher precision | No (SSIM to training 0.21–0.30) |
| **Dark** (lowest mean brain intensity, 46–49 of 255) | Dim, low-contrast slices | Faithful to dim but valid scans in the training data (Discovery 11), not a generator fault | No (0.31–0.46) |
| **Dropped mode** (real AD test slices farthest from any generated slice, 1.61–1.64 × radius) | Large, high-contrast AD slices with enlarged ventricles and pronounced sulci; their nearest generated slices are smaller and lower in contrast | The generator under-reproduces typical AD anatomy, as for every run. 65.7% of real AD test slices have no generated slice in their neighbourhood, against 41.3% of CN slices (default StyleGAN2 at 10,000: 71.9% and 47.0%). A plausible cause: AD slices come from fewer training patients (177 vs 365 CN, with more repeat scans) | Not applicable |

**Recommended configuration versus default StyleGAN2.** Augmentation trades typical-case quality and coverage against rarer but worse outliers: it reaches more real slices of both classes and places fewer slices off the real manifold, but its worst slices are clearly malformed where default StyleGAN2's were merely unusual.

| Uncurated, seed 1 | Uncurated, seed 2 |
|---|---|
| ![Uncurated seed 1](figures/sg2_adni64_augp02_iter017500_uncurated_seed1.png) | ![Uncurated seed 2](figures/sg2_adni64_augp02_iter017500_uncurated_seed2.png) |

*Uncurated 8 × 8 grids from the recommended snapshot. Most slices are realistic and varied in size, position and contrast; a few per grid are dim, and about one per grid is visibly malformed (e.g. the smeared bright slice at the bottom right of seed 2). The autopsy cases are exceptions, not typical output.*

Two practical consequences: the release filter should drop malformed slices as well as copies (Section 11.12), and the dropped-mode row is the concrete form of the AD coverage gap in Section 11.6.

### 11.8 Latent-space analysis

**Interpolation.** For 40 pairs of latents, 9 frames each, every frame was matched to its nearest training slice (`latent.py`). StyleGAN2 interpolates linearly in its style space W; the baseline spherically in z.

| Run, iteration | Paths reaching copy level | Different training neighbours per 9-frame path | Mid-path rise in similarity |
|---|---:|---:|---:|
| Baseline, 10,000 | 0 / 40 | 7.2 | +0.035 |
| StyleGAN2, 10,000 | 0 / 40 | 7.9 | +0.073 |
| StyleGAN2 + fixed p = 0.2, 17,500 (recommended) | 0 / 40 | 7.7 | +0.075 |

Both StyleGAN2 models morph smoothly, and each path passes near about eight different training slices, so they do not snap from one stored image to the next. The slight rise in similarity mid-path is the realism confound again (midpoints in W drift towards the average, most typical slice), not memorisation. In the "generated to training scan" row, the final frame is the inversion of that training slice, so its high similarity (0.84 for the recommended model) is expected by construction.

![Interpolation, recommended configuration](figures/sg2_adni64_augp02_iter017500_interpolation.png)

*Interpolation for the recommended configuration. Default StyleGAN2's is in `figures/sg2_adni64_iter010000_interpolation.png`.*

**Inversion membership test.** For one slice from each of 64 training patients and 64 unseen test patients, the latent that best reproduces the slice was found by gradient descent (500 steps). A generator that memorised its training patients would reproduce their slices better:

| Run, iteration | Training patients | Unseen patients | Difference (95% CI) |
|---|---:|---:|---|
| Baseline, 10,000 | 0.797 | 0.775 | **+0.021 [+0.007, +0.035]** |
| StyleGAN2, 10,000 | 0.863 | 0.858 | +0.004 [−0.005, +0.015] |
| StyleGAN2 + fixed p = 0.2, 17,500 (recommended) | 0.845 | 0.841 | +0.004 [−0.007, +0.015] |

*Reconstruction quality is masked SSIM between each slice and its inversion.*

The baseline reproduces its training patients significantly better than unseen patients; neither StyleGAN2 model shows a detectable difference, while both reproduce real slices much better than the baseline. The recommended model reconstructs slightly less well than default StyleGAN2 (0.845 against 0.863), consistent with its narrower sample set at this checkpoint (recall 0.18, Section 11.5). This independent test agrees with the matched-reference comparison (Section 11.3): the baseline carries a measurable training-patient signal, StyleGAN2 does not, with or without augmentation.

| Baseline (10,000) | StyleGAN2 (10,000) | Recommended (17,500) |
|---|---|---|
| ![Baseline inversion](figures/wgan_adni64_iter010000_inversion.png) | ![StyleGAN2 inversion](figures/sg2_adni64_iter010000_inversion.png) | ![Recommended inversion](figures/sg2_adni64_augp02_iter017500_inversion.png) |

### 11.9 Resolution: 128 × 128

The recommended configuration (fixed p = 0.2) was also trained at 128 × 128, with the audit recalibrated at that resolution (thresholds are resolution-specific).

**The audit discriminates better at 128 px.** With more detail, real strangers resemble each other less (masked median 0.35, against 0.59 at 64 px), so the gap between strangers and copies widens: 95% of strangers score below 0.42, while the copy level is 0.70. The calibration still transfers (test false-flag rate 0.1–1.4%), every near-duplicate is flagged, and so are 94% (CN) and 75% (AD) of same-day repeats. Same-day CN repeats are also less identical than they looked at 64 px (21% above 0.99, against 58%).

![128 px audit ladder](figures/audit_ladder_r128.png)

**Results at 15,100 iterations**, each against its own resolution's real reference:

| | FID (reference) | Precision | Recall | Density | Coverage (reference) | Real AD reached (reference) |
|---|---|---:|---:|---:|---|---|
| 64 px, fixed p = 0.2 | 13.1 (6.3) | 0.555 | 0.183 | 0.668 | 0.491 (0.478) | 0.329 (0.402) |
| 128 px, fixed p = 0.2 | 16.9 (6.0) | 0.508 | **0.016** | 0.556 | 0.394 (0.488) | 0.274 (0.403) |

The 128 px samples show much sharper cortical folds, and the run shows no memorisation signal: its excess stays negative (−0.0024 [−0.0088, +0.0039] at 15,100), no slice comes near copy level (highest masked SSIM 0.52, against 0.70), and its discriminator stayed balanced. But at the same real-image budget it has not converged: FID was still falling steeply (24.1 at 10,000 to 16.9 at 15,100), and its recall is very low, meaning its samples occupy a narrow, dense region of the real data. Its AD coverage is lower than at 64 px.

**Continued to 30,200 iterations**, the 128 px run improves but does not catch up:

| 128 px, fixed p = 0.2 | FID (reference 6.0) | Precision | Recall (reference 0.31) | Coverage (reference 0.49) | Real AD reached (reference 0.40) | Excess (masked, 95% CI) |
|---|---:|---:|---:|---:|---:|---|
| 15,100 iterations | 16.9 | 0.508 | 0.016 | 0.394 | 0.274 | −0.0024 [−0.0088, +0.0039] |
| 27,500 | 13.4 | 0.507 | 0.071 | 0.443 | 0.339 | −0.0023 [−0.0088, +0.0039] |
| 30,200 | 14.2 | 0.489 | 0.071 | 0.435 | 0.312 | −0.0027 [−0.0092, +0.0035] |

FID improves from 16.9 to 13.4–14.2 and coverage reaches about 90% of the reference, but recall stays very low (0.07 against 0.31): the generated set remains much narrower than the real data, and AD coverage (77% of the reference) stays below 64 px at the same budget (85%). Memorisation stays absent throughout: the excess is negative at every snapshot and the most similar slice (masked SSIM 0.55) is far below copy level (0.70). A likely cause of the low diversity is capacity: with the channel formula used here, the 128 × 128 layers have only 64 channels, and the 128 px generator has barely more parameters than the 64 px one (8.63 M vs 8.39 M). **At 128 px the model is sharper and memorises nothing measurable, but it is not yet the better generator**; wider high-resolution layers and longer training are the next steps.

| Samples (15,100 iterations) | Nearest training neighbours (15,100 iterations) |
|---|---|
| ![128 px samples](figures/sg2_adni128_augp02_samples.png) | ![128 px nearest](figures/sg2_adni128_augp02_nearest.png) |
| **Samples (30,200 iterations)** | **Nearest training neighbours (30,200 iterations)** |
| ![128 px samples at 30,200](figures/sg2_adni128_augp02_30k_samples.png) | ![128 px nearest at 30,200](figures/sg2_adni128_augp02_30k_nearest.png) |

### 11.10 Resource profiling

| | WGAN-GP | StyleGAN2 | StyleGAN2, conditional | StyleGAN2, 128 px |
|---|---:|---:|---:|---:|
| Generator parameters | 3,574,656 | 8,386,958 (mapping 2,101,248) | 8,650,638 (mapping 2,364,928) | 8,629,073 |
| Discriminator / critic parameters | 2,764,481 | 5,854,337 | 5,855,105 | 5,973,185 |
| Training time per iteration (batch 64) | 95 ms | 260 ms (263 ms with augmentation) | 260 ms | 420 ms |
| Training time, 15,100 iterations | 25 min | 67–68 min | 67 min | 107 min (another 109 min to 30,200) |
| Peak GPU memory, training | 0.43 GB | 5.6 GB | 5.6 GB | 9.7 GB |
| Generation time per 1,000 slices | 10 ms | 815 ms | not measured | not measured |
| Audit calibration (`audit.py`) | about 2 min at 64 px | | | 7 min at 128 px |

All on one NVIDIA A100-PCIE-40GB. The whole project used about 24 A100-hours across 52 GPU jobs (from `sacct`): about 15.8 h of training, 7.9 h of evaluation and 0.3 h for audit calibration and the other analyses. That is within the 30 A100-hour estimate in the feasibility review. The longest single job was the 128 px continuation (109 minutes of training).

### 11.11 Acceptance criteria

| Criterion (Section 2) | Result |
|---|---|
| 1. Fidelity: lower FID than the baseline; SSIM > 0.6 | **Met.** Every StyleGAN2 variant beats the baseline by about 4× on FID (11.4–13.7 vs 48.4); median nearest-neighbour SSIM 0.80 (whole) and 0.62 (brain) |
| 2. No copies among 3,100 slices | **Not reliably met without a filter.** Most snapshots have none, but five runs produced a single copy-level slice in one or two snapshots each. Met for any cohort screened by the release filter (Section 11.12) |
| 3. Flag rate and excess no higher than the baseline's at equal or better FID | **Met on excess, not on flag rate.** The flag rate measures realism as much as memorisation (Section 11.3); the excess, which separates the two, is significantly lower for StyleGAN2 than for the baseline, and lower still with augmentation |
| 4. Coverage across both classes and slice positions | **Partly met.** Slice positions are covered. Real AD slices are under-covered by every model; augmentation narrows the gap (80–85% of the reference, against 70% without), conditioning does not, with or without augmentation (Section 11.6). A dedicated AD/CN classifier finds no shortage of AD-classified slices, so the gap is in the variety of AD anatomy reproduced, not in the number of AD-like slices |
| 5. One A100, jobs ≤ 12 h, ≤ 24 GPU-hours per configuration | **Met.** 64 px: about 67 minutes and 5.6 GB per run; 128 px: 107 minutes and 9.7 GB per 15,100 iterations |

### 11.12 Recommendation to the project manager

**Use StyleGAN2 with fixed-probability discriminator augmentation (p = 0.2) at 64 × 64, trained to about 17,500 iterations, and screen every cohort before release.**

- **Why this configuration and checkpoint.** In two independent runs it is significantly less pulled towards its training patients than default StyleGAN2 from 7,500 iterations on, and the effect replicates in the conditional model. At 17,500 iterations it combines FID 11.9, real-level coverage (0.480), 85% of the real AD coverage and an excess below the control (−0.0016). Training on to 30,200 iterations gives the best FID of any run (10.8) and better recall, at the cost of a slow upward creep in the excess (+0.0018, not significant); that is a reasonable choice if realism matters more than privacy margin. ADA with target 0.4 is an alternative with similar benefits.
- **Release safeguard (essential, not optional).** Run every generated cohort through the audit (`predict.py`) and drop any slice at or above copy level before sharing. Near-copies are rare (at most one in 3,100 per snapshot) but occur even in augmented runs. Do not judge a cohort by its flag rate alone; use the matched-reference excess. Also drop malformed slices: the recommended model's rare worst outputs are enlarged, stretched or smeared brains (Section 11.7), so a screen on distance from the real data (e.g. slices far outside every real Inception neighbourhood, or with very low similarity to any training slice) should be added, with its threshold calibrated on real validation slices before use.
- **Not yet suitable for training or testing AD classifiers.** Every generator reaches less of the variety of real AD anatomy than of CN anatomy (Section 11.6), especially large, high-contrast AD slices, so a classifier trained on a synthetic cohort would see a narrow picture of AD. Class-conditional generation is the right starting point for class-balanced cohorts: a dedicated classifier recognises the requested class in 86–96% of AD-requested slices. But conditioning did not widen AD coverage, even with augmentation, and part of what the label controls may be slice level rather than disease anatomy (Discovery 9). Wider coverage of AD anatomy, verified with level-matched comparisons, is needed first.
- **Privacy caveat.** The audit barely recognises a patient across visits (Section 7.3), so it cannot rule out a generator reproducing a training patient's anatomy in a new head position. The inversion test (Section 11.8) found no training-patient advantage for StyleGAN2, including the recommended configuration at its recommended checkpoint, which is reassuring but not proof. A feature-space identity check is advisable before any release outside the research team.

### 11.13 Limitations and future work

- **Few repeats.** Only the recommended configuration was replicated (two seeds). Other configurations ran once, so differences of 1–2 FID points between them cannot be attributed to the configuration; seeds alone moved FID by up to 3 points. The large effects (StyleGAN2 vs the baseline; augmentation vs none on the excess) hold across runs.
- **Resolution.** The main comparison is at 64 × 64. At 128 × 128 the recommended configuration stayed far less diverse (recall 0.07) even after 30,200 iterations, probably because its high-resolution layers are narrow (Section 11.9).
- **Checkpoint selection on test data.** Checkpoints were compared on test-set metrics (FID and the matched-reference excess both use the test patients), so the chosen checkpoint's numbers are slightly optimistic. A held-out selection split would remove this bias.
- **Class information.** Inception-v3 features and pixel SSIM are both nearly blind to AD-specific anatomy. The dedicated classifier is better (0.78 test balanced accuracy, also within single slice levels) but moderate and partly level-sensitive, and on generated slices its absolute AD share is affected by the shift away from real data, so only within-model comparisons (conditional label fidelity) are interpreted.
- **Validation sample size.** The audit threshold rests on 68 validation patients and the matched-reference control on 68 test patients, which is why single-run excess intervals are wide; direct model comparisons avoid this.
- **Future work:** AD-balanced or AD-oversampled conditional training to widen AD coverage, audited per class, since showing each of the 177 AD training patients more often is exactly the condition under which memorisation of minority-class patients should rise; a stronger, level-matched AD/CN classifier (e.g. trained only on slice indices shared by both classes); conditioning on slice level as well as class, to separate disease anatomy from level (Discovery 9); wider high-resolution layers and longer training at 128 px; a feature-space identity audit; more seeds per configuration; a held-out split for checkpoint selection.

## 12. Artificial Intelligence Usage Disclosure

This disclosure follows the UQ Library *Guide to acknowledging and referencing AI*, which asks for the AI tool, how it was used, the prompts, the part of the work affected, and the date. A verification column is added as required by the COMP3710 report specification (Section 6.1). The log is updated as the project progresses.

### 12.1 Tools used

| Tool | Provider | Access |
|---|---|---|
| Claude Opus 5.5 | Anthropic | claude.ai web interface (project workspace) |

### 12.2 Usage log

| Date | AI tool | How it was used | Prompt(s) (summarised or quoted) | Part of the work | How I verified it |
|---|---|---|---|---|---|
| 28/09/2026 | Claude Opus 5.5 | **Planned / brainstormed:** compared the Hard-difficulty projects against my previous lab work; I chose StyleGAN2 on ADNI | "Give me my best options, hard difficulty only…"; "how about style GAN? … would it connect with what i've done in the assessments?" | Project choice; Sections 1 and 6 | Checked each option's difficulty band, dataset, and targets against Section 2 of the report specification |
| 28/09/2026 | Claude Opus 5.5 | **Instructions:** step-by-step guidance for forking the repository, the folder skeleton, `.gitignore`, and GitHub SSH access from Rangpur | "maybe we should fork the repo first and setup the vs code environment first?"; "lets do the ssh stuff now with rangpur" | Repository structure, `.gitignore`, Git workflow | Confirmed the `topic-recognition` branch and project folder on GitHub; `ssh -T git@github.com` authenticated successfully; verified GitHub's host key fingerprint against GitHub's published value |
| 28/09/2026 | Claude Opus 5.5 | **Generated code:** shell and Python one-off commands to explore the ADNI folder structure, filename convention, metadata JSON, and patient-level overlap | Requests to inspect the ADNI dataset on Rangpur | Sections 3 and 4 (dataset structure, Discoveries 1–7) | Ran every command myself on Rangpur and reviewed the raw output; checked internal consistency (e.g. scans per patient sums to 680 patients and 1,526 scans) |
| 28/09/2026 | Claude Opus 5.5 | **Brainstormed / planned:** proposed the patient-level re-split and the memorisation threshold calibrated from repeat visits of the same patient (distance ladder); I adopted both | "For the design, I like the measure of how similar two scans of the same person across visits can be used as a principled memorisation threshold, can we include that in the design too?" | Sections 5.2 and 7 | Reasoned through each design choice against the leakage evidence in Section 4; implementation and validation still to come |
| 28/09/2026 | Claude Opus 5.5 | **Generated code:** wrote `data_audit.py` (dataset checks and README figures) | "lets update the readme, have to document the preprocessing steps … could be good to have some graphs generated at this stage too" | `data_audit.py`; `figures/`; Section 8 | Ran it on the full dataset on Rangpur; its statistics matched my earlier independent command outputs (e.g. 216 overlapping patients, 1,526 scans); inspected the generated figures |
| 28/09/2026 | Claude Opus 5.5 | **Drafted / edited:** drafted README Sections 3–8 from my audit results, then updated them with the full audit output (including Discovery 9) | Same as above, plus pasting the `data_audit.py` output | README Sections 3–8 | Checked every number in the README against the Rangpur output |
| 28/09/2026 | Claude Opus 5.5 | **Drafted:** this disclosure table, following UQ Library guidance located via web search | "can you add an AI usage referencing table to the readme in accordance with uq guidelines?? include the date - 28/09/2026" | Section 12 | Checked the required fields against the UQ Library guide linked below |
| 28/09/2026 | Claude Opus 5.5 | **Edited:** changed the AI reference in 12.4 to UQ's general APA format with the tool's web address, and added it to the reference list | "yes add the tool's web address to those references" | Sections 12.4 and 13 | Checked the format against the APA 7th "General AI references" example in the UQ Library guide |
| 28/09/2026 | Claude Opus 5.5 | **Planned:** ordered the next stages (data pipeline → threshold calibration → WGAN-GP baseline → feasibility review → StyleGAN2) | "ok all done, git is up to date. Where to next?" | Project plan; Section 2 (feasibility review) | Checked the plan against the feasibility check-off requirements in Section 3 of the report specification |
| 28/09/2026 | Claude Opus 5.5 | **Generated code:** wrote `dataset.py` (slice indexing, patient-level stratified split, padding/resizing, cached preloading, OASIS verification loader, smoke test) | "yes" (in response to the proposal to write `dataset.py` with the index, patient-level split, and preloading) | `dataset.py`; Sections 5.1 and 5.2 | Indexing and split logic were tested on synthetic data (correct 544/68/68 patient split, deterministic across runs). Ran the smoke test on Rangpur: the leakage assertion passed, the split table matched the designed patient counts (544/68/68), and the real-image grids in `figures/dataset_batch_check.png` and `figures/oasis_batch_check.png` showed correctly preprocessed brain slices. The split became 542/68/68 after the two corrupted CN training scans were excluded (Section 5.1, step 3) |
| 05/10/2026 | Claude Opus 5.5 | **Generated code / analysed:** proposed cropping to the brain region and wrote the brain-extent measurement and crop support in `data_audit.py` and `dataset.py` | "yes" (in response to the proposal to crop to the brain region) | `data_audit.py`; `dataset.py` (`ADNI_CROP`); Discovery 10 | Ran the measurement on Rangpur. Claude's visual estimate of brain size from the batch grid (about 40% of the frame) was wrong; the measurement showed about 65% and a full-frame crop, so I rejected cropping based on the evidence |
| 05/10/2026 | Claude Opus 5.5 | **Generated code:** one-off diagnostic scripts to locate outlier slices, per-scan intensity, edge-touching scans, and the second-darkest scan, plus `scp` commands to retrieve figures | Requests to run and retrieve the diagnostics (e.g. "give me the scp cmds") | Discoveries 11–12 | Inspected every diagnostic figure myself: two scans are corrupted, three edge-touching scans are anatomically normal |
| 05/10/2026 | Claude Opus 5.5 | **Generated code:** relative-intensity exclusion rule applied after the patient split in `dataset.py`; folded the diagnostics into `data_audit.py` | Continuation of the outlier investigation, after I uploaded the figures and output | `dataset.py`; `data_audit.py`; Section 5.1 | Full audit on Rangpur flagged exactly the 2 corrupted scans; smoke test passed the leakage check, removed 2 scans (40 slices) from train only, and left val and test unchanged |
| 05/10/2026 | Claude Opus 5.5 | **Drafted:** README updates for Discoveries 10–12 and Sections 3.4, 5.1, 5.2, 6, and 8 | Pasting the audit and smoke-test output | README Sections 3–6, 8 | Checked every number against the Rangpur output |
| 05/10/2026 | Claude Opus 5.5 | **Generated code:** wrote `audit.py` part 1 (batched GPU SSIM, near-duplicate, same-person and different-person rungs, per-class thresholds) and the Slurm script `audit.sh` | "let me give you access to the repo … Read dataset.py from my fork" | `audit.py`, `audit.sh`; Section 7 | Claude checked its SSIM against scikit-image (agreement within 5 × 10⁻⁶) and the chunked search against brute force on synthetic data; I reviewed the code and ran it on ADNI on Rangpur (job 632476) |
| 05/10/2026 | Claude Opus 5.5 | **Debugged:** diagnosed why the job sat in `PartitionConfig`. Claude's first `audit.sh` copied `--mem` from the course guide and omitted `--account`, and its `source activate` line received the script's arguments | Pasted `squeue`, `scontrol show partition`, `scontrol show job` and `sacctmgr` output | `audit.sh`; Section 7.5 | Confirmed with `scontrol` (partition memory 10M in total, `AllowAccounts=comp3710`, job charged to my personal account) and `sacctmgr`; the corrected job ran with an empty error log |
| 06/10/2026 | Claude Opus 5.5 | **Analysed / generated code:** Claude pointed out that the first run's same-person threshold (~1.0) was invalid (the design it had proposed on 28/09), hypothesised duplicate acquisitions, and wrote a visit-pair diagnostic dating each scan from its metadata path | Pasted the first `audit.py` output and the structure of a metadata `raw` path (patient ID masked) | `audit.py`; Discoveries 13–14 | Ran the diagnostic on Rangpur (job 634430): near-identical pairs occur only on the same day, and different-day visits are misaligned (median best-match offset 4 slices) |
| 06/10/2026 | Claude Opus 5.5 | **Generated code:** restructured the audit around validation-calibrated stranger thresholds, with same-session and different-day pairs as reference rungs. I asked whether this was worthwhile before agreeing. Committed as `0c93c41`, which by mistake reuses the previous commit's message | "do you think it is worthwhile restructuring the threshold? if that will give us better findings and a better stylegan ultimately, i think we should." | `audit.py`; Section 7 | Claude tested it on synthetic patients with planted copies; I ran it on ADNI (job 634470): test false-flag rate 0.3–0.9% against the nominal 1%, and at least 99.2% of near-duplicates flagged |
| 06/10/2026 | Claude Opus 5.5 | **Drafted:** README updates for Discoveries 5, 8, 13 and 14 and Sections 6 and 7, plus these log rows | "yes, lets do it" (after uploading the audit figure) | README Sections 4, 6, 7 and 12 | Checked every number against the output of jobs 632476, 634430 and 634470 |
| 18/10/2026 | Claude Opus 5.5 | **Generated code:** ported my Lab 2 WGAN-GP into `modules.py` (stages scale with resolution) and wrote `train.py`/`train.sh` (iteration-based training, logging, versioned snapshots, resume) | "yes lets do it" and "its in task3_gan https://github.com/cocoagoodwin101-ui/comp3710-lab2-pattern-recognition.git" | `modules.py`, `train.py`, `train.sh`; Section 11.1 | Claude loaded Lab 2 weights into the port (identical outputs). I trained it on OASIS on Rangpur and compared losses and samples with my Lab 2 results before training on ADNI |
| 18/10/2026 | Claude Opus 5.5 | **Generated code:** `predict.py` (generated-sample audit), then neighbour-concentration and copy-level checks, then the matched-reference test. Claude's first concentration comparison turned out to be confounded (test slices come in correlated runs of 20) | "yes" (to each proposed check) | `predict.py`; Section 11.3 | Claude tested each check with planted copies on synthetic data; I ran every version on Rangpur. I accepted Claude's correction that the concentration comparison could not measure diversity |
| 22/10/2026 | Claude Opus 5.5 | **Generated code / analysed:** paired-by-subset excess with a patient-level bootstrap. The result showed Claude's earlier claim ("clearly present from about 10,000 iterations") overstated the baseline's drift | "robustness refinement first" | `predict.py`; Section 11.3 | Ran on Rangpur (job 685336); the 95% intervals include zero, and the README reports the drift as consistent but not significant |
| 22/10/2026 | Claude Opus 5.5 | **Drafted:** the one-page feasibility review and a check-off study sheet; Claude revised the review when I asked whether it was sufficient (added aim and plan, made criterion 3 comparative) | "you sure that one-page feasibility review is sufficient??" | Section 2 | Checked against the four required parts in Section 3 of the specification; to be presented at the check-off on 27/10/2026 |
| 23/10/2026 | Claude Opus 5.5 | **Generated code:** FID and improved precision/recall with a real-vs-real reference; later density/coverage and the trade-off figure | "yes, and also, are we getting the stylegan done now (have we finished the baseline??)" | `predict.py`; Sections 11.2 and 11.4 | Claude checked the metrics on distributions with known answers (identical, shifted, half the modes dropped, diffuse outliers); I ran them on all snapshots on Rangpur |
| 23/10/2026 | Claude Opus 5.5 | **Generated code:** StyleGAN2 generator, discriminator and losses in `modules.py`; StyleGAN2 training in `train.py`; snapshot support in `predict.py` | "yes, committed already, lets move forward with that order" | `modules.py`, `train.py`, `predict.py`; Section 1 | Claude checked the modulated convolution against a per-sample reference, demodulation variance, shapes at 64–256 px and gradients. I verified the implementation by training on OASIS and comparing with the baseline before ADNI |
| 24/10/2026 | Claude Opus 5.5 | **Generated code / analysed:** direct model comparison of excess and single-snapshot evaluation; interpreted StyleGAN2's high flag rate as a realism confound | "yes" | `predict.py`; Section 11.3 | Claude tested the comparison with a planted difference; on Rangpur (job 692112) StyleGAN2's excess was significantly below the baseline's from 7,500 iterations |
| 24/10/2026 | Claude Opus 5.5 | **Generated code / planned:** adaptive discriminator augmentation for greyscale MRI and the `--ada` option; proposed the stronger-R1 run as a test of its discriminator-sharpening hypothesis, which the run then rejected | "yes, if you recommend that, after we finish that how close are we to finishing?" | `modules.py`, `train.py`; Section 11.5 | Claude unit-tested augmentation and the p update; I ran both experiments. The stronger-R1 result contradicted Claude's proposed mechanism, and the README reports the hypothesis as rejected |
| 25/10/2026 | Claude Opus 5.5 | **Generated code:** `embed.py` (joint t-SNE and a class vote) and `autopsy.py` (rule-selected failure cases and uncurated grids) | "yes" | `embed.py`, `autopsy.py`; Sections 11.6 and 11.7 | Tested on synthetic data; ran on Rangpur and inspected every figure myself |
| 25/10/2026 | Claude Opus 5.5 | **Drafted:** README Sections 1, 6, 9, 10 and 11 and these log rows | "yes" (to drafting the README results) | README | Checked every number against the Rangpur job outputs |
| 26/10/2026 | Claude Opus 5.5 | **Analysed / drafted:** diagnosed the failure-autopsy rows and drafted Section 11.7, the Section 11.6 coverage sentence and the resource figures; advised leaving out the "43.1% off-manifold" figure, which has no real-data reference | Pasting the `autopsy.py`, timing and `sacct` output and uploading the autopsy figures | README Sections 11.6–11.8 | Checked each diagnosis against the figure myself, and every number against the job outputs |
| 26/10/2026 | Claude Opus 5.5 | **Planned / generated code:** checked the specification against the README and found latent interpolation missing; wrote `latent.py` (interpolation with nearest-neighbour tracking and an inversion membership test) and the `--aug_p` fixed-probability augmentation option | "about 4 days. i feel like we can try and get it done." | `latent.py`, `modules.py`, `train.py`; Sections 11.5, 11.8 | Claude checked inversion on images the generator itself made (reconstruction SSIM 0.25 → 0.90) and that p stays fixed; I ran both on Rangpur |
| 26/10/2026 | Claude Opus 5.5 | **Generated code:** class-conditional StyleGAN2 (class-embedded mapping network, projection discriminator), labelled training batches, per-class coverage in `predict.py` and class colouring in `embed.py` | "yes write conditional stylegan2 if you can write it without having any of the results/output from those queued jobs" | `modules.py`, `train.py`, `predict.py`, `embed.py`; Section 11.6 | Claude checked old unconditional snapshots still load and that the label changes both networks; I trained and evaluated it on Rangpur |
| 27/10/2026 | Claude Opus 5.5 | **Debugged (Claude's bug):** Claude's edit to `predict.py` moved a `@torch.no_grad()` decorator off `generate`, so StyleGAN2 generation kept gradients and four evaluation jobs ran out of GPU memory | Pasting the `.err` tracebacks | `predict.py` | I found the out-of-memory errors in the `.err` files; Claude fixed the decorator and verified gradients are off; the four jobs were rerun successfully |
| 27/10/2026 | Claude Opus 5.5 | **Analysed:** results of fixed-probability augmentation, ADA target 0.4, conditioning and the latent analysis. Claude revised its own earlier claims: the late drift is reduced by augmentation, not by constraining the discriminator; and its claim of "no copy-level slices" for fixed p = 0.2 was withdrawn after the second seed produced two | Pasting the evaluation logs and figures | Sections 11.3, 11.5, 11.6, 11.8 | Checked every number against the job outputs; the second seed was run specifically to test the first result |
| 27/10/2026 | Claude Opus 5.5 | **Planned:** the 128 px sensitivity run, a second seed, and continuation and combination runs | "yes, wait so after this there are absolutely no more experiments to do?? could we maybe try increasing resolution from 64x64 upwards to see if that will improve results?" | Sections 11.5, 11.9 | Ran all on Rangpur; reported 128 px as not converged rather than as a better or worse model |
| 28/10/2026 | Claude Opus 5.5 | **Drafted:** README updates for the extension experiments and these log rows | "yes lets do that" | README Sections 5, 6, 10, 11, 12 | Checked every number against the job outputs before committing |
| 28/10/2026 | Claude Opus 5.5 | **Analysed / drafted:** results of the continued augmentation run, conditional + augmentation and the continued 128 px run; wrote the text replacing the README placeholders, including a revised checkpoint recommendation (17,500 iterations) and a new limitation (checkpoints chosen on test data), and applied all final edits to this README | Pasting the six job logs and the `sacct` output, uploading the figures, and "can i just give you the readme.md file and you give me back the full copy and pasteable updated finalised one?" | README Sections 11.3–11.13 | Checked every number against the job outputs and read the final README end to end before committing |
| 28/10/2026 | Claude Opus 5.5 | **Reviewed / planned:** read the fork end to end against the specification and marking criteria; found that the failure autopsy, uncurated grids and latent analysis used default StyleGAN2 at 10,000 iterations rather than the recommended checkpoint, an ambiguous ADA row in the fidelity table, and functions without docstrings; proposed the order of remaining work and extensions | "I want to finalise this report … As it stands how close am i to getting full marks? what more can i do?" | Whole README; code | I checked each finding against the README and code on GitHub before acting on it |
| 28/10/2026 | Claude Opus 5.5 | **Generated code:** one-line docstrings for public functions that lacked them (no code changes); two of Claude's first drafts overstated what a function does and were corrected after Claude checked them against the code | "lets as much of it done today as possible!!!" | All `.py` files | Pyflakes and compile checks passed; I reviewed the diff (37 added lines, docstrings only) before committing |
| 28/10/2026 | Claude Opus 5.5 | **Analysed / drafted:** interpreted the autopsy and latent analysis of the recommended checkpoint (jobs 709155, 709168) and drafted Sections 11.7 and 11.8 and the related edits; identified the new "malformed outlier" finding and presented the augmentation-leaking explanation as an untested hypothesis | Pasting both job logs and uploading the five figures | README Sections 1, 11.6–11.8, 11.12 | Checked every number against the two logs; the AD coverage from the autopsy (0.343, 85% of the reference) independently matched the value from `predict.py` |
| 28/10/2026 | Claude Opus 5.5 | **Generated code:** wrote `classify.py` and `classify.sh` (AD/CN CNN, level-only confound baseline, within-level accuracy, faithful-generator reference, conditional label fidelity). Claude's synthetic test exposed two of its own errors, both fixed before the Rangpur run: a wrong parameter count in a docstring, and a confound baseline built on within-window position, which cannot detect the class-dependent slice level of Discovery 9 | "yes" (to the proposed classifier), "lets as much of it done today as possible!!!" | `classify.py`, `classify.sh` | Tested on synthetic data with a planted anatomy difference and a planted level confound: the level-only baseline detected the confound, the within-level check isolated the anatomy signal, and a positive-control generator scored 98–102% of the expected share. Ran on Rangpur (job 709310) |
| 28/10/2026 | Claude Opus 5.5 | **Analysed / drafted:** interpreted the classifier results and drafted the new Section 11.6 block and the revisions they forced: the Inception class vote's "too few AD-like slices" conclusion, which Claude had drafted on 25/10, was withdrawn, and the claim that conditioning has a small effect was corrected | Pasting the job log and uploading the figure | README Sections 1, 6, 10, 11.6, 11.7, 11.11–11.13 | Checked every number against job 709310; checked that the baseline's AD share matched StyleGAN2's before concluding that the absolute share does not rank generators |
| 28/10/2026 | Claude Opus 5.5 | **Reviewed / edited:** found works used or cited in the README but missing from the reference list (projection discriminator, FID, precision/recall, density/coverage, R1, Inception-v3, t-SNE) and added them | "is that everything you need?" | Sections 1, 6, 11.2, 13 | Checked each title, venue and year against the paper |

### 12.3 Verification approach

AI-generated commands and code were never trusted on their own output alone. Every statistic reported here comes from commands I ran on Rangpur, and counts were cross-checked between independent methods (one-off shell commands and `data_audit.py`). Design suggestions were adopted only after I checked them against the dataset evidence. I take full responsibility for all code, analysis, and claims in this project.

### 12.4 Reference (APA 7th)

Anthropic. (2026). *Claude* [Large language model]. https://claude.ai

Guidance followed: The University of Queensland Library. *Guide to acknowledging and referencing AI.* https://guides.library.uq.edu.au/referencing/acknowledging-and-referencing-ai

## 13. References

- T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," CVPR, 2019.
- T. Karras et al., "Analyzing and improving the image quality of StyleGAN," CVPR, 2020.
- S. Liu, S. Chandra, et al., "Style-based generative models for medical neuroimaging," IEEE Transactions on Medical Imaging, 2023.
- Alzheimer's Disease Neuroimaging Initiative (ADNI), http://adni.loni.usc.edu
- Anthropic. (2026). *Claude* [Large language model]. https://claude.ai
- Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.
- I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved training of Wasserstein GANs," in Proc. NeurIPS, 2017.
- T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila, "Training generative adversarial networks with limited data," in Proc. NeurIPS, 2020.
- T. Miyato and M. Koyama, "cGANs with projection discriminator," in Proc. ICLR, 2018.
- M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in Proc. NeurIPS, 2017.
- T. Kynkäänniemi, T. Karras, S. Laine, J. Lehtinen, and T. Aila, "Improved precision and recall metric for assessing generative models," in Proc. NeurIPS, 2019.
- M. F. Naeem, S. J. Oh, Y. Uh, Y. Choi, and J. Yoo, "Reliable fidelity and diversity metrics for generative models," in Proc. ICML, 2020.
- L. Mescheder, A. Geiger, and S. Nowozin, "Which training methods for GANs do actually converge?," in Proc. ICML, 2018.
- C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception architecture for computer vision," in Proc. CVPR, 2016.
- L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," Journal of Machine Learning Research, vol. 9, pp. 2579–2605, 2008.