# EEG Neural Signal Epoching — Code and Notes (repo_for_BCI)

**[English](README.md) | [简体中文](README.zh-CN.md)**

Code and notes from BCI experiments (joint work with a senior student) plus a self-run neural signal epoching task. The repository records the VR-trigger-mark based EEG segmentation workflow together with the background knowledge behind it.

## Repository Layout

| Path | Description |
|---|---|
| [`epoching.py`](epoching.py) | Detection of VR trigger marks (the `HL 2` channel) and splitting of continuous EEG into labeled epochs |
| [`notes.md`](notes.md) | Original handwritten study / experiment notes; the organized version is embedded in the "Study Notes" section below |
| [`README.md`](README.md) / [`README.zh-CN.md`](README.zh-CN.md) | Bilingual documentation (this file is English) |

## Experiments Covered

The same script was used across three VR-stimulation EEG experiments:

| # | Experiment | Classes produced |
|---|---|---|
| 1 | Handwritten Chinese character stroke classification | 横 / 竖 / 撇 / 捺 / 点 / 其他 |
| 2 | Emotion analysis (horror VR videos) | `V_49`, `V_92`, `V_212`, `V_208` (named by video length) |
| 3 | Visual stimulation with different object categories | `V1` … `V6` |

All three experiments recorded ECG in addition to EEG.

## Pipeline (`epoching.py`)

Two input files are used per recording:

- a version that **keeps** the `HL 2` trigger channel — used to locate the trigger points
- a version with `HL 2` **removed** (hand-saved from EEGLAB) — used to cut the data

### Trigger detection (improved dual-mode)

| Parameter | Value | Purpose |
|---|---|---|
| `baseline` | `-5550e-6` V | resting baseline |
| `relative_thresh` | `2200e-6` V | threshold on the deviation from the baseline |
| `gap_fill_sec` | `0.06` s | `binary_closing` to bridge short drops inside one block |
| `short_block_sec` / `long_block_sec` | `0.6` s / `0.8` s | block-length classification |
| `short_min_gap_sec` / `long_min_gap_sec` | `0.3` s / `2.0` s | dead zone between accepted triggers |

Steps:

1. Read the `HL 2` channel with `mne.io.read_raw_eeglab`, flatten it to 1-D and take `sfreq`
2. Threshold on `abs(data - baseline) > relative_thresh`
3. `binary_closing` to merge sub-60 ms gaps that belong to the same mark
4. `scipy.ndimage.label` to extract contiguous blocks, dropping blocks shorter than 5 samples
5. Apply the short / long dead zone so that one physical mark yields exactly one trigger
6. Visualize the signal, the baseline and the detected triggers with matplotlib

### Splitting into epochs

- `split_and_save(start, end, n_segments, category)` cuts a timestamp range into equal parts and writes every part into the folder named after its category
- intervals between consecutive marks are simply discarded (舍弃)
- `split_and_save_with_skip(..., skip_sec=3.0)` additionally drops the first N seconds of every segment

## Environment

```bash
pip install mne numpy scipy matplotlib
```

Python 3.10+ is recommended.

## Remarks

- `epoching.py` also records the EEGLAB "force old format" trick — MATLAB v7.3 `.set`/`.mat` files cannot be read directly, so the file must be re-saved with `version = '7'`
- Excluded from version control via `.gitignore`: `.idea/`, `.venv/`, `.github/`, `docs/` — the latter two are leftover scaffolding folders that are kept local-only
- The original handwritten notes are archived in [`notes.md`](notes.md); an organized version follows below

## Study Notes

### 1. EEG Basics

#### 1.1 Electrode Positions

**10-20 system**

| Position | Share of the nasion–inion line |
|---|---|
| Above the nasion | 10% |
| FP (prefrontal) | 20% |
| F (frontal) | 20% |
| C (central) | 20% |
| P (parietal) | 20% |
| O (occipital), downwards | 10% |
| Z | Midline |

- Odd numbers sit on the left of the midline, even (mirrored) numbers on the right (e.g. C1 and C2 are symmetric about the midline).

**10-10 system**

- Splits the 20% regions of the 10-20 system into two 10% halves, giving a denser electrode grid.

#### 1.2 Brain Regions and Functions

| Region | Code | Function |
|---|---|---|
| Prefrontal lobe | FP | Higher-order thinking |
| Frontal lobe | F | — |
| Central region | C | Motor |
| Parietal lobe | P | — |
| Occipital lobe | O | Vision |
| Temporal lobe | — | Below the parietal/frontal lobes; visual memory, language, emotion, long-term memory (hippocampus) |

#### 1.3 EEG Frequency Bands

**The 5 bands established by clinical experience**

| Band | Frequency | Amplitude | Associated state |
|---|---|---|---|
| delta | 0.1–4 Hz | 20–200 μV | Deep sleep, memory formation |
| theta | 4–8 Hz | 10–50 μV | Motor imagery, memory encoding (stronger on difficult tasks), fatigue |
| alpha | 8–12(13) Hz | 20–100 μV | Calm; easy to detect over occipital/parietal areas; the most prominent rhythm |
| beta | 12(13)–30 Hz | 5–20 μV | Movement, mirror movement |
| gamma | > 30 Hz | Low amplitude | Strongly associated with focused attention |

**Even-band splitting (an engineering approach)**

- Split into bands every 2 Hz, giving 25 bands (matching China's 50 Hz mains frequency).

#### 1.4 What EEG Really Measures

- EEG is bioelectricity in nature, but limited by today's technology what we actually record is essentially **postsynaptic membrane potentials**.

| Type | How it is recorded |
|---|---|
| EEG | Measured on the scalp |
| ECoG (electrocorticography) | Measured on the meninges without damaging them |
| Local field potential | Electrodes pushed deep into a brain region |

#### 1.5 Artifacts

- Besides the EEG signal itself, the recording also contains artifacts (noise).
- Artifact removal is therefore required.

#### 1.6 Hardware

**EEG cap**

- Active electrode
- Reference electrode
- Ground electrode

**Amplifier**

| Spec | Requirement | Note |
|---|---|---|
| Input impedance | > 10 GΩ | — |
| CMRR (common-mode rejection ratio) | > 100 dB | Amplifies the differential signal, rejects the common-mode signal |
| Frequency response | — | No information above 1000 Hz is needed |

**ADC (analog-to-digital converter)**

- Purpose: convert the analog signal into a digital one.
- Resolution: expressed in bits (2^bits levels), typically greater than 20.

### 2. Experiment Workflow

#### 2.1 Goal and Paradigm

- The experimenter should either work out a paradigm through practice or adopt a well-established one.
- Set the experimental goal first, then decide the protocol and paradigm.

#### 2.2 Four Acquisition Paradigms

**1) SSVEP (steady-state visual evoked potential)**

- Recorded over the occipital lobe, accuracy above 95%, driven by a PPT light stimulus.
- Pros: high recognition accuracy.
- Cons: binary classification only; not suitable for long-term wearing; risk of photosensitive epilepsy.

**2) Motor paradigms**

- Motor imagery (MI)
- Motor execution (ME)
- Development direction: brain-controlled object grasping.

**3) Affective paradigms**

**4) Disease diagnosis**

#### 2.3 Acquisition Advice

- **Electrodes**: wet Ag/AgCl electrodes are better; conductive gel beats saline.
- **Cap placement**: the front sits at the nasion, the back at the inion.
- **Subject comfort is a key guarantee of high-quality data**:
  1. Chat with the subject and help them relax
  2. Keep the session at a reasonable length
  3. Offer snacks, drinks and background music to relax them
  4. Offer a reward, and be ready to pay extra afterwards
- **Blinking**: ask the subject to control their blink cycle and blink during rest.
- **Monitor distance**: 1–2 m.
- **Do not use too many electrodes** — it only adds interference.

#### 2.4 Hardware Classification

| Type | Characteristics |
|---|---|
| Invasive | Good data quality; high cost, harm to the subject |
| Non-invasive | Portable, safe, high freedom for the experimenter; downside is drift |
| Interventional | — |

**Invasive**

- **Fully invasive**: surgically implanted, records spike discharges, electrodes inserted into the brain region (Prof. Hong Bo, Tsinghua University).
- **Semi-invasive**: the meninges are left intact and the electrode sits on top of them (Zhejiang University).
  - Pros: good data quality
  - Cons: high cost, harm to the subject
  - Techniques: microelectrode arrays, deep brain stimulation
  - Avoids interference from hair, scalp, skull, meninges and air cavities

**Non-invasive**

- Pros: portable, safe, high freedom for the experimenter.
- Cons: drift — signal drift (the EEG drifting during a trial) and electrode drift (the cap physically moving during the experiment).
- Channel counts of 2, 3, 16, 64, 256 are available; more channels give better data quality.

### 3. Data Preprocessing

- EEG signal = brain activity signal − the bioelectric signal picked up by the reference electrode.

#### 3.1 Recommendations

| Item | Recommendation |
|---|---|
| Sampling rate | 250 Hz or 1 kHz is enough |
| Environment and noise | Shielded room, active electrode system, notch filter |
| ADC | 20 bit or more; ideally no high-pass is needed; if the resolution is insufficient, use a 0.01 Hz or 0.1 Hz high-pass |
| Online acquisition | Keep the parameters permissive and filter as little as possible |

#### 3.2 Standard Steps

1. Import the EEG data into the computer (CSV, mat, daq, fif)
2. Plot a 3-D head map using the electrode positions of this session (positions supplied by an external database)
3. Filtering (mains/notch, low-pass, high-pass, band-pass, Butterworth)
4. Artifact removal (EOG, EMG)
5. *(optional)* Re-referencing — the reference electrode normally works fine; if it is broken, re-reference to the CZ electrode (the reference electrode sits on the earlobe)
6. **Epoching** — cut the continuous recording into samples according to the paradigm's timestamps
7. Reject bad segments and interpolate bad channels — drop noisy segments, fill in from other samples
8. *(optional)* Visualize the results
9. Save the data

#### 3.3 EEGLAB Walkthrough (MATLAB)

Start-up: set the path in MATLAB → load the EEGLAB folder → type `eeglab` in the command window.

**1. Import the data and inspect its basic contents**

- Use the `extensions - loadcurry` plug-in
- `import data` → `using EEG function and plugins` → `from Neuroscan file`

**2. Enter events and electrode locations**

- Events: the `.ceo` file normally holds the events; load it via `file - import event info - from ASCII file`
- Locations: `.dpo` is the electrode-position file, but the loadcurry plug-in already loads the bundled position file during the Import data step
  - Inspect the positions via `edit - channel locations - plot 2D/3D`
  - Channels F11, F12, FT11, FT12 turned out to have no position information
  - Trying `look up locs` in `edit - channel locations` (several files with Neuroscan headers) did not fill in the missing positions — it actually wiped the existing ones

**3. Re-referencing and downsampling**

- Re-reference: `tool - re reference` → pick a channel as the new reference. Curry 9's reference electrode was good, so no re-referencing was needed
- Downsampling: `tool - change sampling rate` → 250 Hz

**4. Filtering**

- `tool - filter the data` → band-pass filtering; channels that should not be filtered can be excluded
- After checking the literature this was set to 0.1–40 Hz (formal experiments usually use 8–30 Hz)

**5. Visualization**

- `plot - channel data (scroll)` shows all channels and lets you hunt for bad channels and bad segments

**6. Removing bad channels**

- **Delete outright**: manually `edit - select data - channel range`, enter the unwanted channels and tick remove (when a channel has no data at all and only a few channels are affected)
- **Interpolate**: `tools - interpolate electrodes - select from data channel` (when a bad channel still holds usable data worth recovering, or when many channels are bad)

**7. Removing bad segments**

- Select a segment in the visualization window → `reject` to drop it
  - norm and stack normalization are the most intuitive views for spotting artifacts over long stretches (large muscle artifacts, eye-movement artifacts)
  - How do you tell which segment is bad? Note down when the subject did something else during the experiment
- Run ICA via `tool - decompose data by ICA`
- `classify components by ICLabel` automatically flags eye-movement artifacts for removal

**8. Splitting the experimental data**

- Use the version that **keeps HL 2** to locate the split points, and the version with **HL 2 removed** to cut the recording into pieces and save them into folders
- See "5. Concrete Epoching Scheme" below for the details
- Open question: how do you verify each step? What if something goes wrong?

#### 3.4 Curry 9

- Curry 9 ships with its own preprocessing features.

### 4. Data Analysis

- When the number of usable samples is small, **weighted averaging over neighbouring brain regions** is applied first, and the result is then handed to machine learning / deep learning.
- The step that lifts the resolution of ML / DL is **feature extraction**, with these families of methods:
  - Time-domain analysis
  - Frequency-domain analysis
  - Spatial-domain analysis
  - Time-frequency analysis

The end goal: draw conclusions and make predictions.

### 5. Concrete Epoching Scheme: Handwritten Chinese Character Experiment

- Timestamps are numbered with plain integers (they correspond to `t[0]`, `t[1]`, … in `epoching.py`).
- **The interval between two neighbouring markers** is one piece of content, and **every other interval is discarded**.

| Timestamp interval | Treatment | Saved to |
|---|---|---|
| 1–2 | Discard | — |
| 2–3 | Split into 25 equal parts | `横` (horizontal) |
| 3–4 | Discard | — |
| 4–5 | Split into 25 equal parts | `竖` (vertical) |
| 5–6 | Discard | — |
| 6–7 | Split into 25 equal parts | `撇` (left-falling) |
| 7–8 | Discard | — |
| 8–9 | Split into 25 equal parts | `捺` (right-falling) |
| 9–10 | Discard | — |
| 10–11 | Split into 25 equal parts | `点` (dot) |
| 11–12 | Discard | — |
| 12–13 | Split into 40 equal parts (cycling 横, 竖, 撇, 捺 ten times) | matching stroke folder |
| 13–14 | Discard | — |
| 14–15 | Split into 90 equal parts (cycling 撇, 横, 竖, 撇, 点, 点, 点, 横, 竖 ten times) | matching stroke folder |
| 15–16 | Discard | — |
| 16–17 | Split into 10 equal parts | — |

The remaining intervals (23–24, 26–27, 28–29, 30–31, …) already have complete class sequences and call examples in `epoching.py`; just adapt them.

### 6. Open Questions

- How can the epoching process be verified step by step? How are mistakes traced and rolled back?
- Paths and thresholds in `epoching.py` currently have to be edited by hand; they could be turned into configuration options later.

