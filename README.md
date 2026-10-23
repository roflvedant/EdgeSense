# EdgeSense

Local acoustic machine fault detection for CSCI 49000-AIT, Fall 2026.

**Team:** Muhammad Ahmed Ahsan and Vedant Patil  
**Repository:** https://github.com/roflvedant/EdgeSense

EdgeSense analyzes prerecorded machine audio and classifies each one-second window as Normal or Abnormal. It uses MFCC features with logistic regression and compares the results against an RMS-energy baseline. Abnormal predictions produce a local inspection warning.

## Setup

Use Python 3.12 and Git. Open PowerShell and run:

```powershell
git clone https://github.com/roflvedant/EdgeSense.git
cd EdgeSense
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repository includes the selected audio recordings, trained model, and saved results. You do not need to download the dataset or retrain the model to run the demo.

Run all commands below from the repository root, where `requirements.txt` is located. The commands call the virtual environment's Python directly, so activation is not required.

## Run the demo

```powershell
.\.venv\Scripts\python.exe scripts/demo.py
```

The demo loads the saved model and analyzes one normal and one abnormal validation recording. Each ten-second recording produces ten one-second predictions.

The output includes:

- The windowâ€™s starting time.
- A Normal or Abnormal prediction.
- The modelâ€™s abnormal score.
- An inspection warning for abnormal predictions.

The script also writes:

- `results/demo_normal.json`
- `results/demo_abnormal.json`

These are validation examples, not reserved test recordings. The score is not calibrated confidence, and warnings do not control equipment.

## Run verification

```powershell
.\.venv\Scripts\python.exe scripts/verify.py
```

The verification script checks:

- All 120 audio files against their saved hashes.
- Recording-level split isolation.
- The training, validation, and test recording counts.
- Metrics recalculated from saved validation predictions.
- Finite 40-dimensional features and rejection of audio shorter than one second.

Successful checks print `PASS` messages.

## Save the console output

The committed demo and verification logs were regenerated on Vedant's Windows laptop using:

```powershell
.\.venv\Scripts\python.exe scripts/demo.py | Out-File -Encoding utf8 results/demo_console.txt

.\.venv\Scripts\python.exe scripts/verify.py | Out-File -Encoding utf8 results/verification.txt

Get-Content results/verification.txt
```

Running the scripts without redirection prints their output without replacing these text logs. Paths inside demo output reflect the computer used to generate it.

## Training and evaluation

The saved experiment was run on October 22, 2026, on Windows 11 using Python 3.12.10 and the dependencies in `requirements.txt`.

To reproduce training:

```powershell
.\.venv\Scripts\python.exe src/edgesense.py train
```

The dataset contains 80 normal and 40 abnormal recordings from MIMII valve ID 00 at +6 dB SNR. The included audio retains microphone channel 0 as mono audio at 16 kHz.

Recordings are split before windowing:

| Split | Normal recordings | Abnormal recordings | Total |
|---|---:|---:|---:|
| Training | 56 | 28 | 84 |
| Validation | 12 | 6 | 18 |
| Reserved test | 12 | 6 | 18 |

The model uses 40 features: the mean and standard deviation of 20 MFCC coefficients. A scaler is fitted on training features and applied before balanced logistic regression.

Both classification thresholds are selected using validation F1. The reserved test recordings are not opened for feature extraction or evaluation by the training command.

Retraining replaces the saved model and validation outputs. Rerun the demo and verification afterward if their saved outputs should reflect the new run.

## Saved results

Results are recorded in `results/metrics.json`.

| Validation metric | RMS baseline | MFCC classifier |
|---|---:|---:|
| Abnormal precision | 0.341 | 0.729 |
| Abnormal recall | 1.000 | 0.850 |
| Abnormal F1 | 0.508 | 0.785 |

The classifier improves on the RMS baseline but remains below the proposed F1 target of 0.85. These are preliminary validation results, not final test results.

### Windows laptop timing

| Measurement | Result |
|---|---:|
| Mean MFCC extraction plus inference | 1.76 ms |
| 95th percentile | 2.42 ms |
| Maximum | 3.18 ms |
| Mean model-only inference | 0.34 ms |
| Saved model bundle size | 2,867 bytes |

Timing was measured over 50 validation windows after warmup, with numerical libraries limited to one thread. Raw samples are stored in `results/latency_samples.csv`.

These measurements exclude audio acquisition, WAV decoding, disk access, process startup, and JIT compilation. They measure warm processing time, not total alert delay. Timing will vary between computers.

## Analyze another recording

```powershell
.\.venv\Scripts\python.exe src/edgesense.py infer --audio "path\to\recording.wav" --output prediction.json
```

Recordings must be at least one second long. The loader uses the first microphone channel and resamples input to 16 kHz when necessary. It analyzes complete one-second windows and reports any discarded incomplete tail.

## Repository contents

| Path | Contents |
|---|---|
| `src/edgesense.py` | Training, feature extraction, baseline, evaluation, timing, and inference |
| `scripts/demo.py` | Demonstration using the saved model |
| `scripts/verify.py` | Data integrity, split, metric, and feature checks |
| `scripts/download_subset.py` | Reproducible download of the selected dataset subset |
| `data/` | Selected WAV recordings and provenance manifest |
| `models/edgesense.joblib` | Trained scaler, classifier, thresholds, and configuration |
| `results/metrics.json` | Evaluation results and runtime environment |
| `results/latency_samples.csv` | Raw laptop timing samples |
| `results/validation_predictions.csv` | Per-window validation predictions |
| `results/splits.csv` | Recording-level split assignments |
| `results/confusion_matrices.png` | Validation confusion matrices |
| `results/demo_normal.json`, `results/demo_abnormal.json` | Demo predictions |
| `results/demo_console.txt`, `results/verification.txt` | Saved Windows console output |
| `results/experiment_notes.json` | Experiment notes |
| `docs/EdgeSense Midterm Progress Report FINALLLLL.pdf` | Midterm report |
| `requirements.txt` | Pinned dependencies |
| `DATA_LICENSE.md` | Dataset attribution and licensing |

## Limitations

The current experiment uses a small subset from one machine ID and one noise condition. Its class prevalence differs from the full dataset.

Windows inherit recording labels rather than per-second fault annotations. Neighboring windows are correlated. Recording-level isolation prevents the same file from crossing splits but does not prove independence between sessions or physical machines.

Live microphone input, MQTT, Raspberry Pi deployment, a neural network, and reserved-test evaluation are not implemented in this midterm prototype.

## Dataset attribution

MIMII public 1.0 was created by Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, Hitachi, Ltd.

- [Original dataset](https://doi.org/10.5281/zenodo.3384388)
- [Original valve archive](https://zenodo.org/records/3384388/files/6_dB_valve.zip)
- [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/)

The included audio retains channel 0 from the original recordings and is distributed under the same CC BY-SA 4.0 license. See `DATA_LICENSE.md` and `data/manifest.json` for details. No synthetic audio or pretrained model is used.

