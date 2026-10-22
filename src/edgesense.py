"""EdgeSense offline training, validation, and local acoustic inference."""
import argparse
import csv
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

RATE = 16000
CONFIG = {'sample_rate': RATE, 'window_seconds': 1, 'n_mfcc': 20,
          'n_fft': 512, 'hop_length': 160, 'n_mels': 40, 'center': False}


def load_audio(path):
    rate, signal = wavfile.read(path)
    if signal.ndim == 2:
        signal = signal[:, 0]
    if np.issubdtype(signal.dtype, np.signedinteger):
        signal = signal.astype(np.float32) / float(np.iinfo(signal.dtype).max + 1)
    elif signal.dtype == np.uint8:
        signal = (signal.astype(np.float32) - 128) / 128
    else:
        signal = signal.astype(np.float32)
    if not np.all(np.isfinite(signal)) or len(signal) == 0:
        raise ValueError('Audio must contain finite, nonempty samples')
    if rate != RATE:
        signal = librosa.resample(signal, orig_sr=rate, target_sr=RATE)
    return signal


def windows(signal):
    count = len(signal) // RATE
    if count == 0:
        raise ValueError('Audio must be at least one second long')
    return signal[:count * RATE].reshape(count, RATE)


def features(window):
    coeff = librosa.feature.mfcc(y=window, sr=RATE, n_mfcc=20, n_fft=512,
                                hop_length=160, n_mels=40, center=False)
    return np.concatenate([coeff.mean(axis=1), coeff.std(axis=1)]).astype(np.float32)


def split_recordings(rows):
    labels = [int(row['label'] == 'abnormal') for row in rows]
    train, rest = train_test_split(np.arange(len(rows)), test_size=0.30,
                                  stratify=labels, random_state=42)
    valid, test = train_test_split(rest, test_size=0.50,
                                 stratify=np.array(labels)[rest], random_state=42)
    return {**{int(i): 'train' for i in train}, **{int(i): 'validation' for i in valid},
            **{int(i): 'test' for i in test}}


def scores(labels, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average='binary', zero_division=0)
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
            'confusion_matrix': confusion_matrix(labels, predicted, labels=[0, 1]).tolist()}


def choose_threshold(labels, values, thresholds):
    # On an F1 tie, keep the first (lowest) threshold, favoring recall.
    return float(max(thresholds, key=lambda t: f1_score(labels, values >= t, zero_division=0)))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(root):
    data = root / 'data'
    output = root / 'results'
    models = root / 'models'
    output.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    manifest = json.loads((data / 'manifest.json').read_text())
    rows = manifest['files']
    assignment = split_recordings(rows)
    split_rows = [{'path': row['path'], 'label': row['label'], 'split': assignment[i],
                   'source_member': row['source_member']} for i, row in enumerate(rows)]
    write_csv(output / 'splits.csv', split_rows)
    feature_rows, rms, labels, splits, ids, offsets = [], [], [], [], [], []
    # Never open test WAV files in the midterm experiment.
    for i, row in enumerate(rows):
        if assignment[i] == 'test':
            continue
        for offset, window in enumerate(windows(load_audio(data / row['path']))):
            feature_rows.append(features(window))
            rms.append(float(np.sqrt(np.mean(window ** 2))))
            labels.append(int(row['label'] == 'abnormal'))
            splits.append(assignment[i])
            ids.append(row['path'])
            offsets.append(offset)
        if (i + 1) % 20 == 0:
            print('Processed recording', i + 1, flush=True)
    x, y, energy, split = np.array(feature_rows), np.array(labels), np.array(rms), np.array(splits)
    train_mask, val_mask = split == 'train', split == 'validation'
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced',
                          C=1.0, max_iter=2000, random_state=42))
    model.fit(x[train_mask], y[train_mask])
    probability = model.predict_proba(x[val_mask])[:, 1]
    model_threshold = choose_threshold(y[val_mask], probability, np.linspace(0.05, 0.95, 91))
    candidates = np.r_[0, np.quantile(energy[train_mask], np.linspace(0, 1, 101)),
                       np.nextafter(energy[train_mask].max(), np.inf)]
    energy_threshold = choose_threshold(y[val_mask], energy[val_mask], candidates)
    predicted = (probability >= model_threshold).astype(int)
    baseline_predicted = (energy[val_mask] >= energy_threshold).astype(int)
    bundle = {'pipeline': model, 'threshold': model_threshold,
              'rms_threshold': energy_threshold, 'config': CONFIG,
              'scope': 'MIMII valve id_00 +6 dB; microphone channel 0'}
    model_path = models / 'edgesense.joblib'
    joblib.dump(bundle, model_path)
    val_indices = np.where(val_mask)[0]
    prediction_rows = [{'path': ids[idx], 'window_start_seconds': offsets[idx],
                        'true_label': int(y[idx]), 'abnormal_probability': float(probability[j]),
                        'model_prediction': int(predicted[j]), 'rms': float(energy[idx]),
                        'baseline_prediction': int(baseline_predicted[j])}
                       for j, idx in enumerate(val_indices)]
    write_csv(output / 'validation_predictions.csv', prediction_rows)
    # Timing uses real validation windows after feature-library and model warmup.
    val_recordings = [row for i, row in enumerate(rows) if assignment[i] == 'validation']
    timing_windows = [w for row in val_recordings[:5]
                      for w in windows(load_audio(data / row['path']))]
    model_times, pipeline_times = [], []
    with threadpool_limits(limits=1):
        model.predict_proba([features(timing_windows[0])])
        for window in timing_windows:
            begin = time.perf_counter()
            vector = features(window)
            after_features = time.perf_counter()
            model.predict_proba([vector])
            end = time.perf_counter()
            pipeline_times.append((end - begin) * 1000)
            model_times.append((end - after_features) * 1000)
    result = {'run_date': '2026-10-22', 'evaluation_split': 'validation',
              'test_evaluated': False, 'window_seconds': 1, 'feature_dimensions': 40,
              'model': scores(y[val_mask], predicted),
              'rms_baseline': scores(y[val_mask], baseline_predicted),
              'thresholds': {'model': model_threshold, 'rms_baseline': energy_threshold},
              'model_size_bytes': model_path.stat().st_size,
              'timing': {'windows': len(timing_windows), 'threads': 1,
                         'warm_feature_and_model_mean_ms': float(np.mean(pipeline_times)),
                         'warm_feature_and_model_p95_ms': float(np.percentile(pipeline_times, 95)),
                         'warm_feature_and_model_max_ms': float(np.max(pipeline_times)),
                         'model_only_mean_ms': float(np.mean(model_times)),
                         'excludes': 'WAV decoding, disk I/O, acquisition, process startup and JIT compilation'},
              'split_counts': {name: {'recordings': sum(assignment[i] == name for i in range(len(rows))),
                                      'normal_recordings': sum(assignment[i] == name and r['label'] == 'normal' for i, r in enumerate(rows)),
                                      'abnormal_recordings': sum(assignment[i] == name and r['label'] == 'abnormal' for i, r in enumerate(rows))}
                               for name in ['train', 'validation', 'test']},
              'validation_windows': int(val_mask.sum()), 'training_windows': int(train_mask.sum()),
              'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                              'processor': platform.processor(),
                              'packages': {name: importlib.metadata.version(name) for name in
                                           ['numpy', 'scipy', 'librosa', 'scikit-learn', 'joblib', 'matplotlib']}}}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    write_csv(output / 'latency_samples.csv', [{'sample': i, 'feature_and_model_ms': p,
                                              'model_only_ms': m}
                                             for i, (p, m) in enumerate(zip(pipeline_times, model_times))])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, name, metric in zip(axes, ['RMS baseline', 'MFCC logistic regression'],
                                [result['rms_baseline'], result['model']]):
        matrix = np.array(metric['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha='center', va='center',
                        color='white' if matrix[row, col] > matrix.max() / 2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Abnormal'],
               yticklabels=['Normal', 'Abnormal'], xlabel='Predicted', ylabel='True', title=name)
    fig.suptitle('Preliminary validation windows only')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrices.png', dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


def infer(audio_path, model_path, output=None):
    bundle = joblib.load(model_path)
    signal = load_audio(audio_path)
    predictions = []
    with threadpool_limits(limits=1):
        for offset, window in enumerate(windows(signal)):
            probability = float(bundle['pipeline'].predict_proba([features(window)])[0, 1])
            abnormal = probability >= bundle['threshold']
            predictions.append({'window_start_seconds': offset, 'prediction': 'Abnormal' if abnormal else 'Normal',
                                'abnormal_probability': round(probability, 6),
                                'warning': 'Inspect equipment' if abnormal else 'No warning'})
    result = {'audio': str(audio_path), 'threshold': bundle['threshold'],
              'probability_note': 'Model score is not calibrated confidence',
              'dropped_tail_samples': int(len(signal) % RATE), 'windows': predictions}
    if output:
        Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'infer'])
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'train':
        with threadpool_limits(limits=1):
            train(args.root)
    else:
        if args.audio is None:
            parser.error('--audio is required for infer')
        infer(args.audio, args.root / 'models/edgesense.joblib', args.output)


if __name__ == '__main__':
    main()
