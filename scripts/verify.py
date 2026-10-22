"""Check recorded provenance, split isolation, and independently recomputed metrics."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from edgesense import load_audio, windows, features

manifest = json.loads((ROOT / 'data/manifest.json').read_text())
rows = list(csv.DictReader((ROOT / 'results/splits.csv').open()))
assert len(rows) == len(manifest['files']) == 120
assert len({row['source_member'] for row in rows}) == 120
source_hashes = {row['path']: row['source_sha256'] for row in manifest['files']}
hash_splits = {}
for row in rows:
    hash_splits.setdefault(source_hashes[row['path']], set()).add(row['split'])
assert all(len(splits) == 1 for splits in hash_splits.values())
for row in manifest['files']:
    path = ROOT / 'data' / row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['mono_sha256']
assert {name: sum(row['split'] == name for row in rows) for name in
        ['train', 'validation', 'test']} == {'train': 84, 'validation': 18, 'test': 18}
print('PASS: 120 derived WAV hashes match the provenance manifest.')
print('PASS: source files and original-byte hashes do not cross recording splits.')
print('PASS: train / validation / test recording counts are 84 / 18 / 18.')
validation = {row['path'] for row in rows if row['split'] == 'validation'}
predictions = list(csv.DictReader((ROOT / 'results/validation_predictions.csv').open()))
assert {row['path'] for row in predictions} == validation
assert len(predictions) == 180
metrics = json.loads((ROOT / 'results/metrics.json').read_text())
assert not metrics['test_evaluated']
y = [int(row['true_label']) for row in predictions]
for name, column in [('model', 'model_prediction'), ('rms_baseline', 'baseline_prediction')]:
    predicted = [int(row[column]) for row in predictions]
    expected = metrics[name]
    assert np.isclose(f1_score(y, predicted), expected['f1'])
    assert np.isclose(precision_score(y, predicted, zero_division=0), expected['precision'])
    assert np.isclose(recall_score(y, predicted, zero_division=0), expected['recall'])
    assert confusion_matrix(y, predicted, labels=[0, 1]).tolist() == expected['confusion_matrix']
print('PASS: precision, recall, F1 and confusion matrices recompute from saved predictions.')
sample = ROOT / 'data' / next(iter(validation))
vector = features(windows(load_audio(sample))[0])
assert vector.shape == (40,) and np.isfinite(vector).all()
try:
    windows(np.zeros(100, dtype=np.float32))
except ValueError:
    pass
else:
    raise AssertionError('Short audio was silently accepted')
print('PASS: finite 40-dimensional MFCC features; too-short audio rejected.')
