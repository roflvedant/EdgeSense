"""Run the trained model on one validation recording of each class."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from edgesense import infer

rows = list(csv.DictReader((ROOT / 'results/splits.csv').open()))
for label in ['normal', 'abnormal']:
    row = next(row for row in rows if row['split'] == 'validation' and row['label'] == label)
    print('\nDEMONSTRATION:', label.upper(), row['path'])
    infer(ROOT / 'data' / row['path'], ROOT / 'models/edgesense.joblib',
          ROOT / 'results' / f'demo_{label}.json')
