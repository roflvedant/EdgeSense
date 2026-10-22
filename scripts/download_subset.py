"""Fetch a reproducible MIMII subset without downloading the full 6.9 GB ZIP."""
import argparse
import concurrent.futures
import hashlib
import io
import json
import random
import struct
import tempfile
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile

URL = 'https://zenodo.org/records/3384388/files/6_dB_valve.zip?download=1'


def byte_range(start=None, end=None):
    value = 'bytes=-65536' if start is None else f'bytes={start}-{end}'
    for attempt in range(5):
        try:
            request = urllib.request.Request(URL, headers={'Range': value})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise RuntimeError('Server must support HTTP byte ranges')
                return response.read(), response.headers['Content-Range']
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def index():
    tail, header = byte_range()
    size = int(header.split('/')[-1])
    eocd = tail.rfind(b'PK\x05\x06')
    _, _, _, _, _, directory_size, offset, _ = struct.unpack_from('<4s4H2LH', tail, eocd)
    if offset == 0xFFFFFFFF:
        location = tail.rfind(b'PK\x06\x06')
        directory_size, offset = struct.unpack_from('<QQ', tail, location + 40)
    directory, _ = byte_range(offset, offset + directory_size - 1)
    with tempfile.TemporaryFile() as sparse:
        sparse.truncate(size)
        sparse.seek(size - len(tail))
        sparse.write(tail)
        sparse.seek(offset)
        sparse.write(directory)
        sparse.flush()
        with zipfile.ZipFile(sparse) as archive:
            return archive.infolist()


def download(info, output):
    label = 'abnormal' if '/abnormal/' in info.filename else 'normal'
    destination = output / label / Path(info.filename).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Includes room for the local header and ZIP64 extra fields.
    block, _ = byte_range(info.header_offset, info.header_offset + info.compress_size + 2047)
    if block[:4] != b'PK\x03\x04':
        raise ValueError('Bad local ZIP header')
    name_length, extra_length = struct.unpack_from('<HH', block, 26)
    start = 30 + name_length + extra_length
    compressed = block[start:start + info.compress_size]
    raw = zlib.decompress(compressed, -15) if info.compress_type == 8 else compressed
    if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
        raise ValueError('Archive member failed length or CRC check')
    rate, audio = wavfile.read(io.BytesIO(raw))
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    mono = audio if audio.ndim == 1 else audio[:, 0]
    wavfile.write(destination, rate, mono)
    return {'path': destination.relative_to(output).as_posix(), 'label': label,
            'source_member': info.filename, 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'mono_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'sample_rate': rate, 'samples': len(mono), 'original_channels': channels,
            'selected_channel_zero_based': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--normal', type=int, default=80)
    parser.add_argument('--abnormal', type=int, default=40)
    args = parser.parse_args()
    members = index()
    print('Archive indexed:', len(members), 'members', flush=True)
    rng = random.Random(42)
    chosen = []
    for label, count in [('normal', args.normal), ('abnormal', args.abnormal)]:
        available = sorted([i for i in members if '/id_00/' in '/' + i.filename
                            and f'/{label}/' in i.filename and i.filename.endswith('.wav')],
                           key=lambda i: i.filename)
        print(label, 'available:', len(available), flush=True)
        if len(available) < count:
            raise ValueError('Not enough source recordings')
        chosen.extend(rng.sample(available, count))
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, member, args.output) for member in chosen]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0:
                print('Downloaded', len(rows), '/', len(chosen), flush=True)
    manifest = {'dataset_doi': '10.5281/zenodo.3384388', 'archive_url': URL,
                'machine': 'valve', 'machine_id': '00', 'snr_db': 6, 'selection_seed': 42,
                'license': 'CC BY-SA 4.0', 'modification': 'Retained microphone channel 0 as mono PCM16',
                'files': sorted(rows, key=lambda row: row['path'])}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Saved manifest and', len(rows), 'verified real recordings.', flush=True)


if __name__ == '__main__':
    main()
