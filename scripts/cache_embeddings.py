"""Freeze a CNN/Conformer epoch encoder and cache on the real recording timeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from psg_only.constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
from psg_only.data import night_statistics, read_manifest, train_stats, validate_inputs
from psg_only.evaluate import load_model
from psg_only.provenance import file_hash
from psg_only.train import device_for_run


def cache_embeddings(checkpoint, output, batch_size=64, train_subject_limit=None, val_subject_limit=None):
    args = argparse.Namespace(checkpoint=checkpoint, output=output, batch_size=batch_size,
                              train_subject_limit=train_subject_limit, val_subject_limit=val_subject_limit)
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive.')
    device = device_for_run()
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint['model_type'] not in {'b0', 'conformer_epoch'}:
        raise ValueError('Embedding cache requires a single-epoch CNN or Conformer.')
    if checkpoint['config'].get('smoke'):
        args.train_subject_limit = min(args.train_subject_limit or 8, 8)
        args.val_subject_limit = min(args.val_subject_limit or 2, 2)
    model.requires_grad_(False)
    data = checkpoint['config']['data']
    validate_inputs(data['manifest'], data['cache_root'], data['stats'])
    # 학습에 쓴 정규화와 embedding 생성 정규화가 어긋나면 실험이 조용히 망가진다.
    # 체크포인트 config 의 night_norm 을 그대로 따르고, provenance 에도 박아
    # 서로 다른 정규화의 캐시가 섞이지 않게 한다.
    night_norm = bool(data.get('night_norm', False))
    identity = dict(checkpoint_sha256=file_hash(args.checkpoint), config_hash=checkpoint['config_hash'],
                    manifest_hash=file_hash(data['manifest']), stats_hash=file_hash(data['stats']),
                    source_cache=Path(data['cache_root']).name,
                    normalization='night' if night_norm else 'train_global')
    if checkpoint['model_type'] == 'conformer_epoch':
        identity.update(encoder_type='conformer_epoch', cache_schema_version='embedding-cache-v1')
    if any(checkpoint['config'].get(k) != identity[k] for k in ('manifest_hash', 'stats_hash')):
        raise ValueError('B0 inputs changed since training; retrain with pinned inputs.')
    frame = read_manifest(data['manifest'])
    mean, std = train_stats(data['stats'])
    for split in ('train', 'val'):
        subjects = sorted(frame.loc[frame.split == split, 'subject_id'].unique())
        limit = args.train_subject_limit if split == 'train' else args.val_subject_limit
        if limit is not None:
            subjects = subjects[:limit]
        for number, sid in enumerate(subjects, 1):
            out = Path(args.output) / sid
            metadata_path = out / 'metadata.json'
            if metadata_path.exists():
                old = json.loads(metadata_path.read_text())
                if any(old.get(k) != v for k, v in identity.items()) or old.get('class_order_version') != CLASS_ORDER_VERSION:
                    raise FileExistsError('Existing cache has different provenance; choose a new output.')
                if any(old.get('files', {}).get(n) != file_hash(out / n) for n in ('embeddings.npy','labels.npy','valid.npy')):
                    raise ValueError('Existing embedding cache is incomplete or corrupt.')
                continue
            rows = frame.loc[(frame.split == split) & (frame.subject_id == sid)].sort_values('subject_epoch_index')
            source = Path(data['cache_root']) / sid
            subject_mean, subject_std = night_statistics(source) if night_norm else (mean, std)
            mels = np.load(source / 'mels.npy', mmap_mode='r')
            source_valid = np.load(source / 'valid.npy', mmap_mode='r')
            cache_idx = rows.subject_epoch_index.to_numpy()
            epoch_idx = rows.epoch_index.to_numpy()
            total = int(epoch_idx.max()) + 1
            embeddings = np.zeros((total, 192), dtype=np.float16)
            labels = np.full(total, -100, dtype=np.int64)
            valid = np.zeros(total, dtype=bool)
            valid[epoch_idx] = source_valid[cache_idx]
            labels[epoch_idx] = np.asarray((0, 2, 3, 1))[rows.label_index.to_numpy()]
            labels[~valid] = -100
            for start in range(0, len(rows), args.batch_size):
                ci = cache_idx[start:start + args.batch_size]
                ei = epoch_idx[start:start + args.batch_size]
                x = torch.from_numpy(np.asarray(mels[ci], dtype=np.float32).copy()).unsqueeze(1)
                with torch.no_grad():
                    _, embedding = model(((x - subject_mean) / subject_std).to(device))
                embeddings[ei] = embedding.cpu().numpy().astype(np.float16)
            if not np.isfinite(embeddings).all():
                raise ValueError('Non-finite embeddings.')
            out.mkdir(parents=True, exist_ok=True)
            for name, array in [('embeddings', embeddings), ('labels', labels), ('valid', valid)]:
                np.save(out / f'{name}.npy', array)
            payload = dict(identity, class_order=list(CANONICAL_CLASSES), class_order_version=CLASS_ORDER_VERSION,
                           subject_id=sid, shape=list(embeddings.shape), dtype='float16',
                           files={n: file_hash(out / n) for n in ('embeddings.npy','labels.npy','valid.npy')})
            temporary = out / 'metadata.json.tmp'
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(metadata_path)
            print(json.dumps(dict(stage="cache_embeddings", split=split, completed=number, subjects=len(subjects))), flush=True)
    return Path(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-subject-limit', type=int)
    parser.add_argument('--val-subject-limit', type=int)
    args = parser.parse_args()
    cache_embeddings(args.checkpoint, args.output, args.batch_size,
                     args.train_subject_limit, args.val_subject_limit)

if __name__ == '__main__':
    main()
