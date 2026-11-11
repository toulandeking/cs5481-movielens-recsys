"""Export demo and full-catalog test Top-10 from frozen, unmasked scores.

python -m src.export_recommendations --model lrurec --scores <scores.npy> --reference <test.json>
python -m src.export_recommendations --model itemcf
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.adapters.base import all_rated_mask
from src.contract import K, N_ITEMS, N_USERS
from src.evaluate import evaluate_matrix, load_ground_truth


def export(model: str, scores: np.ndarray, destination: Path, reference: Path, source: str) -> dict:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (N_USERS, N_ITEMS) or not np.isfinite(scores).all():
        raise ValueError('Expected a finite, unmasked (6040, 3416) score matrix')
    truth = load_ground_truth('test')
    expected = json.loads(reference.read_text(encoding='utf-8'))['reports']['full']
    destination.mkdir(parents=True, exist_ok=True)
    tables = {}
    for filename, mask in [('top10.parquet', all_rated_mask()),
                           ('top10_eval.parquet', truth['seen'])]:
        masked = scores.copy()
        masked[mask] = -np.inf
        # Stable full sort: ties use ascending canonical iid, including the kth boundary.
        indices = np.argsort(-masked, axis=1, kind='stable')[:, :K]
        table = pd.DataFrame({
            'uid': np.repeat(np.arange(N_USERS, dtype=np.int32), K),
            'rank': np.tile(np.arange(1, K + 1, dtype=np.int8), N_USERS),
            'iid': indices.astype(np.int32).ravel(),
            'score': np.take_along_axis(scores, indices, axis=1).ravel(),
        })
        assert len(table) == 60400
        assert not mask[table.uid, table.iid].any()
        assert not table.duplicated(['uid', 'iid']).any()
        tables[filename] = table

    evaluation = tables['top10_eval.parquet']
    hits = evaluation.iid.to_numpy() == truth['targets'][evaluation.uid.to_numpy()]
    recall = float(hits.sum() / N_USERS)
    ndcg = float((1 / np.log2(evaluation['rank'].to_numpy()[hits].astype(np.float64) + 1)).sum() / N_USERS)
    report = evaluate_matrix(scores, mode='full', split='test', model=model)
    for key, actual in [('recall', recall), ('ndcg', ndcg)]:
        if not np.isclose(actual, getattr(report, key), atol=1e-12, rtol=0):
            raise ValueError(f'Export differs from evaluate.py: {key}')
        if not np.isclose(actual, expected[key], atol=1e-12, rtol=0):
            raise ValueError(f'Export differs from frozen experiment: {key}')
    files = {}
    for filename, table in tables.items():
        path = destination / filename
        table.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(table, pd.read_parquet(path))
        files[filename] = {'rows': len(table), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    verification = {
        'model': model, 'source': source, 'reference': reference.as_posix(),
        'columns': ['uid', 'rank', 'iid', 'score'], 'uid_range': [0, N_USERS - 1],
        'iid_range': [0, N_ITEMS - 1], 'rank_range': [1, K],
        'demo_excludes': ['train', 'val', 'test'], 'eval_excludes': ['train', 'val'],
        'context': 'train only; unchanged from frozen experiment',
        'eval_mode': 'full catalog (not 101 candidates)',
        'test_hits': int(hits.sum()), 'recall_at_10': recall, 'ndcg_at_10': ndcg,
        'matches_evaluate_py_and_frozen_report': True, 'files': files,
    }
    (destination / 'top10_verification.json').write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['itemcf', 'lightfm', 'lrurec'])
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    reference = args.reference or Path(f'outputs/metrics/{args.model}_test.json')
    if args.scores:
        scores = np.load(args.scores, allow_pickle=False)
        source = args.scores.as_posix()
    elif args.model == 'itemcf':
        from src.adapters.itemcf import ItemCFAdapter
        params_path = Path('outputs/metrics/itemcf_params.json')
        params = json.loads(params_path.read_text(encoding='utf-8'))['params']
        adapter = ItemCFAdapter(**params)
        adapter.fit()
        scores = adapter.score_matrix()
        source = 'ItemCF fit on train with outputs/metrics/itemcf_params.json'
    else:
        parser.error('--scores is required: use frozen scores, do not silently retrain')
    result = export(args.model, scores, args.out or Path('results') / args.model, reference, source)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
