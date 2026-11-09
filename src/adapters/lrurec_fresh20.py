"""Train a NEW twenty-trial experiment without loading any prior checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def configure(run_dir):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import src.contract as contract

    contract.OUTPUT_DIR = Path(run_dir)
    adapter = importlib.import_module("src.adapters.lrurec")
    evaluator = importlib.import_module("src.evaluate")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if adapter.MET != Path(run_dir) / "metrics" or evaluator.OUTPUT_DIR != Path(
        run_dir
    ):
        raise RuntimeError("Output routing was initialized before the fresh experiment")
    return adapter, evaluator, contract


def train_new(trial, params, run_dir):
    adapter, _, _ = configure(run_dir)
    path = adapter.MODELS / f"lrurec_trial_{trial:02d}.pt"
    if path.exists():
        raise RuntimeError(
            f"Fresh experiment refuses an existing trial checkpoint: {path}"
        )
    print(
        f"FRESH trial {trial}/20: random initialization, no checkpoint reuse; {params}",
        flush=True,
    )
    model = adapter.LRURecAdapter(**params)
    model.fit()
    model.save(path)
    pd.DataFrame(model.history).to_csv(
        adapter.MET / f"lrurec_trial_{trial:02d}_epochs.csv", index=False
    )
    return dict(
        trial=trial,
        **params,
        best_epoch=model.best_epoch,
        val_recall=model.best_val_recall,
        val_ndcg=model.best_val_ndcg,
        fit_seconds=model.fit_seconds,
        training_windows=model.training_windows,
        initialization="fresh_random",
        checkpoint_reused=False,
    )


def verify_run(run_dir, original_out):
    adapter, _, contract = configure(run_dir)
    import numpy as np

    manifest = json.loads(
        (adapter.MET / "lrurec_fresh20_manifest.json").read_text(encoding="utf-8")
    )
    complete = json.loads(
        (adapter.MET / "lrurec_fresh20_complete.json").read_text(encoding="utf-8")
    )
    table = pd.read_csv(adapter.MET / "lrurec_tuning_val.csv")
    assert complete["trials"] == 20 and complete["old_trials_counted"] == 0
    assert len(table) == 20 and sorted(table.trial.tolist()) == list(range(1, 21))
    assert not table.checkpoint_reused.any()
    assert set(table.initialization) == {"fresh_random"}
    keys = ["hidden_size", "num_layers", "maxlen", "dropout", "learning_rate"]
    assert len(table[keys].drop_duplicates()) == 20
    for plan in manifest["plans"]:
        row = table[table.trial == plan["trial"]].iloc[0]
        for key, value in plan["params"].items():
            assert row[key] == value, (plan["trial"], key)
        assert (adapter.MODELS / f"lrurec_trial_{plan['trial']:02d}.pt").is_file()
    chosen = json.loads(
        (adapter.MET / "lrurec_params.json").read_text(encoding="utf-8")
    )
    best = table.sort_values(
        ["val_recall", "val_ndcg", "trial"], ascending=[False, False, True]
    ).iloc[0]
    assert chosen["selected_trial"] == int(best.trial)
    run = json.loads((adapter.MET / "lrurec_run.json").read_text(encoding="utf-8"))
    assert (
        run["test_evaluations_in_this_run"] == 1 and run["old_checkpoints_loaded"] == 0
    )
    assert (
        run["checkpoint_sha256"]
        == hashlib.sha256((adapter.MODELS / "lrurec_best.pt").read_bytes()).hexdigest()
    )
    model = adapter.LRURecAdapter.load(adapter.MODELS / "lrurec_best.pt")
    scores = np.load(run_dir / "scores" / "lrurec_test.npy")
    assert (
        scores.shape == (contract.N_USERS, contract.N_ITEMS)
        and np.isfinite(scores).all()
    )
    np.testing.assert_array_equal(model.score_matrix(), scores)
    top = pd.read_parquet(run_dir / "top10" / "lrurec.parquet")
    assert len(top) == contract.N_USERS * 10 and (top.groupby("uid").size() == 10).all()
    assert not top.duplicated(["uid", "iid"]).any()
    assert (
        len(
            top.merge(
                contract.read_canon_table("ratings")[["uid", "iid"]], on=["uid", "iid"]
            )
        )
        == 0
    )
    np.testing.assert_array_equal(top.score.to_numpy(), scores[top.uid, top.iid])
    reports = json.loads(
        (adapter.MET / "lrurec_test.json").read_text(encoding="utf-8")
    )["reports"]
    buckets = pd.read_csv(adapter.MET / "lrurec_popularity_buckets.csv")
    assert set(reports) == {"full", "candidates"}
    assert int(buckets.test_n.sum()) == contract.N_USERS
    assert (
        abs(
            (buckets.test_n * buckets["test_hit@10"].fillna(0)).sum() / contract.N_USERS
            - reports["full"]["recall"]
        )
        < 1e-12
    )
    for mode in reports.values():
        assert mode["hr"] == mode["recall"] and 0 <= mode["ndcg"] <= mode["recall"]
    for name, digest in manifest["original_frozen_sha256"].items():
        assert hashlib.sha256((original_out / name).read_bytes()).hexdigest() == digest
    assert (
        hashlib.sha256(contract.canon_path("ratings").read_bytes()).hexdigest()
        == manifest["canonical_ratings_sha256"]
    )
    print(
        "FRESH20 CHECK PASS: 20 NEW distinct trials, no old checkpoints counted, correct selection, checkpoint round trip, clean top10, consistent reports, original experiment preserved."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the completed run without training or test reevaluation",
    )
    parser.add_argument(
        "--top", action="store_true", help="Show recommendations from this fresh run"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Rebuild diagnostics from saved scores"
    )
    parser.add_argument("--uid", type=int, default=42)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    import src.contract as contract

    original_out = contract.OUTPUT_DIR.resolve()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_relative_to(original_out) or run_dir == original_out:
        parser.error(
            "--run-dir must be a new subdirectory of the repository outputs directory"
        )
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    if sum([args.check, args.top, args.analyze]) > 1:
        parser.error("Choose --check, --top or --analyze")
    if args.check:
        verify_run(run_dir, original_out)
        return
    if args.analyze:
        adapter, _, _ = configure(run_dir)
        adapter.analyze()
        return
    if args.top:
        if not 0 <= args.uid < contract.N_USERS:
            parser.error("uid outside canon range")
        top = pd.read_parquet(run_dir / "top10" / "lrurec.parquet")
        print(
            top[top.uid == args.uid]
            .merge(contract.read_canon_table("movies"), on="iid")
            .to_string(index=False)
        )
        return
    if args.device == "cuda" and args.workers != 1:
        parser.error("Use --workers 1 for a single GPU")
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter, evaluator, contract = configure(run_dir)
    import numpy as np
    import torch

    EXTRA_GRID = adapter.EXTRA_GRID

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this environment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configs = adapter.GRID + EXTRA_GRID
    assert len(configs) == 20 and len({tuple(sorted(c.items())) for c in configs}) == 20
    params = [
        adapter.DEFAULTS
        | c
        | dict(epochs=args.epochs, patience=args.patience, device=args.device)
        for c in configs
    ]
    adapter.MET.mkdir()
    adapter.MODELS.mkdir()
    protected = [
        "metrics/lrurec_test.json",
        "metrics/lrurec_params.json",
        "metrics/lrurec_run.json",
        "models/lrurec_best.pt",
        "scores/lrurec_test.npy",
        "top10/lrurec.parquet",
    ]
    original_hashes = {
        name: hashlib.sha256((original_out / name).read_bytes()).hexdigest()
        for name in protected
        if (original_out / name).is_file()
    }
    manifest = dict(
        started_at=datetime.now(timezone.utc).isoformat(),
        experiment="fresh20",
        trials=20,
        plans=[dict(trial=i, params=p) for i, p in enumerate(params, 1)],
        old_trials_counted=0,
        old_checkpoints_loaded=0,
        workers=args.workers,
        selection="max full val Recall@10, then NDCG@10, then earliest trial",
        protocol="train-only context; common canon; shared full/candidates evaluator",
        repeated_experiment_after_prior_test_viewed=True,
        original_frozen_sha256=original_hashes,
        canonical_ratings_sha256=hashlib.sha256(
            contract.canon_path("ratings").read_bytes()
        ).hexdigest(),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        ),
    )
    adapter.write_json(adapter.MET / "lrurec_fresh20_manifest.json", manifest)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(train_new, i, p, str(run_dir)) for i, p in enumerate(params, 1)
        ]
        for job in as_completed(jobs):
            row = job.result()
            rows.append(row)
            pd.DataFrame(rows).sort_values("trial").to_csv(
                adapter.MET / "lrurec_tuning_val.csv", index=False
            )
            print(
                f"FRESH completed {len(rows)}/20; trial={row['trial']}, val_R@10={row['val_recall']:.6f}",
                flush=True,
            )
    table = pd.DataFrame(rows).sort_values("trial")
    assert len(table) == 20 and not table.checkpoint_reused.any()
    winner = table.sort_values(
        ["val_recall", "val_ndcg", "trial"], ascending=[False, False, True]
    ).iloc[0]
    trial = int(winner.trial)
    model = adapter.LRURecAdapter.load(adapter.MODELS / f"lrurec_trial_{trial:02d}.pt")
    model.save(adapter.MODELS / "lrurec_best.pt")
    adapter.write_json(
        adapter.MET / "lrurec_params.json",
        dict(
            selected_trial=trial,
            params=model.params,
            best_epoch=model.best_epoch,
            val_recall=model.best_val_recall,
            val_ndcg=model.best_val_ndcg,
            selection=manifest["selection"],
            experiment="fresh20",
            old_trials_counted=0,
            repeated_experiment_after_prior_test_viewed=True,
            environment=manifest["environment"],
        ),
    )
    # Configuration is now frozen, after ALL twenty NEW validation trials.
    scores = model.score_matrix()
    assert (
        scores.shape == (contract.N_USERS, contract.N_ITEMS)
        and np.isfinite(scores).all()
    )
    reports = evaluator.evaluate_all(scores, split="test", model="LRURec-fresh20")
    evaluator.save_report(reports, "lrurec_test")
    evaluator.print_reports(reports)
    (run_dir / "scores").mkdir()
    np.save(run_dir / "scores" / "lrurec_test.npy", scores)
    (run_dir / "top10").mkdir()
    top = model.export_top10()
    top.to_parquet(run_dir / "top10" / "lrurec.parquet", index=False)
    adapter.write_json(
        adapter.MET / "lrurec_run.json",
        dict(
            experiment="fresh20",
            trials=20,
            old_trials_counted=0,
            old_checkpoints_loaded=0,
            selected_trial=trial,
            best_epoch=model.best_epoch,
            checkpoint_sha256=hashlib.sha256(
                (adapter.MODELS / "lrurec_best.pt").read_bytes()
            ).hexdigest(),
            fit_seconds=model.fit_seconds,
            total_fit_seconds=float(table.fit_seconds.sum()),
            training_windows=model.training_windows,
            score_shape=list(scores.shape),
            top10_rows=len(top),
            test_evaluations_in_this_run=1,
            repeated_experiment_after_prior_test_viewed=True,
        ),
    )
    adapter.analyze()
    current_hashes = {
        name: hashlib.sha256((original_out / name).read_bytes()).hexdigest()
        for name in original_hashes
    }
    assert original_hashes == current_hashes, (
        "Original experiment changed during fresh run"
    )
    adapter.write_json(
        adapter.MET / "lrurec_fresh20_complete.json",
        dict(
            completed_at=datetime.now(timezone.utc).isoformat(),
            trials=20,
            old_trials_counted=0,
            old_checkpoints_loaded=0,
            original_experiment_preserved=True,
            selected_trial=trial,
            val_recall=model.best_val_recall,
            val_ndcg=model.best_val_ndcg,
            full_recall=reports["full"].recall,
            full_ndcg=reports["full"].ndcg,
        ),
    )
    print("FRESH TWENTY-TRIAL EXPERIMENT COMPLETE", str(run_dir), flush=True)


if __name__ == "__main__":
    main()
