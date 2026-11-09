"""LRURec on the frozen canon split. See LRUREC.md for the protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.adapters.base import AdapterBase, _check_matrix
from src.adapters.lrurec_model import LRU
from src.contract import N_ITEMS, N_USERS, OUTPUT_DIR, SEED, read_canon_table
from src.evaluate import (
    _rank_metrics,
    evaluate_all,
    evaluate_matrix,
    load_ground_truth,
    print_reports,
    save_report,
)

MET = OUTPUT_DIR / "metrics"
MODELS = OUTPUT_DIR / "models"
DEFAULTS = dict(
    hidden_size=64,
    num_layers=2,
    maxlen=200,
    dropout=0.2,
    learning_rate=0.001,
    epochs=5,
    patience=2,
    batch_size=32,
    seed=SEED,
    threads=4,
    device="cpu",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def histories(ratings, include_val=False):
    allowed = ["train", "val"] if include_val else ["train"]
    rows = ratings[ratings["split"].isin(allowed)].sort_values(
        ["uid", "ts", "iid"], kind="stable"
    )
    result = [[] for _ in range(N_USERS)]
    for uid, group in rows.groupby("uid", sort=False):
        result[int(uid)] = (group["iid"].to_numpy(dtype=np.int64) + 1).tolist()
    return result


def training_pairs(sequences, maxlen):
    """Nonoverlapping target windows cover every train transition exactly once.
    The previous item remains available across window boundaries. No target
    exists for left padding or for the first event of each user's history.
    """
    xs, ys = [], []
    for seq in sequences:
        for start in range(1, len(seq), maxlen):
            end = min(start + maxlen, len(seq))
            x, y = seq[start - 1 : end - 1], seq[start:end]
            xs.append([0] * (maxlen - len(x)) + x)
            ys.append([0] * (maxlen - len(y)) + y)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def inference_tokens(sequences, maxlen):
    return torch.tensor(
        [[0] * max(0, maxlen - len(s)) + s[-maxlen:] for s in sequences],
        dtype=torch.long,
    )


class LRURecAdapter(AdapterBase):
    name = "LRURec"

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"Unknown parameters: {unknown}")
        self.params = DEFAULTS | kwargs
        for key in [
            "hidden_size",
            "num_layers",
            "maxlen",
            "epochs",
            "patience",
            "batch_size",
            "threads",
        ]:
            if self.params[key] < 1:
                raise ValueError(f"{key} must be positive")
        if not 0 <= self.params["dropout"] < 1:
            raise ValueError("invalid dropout")
        if self.params["learning_rate"] <= 0:
            raise ValueError("invalid learning_rate")
        self.context = "train"  # Common primary benchmark uses train-only context.
        self.model = None
        self._scores = None
        self.history = []

    def _initialize(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        p = self.params
        if p["device"].startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        random.seed(p["seed"])
        np.random.seed(p["seed"])
        torch.manual_seed(p["seed"])
        torch.set_num_threads(p["threads"])
        torch.use_deterministic_algorithms(True)
        args = SimpleNamespace(
            num_items=N_ITEMS,
            bert_hidden_units=p["hidden_size"],
            bert_num_blocks=p["num_layers"],
            bert_dropout=p["dropout"],
            bert_attn_dropout=p["dropout"],
            dataset_code="ml-1m",
        )
        self.device = torch.device(p["device"])
        self.model = LRU(args).to(self.device)
        self._ratings = read_canon_table("ratings")
        self._train = histories(self._ratings)
        self._tokens = inference_tokens(self._train, p["maxlen"])

    def fit(self):
        self._initialize()
        p = self.params
        x, y = training_pairs(self._train, p["maxlen"])
        generator = torch.Generator().manual_seed(p["seed"])
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=p["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=p["learning_rate"])
        best, stale, best_state = -1.0, 0, None
        self.history = []
        begin = time.perf_counter()
        for epoch in range(1, p["epochs"] + 1):
            tick = time.perf_counter()
            self.model.train()
            total, count = 0.0, 0
            for tokens, labels in loader:
                tokens, labels = tokens.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = self.model(tokens, labels=labels)
                target = labels[labels > 0]
                loss = nn.functional.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * target.numel()
                count += target.numel()
            scores = self._score_tokens(self._tokens)
            report = evaluate_matrix(scores, "full", split="val", model=self.name)
            row = dict(
                epoch=epoch,
                loss=total / count,
                val_recall=report.recall,
                val_ndcg=report.ndcg,
                seconds=time.perf_counter() - tick,
            )
            self.history.append(row)
            print(
                f"epoch={epoch} loss={row['loss']:.4f} val_R@10={report.recall:.5f} "
                f"val_NDCG@10={report.ndcg:.5f} seconds={row['seconds']:.1f}",
                flush=True,
            )
            # Prespecified primary selection: full val Recall; tie: NDCG.
            key = (report.recall, report.ndcg)
            if best_state is None or key > best:
                best, stale = key, 0
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
            else:
                stale += 1
            if stale >= p["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.fit_seconds = time.perf_counter() - begin
        self._scores = None
        self.best_val_recall, self.best_val_ndcg = best
        self.training_windows = len(x)
        return self

    def _score_tokens(self, tokens):
        self.model.eval()
        batches = []
        with torch.no_grad():
            for batch in tokens.split(self.params["batch_size"]):
                logits, _ = self.model(batch.to(self.device), only_last=True)
                batches.append(logits[:, 1:].cpu().numpy().astype(np.float32))
        return _check_matrix(np.concatenate(batches))

    def score_matrix(self):
        if self.model is None:
            raise RuntimeError("Call fit() or load() first")
        if self._scores is None:
            self._scores = self._score_tokens(self._tokens)
        return self._scores.copy()

    def predict_scores(self, uid, iid):
        return self.score_matrix()[np.asarray(uid), np.asarray(iid)]

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                params=self.params,
                state_dict=self.model.state_dict(),
                history=self.history,
                best_epoch=self.best_epoch,
                best_val_recall=self.best_val_recall,
                best_val_ndcg=self.best_val_ndcg,
                fit_seconds=self.fit_seconds,
                training_windows=self.training_windows,
            ),
            path,
        )

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(**data["params"])
        obj._initialize()
        obj.model.load_state_dict(data["state_dict"])
        obj.model.eval()
        for k in [
            "history",
            "best_epoch",
            "best_val_recall",
            "best_val_ndcg",
            "fit_seconds",
            "training_windows",
        ]:
            setattr(obj, k, data[k])
        return obj


# Ten prespecified trials cover all five required axes; full data, shared budget.
GRID = [
    dict(hidden_size=32, num_layers=1, maxlen=100, dropout=0.1, learning_rate=0.001),
    dict(hidden_size=32, num_layers=1, maxlen=200, dropout=0.1, learning_rate=0.001),
    dict(hidden_size=32, num_layers=2, maxlen=200, dropout=0.1, learning_rate=0.001),
    dict(hidden_size=64, num_layers=1, maxlen=200, dropout=0.1, learning_rate=0.001),
    dict(hidden_size=64, num_layers=2, maxlen=200, dropout=0.2, learning_rate=0.001),
    dict(hidden_size=32, num_layers=1, maxlen=200, dropout=0.2, learning_rate=0.001),
    dict(hidden_size=32, num_layers=1, maxlen=200, dropout=0.1, learning_rate=0.0005),
    dict(hidden_size=64, num_layers=1, maxlen=100, dropout=0.2, learning_rate=0.001),
    dict(hidden_size=32, num_layers=2, maxlen=100, dropout=0.2, learning_rate=0.001),
    dict(hidden_size=32, num_layers=1, maxlen=300, dropout=0.1, learning_rate=0.001),
]


# The second half of the complete twenty-configuration search.
EXTRA_GRID = [
    {
        "hidden_size": 32,
        "num_layers": 1,
        "maxlen": 100,
        "dropout": 0.0,
        "learning_rate": 0.001,
    },
    {
        "hidden_size": 32,
        "num_layers": 1,
        "maxlen": 200,
        "dropout": 0.0,
        "learning_rate": 0.001,
    },
    {
        "hidden_size": 64,
        "num_layers": 1,
        "maxlen": 100,
        "dropout": 0.1,
        "learning_rate": 0.001,
    },
    {
        "hidden_size": 64,
        "num_layers": 1,
        "maxlen": 200,
        "dropout": 0.0,
        "learning_rate": 0.001,
    },
    {
        "hidden_size": 64,
        "num_layers": 1,
        "maxlen": 200,
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    {
        "hidden_size": 32,
        "num_layers": 1,
        "maxlen": 100,
        "dropout": 0.1,
        "learning_rate": 0.0005,
    },
    {
        "hidden_size": 64,
        "num_layers": 1,
        "maxlen": 100,
        "dropout": 0.1,
        "learning_rate": 0.0005,
    },
    {
        "hidden_size": 32,
        "num_layers": 2,
        "maxlen": 100,
        "dropout": 0.1,
        "learning_rate": 0.001,
    },
    {
        "hidden_size": 32,
        "num_layers": 1,
        "maxlen": 50,
        "dropout": 0.1,
        "learning_rate": 0.001,
    },
    {
        "hidden_size": 64,
        "num_layers": 1,
        "maxlen": 50,
        "dropout": 0.1,
        "learning_rate": 0.001,
    },
]


def tune(args):
    MET.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    if (MET / "lrurec_test.json").exists():
        raise RuntimeError(
            "Final test already exists; do not tune after viewing test results."
        )
    rows = []
    best = (-1.0, -1.0)
    for trial, config in enumerate(GRID, 1):
        path = MODELS / f"lrurec_trial_{trial:02d}.pt"
        wanted = (
            DEFAULTS
            | config
            | dict(
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                threads=args.threads,
                device=args.device,
            )
        )
        if path.exists():
            model = LRURecAdapter.load(path)
            if model.params != wanted:
                raise RuntimeError(f"Existing trial {trial} has different settings")
        else:
            print(f"\nTrial {trial}/10: {wanted}", flush=True)
            model = LRURecAdapter(**wanted)
            model.fit()
            model.save(path)
        row = dict(
            trial=trial,
            **model.params,
            best_epoch=model.best_epoch,
            val_recall=model.best_val_recall,
            val_ndcg=model.best_val_ndcg,
            fit_seconds=model.fit_seconds,
            training_windows=model.training_windows,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(MET / "lrurec_tuning_val.csv", index=False)
        pd.DataFrame(model.history).to_csv(
            MET / f"lrurec_trial_{trial:02d}_epochs.csv", index=False
        )
        key = (row["val_recall"], row["val_ndcg"])
        if key > best:
            best = key
            model.save(MODELS / "lrurec_best.pt")
            write_json(
                MET / "lrurec_params.json",
                dict(
                    selected_trial=trial,
                    params=model.params,
                    best_epoch=model.best_epoch,
                    val_recall=key[0],
                    val_ndcg=key[1],
                    selection="max full val Recall@10, tie full val NDCG@10; earliest trial if tied",
                    inference_context="train only for both val and test (common benchmark)",
                    environment=dict(
                        python=platform.python_version(),
                        torch=torch.__version__,
                        numpy=np.__version__,
                        pandas=pd.__version__,
                    ),
                    upstream_commit="e02a559bd8c42342f5c53d69cefea2589becb22c",
                ),
            )
    print(
        "Selected:",
        (MET / "lrurec_params.json").read_text(encoding="utf-8"),
        flush=True,
    )


def final_eval():
    path = MET / "lrurec_test.json"
    if path.exists():
        print("Final test already recorded. Reusing saved report without reevaluation.")
        print(path.read_text(encoding="utf-8"))
        return
    trials = pd.read_csv(MET / "lrurec_tuning_val.csv")
    if len(trials) != len(GRID):
        raise RuntimeError("Complete all ten validation trials before final test.")
    selected = json.loads((MET / "lrurec_params.json").read_text(encoding="utf-8"))
    model = LRURecAdapter.load(MODELS / "lrurec_best.pt")
    if model.params != selected["params"]:
        raise RuntimeError("Checkpoint does not match frozen selected configuration.")
    scores = model.score_matrix()
    if not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite scores: aborting before final test")
    reports = evaluate_all(scores, split="test", model="LRURec")
    save_report(reports, "lrurec_test")
    print_reports(reports)
    (OUTPUT_DIR / "scores").mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "scores" / "lrurec_test.npy", scores)
    np.save(
        OUTPUT_DIR / "scores" / "lrurec_val.npy", scores
    )  # identical train-only inference
    top = model.export_top10()
    (OUTPUT_DIR / "top10").mkdir(parents=True, exist_ok=True)
    top.to_parquet(OUTPUT_DIR / "top10" / "lrurec.parquet", index=False)
    write_json(
        MET / "lrurec_run.json",
        dict(
            checkpoint_sha256=hashlib.sha256(
                (MODELS / "lrurec_best.pt").read_bytes()
            ).hexdigest(),
            fit_seconds=model.fit_seconds,
            best_epoch=model.best_epoch,
            training_windows=model.training_windows,
            context="train only; val excluded from test ranking by shared evaluator",
            score_shape=list(scores.shape),
            top10_rows=len(top),
        ),
    )


def analyze():
    """Diagnostics reuse frozen scores; never retrain or select on test."""
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR.parent / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.load(OUTPUT_DIR / "scores" / "lrurec_test.npy")
    ratings = read_canon_table("ratings")
    pop = np.bincount(
        ratings.loc[ratings["split"] == "train", "iid"], minlength=N_ITEMS
    )
    edges = [0, 20, 100, 500, 1500, np.inf]
    labels = ["<20", "20-99", "100-499", "500-1499", ">=1500"]
    groups = np.digitize(pop, edges) - 1
    rows = []
    truths = {split: load_ground_truth(split) for split in ["val", "test"]}
    masked = {}
    for split, gt in truths.items():
        masked[split] = scores.copy()
        masked[split][gt["seen"]] = -np.inf
    for bucket, label in enumerate(labels):
        row = {"target_popularity": label}
        for split in ["val", "test"]:
            gt = truths[split]
            selected = groups[gt["targets"]] == bucket
            n = int(selected.sum())
            row[f"{split}_n"] = n
            if n:
                rep = _rank_metrics(
                    masked[split][selected], gt["targets"][selected], 10
                )
                row[f"{split}_hit@10"] = rep["recall"]
                row[f"{split}_ndcg@10"] = rep["ndcg"]
            else:
                row[f"{split}_hit@10"] = np.nan
                row[f"{split}_ndcg@10"] = np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(MET / "lrurec_popularity_buckets.csv", index=False)
    figdir = OUTPUT_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, table["val_hit@10"], 0.4, label="Validation")
    ax.bar(x + 0.2, table["test_hit@10"], 0.4, label="Test")
    ax.set(
        xticks=x,
        xticklabels=labels,
        xlabel="Target train interactions",
        ylabel="Recall@10",
        title="LRURec: popularity bias",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "lrurec_popularity_bias.png", dpi=160)
    plt.close(fig)
    tuning = pd.read_csv(MET / "lrurec_tuning_val.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    is_extension = (
        "phase" in tuning.columns
        and (tuning.phase == "post_test_validation_extension").any()
    )
    colors = [
        "#EE7733" if is_extension and trial > 10 else "#4477AA"
        for trial in tuning.trial
    ]
    ax.bar(tuning.trial.astype(str), tuning.val_recall, color=colors)
    if is_extension:
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="#4477AA", label="Initial 10"),
                Patch(color="#EE7733", label="Additional val-only 10"),
            ]
        )
    ax.set(
        xlabel="Prespecified configuration (see tuning CSV)",
        ylabel="Best validation Recall@10",
        title="LRURec: 20 freshly trained configurations"
        if "initialization" in tuning.columns
        else "LRURec: validation parameter sensitivity",
    )
    fig.tight_layout()
    fig.savefig(figdir / "lrurec_tuning_sensitivity.png", dpi=160)
    plt.close(fig)
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ["tune", "eval", "analyze", "top", "smoke"]:
        group.add_argument("--" + flag, action="store_true")
    parser.add_argument("--uid", type=int, default=42)
    for flag in ["epochs", "patience", "batch_size", "threads"]:
        parser.add_argument(
            "--" + flag.replace("_", "-"), type=int, default=DEFAULTS[flag]
        )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.tune:
        tune(args)
    elif args.eval:
        final_eval()
    elif args.analyze:
        analyze()
    elif args.top:
        if not 0 <= args.uid < N_USERS:
            parser.error("uid outside canon range")
        top = pd.read_parquet(OUTPUT_DIR / "top10" / "lrurec.parquet")
        print(
            top[top.uid == args.uid]
            .merge(read_canon_table("movies"), on="iid")
            .to_string(index=False)
        )
    elif args.smoke:
        model = LRURecAdapter(
            hidden_size=16,
            num_layers=1,
            maxlen=50,
            epochs=1,
            batch_size=args.batch_size,
            threads=args.threads,
            device=args.device,
        )
        model.fit()
        assert model.score_matrix().shape == (N_USERS, N_ITEMS)
        assert np.isfinite(model.score_matrix()).all()
        model.save(MODELS / "lrurec_smoke.pt")
        restored = LRURecAdapter.load(MODELS / "lrurec_smoke.pt")
        np.testing.assert_array_equal(model.score_matrix(), restored.score_matrix())
        print(
            "SMOKE PASS: full-data training, val scoring, finite scores, checkpoint round trip. No test evaluation."
        )


if __name__ == "__main__":
    main()
