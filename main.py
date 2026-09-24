# -*- coding: utf-8 -*-
"""
ConfDS E-Nose Drift Adaptation
==============================
Public evaluation script accompanying:

"Progressive Domain Adaptation under Sensor Drift for Gas Classification
 with Limited Calibration Labels", Eurosensors 2026.

Author
------
Han Fan

Affiliation
-----------
Munich Institute of Robotics and Machine Intelligence,
Chair of Perception for Intelligent Systems,
Technical University of Munich, Germany

Contact
-------
han.fan@tum.de

Protocol
--------
Batch 01 is the labelled source domain. For each target batch b02--b10,
the CDAN implementation initializes a fresh model and performs batch-wise
adaptation. Adapted parameters from one target batch are NOT propagated to
the next target batch. Available source--target transfer-pair constraints are
accumulated up to the current target batch.

A diverse pool of CDAN experts is trained by varying selected adaptation
settings. Their class-probability outputs are fused using confidence-aware
Dawid--Skene EM (ConfDS). The only target labels supplied to ConfDS are those
associated with the target members of the allowed transfer-pair subset.

Reproducibility note
--------------------
This public runner uses a FIXED ConfDS confidence threshold (default 0.80) and
retains the source-only expert pruning used in the original final experiment.
The pruning criterion uses labelled source data only and therefore does not
consume target calibration labels or target-test labels. The runner does not
sweep the ConfDS threshold using full target-test labels and does not perform
oracle evaluation.

The --smoke option provides a fast end-to-end software test: one run, b02 only,
two experts, and 20 optimizer iterations per expert by default. Smoke-test
accuracy is not scientifically meaningful; it is only intended to verify that
the software pipeline executes correctly.
"""

import os

# Set before importing torch. This improves reproducibility on CUDA where
# deterministic algorithms require a CuBLAS workspace configuration.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import csv
import importlib.util
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from confds import confidence_aware_soft_ds


# =============================================================================
# Public experiment configuration
# =============================================================================

PAIR_N = 10
RUN_SEEDS = [0, 1, 2]
PAIR_SHUFFLE_SEEDS = [1337, 2023]

# Fixed a priori in this public reference runner. Do not sweep this value and
# select it using full target-test accuracy.
CONFDS_TAU = 0.5

# Source-only expert pruning retained from the original final experiment.
# This uses source labels only; it does not consume target labels.
ENABLE_SRC_PRUNE = True
SRC_PRUNE_THRESH = 0.95
SRC_PRUNE_KEEP_MIN = 10
SRC_EVAL_MAX = 6000


# =============================================================================
# Reproducibility and import helpers
# =============================================================================

def configure_determinism() -> None:
    """Best-effort deterministic execution without hard-failing on unsupported ops."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Compatibility fallback for older PyTorch releases.
        torch.use_deterministic_algorithms(False)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_fresh(module_name: str, path: Path):
    """Import a Python file as a fresh module so expert overrides do not leak."""
    if not path.exists():
        raise FileNotFoundError(f"Required implementation file not found: {path}")

    unique_name = f"{module_name}_{int(time.time() * 1e6)}"
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for: {path}")

    module = importlib.util.module_from_spec(spec)
    # Register before execution; this is safer for modules that use dataclasses.
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Calibration-subset helpers
# =============================================================================

def _row_key(x: np.ndarray, decimals: int = 6) -> Tuple:
    return tuple(np.round(np.asarray(x, dtype=np.float64), decimals=decimals).tolist())


def map_pair_rows_to_batch_indices(
    x_batch: np.ndarray,
    x_pairs: np.ndarray,
    decimals: int = 6,
) -> np.ndarray:
    """Map paired target rows back to indices in the current target batch."""
    if x_pairs is None or x_pairs.size == 0:
        return np.zeros((0,), dtype=np.int64)

    bucket: Dict[Tuple, int] = {}
    for i in range(x_batch.shape[0]):
        key = _row_key(x_batch[i], decimals)
        if key not in bucket:
            bucket[key] = int(i)

    indices: List[int] = []
    for row in x_pairs:
        key = _row_key(row, decimals)
        if key in bucket:
            indices.append(bucket[key])
        else:
            # Numerical fallback if small floating-point differences remain.
            d2 = np.sum((x_batch - row) ** 2, axis=1)
            indices.append(int(np.argmin(d2)))

    return np.asarray(indices, dtype=np.int64)


def align_probabilities(p_raw: np.ndarray, class_order: np.ndarray) -> np.ndarray:
    """Align already-computed probabilities to the physical class order.

    This helper is retained for compatibility. New classifier-head capture uses
    ``probabilities_from_logits`` below, which is numerically safer when the
    legacy head contains an unused output column.
    """
    p_raw = np.asarray(p_raw, dtype=np.float64)
    class_order = np.asarray(class_order, dtype=np.int64)

    if p_raw.ndim != 2:
        raise ValueError("Classifier probabilities must have shape (N, C).")

    if class_order.size and np.min(class_order) >= 0 and np.max(class_order) < p_raw.shape[1]:
        p = p_raw[:, class_order]
    elif p_raw.shape[1] == len(class_order):
        p = p_raw
    else:
        raise ValueError(
            f"Cannot align classifier output shape {p_raw.shape} with "
            f"class order {class_order.tolist()}."
        )

    # Normalize by the actual retained-class mass.  Do not floor a small
    # positive denominator to 1e-12: doing so makes the row cease to sum to 1.
    denom = p.sum(axis=1, keepdims=True)
    if np.any(denom <= 0.0):
        raise FloatingPointError(
            "Retained-class probability mass is zero. Use probabilities_from_logits() "
            "for numerically stable classifier-head conversion."
        )
    return p / denom


def probabilities_from_logits(logits: torch.Tensor, class_order: np.ndarray) -> np.ndarray:
    """Return stable probabilities over the physical classes only.

    The UCI MATLAB representation used by the original experiments has labels
    1,...,6.  The legacy classifier therefore has seven outputs because it was
    created with ``max(label) + 1``; output 0 is unused.

    Mathematically, selecting classes 1,...,6 from a seven-way softmax and then
    renormalizing is exactly the same as applying softmax directly to logits
    1,...,6.  Performing the selection *before* softmax avoids float32
    underflow when the unused logit 0 becomes very large for a target sample.
    It does not change training or the intended six-class posterior.
    """
    class_order = np.asarray(class_order, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"Classifier logits must have shape (N,C); got {tuple(logits.shape)}.")

    if (
        class_order.size
        and np.min(class_order) >= 0
        and np.max(class_order) < logits.shape[1]
    ):
        idx = torch.as_tensor(class_order, dtype=torch.long, device=logits.device)
        valid_logits = logits.index_select(1, idx)
    elif logits.shape[1] == len(class_order):
        valid_logits = logits
    else:
        raise ValueError(
            f"Cannot align classifier logits shape {tuple(logits.shape)} with "
            f"class order {class_order.tolist()}."
        )

    # Compute in float64 for a stable public reference implementation.
    p = torch.softmax(valid_logits.to(torch.float64), dim=1).cpu().numpy()
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("Non-finite classifier probabilities.")
    return p


# =============================================================================
# Diverse expert pool used by the final experiment
# =============================================================================

def build_expert_configs(run_seed: int) -> List[Dict]:
    """Return the 24 expert configurations used by the diverse pool."""
    configs: List[Dict] = []

    # Nine base-like experts: 3 training seeds x 3 nearby configurations.
    for offset in (0, 1, 2):
        seed = int(run_seed * 100 + offset)
        configs.extend(
            [
                dict(name=f"base_tau0995_s{seed}", seed=seed, PSEUDO_TAU=0.995),
                dict(name=f"base_tau0990_s{seed}", seed=seed, PSEUDO_TAU=0.990),
                dict(
                    name=f"base_tau0995_r4_s{seed}",
                    seed=seed,
                    PSEUDO_TAU=0.995,
                    DSBN_BLEND_RAMP=4,
                ),
            ]
        )

    # Fifteen mechanism-diverse experts.
    seed = int(run_seed * 100)
    variants = [
        ("adv002_conf085", dict(LAMBDA_ADV=0.02, CONF_GATING=0.85)),
        ("adv005_conf085", dict(LAMBDA_ADV=0.05, CONF_GATING=0.85)),
        ("adv002_conf000", dict(LAMBDA_ADV=0.02, CONF_GATING=0.00)),
        ("noPseudo", dict(USE_PSEUDO=False)),
        ("tau0975", dict(PSEUDO_TAU=0.975)),
        ("pseudoW010", dict(LAMBDA_PSEUDO=0.10)),
        ("noCorr_pairBase", dict(USE_DCAE_CORR=False, PAIR_ALIGN_ON="base")),
        ("corr_pairBase", dict(PAIR_ALIGN_ON="base")),
        ("multiDomain", dict(ADV_BINARY_DOMAIN=False)),
        (
            "multiDomain_corrFullID",
            dict(ADV_BINARY_DOMAIN=False, CORR_USE_FULL_DOMAIN_ID=True),
        ),
        (
            "binDomain_corrFullID",
            dict(ADV_BINARY_DOMAIN=True, CORR_USE_FULL_DOMAIN_ID=True),
        ),
        ("dom_none", dict(DOMAIN_COND_MODE="none")),
        ("dom_dsbn", dict(DOMAIN_COND_MODE="dsbn")),
        ("dom_film", dict(DOMAIN_COND_MODE="film")),
        ("pairStrong", dict(LAMBDA_PAIR=0.08)),
    ]

    for name, overrides in variants:
        cfg = dict(name=name, seed=seed)
        cfg.update(overrides)
        configs.append(cfg)

    if len(configs) != 24:
        raise RuntimeError(f"Expected 24 experts, constructed {len(configs)}.")
    return configs


# =============================================================================
# Train one expert and capture its target class-probability outputs
# =============================================================================

def run_one_expert(
    cdan_path: Path,
    data: List[np.ndarray],
    labels: List[np.ndarray],
    transfer_pairs,
    class_order: np.ndarray,
    cfg: Dict,
) -> Tuple[List[np.ndarray], float, List[float]]:
    """Train one expert and capture target probabilities plus source-head accuracy.

    Source-head accuracy is evaluated after each target-step training using only
    the labelled source batch (b01). Its mean is used by the source-only pruning
    rule from the original final experiment.
    """
    cdan = import_fresh(f"cdan_{cfg['name']}", cdan_path)

    # Apply expert-specific mechanism/weight overrides.
    for key, value in cfg.items():
        if key in ("name", "seed"):
            continue
        if not hasattr(cdan, key):
            raise AttributeError(
                f"Expert configuration '{cfg['name']}' requests '{key}', but "
                f"{cdan_path.name} does not define that setting."
            )
        setattr(cdan, key, value)

    # Keep the transfer-pair budget strict in the training module as well.
    if hasattr(cdan, "N_TRANS_SMP"):
        cdan.N_TRANS_SMP = int(PAIR_N)

    # train_eval_incremental() only invokes head_acc() when this is enabled.
    cdan.REPORT_HEAD_ACC = True

    seed = int(cfg.get("seed", 0))
    seed_all(seed)

    target_probabilities: List[np.ndarray] = []
    source_head_accuracies: List[float] = []
    original_head_acc = cdan.head_acc

    # Source-only evaluation data. The UCI source batch is small, but preserve
    # the cap used by the original runner for generality.
    x_src = data[0]
    y_src = labels[0]
    if SRC_EVAL_MAX is not None and x_src.shape[0] > int(SRC_EVAL_MAX):
        rng = np.random.RandomState(12345 + seed)
        src_idx = rng.choice(
            x_src.shape[0],
            size=int(SRC_EVAL_MAX),
            replace=False,
        )
        x_src_eval = x_src[src_idx]
        y_src_eval = y_src[src_idx]
    else:
        x_src_eval = x_src
        y_src_eval = y_src

    @torch.no_grad()
    def capture_head(model, x: np.ndarray, y: np.ndarray, device: str) -> float:
        """Capture target probabilities and source-head retention accuracy."""
        model.eval()

        # --- target head outputs ---
        xt = torch.as_tensor(x, device=device, dtype=torch.float32)
        d_t = torch.ones((xt.size(0),), device=device, dtype=torch.long)
        z_t = model.encode(xt, d_cond=d_t, d_corr=d_t)
        logits_t = model.predict(z_t)

        # Stable six-class posterior from the physically valid logits.
        p_t = probabilities_from_logits(logits_t, class_order)
        p_t = np.array(p_t, dtype=np.float64, copy=True)
        target_probabilities.append(p_t)

        pred_t = class_order[np.argmax(p_t, axis=1)]
        target_acc = float(np.mean(pred_t == y))

        # --- source head accuracy: source labels only ---
        xs = torch.as_tensor(x_src_eval, device=device, dtype=torch.float32)
        d_s = torch.zeros((xs.size(0),), device=device, dtype=torch.long)
        z_s = model.encode(xs, d_cond=d_s, d_corr=d_s)
        logits_s = model.predict(z_s)
        p_s = probabilities_from_logits(logits_s, class_order)
        pred_s = class_order[np.argmax(p_s, axis=1)]
        source_head_accuracies.append(float(np.mean(pred_s == y_src_eval)))

        return target_acc

    cdan.head_acc = capture_head
    try:
        cdan.train_eval_incremental(data, labels, transfer_pairs, seed=seed)
    finally:
        cdan.head_acc = original_head_acc

    expected = len(data) - 1
    if len(target_probabilities) != expected:
        raise RuntimeError(
            f"Expert '{cfg['name']}' produced outputs for "
            f"{len(target_probabilities)}/{expected} target batches."
        )
    if len(source_head_accuracies) != expected:
        raise RuntimeError(
            f"Expert '{cfg['name']}' produced source accuracies for "
            f"{len(source_head_accuracies)}/{expected} target batches."
        )

    for bi, p in enumerate(target_probabilities, start=2):
        if not np.all(np.isfinite(p)):
            raise FloatingPointError(
                f"Non-finite probabilities from {cfg['name']} on b{bi:02d}."
            )
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
            raise FloatingPointError(
                f"Probabilities do not sum to one for {cfg['name']} on b{bi:02d}."
            )

    source_mean = float(np.mean(source_head_accuracies))
    return target_probabilities, source_mean, source_head_accuracies


# =============================================================================
# Output helpers
# =============================================================================

def write_run_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, A: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    means = A.mean(axis=0)
    stds = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(A.shape[1])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "mean_accuracy", "std_accuracy", "n_runs"])
        for j, (m, s) in enumerate(zip(means, stds), start=2):
            writer.writerow([f"b{j:02d}", f"{m:.8f}", f"{s:.8f}", A.shape[0]])
        run_means = A.mean(axis=1)
        overall_std = run_means.std(ddof=1) if len(run_means) > 1 else 0.0
        writer.writerow(["overall", f"{run_means.mean():.8f}", f"{overall_std:.8f}", A.shape[0]])


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ConfDS e-nose drift-adaptation experiment."
    )
    parser.add_argument(
        "--data",
        default="UciData.mat",
        help="Path to the prepared UCI MATLAB file (default: UciData.mat).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=CONFDS_TAU,
        help=(
            "Fixed ConfDS high/low confidence threshold (default: 0.80). "
            "Do not tune this value using full target-test labels."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for per-run and summary CSV files (default: results).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast end-to-end software test: first target batch only, first two "
            "experts, one seed combination, and few training iterations."
        ),
    )
    parser.add_argument(
        "--smoke-iters",
        type=int,
        default=20,
        help="Optimizer iterations per expert in --smoke mode (default: 20).",
    )
    parser.add_argument(
        "--smoke-experts",
        type=int,
        default=2,
        help="Number of experts used in --smoke mode (default: 2).",
    )
    parser.add_argument(
        "--no-source-prune",
        action="store_true",
        help=(
            "Disable the source-only expert pruning used in the original final "
            "experiment. By default pruning is enabled for full runs."
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.tau < 1.0):
        parser.error("--tau must be between 0 and 1.")
    if args.smoke_iters < 1:
        parser.error("--smoke-iters must be >= 1.")
    if args.smoke_experts < 1:
        parser.error("--smoke-experts must be >= 1.")

    configure_determinism()

    root = Path(__file__).resolve().parent
    cdan_path = root / "cdan_expert.py"
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    if not cdan_path.exists():
        raise FileNotFoundError(
            f"Expected {cdan_path.name} next to {Path(__file__).name}."
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    pair_seeds = [PAIR_SHUFFLE_SEEDS[0]] if args.smoke else PAIR_SHUFFLE_SEEDS
    run_seeds = [RUN_SEEDS[0]] if args.smoke else RUN_SEEDS
    total_runs = len(pair_seeds) * len(run_seeds)

    print("=== ConfDS E-Nose Drift Adaptation ===")
    print("Author          : Han Fan")
    print("Affiliation     : Technical University of Munich")
    print("Data            :", data_path)
    print("Mode            :", "SMOKE TEST" if args.smoke else "FULL EXPERIMENT")
    print("ConfDS tau      :", args.tau, "(fixed; no target-test sweep)")
    print("Pair budget cap :", PAIR_N)
    print(
        "Source pruning  :",
        "OFF (--no-source-prune)"
        if args.no_source_prune
        else f"ON (source acc >= {SRC_PRUNE_THRESH:.2f}; keep at least {SRC_PRUNE_KEEP_MIN})",
    )
    print("CUDA available  :", torch.cuda.is_available())
    if args.smoke:
        print("Smoke settings  :", f"b02 only, {args.smoke_experts} experts, {args.smoke_iters} iterations")
    print("")

    all_runs: List[List[float]] = []
    csv_rows: List[Dict] = []
    pruning_rows: List[Dict] = []
    run_id = 0

    for pair_seed in pair_seeds:
        for run_seed in run_seeds:
            run_id += 1
            print(
                f"\n=== Run {run_id}/{total_runs}: "
                f"training seed={run_seed}, pair-shuffle seed={pair_seed} ==="
            )
            seed_all(10000 + int(run_seed) + int(pair_seed))

            # Load through the exact CDAN loader so preprocessing and pair
            # selection are identical to expert training.
            loader = import_fresh("cdan_loader", cdan_path)
            loader.PAIR_SHUFFLE_SEED = int(pair_seed)
            loader.N_TRANS_SMP = int(PAIR_N)

            data, labels, transfer_pairs = loader.load_data(
                str(data_path),
                n_trans_smp=int(PAIR_N),
                data_divider=float(loader.DATA_DIVIDER),
                baseline_n=int(loader.BASELINE_N),
            )

            if len(data) != len(labels):
                raise ValueError("Number of data batches and label batches does not match.")
            if len(data) < 2:
                raise ValueError("At least one source and one target batch are required.")

            # Smoke mode intentionally limits the evaluation to b01 -> b02.
            if args.smoke:
                data = data[:2]
                labels = labels[:2]
                transfer_pairs = transfer_pairs[:1]

            class_order = np.unique(labels[0]).astype(np.int64)
            class_order.sort()
            print("Source shape    :", data[0].shape)
            print("Target batches  :", len(data) - 1)
            print("Classes         :", class_order.tolist())

            # Calibration labels are exactly the target members of the allowed
            # transfer-pair subset; no extra target samples are labelled.
            calibration_indices: List[np.ndarray] = []
            for b in range(1, len(data)):
                xt_pair = transfer_pairs[b - 1][1]
                idx = map_pair_rows_to_batch_indices(data[b], xt_pair, decimals=6)
                calibration_indices.append(idx)

            cal_sizes = [int(x.size) for x in calibration_indices]
            pair_sizes = [int(transfer_pairs[b][1].shape[0]) for b in range(len(transfer_pairs))]
            print("Pair sizes      :", pair_sizes)
            print("Calibration n   :", cal_sizes)

            if cal_sizes != pair_sizes:
                raise RuntimeError(
                    "Calibration-set sizes do not match the paired-target subset sizes."
                )

            configs = build_expert_configs(run_seed)
            if args.smoke:
                configs = configs[: min(args.smoke_experts, len(configs))]
                for cfg in configs:
                    cfg["MAX_ITERS"] = int(args.smoke_iters)

            print("Experts         :", len(configs))

            # Train all experts and record source-only retention scores.
            expert_records: List[Dict] = []
            for j, cfg in enumerate(configs, start=1):
                print(f"\n[Expert {j:02d}/{len(configs):02d}] {cfg['name']}")
                probabilities, source_acc_mean, source_acc_per_batch = run_one_expert(
                    cdan_path,
                    data,
                    labels,
                    transfer_pairs,
                    class_order,
                    cfg,
                )
                print(f"[SOURCE] mean head accuracy={source_acc_mean:.4f}")
                expert_records.append(
                    dict(
                        name=cfg["name"],
                        probabilities=probabilities,
                        source_acc_mean=source_acc_mean,
                        source_acc_per_batch=source_acc_per_batch,
                    )
                )

            # -------------------------------------------------------------
            # Source-only expert pruning used in the original final runner.
            # No target labels are consulted here.
            # -------------------------------------------------------------
            source_scores = np.asarray(
                [r["source_acc_mean"] for r in expert_records],
                dtype=np.float64,
            )
            keep = np.ones(len(expert_records), dtype=bool)

            pruning_enabled = ENABLE_SRC_PRUNE and not args.no_source_prune
            if pruning_enabled:
                keep = source_scores >= float(SRC_PRUNE_THRESH)
                keep_min = min(int(SRC_PRUNE_KEEP_MIN), len(expert_records))
                if int(np.sum(keep)) < keep_min:
                    order = np.argsort(-source_scores, kind="stable")
                    keep = np.zeros(len(expert_records), dtype=bool)
                    keep[order[:keep_min]] = True

                print(
                    f"[PRUNE] kept {int(np.sum(keep))}/{len(expert_records)} experts "
                    f"using source-only threshold={SRC_PRUNE_THRESH:.2f}"
                )
            else:
                print(f"[PRUNE] disabled; kept all {len(expert_records)} experts")

            for rec, is_kept in zip(expert_records, keep.tolist()):
                pruning_rows.append(
                    dict(
                        run_id=run_id,
                        train_seed=run_seed,
                        pair_shuffle_seed=pair_seed,
                        expert=rec["name"],
                        source_head_acc_mean=f"{rec['source_acc_mean']:.8f}",
                        kept=bool(is_kept),
                        threshold=f"{SRC_PRUNE_THRESH:.6f}",
                        pruning_enabled=bool(pruning_enabled),
                        smoke_test=bool(args.smoke),
                    )
                )

            kept_records = [
                rec for rec, is_kept in zip(expert_records, keep.tolist()) if is_kept
            ]
            if not kept_records:
                raise RuntimeError("Source pruning removed all experts.")

            # expert_outputs[m][bi] -> probability matrix (N_b, C)
            expert_outputs: List[List[np.ndarray]] = [
                rec["probabilities"] for rec in kept_records
            ]
            kept_expert_names = [rec["name"] for rec in kept_records]
            print("Fusion experts  :", len(expert_outputs))

            # Map the dataset's class labels to 0..C-1 indices expected by ConfDS.
            label_to_index = {int(label): i for i, label in enumerate(class_order.tolist())}

            batch_accuracy: List[float] = []
            for bi in range(len(data) - 1):
                # [M, N, C]
                expert_probabilities = np.stack(
                    [expert_outputs[m][bi] for m in range(len(expert_outputs))],
                    axis=0,
                )

                y_labels = labels[bi + 1].astype(np.int64)
                y = np.asarray(
                    [label_to_index[int(v)] for v in y_labels],
                    dtype=np.int64,
                )

                cal_idx = calibration_indices[bi]
                cal_y = y[cal_idx]

                q = confidence_aware_soft_ds(
                    expert_probabilities,
                    cal_idx,
                    cal_y,
                    tau=float(args.tau),
                    n_iter=30,
                    dirichlet_alpha=1.0,
                    clamp_labeled=True,
                )

                if q.shape != (len(y), len(class_order)):
                    raise RuntimeError(
                        f"Unexpected ConfDS posterior shape on b{bi+2:02d}: {q.shape}."
                    )
                if not np.all(np.isfinite(q)):
                    raise FloatingPointError(f"Non-finite ConfDS posterior on b{bi+2:02d}.")
                if not np.allclose(q.sum(axis=1), 1.0, atol=1e-5):
                    raise FloatingPointError(
                        f"ConfDS posterior does not sum to one on b{bi+2:02d}."
                    )

                prediction = np.argmax(q, axis=1)
                accuracy = float(np.mean(prediction == y))
                batch_accuracy.append(accuracy)

                print(
                    f"[ConfDS] b{bi+2:02d}: accuracy={accuracy:.4f}, "
                    f"calibration labels={len(cal_idx)}"
                )

                csv_rows.append(
                    dict(
                        run_id=run_id,
                        train_seed=run_seed,
                        pair_shuffle_seed=pair_seed,
                        batch=f"b{bi+2:02d}",
                        accuracy=f"{accuracy:.8f}",
                        calibration_labels=len(cal_idx),
                        n_experts=len(expert_outputs),
                        source_pruning=bool(pruning_enabled),
                        source_prune_threshold=f"{SRC_PRUNE_THRESH:.6f}",
                        confds_tau=f"{args.tau:.6f}",
                        smoke_test=bool(args.smoke),
                    )
                )

            all_runs.append(batch_accuracy)
            print(f"Run mean target accuracy: {np.mean(batch_accuracy):.6f}")

    A = np.asarray(all_runs, dtype=np.float64)
    batch_mean = A.mean(axis=0)
    batch_std = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(A.shape[1])
    run_means = A.mean(axis=1)
    overall_std = run_means.std(ddof=1) if len(run_means) > 1 else 0.0

    print("\n=== FINAL SUMMARY ===")
    print("Mode       :", "SMOKE TEST" if args.smoke else "FULL EXPERIMENT")
    print("n_runs     :", A.shape[0])
    print("fixed tau  :", args.tau)
    for j, (mean, std) in enumerate(zip(batch_mean, batch_std), start=2):
        print(f"b{j:02d}: {mean:.4f} ± {std:.4f}")
    print(f"Overall: {run_means.mean():.6f} ± {overall_std:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "full"
    runs_csv = output_dir / f"confds_{suffix}_runs.csv"
    summary_csv = output_dir / f"confds_{suffix}_summary.csv"
    pruning_csv = output_dir / f"confds_{suffix}_source_pruning.csv"
    write_run_csv(runs_csv, csv_rows)
    write_summary_csv(summary_csv, A)
    write_run_csv(pruning_csv, pruning_rows)
    print("\nSaved:")
    print(" ", runs_csv)
    print(" ", summary_csv)
    print(" ", pruning_csv)

    if args.smoke:
        print(
            "\nSMOKE TEST PASSED: the end-to-end training -> expert probabilities -> "
            "ConfDS -> evaluation pipeline completed without structural/numerical errors."
        )
        print("Smoke-test accuracy is not intended for scientific interpretation.")


if __name__ == "__main__":
    main()
