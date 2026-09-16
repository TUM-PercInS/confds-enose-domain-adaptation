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
This public runner uses a FIXED ConfDS confidence threshold (default 0.80).
It does not sweep the threshold using full target-test labels, does not perform
oracle evaluation, and does not apply development-only source pruning.

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
CONFDS_TAU = 0.80


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
    """Align classifier-output columns with the observed source class order."""
    p_raw = np.asarray(p_raw, dtype=np.float64)
    class_order = np.asarray(class_order, dtype=np.int64)

    if p_raw.ndim != 2:
        raise ValueError("Classifier probabilities must have shape (N, C).")

    # Common case: labels are zero-based class IDs and select output columns.
    if class_order.size and np.min(class_order) >= 0 and np.max(class_order) < p_raw.shape[1]:
        p = p_raw[:, class_order]
    # Alternative common case: labels are 1..C while the head has C outputs.
    elif p_raw.shape[1] == len(class_order):
        p = p_raw
    else:
        raise ValueError(
            f"Cannot align classifier output shape {p_raw.shape} with "
            f"class order {class_order.tolist()}."
        )

    denom = np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p / denom


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
) -> List[np.ndarray]:
    """Train one expert and return its probability matrix for every target batch."""
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
    original_head_acc = cdan.head_acc

    @torch.no_grad()
    def capture_head(model, x: np.ndarray, y: np.ndarray, device: str) -> float:
        """Capture the same classifier-head output used in the final experiment."""
        model.eval()
        xt = torch.as_tensor(x, device=device, dtype=torch.float32)

        # This reproduces the capture convention used in the final experiment:
        # target routing ID = 1 at classifier-head evaluation.
        d = torch.ones((xt.size(0),), device=device, dtype=torch.long)
        z = model.encode(xt, d_cond=d, d_corr=d)
        logits = model.predict(z)
        p_raw = torch.softmax(logits, dim=1).cpu().numpy()
        p = align_probabilities(p_raw, class_order)
        target_probabilities.append(p)

        pred_labels = class_order[np.argmax(p, axis=1)]
        return float(np.mean(pred_labels == y))

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

    for bi, p in enumerate(target_probabilities, start=2):
        if not np.all(np.isfinite(p)):
            raise FloatingPointError(f"Non-finite probabilities from {cfg['name']} on b{bi:02d}.")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
            raise FloatingPointError(f"Probabilities do not sum to one for {cfg['name']} on b{bi:02d}.")

    return target_probabilities


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
    print("CUDA available  :", torch.cuda.is_available())
    if args.smoke:
        print("Smoke settings  :", f"b02 only, {args.smoke_experts} experts, {args.smoke_iters} iterations")
    print("")

    all_runs: List[List[float]] = []
    csv_rows: List[Dict] = []
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

            # expert_outputs[m][bi] -> probability matrix (N_b, C)
            expert_outputs: List[List[np.ndarray]] = []
            for j, cfg in enumerate(configs, start=1):
                print(f"\n[Expert {j:02d}/{len(configs):02d}] {cfg['name']}")
                probabilities = run_one_expert(
                    cdan_path,
                    data,
                    labels,
                    transfer_pairs,
                    class_order,
                    cfg,
                )
                expert_outputs.append(probabilities)

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
                        n_experts=len(configs),
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
    write_run_csv(runs_csv, csv_rows)
    write_summary_csv(summary_csv, A)
    print("\nSaved:")
    print(" ", runs_csv)
    print(" ", summary_csv)

    if args.smoke:
        print(
            "\nSMOKE TEST PASSED: the end-to-end training -> expert probabilities -> "
            "ConfDS -> evaluation pipeline completed without structural/numerical errors."
        )
        print("Smoke-test accuracy is not intended for scientific interpretation.")


if __name__ == "__main__":
    main()
