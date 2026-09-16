# -*- coding: utf-8 -*-
"""
Confidence-aware soft Dawid--Skene fusion (ConfDS).

Eurosensors 2026 implementation.

Author
------
Han Fan

Affiliation
-----------
Munich Institute of Robotics and Machine Intelligence,
Chair of Perception for Intelligent Systems,
Technical University of Munich,
Germany

Contact
-------
han.fan@tum.de

Description
-----------
Each expert provides a class-probability distribution for every target
sample. Expert predictions are separated into high- and low-confidence
channels. A separate confusion matrix is estimated for each expert and
confidence channel using expectation--maximization (EM).

A small labelled calibration subset is used as fixed anchors.

The output q[i, k] is the fused posterior probability that target sample i
belongs to class k. The final classification is argmax_k q[i, k].
"""

from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    denom = np.sum(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def confidence_aware_soft_ds(
    expert_probabilities: np.ndarray,
    calibration_indices: np.ndarray,
    calibration_labels: np.ndarray,
    *,
    tau: float = 0.80,
    n_iter: int = 30,
    dirichlet_alpha: float = 1.0,
    clamp_labeled: bool = True,
) -> np.ndarray:
    """
    Confidence-aware soft Dawid--Skene fusion.

    Parameters
    ----------
    expert_probabilities : np.ndarray, shape (M, N, C)
        Soft class probabilities from M experts for N target samples
        and C classes.

    calibration_indices : np.ndarray, shape (K,)
        Indices of the K labelled calibration samples in the target batch.

    calibration_labels : np.ndarray, shape (K,)
        Ground-truth class indices for the calibration samples.

    tau : float
        Confidence threshold separating high- and low-confidence
        expert predictions.

    n_iter : int
        Maximum number of EM iterations.

    dirichlet_alpha : float
        Symmetric Dirichlet smoothing parameter for the expert
        confusion matrices.

    clamp_labeled : bool
        If True, posterior probabilities of calibration samples are
        fixed to their known one-hot labels during EM.

    Returns
    -------
    q : np.ndarray, shape (N, C)
        Fused class posterior probabilities.
    """

    P = np.asarray(expert_probabilities, dtype=np.float64)

    if P.ndim != 3:
        raise ValueError(
            "expert_probabilities must have shape (M, N, C)."
        )

    M, N, C = P.shape

    # Ensure proper probability distributions.
    P = np.clip(P, 1e-12, None)
    P = _normalize(P, axis=2)

    calibration_indices = np.asarray(
        calibration_indices, dtype=np.int64
    ).ravel()

    calibration_labels = np.asarray(
        calibration_labels, dtype=np.int64
    ).ravel()

    if len(calibration_indices) != len(calibration_labels):
        raise ValueError(
            "calibration_indices and calibration_labels must have "
            "the same length."
        )

    # -------------------------------------------------------------
    # Confidence state
    #
    # bins[m, i] = 1 : expert m is high-confidence on sample i
    # bins[m, i] = 0 : expert m is low-confidence on sample i
    # -------------------------------------------------------------

    confidence = np.max(P, axis=2)
    bins = (confidence >= float(tau)).astype(np.int64)

    # -------------------------------------------------------------
    # Initialization:
    # simple mean of expert probability distributions.
    # -------------------------------------------------------------

    q = np.mean(P, axis=0)
    q = _normalize(q, axis=1)

    # Calibration labels act as fixed anchors.
    if clamp_labeled and len(calibration_indices) > 0:

        q[calibration_indices] = 0.0

        q[
            calibration_indices,
            calibration_labels
        ] = 1.0

    # theta[m, h, k, l]
    #
    # m : expert
    # h : confidence channel (0 low, 1 high)
    # k : latent/true class
    # l : class supported by expert output
    #
    # Thus each expert has TWO C x C confusion matrices.

    theta = np.full(
        (M, 2, C, C),
        1.0 / C,
        dtype=np.float64
    )

    for _ in range(int(n_iter)):

        # =========================================================
        # M-step
        #
        # Estimate confidence-specific confusion matrices from
        # current latent posteriors q.
        #
        # Because expert predictions are soft probabilities,
        # they contribute fractional counts.
        # =========================================================

        counts = np.full_like(
            theta,
            float(dirichlet_alpha)
        )

        for m in range(M):

            for h in (0, 1):

                ids = np.flatnonzero(
                    bins[m] == h
                )

                if len(ids) == 0:
                    continue

                # q[ids].T      : C x n
                # P[m, ids]     : n x C
                #
                # result        : C x C

                counts[m, h] += (
                    q[ids].T @ P[m, ids]
                )

        theta = _normalize(
            counts,
            axis=3
        )

        # =========================================================
        # E-step
        #
        # Infer fused class posterior for every target sample.
        # =========================================================

        class_prior = (
            np.sum(q, axis=0) + 1.0
        )

        class_prior = _normalize(
            class_prior,
            axis=0
        )

        log_q = np.log(
            class_prior + 1e-12
        )[None, :]

        log_q = np.repeat(
            log_q,
            N,
            axis=0
        )

        for m in range(M):

            for i in range(N):

                h = bins[m, i]

                # Confidence-specific confusion matrix
                # for expert m.
                th = theta[m, h]

                # Expected log-likelihood under the expert's
                # soft probability output.

                log_q[i] += np.sum(
                    P[m, i][None, :]
                    * np.log(th + 1e-12),
                    axis=1
                )

        # Numerically stable softmax.

        log_q -= np.max(
            log_q,
            axis=1,
            keepdims=True
        )

        q_new = np.exp(log_q)

        q_new = _normalize(
            q_new,
            axis=1
        )

        # Keep calibration labels fixed.

        if (
            clamp_labeled
            and len(calibration_indices) > 0
        ):

            q_new[calibration_indices] = 0.0

            q_new[
                calibration_indices,
                calibration_labels
            ] = 1.0

        # Convergence test.

        if np.max(
            np.abs(q_new - q)
        ) < 1e-7:

            q = q_new
            break

        q = q_new

    return q


def fused_prediction(q: np.ndarray) -> np.ndarray:
    """
    Convert fused posterior distributions into hard class predictions.
    """
    return np.argmax(q, axis=1)