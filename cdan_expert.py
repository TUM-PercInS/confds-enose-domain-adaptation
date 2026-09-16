# -*- coding: utf-8 -*-
"""
CDAN expert implementation for:

"Progressive Domain Adaptation under Sensor Drift for Gas Classification
 with Limited Calibration Labels"
Eurosensors 2026.

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

Implementation note
-------------------
The evaluation is batch-wise. For each target batch, a new expert model is
initialized and adapted using the labelled source domain (Batch 01), the
current target batch, and the available paired transfer measurements.

Adapted model parameters from one target batch are NOT propagated to the
next target batch. Transfer-pair constraints available up to the current
batch are accumulated for pairwise feature alignment.

The implementation contains:
    - source-label classification;
    - conditional adversarial domain adaptation (CDAN);
    - paired source--target feature alignment;
    - EMA-teacher pseudo-label self-training;
    - optional domain-conditioned FiLM / DSBN;
    - a lightweight domain-conditioned correction module.

The target labels used for final evaluation are not used by the CDAN
adaptation losses.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict

import numpy as np
import math
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

import torch
from torch import nn
import torch.nn.functional as F


# =============================================================================
# USER SETTINGS
# =============================================================================
ALIGN_CURRENT_ONLY = True

MAT_PATH = "UciData.mat"   # set absolute path if needed
SEED = 0


# Fixed-seed shuffle for unlabeled transfer pairs (Tsrc/Ttar); uses NO class labels.
PAIR_SHUFFLE_SEED = 2023 #1337
DATA_DIVIDER = 2.0
N_TRANS_SMP = 10  # caps unlabeled transfer pairs (Tsrc/Ttar) used for pair alignment; NO target labels used

BASELINE_N = 0    # keep 0 unless you know you want row-feature baseline correction

# downstream evaluation classifier to match your DCAE style
LR_MAX_ITER = 5000
LR_SOLVER = "lbfgs"

# ---- Training budget (per target batch b) ----
MAX_ITERS = 2000        # optimizer steps per target batch
BATCH_SRC = 256         # labeled source minibatch
BATCH_PER_DOMAIN = 256  # per-domain minibatch for domain loss (balanced)

LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

HIDDEN = [256, 128]     # encoder MLP hidden sizes (tabular -> usually larger than SAE)
FEATURE_DIM = None      # if None, inferred from last hidden in HIDDEN

# ---- Adversarial settings ----
LAMBDA_ADV = 0.0        # IMPORTANT: keep small first; too large often destroys class info
USE_GRL_SCHEDULE = True
ADV_WARMUP_FRAC = 0.30  # fraction of iterations with lambda_adv ~ 0 -> stabilize classifier first

# ---- Conditional settings (CDAN) ----
USE_CONDITIONAL = True          # True=CDAN, False=DANN
COND_MAP = "outer"              # "outer" (classic) or "concat"
DETACH_PRED_FOR_DOMAIN = True   # detach p(y|x) when feeding domain disc (stabilizes on tabular)


# ---- Domain-conditioned encoder (optional) ----
# 'none' (default): plain encoder
# 'film': input FiLM (domain-conditioned affine)
# 'dsbn': domain-specific batchnorm inside encoder
# 'film+dsbn': both
DOMAIN_COND_MODE = "film+dsbn"

DSBN_BLEND = True
DSBN_BLEND_START_BATCH = 2   # 从 batch3 才开始引入 DSBN（batch2 alpha=0）
DSBN_BLEND_RAMP = 8         # 3 个 batch 内线性升到最大
DSBN_BLEND_MAX_ALPHA = 1.0   # 最终 DSBN 占比


# DSBN stability knobs (no class labels used)
DSBN_FREEZE_AFFINE = True

# ---------------- DCAE-like correction head (Exp2/Exp3) ----------------
# Provide an explicit small correction head on the encoder output so we can ablate:
#   - pair loss on base feature vs corrected feature (PAIR_ALIGN_ON)
#   - train only the correction module after a source-only warmup (FREEZE_BASE_AFTER)
USE_DCAE_CORR = True
DCAE_COR_N_UNIT = 6          # similar scale to DCAE N_COR_UNIT
DCAE_COR_ACT = "tanh"        # "tanh" | "relu" | "none"
PAIR_ALIGN_ON = "corr"       # "base" | "corr"
FREEZE_BASE_AFTER = 0      # 0 disables; otherwise freeze encoder MLP after this many iters per batch
TRAIN_CLS_IN_STAGE2 = True   # keep training classifier head in stage2
TRAIN_FILM_IN_STAGE2 = False # also allow FiLM to adapt in stage2 (usually keep False for DCAE-like)
LAMBDA_CORR_REG = 1e-3       # penalize correction magnitude (keeps it "small")
  # freeze BN weight/bias; only running stats adapt per domain (AdaBN-style)
DSBN_CALIBRATE = True      # after each incremental training step, refresh DSBN running stats on the full current target batch
DSBN_CALIBRATE_BATCH = 512 # batch size for the calibration forward pass
# ---- Domain discriminator label space ----
ADV_BINARY_DOMAIN = True        # True: source(0) vs target(1) (weaker: all targets share same cond); False: multi-class domain id 0..B-1 (recommended for Exp2/3)
# When ADV_BINARY_DOMAIN=True, keep FiLM/DSBN in *binary* space (robust stats), but allow the
# DCAE-like correction head to use the *full* original domain IDs (0..b). This avoids DSBN "tiny groups"
# in pair alignment while retaining DCAE-like per-batch correction.
CORR_USE_FULL_DOMAIN_ID = False

# Soft stage-2 (alternative to hard FREEZE_BASE_AFTER): after STAGE2_AFTER steps, keep training the base
# encoder but with a smaller LR and (optionally) L2-SP regularization to a snapshot.
STAGE2_AFTER = 0              # 0 disables
LR_BASE_MULT_STAGE2 = 0.1     # base LR = LR * this
L2SP_WEIGHT = 0.0             # 0 disables; try 1e-6 ~ 1e-4

# ---- Self-training (pseudo labels) ----
USE_PSEUDO = True
PSEUDO_TAU = 0.995               # confidence threshold for pseudo labels
LAMBDA_PSEUDO = 0.3150            # weight for pseudo-label CE
USE_EMA_TEACHER = True
EMA_DECAY = 0.999

# ---- Optional pair alignment using provided (Tsrc,Ttar) ----
USE_PAIR_ALIGN = True #True
LAMBDA_PAIR = 0.0315
PAIR_MAX_K = 512


# ---- Extra regularizers (optional) ----
LAMBDA_ENT = 0.00       # target entropy minimization (0 disables). Try 0.001~0.01 if CDAN unstable.
CONF_GATING = 0.85      # if >0, only apply domain loss on samples with max prob >= CONF_GATING (0 disables)

# ---- Which accuracy to report ----
# To match your DCAE pipeline, we report LR-on-features accuracy by default.
REPORT_HEAD_ACC = False   # set True to also print end-to-end classifier-head acc

# ---- Optional: run DCAE reference (requires your existing dcae_py312 + eval_dcae_fixed code) ----
RUN_DCAE_REFERENCE = True


# =============================================================================
# Helpers (mirrors your DCAE script loader)
# =============================================================================
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """Exponential moving average teacher update (Mean Teacher)."""
    with torch.no_grad():
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.data.mul_(decay).add_(p_s.data, alpha=(1.0 - decay))

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_lr():
    return LogisticRegression(solver=LR_SOLVER, max_iter=LR_MAX_ITER)

def _baseline_correct_first_n(x: np.ndarray, n: int) -> np.ndarray:
    if n is None or n <= 0:
        return x
    n = int(min(n, x.shape[0]))
    base = np.mean(x[:n], axis=0, keepdims=True)
    return x - base

def load_data(mat_path: str, *, n_trans_smp: int, data_divider: float, baseline_n: int):
    dd = loadmat(mat_path)
    data0 = dd["data"][0].tolist()
    label0 = dd["label"][0].tolist()
    labels = [np.int32(t.flatten()) for t in label0]

    # Transfer pairs (optional; used only if USE_PAIR_ALIGN=True)
    t_src = dd.get("Tsrc", None)
    t_tar = dd.get("Ttar", None)

    # optional baseline correction
    data_bc = []
    for b in range(len(labels)):
        xb = np.asarray(data0[b], dtype=np.float32)
        xb = _baseline_correct_first_n(xb, baseline_n)
        data_bc.append(xb)

    scaler0 = preprocessing.StandardScaler().fit(data_bc[0])

    data = []
    for b in range(len(labels)):
        x = scaler0.transform(data_bc[b]) / float(data_divider)
        data.append(x.astype(np.float32))

    tr_smp = []
    if t_src is not None and t_tar is not None:
        t_src = t_src[0].tolist()
        t_tar = t_tar[0].tolist()
        for b in range(len(t_src)):
            xs0 = np.asarray(t_src[b], dtype=np.float32)
            xt0 = np.asarray(t_tar[b], dtype=np.float32)
            xs0 = _baseline_correct_first_n(xs0, baseline_n)
            xt0 = _baseline_correct_first_n(xt0, baseline_n)

            # Deterministic shuffle of unlabeled transfer pairs to avoid order bias.
            # IMPORTANT: this uses NO class labels (unsupervised); it only reorders precomputed pairs.
            rngp = np.random.RandomState(int(PAIR_SHUFFLE_SEED) + 1000 * int(b))
            n_pair = min(xs0.shape[0], xt0.shape[0])
            perm = rngp.permutation(n_pair)
            xs0 = xs0[perm]
            xt0 = xt0[perm]

            xs = scaler0.transform(xs0)[:n_trans_smp] / float(data_divider)
            xt = scaler0.transform(xt0)[:n_trans_smp] / float(data_divider)
            tr_smp.append((xs.astype(np.float32), xt.astype(np.float32)))
    else:
        # one pair-set per target batch (b=1..B-1)
        tr_smp = [(
            np.zeros((0, data[0].shape[1]), np.float32),
            np.zeros((0, data[0].shape[1]), np.float32),
        ) for _ in range(len(data)-1)]

    return data, labels, tr_smp


def eval_raw_lr(data: List[np.ndarray], labels: List[np.ndarray]) -> Tuple[List[float], float]:
    lr = make_lr()
    lr.fit(data[0], labels[0])
    acc = []
    for b in range(1, len(labels)):
        acc.append(float(lr.score(data[b], labels[b])))
    return acc, float(np.mean(acc))

def fmt_acc_list(xs: List[float]) -> str:
    return " ".join(f"{x:.4f}" for x in xs)


# =============================================================================
# DANN/CDAN components
# =============================================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambd = 1.0

    def set_lambda(self, lambd: float) -> None:
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradReverse.apply(x, self.lambd)

def grl_schedule(progress: float) -> float:
    # DANN schedule: 2/(1+exp(-10p))-1
    p = float(np.clip(progress, 0.0, 1.0))
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

def adv_warmup(progress: float, warmup_frac: float) -> float:
    """0 during warmup, then ramp to 1 linearly."""
    if warmup_frac <= 0:
        return 1.0
    if progress <= warmup_frac:
        return 0.0
    return float((progress - warmup_frac) / max(1e-8, 1.0 - warmup_frac))


class DCAECorrectionHead(nn.Module):
    """Low-rank additive correction conditioned on a *scalar* domain index (DCAE-like).

    delta(d) = cor([1, s(d)]) @ Wd1, where cor(u) = act(u @ Wd0)
    - Wd0 is zero-initialized => identity/no correction at start.
    - Uses only domain ID (no class labels).
    """
    def __init__(self, z_dim: int, n_cor_unit: int = 6, act: str = "tanh"):
        super().__init__()
        self.z_dim = int(z_dim)
        self.n_cor_unit = int(n_cor_unit)
        self.Wd0 = nn.Parameter(torch.zeros(2, self.n_cor_unit))  # zero-init like DCAE
        self.Wd1 = nn.Parameter(torch.empty(self.n_cor_unit, self.z_dim))
        bound = math.sqrt(6.0 / (self.Wd1.shape[0] + self.Wd1.shape[1]))
        nn.init.uniform_(self.Wd1, -bound, bound)
        act = str(act).lower()
        if act in ("none", "linear", ""):
            self.act = None
        elif act == "relu":
            self.act = F.relu
        else:
            self.act = torch.tanh

    def delta(self, d: torch.Tensor) -> torch.Tensor:
        if d.dim() == 0:
            d = d.expand(1)
        # Map domain id -> monotonic scalar index (source=1, target k -> k+1)
        s = d.float() + 1.0
        ones = torch.ones_like(s)
        dom_ft = torch.stack([ones, s], dim=1)  # [B,2]
        cor = dom_ft @ self.Wd0  # [B,k]
        if self.act is not None:
            cor = self.act(cor)
        return cor @ self.Wd1  # [B,z_dim]


class EncoderMLP(nn.Module):
    """MLP encoder with optional DSBN (Soft-DSBN, *identity-preserving*).

    Design goal (critical for your ablations):
      - If alpha==0 (i.e., before DSBN_BLEND_START_BATCH), DSBN must have *zero* effect.
        => encoder behaves exactly like a plain MLP (Linear -> ReLU -> Dropout).
      - If alpha>0, DSBN is blended in on the pre-activation:
            x_lin <- Linear(x)
            x_dom <- DSBN(x_lin, d)
            x_lin <- (1-alpha)*x_lin + alpha*x_dom
        Then apply ReLU/Dropout.

    This avoids the earlier issue where a "shared BN" changed behavior even when DSBN was "off".
    """

    def __init__(self, n_in: int, hidden: List[int], dropout: float, *, n_domains: int, use_dsbn: bool):
        super().__init__()
        self.use_dsbn = bool(use_dsbn)
        self.dropout = float(dropout) if dropout is not None else 0.0
        self.n_domains = int(n_domains)

        self.fcs = nn.ModuleList()
        self.dsbn = nn.ModuleList()

        h_prev = int(n_in)
        for h in hidden:
            h = int(h)
            self.fcs.append(nn.Linear(h_prev, h))
            if self.use_dsbn:
                self.dsbn.append(DomainSpecificBatchNorm1d(h, self.n_domains, freeze_affine=bool(DSBN_FREEZE_AFFINE)))
            else:
                self.dsbn.append(nn.Identity())
            h_prev = h
        self.out_dim = h_prev

    def _dsbn_alpha(self, cur_batch: int) -> float:
        """Blend weight alpha in [0, DSBN_BLEND_MAX_ALPHA]."""
        if not self.use_dsbn or not bool(DSBN_BLEND):
            return float(DSBN_BLEND_MAX_ALPHA if self.use_dsbn else 0.0)
        b0 = int(DSBN_BLEND_START_BATCH)
        ramp = max(1, int(DSBN_BLEND_RAMP))
        if cur_batch < b0:
            return 0.0
        t = min(1.0, float(cur_batch - b0 + 1) / float(ramp))
        return float(DSBN_BLEND_MAX_ALPHA) * t

    def forward(self, x: torch.Tensor, d: Optional[torch.Tensor] = None, *, cur_batch: int = 1) -> torch.Tensor:
        alpha = self._dsbn_alpha(int(cur_batch))
        for i, fc in enumerate(self.fcs):
            x_lin = fc(x)
            if self.use_dsbn and (d is not None) and (alpha > 0.0):
                # DomainSpecificBatchNorm1d already handles batch-size=1 safely.
                x_dom = self.dsbn[i](x_lin, d)
                x_lin = (1.0 - alpha) * x_lin + alpha * x_dom
            x = F.relu(x_lin)
            if self.dropout and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class DANN_CDAN(nn.Module):
    def __init__(self, n_in: int, n_classes: int, n_domains_total: int):
        super().__init__()
        self.n_domains_cond = int(2 if ADV_BINARY_DOMAIN else n_domains_total)
        _mode = str(DOMAIN_COND_MODE).lower()
        # Accept synonyms: "both" means film+dsbn
        self.use_film = ("film" in _mode) or ("both" in _mode)
        self.use_dsbn = ("dsbn" in _mode) or ("both" in _mode)
        self.film = DomainFiLM(self.n_domains_cond, int(n_in)) if self.use_film else None
        self.enc = EncoderMLP(n_in, HIDDEN, DROPOUT, n_domains=self.n_domains_cond, use_dsbn=self.use_dsbn)
        self.cls = nn.Linear(self.enc.out_dim, int(n_classes))
        self.grl = GRL()

        # Optional DCAE-like correction on encoder output
        self.corr = DCAECorrectionHead(self.enc.out_dim, n_cor_unit=int(DCAE_COR_N_UNIT), act=str(DCAE_COR_ACT)) if bool(USE_DCAE_CORR) else None

        if USE_CONDITIONAL:
            if COND_MAP.lower() == "outer":
                dom_in = int(self.enc.out_dim) * int(n_classes)
            else:
                dom_in = int(self.enc.out_dim) + int(n_classes)
        else:
            dom_in = int(self.enc.out_dim)

        self.dom = nn.Sequential(
            nn.Linear(dom_in, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, (2 if ADV_BINARY_DOMAIN else int(n_domains_total))),
        )

    def encode_base(self, x: torch.Tensor, d: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Base encoder (FiLM + EncoderMLP/DSBN), NO correction head.
        if d is None:
            d = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
        elif d.dim() == 0:
            d = d.expand(x.size(0)).to(device=x.device, dtype=torch.long)
        else:
            d = d.to(device=x.device, dtype=torch.long)
        if self.film is not None:
            x = self.film(x, d)
        return self.enc(x, d if self.use_dsbn else None, cur_batch=int(getattr(self, 'cur_batch', 1)))

    def _apply_corr(self, z: torch.Tensor, d_corr: torch.Tensor) -> torch.Tensor:
        if self.corr is None:
            return z
        return z + self.corr.delta(d_corr)
    def encode(self, x: torch.Tensor, d_cond: Optional[torch.Tensor] = None, d_corr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Default encoding used for features/classification: base (FiLM/DSBN) + (optional) correction.
        # d_cond: domain id used for FiLM/DSBN routing (binary or multi)
        # d_corr: domain id used for the correction head (can be full domain id even if d_cond is binary)
        if d_cond is None:
            d_cond = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
        elif d_cond.dim() == 0:
            d_cond = d_cond.expand(x.size(0)).to(device=x.device, dtype=torch.long)
        else:
            d_cond = d_cond.to(device=x.device, dtype=torch.long)

        if d_corr is None:
            d_corr = d_cond
        elif d_corr.dim() == 0:
            d_corr = d_corr.expand(x.size(0)).to(device=x.device, dtype=torch.long)
        else:
            d_corr = d_corr.to(device=x.device, dtype=torch.long)

        z0 = self.encode_base(x, d_cond)
        return self._apply_corr(z0, d_corr)

    def freeze_base(self, *, train_corr: bool = True, train_cls: bool = True, train_film: bool = False) -> None:
        # Freeze encoder MLP weights; keep correction/head trainable as configured.
        for p in self.enc.fcs.parameters():
            p.requires_grad_(False)
        if self.film is not None:
            for p in self.film.parameters():
                p.requires_grad_(bool(train_film))
        if self.corr is not None:
            for p in self.corr.parameters():
                p.requires_grad_(bool(train_corr))
        for p in self.cls.parameters():
            p.requires_grad_(bool(train_cls))


    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.cls(z)

    def domain_logits(self, z: torch.Tensor, p: Optional[torch.Tensor]) -> torch.Tensor:
        if USE_CONDITIONAL:
            assert p is not None
            p_in = p.detach() if DETACH_PRED_FOR_DOMAIN else p
            if COND_MAP.lower() == "outer":
                op = torch.bmm(z.unsqueeze(2), p_in.unsqueeze(1))  # [B,H,C]
                feat = op.view(op.size(0), -1)
            else:
                feat = torch.cat([z, p_in], dim=1)
        else:
            feat = z
        feat = self.grl(feat)
        return self.dom(feat)


def entropy_loss(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    ent = -torch.sum(p * torch.log(p + eps), dim=1)
    return torch.mean(ent)

def sample_idx(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    k = int(min(k, n))
    return rng.choice(n, size=(k,), replace=(k > n))


def calibrate_dsbn(model: nn.Module, x_np: np.ndarray, dom_id: int, *, device: torch.device, batch_size: int = 512) -> None:
    """Refresh DSBN running stats on a full (unlabeled) target batch.

    This uses NO class labels; it only forwards x with a fixed domain id to update BatchNorm running_mean/var.
    """
    if not getattr(model, "use_dsbn", False):
        return
    model_was_training = model.training
    # Disable dropout during calibration while keeping BN in train mode (to update running stats).
    enc = getattr(model, "enc", None)
    saved_dropout = getattr(enc, "dropout", None)
    if enc is not None and saved_dropout is not None:
        enc.dropout = 0.0
    model.train()
    with torch.no_grad():
        n = int(x_np.shape[0])
        for i in range(0, n, int(batch_size)):
            xb = torch.as_tensor(x_np[i:i+int(batch_size)], device=device, dtype=torch.float32)
            d = torch.full((xb.size(0),), int(dom_id), device=device, dtype=torch.long)
            _ = model.encode(xb, d_cond=d, d_corr=torch.full((xb.size(0),), int(dom_id), device=device, dtype=torch.long))
    # restore
    if enc is not None and saved_dropout is not None:
        enc.dropout = saved_dropout
    model.train(model_was_training)


@torch.no_grad()
def lr_on_features_acc(model: DANN_CDAN, data: List[np.ndarray], labels: List[np.ndarray], upto_b: int, device: str) -> float:
    """Train LR on *source features* and evaluate on batch upto_b (single batch)."""
    model.eval()
    x0 = torch.as_tensor(data[0], device=device, dtype=torch.float32)
    d0 = torch.zeros((x0.size(0),), device=device, dtype=torch.long)
    z0 = model.encode(x0, d_cond=d0, d_corr=d0).cpu().numpy()
    lr = make_lr()
    lr.fit(z0, labels[0])

    xb = torch.as_tensor(data[upto_b], device=device, dtype=torch.float32)
    db_cond = torch.full((xb.size(0),), (1 if ADV_BINARY_DOMAIN else int(upto_b)), device=device, dtype=torch.long)
    db_corr = torch.full((xb.size(0),), (int(upto_b) if (ADV_BINARY_DOMAIN and CORR_USE_FULL_DOMAIN_ID) else (1 if ADV_BINARY_DOMAIN else int(upto_b))), device=device, dtype=torch.long)
    zb = model.encode(xb, d_cond=db_cond, d_corr=db_corr).cpu().numpy()
    return float(lr.score(zb, labels[upto_b]))

@torch.no_grad()
def head_acc(model: DANN_CDAN, x: np.ndarray, y: np.ndarray, device: str) -> float:
    model.eval()
    xt = torch.as_tensor(x, device=device, dtype=torch.float32)
    # For reporting head accuracy we treat inputs as 'target' domain (binary=1).
    d = torch.full((xt.size(0),), (1 if ADV_BINARY_DOMAIN else 1), device=device, dtype=torch.long)
    z = model.encode(xt, d_cond=d, d_corr=d)
    logits = model.predict(z)
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    return float(np.mean(pred == y))


def train_eval_incremental(
    data: List[np.ndarray],
    labels: List[np.ndarray],
    tr_smp: List[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> Tuple[List[float], float, Optional[List[float]]]:
    """
    Match DCAE script protocol:
    For each target batch b=1..B-1:
      - Train/adapt using source labels + unlabeled targets (1..b) for domain loss
      - Evaluate on batch b
    """

    set_seed(seed)
    rng = np.random.RandomState(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_src = data[0]
    y_src = labels[0]
    # The MATLAB benchmark representation uses one-based class labels (1,...,6).
    # For fidelity to the experiments reported in the paper, the classifier
    # therefore retains the original max(label)+1 output convention.
    n_classes = int(np.max(y_src) + 1)
    n_domains_total = len(data)  # domain ids 0..B-1

    acc_lr: List[float] = []
    acc_head: List[float] = []

    for b in range(1, len(data)):
        # Re-init model each b (mirrors your DCAE eval which retrains each target step)
        model = DANN_CDAN(n_in=int(x_src.shape[1]), n_classes=int(n_classes), n_domains_total=int(n_domains_total)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # EMA teacher for pseudo-labeling (optional)
        teacher = None
        if USE_EMA_TEACHER and USE_PSEUDO:
            teacher = DANN_CDAN(n_in=int(x_src.shape[1]), n_classes=int(n_classes), n_domains_total=int(n_domains_total)).to(device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()

        # Current target batch index (2..B). Used for DSBN ramping/blending.
        # NOTE: this uses ONLY the batch/domain index; it does NOT use class labels.
        model.cur_batch = int(b + 1)
        if teacher is not None:
            teacher.cur_batch = int(b + 1)


        # domains seen so far: 0..b dom_arrays = [data[k] for k in range(0, b + 1)]
        if ALIGN_CURRENT_ONLY:
            dom_ids = [0, b]  # use only source (0) and the *current* target domain (b)
        else:
            dom_ids = list(range(0, b + 1))  # cumulative targets 1..b
        # NOTE: dom_ids holds the *original* domain IDs; always index `data` by domain ID.


        # Accumulate transfer pairs up to current target batch b (for fairer comparison to pair-based methods)
        pair_src = np.zeros((0, x_src.shape[1]), dtype=np.float32)
        pair_tar = np.zeros((0, x_src.shape[1]), dtype=np.float32)
        pair_src_dom_cond = np.zeros((0,), dtype=np.int64)
        pair_src_dom_corr = np.zeros((0,), dtype=np.int64)
        pair_tar_dom_cond = np.zeros((0,), dtype=np.int64)
        pair_tar_dom_corr = np.zeros((0,), dtype=np.int64)
        if USE_PAIR_ALIGN and (b >= 1) and (len(tr_smp) >= b):
            for k in range(1, b + 1):
                xs_k, xt_k = tr_smp[k - 1]
                if xs_k.size and xt_k.size:
                    pair_src = np.vstack([pair_src, xs_k])
                    pair_tar = np.vstack([pair_tar, xt_k])
                    pair_src_dom_cond = np.hstack([pair_src_dom_cond, np.zeros((xs_k.shape[0],), dtype=np.int64)])
                    pair_src_dom_corr = np.hstack([pair_src_dom_corr, np.zeros((xs_k.shape[0],), dtype=np.int64)])
                    pair_tar_dom_cond = np.hstack([pair_tar_dom_cond, np.full((xt_k.shape[0],), (1 if ADV_BINARY_DOMAIN else int(k)), dtype=np.int64)])
                    pair_tar_dom_corr = np.hstack([pair_tar_dom_corr, np.full((xt_k.shape[0],), (int(k) if (ADV_BINARY_DOMAIN and CORR_USE_FULL_DOMAIN_ID) else (1 if ADV_BINARY_DOMAIN else int(k))), dtype=np.int64)])

        model.train()
        for it in range(int(MAX_ITERS)):
            # ---- stage switch (Exp3): source-only warmup then freeze base encoder ----
            if (FREEZE_BASE_AFTER and int(FREEZE_BASE_AFTER) > 0) and (it == int(FREEZE_BASE_AFTER)):
                model.freeze_base(train_corr=True, train_cls=bool(TRAIN_CLS_IN_STAGE2), train_film=bool(TRAIN_FILM_IN_STAGE2))
                opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)


            progress = (it + 1) / float(MAX_ITERS)

            # GRL lambda schedule
            grl_lam = grl_schedule(progress) if USE_GRL_SCHEDULE else 1.0
            # warmup multiplier (0 -> 1)
            wmul = adv_warmup(progress, ADV_WARMUP_FRAC)
            model.grl.set_lambda(grl_lam * wmul)

            # ----- supervised loss on source -----
            idx_s = sample_idx(x_src.shape[0], int(BATCH_SRC), rng)
            xs = torch.as_tensor(x_src[idx_s], device=device, dtype=torch.float32)
            ys = torch.as_tensor(y_src[idx_s], device=device, dtype=torch.long)

            z_s = model.encode(xs, d_cond=torch.zeros((xs.size(0),), device=device, dtype=torch.long), d_corr=torch.zeros((xs.size(0),), device=device, dtype=torch.long))
            logits_s = model.predict(z_s)
            loss_cls = F.cross_entropy(logits_s, ys)

            # ----- domain loss (balanced per domain) -----
            xs_dom_list = []
            dom_lab_list = []
            dom_corr_list = []
            for dom_id in dom_ids: #range(0, b + 1):
                a = data[dom_id]
                idx = sample_idx(a.shape[0], int(BATCH_PER_DOMAIN), rng)
                xs_dom_list.append(a[idx])
                # dom_lab_list.append(np.full((idx.shape[0],), (0 if (ADV_BINARY_DOMAIN and dom_id == 0) else (1 if ADV_BINARY_DOMAIN else dom_id)), dtype=np.int64))
                dom_lab_list.append(np.full((idx.shape[0],), (0 if dom_id == 0 else 1) if ADV_BINARY_DOMAIN else dom_id, dtype=np.int64))
                dom_corr_list.append(np.full((idx.shape[0],), (dom_id if (ADV_BINARY_DOMAIN and CORR_USE_FULL_DOMAIN_ID) else ((0 if dom_id == 0 else 1) if ADV_BINARY_DOMAIN else dom_id)), dtype=np.int64))
            
            xdom = np.vstack(xs_dom_list).astype(np.float32)
            dlab = np.concatenate(dom_lab_list).astype(np.int64)
            dcorr = np.concatenate(dom_corr_list).astype(np.int64)

            xdom_t = torch.as_tensor(xdom, device=device, dtype=torch.float32)
            dlab_t = torch.as_tensor(dlab, device=device, dtype=torch.long)
            dcorr_t = torch.as_tensor(dcorr, device=device, dtype=torch.long)

            z_d = model.encode(xdom_t, d_cond=dlab_t, d_corr=dcorr_t)
            logits_y_d = model.predict(z_d)
            p_d = F.softmax(logits_y_d, dim=1)

            # optional confidence gating for conditional DA stability
            if CONF_GATING and CONF_GATING > 0:
                conf = torch.max(p_d.detach(), dim=1).values
                mask = conf >= float(CONF_GATING)
                if torch.any(mask):
                    z_d2 = z_d[mask]
                    p_d2 = p_d[mask]
                    dlab2 = dlab_t[mask]
                    dom_logits = model.domain_logits(z_d2, p_d2 if USE_CONDITIONAL else None)
                    loss_dom = F.cross_entropy(dom_logits, dlab2)
                else:
                    loss_dom = torch.tensor(0.0, device=device)
            else:
                dom_logits = model.domain_logits(z_d, p_d if USE_CONDITIONAL else None)
                loss_dom = F.cross_entropy(dom_logits, dlab_t)

            # optional entropy minimization on targets (exclude domain 0)
            loss_ent = torch.tensor(0.0, device=device)
            if LAMBDA_ENT and LAMBDA_ENT > 0:
                mask_t = (dlab_t != 0)
                if torch.any(mask_t):
                    loss_ent = entropy_loss(p_d[mask_t])

            # total

            # ----- pseudo-label self-training on target (high-confidence only) -----
            loss_pseudo = torch.tensor(0.0, device=device)
            if USE_PSEUDO and (LAMBDA_PSEUDO > 0) and not (FREEZE_BASE_AFTER and int(FREEZE_BASE_AFTER) > 0 and it < int(FREEZE_BASE_AFTER)):
                if ALIGN_CURRENT_ONLY:
                    dom_pick = b
                else:
                    dom_pick = rng.randint(1, b + 1)
                xt_all = data[dom_pick]
                idx_t = sample_idx(xt_all.shape[0], int(BATCH_PER_DOMAIN), rng)
                xt_mb = torch.as_tensor(xt_all[idx_t], device=device, dtype=torch.float32)

                with torch.no_grad():
                    if teacher is not None:
                        zt_te = teacher.encode(xt_mb, d_cond=torch.full((xt_mb.size(0),), (1 if ADV_BINARY_DOMAIN else int(dom_pick)), device=device, dtype=torch.long),
                                            d_corr=torch.full((xt_mb.size(0),), (int(dom_pick) if (ADV_BINARY_DOMAIN and CORR_USE_FULL_DOMAIN_ID) else (1 if ADV_BINARY_DOMAIN else int(dom_pick))), device=device, dtype=torch.long))
                        logit_te = teacher.predict(zt_te)
                    else:
                        zt_te = model.encode(xt_mb, d_cond=torch.full((xt_mb.size(0),), (1 if ADV_BINARY_DOMAIN else int(dom_pick)), device=device, dtype=torch.long),
                                        d_corr=torch.full((xt_mb.size(0),), (int(dom_pick) if (ADV_BINARY_DOMAIN and CORR_USE_FULL_DOMAIN_ID) else (1 if ADV_BINARY_DOMAIN else int(dom_pick))), device=device, dtype=torch.long))
                        logit_te = model.predict(zt_te)
                    p_te = F.softmax(logit_te, dim=1)
                    conf, yhat = torch.max(p_te, dim=1)
                    mask_pl = conf >= float(PSEUDO_TAU)

                if torch.any(mask_pl):
                    zt = model.encode(xt_mb[mask_pl], d_cond=torch.full((int(mask_pl.sum().item()),), (1 if ADV_BINARY_DOMAIN else int(dom_pick)), device=device, dtype=torch.long),
                                     d_corr=torch.full((int(mask_pl.sum().item()),), (int(dom_pick) if (ADV_BINARY_DOMAIN and CORR_USE_FULL_DOMAIN_ID) else (1 if ADV_BINARY_DOMAIN else int(dom_pick))), device=device, dtype=torch.long))
                    logit_st = model.predict(zt)
                    loss_pseudo = F.cross_entropy(logit_st, yhat[mask_pl])

            # ----- optional pair feature alignment (uses provided Tsrc/Ttar pairs) -----
            loss_pair = torch.tensor(0.0, device=device)
            loss_corr_reg = torch.tensor(0.0, device=device)
            if USE_PAIR_ALIGN and (LAMBDA_PAIR > 0) and (pair_src.shape[0] > 0) and not (FREEZE_BASE_AFTER and int(FREEZE_BASE_AFTER) > 0 and it < int(FREEZE_BASE_AFTER)):
                k = int(min(PAIR_MAX_K, pair_src.shape[0]))
                ip = sample_idx(pair_src.shape[0], k, rng)
                xs_p = torch.as_tensor(pair_src[ip], device=device, dtype=torch.float32)
                xt_p = torch.as_tensor(pair_tar[ip], device=device, dtype=torch.float32)
                ds_cond = torch.as_tensor(pair_src_dom_cond[ip], device=device, dtype=torch.long)
                ds_corr = torch.as_tensor(pair_src_dom_corr[ip], device=device, dtype=torch.long)
                dt_cond = torch.as_tensor(pair_tar_dom_cond[ip], device=device, dtype=torch.long)
                dt_corr = torch.as_tensor(pair_tar_dom_corr[ip], device=device, dtype=torch.long)
                if str(PAIR_ALIGN_ON).lower() == 'base':
                    zps = model.encode_base(xs_p, d=ds_cond)
                    zpt = model.encode_base(xt_p, d=dt_cond)
                else:
                    zps = model.encode(xs_p, d_cond=ds_cond, d_corr=ds_corr)
                    zpt = model.encode(xt_p, d_cond=dt_cond, d_corr=dt_corr)
                loss_pair = torch.mean((zps - zpt) ** 2)
                loss_corr_reg = torch.tensor(0.0, device=device)
                if (LAMBDA_CORR_REG > 0) and (getattr(model, 'corr', None) is not None):
                    dps = model.corr.delta(ds_corr)
                    dpt = model.corr.delta(dt_corr)
                    loss_corr_reg = 0.5 * (torch.mean(dps ** 2) + torch.mean(dpt ** 2))



            loss = loss_cls + float(LAMBDA_ADV) * loss_dom + float(LAMBDA_ENT) * loss_ent \
                   + float(LAMBDA_PSEUDO) * loss_pseudo + float(LAMBDA_PAIR) * loss_pair + float(LAMBDA_CORR_REG) * loss_corr_reg


            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if teacher is not None:
                update_ema(model, teacher, float(EMA_DECAY))

        # Evaluate on batch b
        # Optional DSBN calibration on the *current* target batch (unlabeled): refresh running stats.
        if DSBN_CALIBRATE and getattr(model, "use_dsbn", False):
            dom_eval = (1 if ADV_BINARY_DOMAIN else int(b))
            calibrate_dsbn(model, data[b].astype(np.float32), dom_eval, device=device, batch_size=int(DSBN_CALIBRATE_BATCH))

        acc_b_lr = lr_on_features_acc(model, data, labels, upto_b=b, device=device)
        acc_lr.append(acc_b_lr)

        if REPORT_HEAD_ACC:
            acc_b_head = head_acc(model, data[b], labels[b], device=device)
            acc_head.append(acc_b_head)

        print(f"[{'CDAN' if USE_CONDITIONAL else 'DANN'}] batch {b+1:02d}  acc(LR-on-feat)={acc_b_lr:.4f}")

    avg_lr = float(np.mean(acc_lr)) if acc_lr else float("nan")
    avg_head = float(np.mean(acc_head)) if acc_head else None
    return acc_lr, avg_lr, (acc_head if REPORT_HEAD_ACC else None)


# =============================================================================
# Optional: DCAE reference (import the same eval as your DCAE script)
# =============================================================================

def maybe_run_dcae_reference(data: List[np.ndarray], labels: List[np.ndarray]) -> Optional[Tuple[List[float], float]]:
    if not RUN_DCAE_REFERENCE:
        return None

    # Import your DCAE script functions/config by importing the file as a module.
    # This keeps the exact same reference implementation.
    import importlib.util, pathlib
    src_path = pathlib.Path(__file__).with_name("test_uci_py312_improvement_evaluation.py")
    if not src_path.exists():
        print("[WARN] RUN_DCAE_REFERENCE=True but test_uci_py312_improvement_evaluation.py not found next to this file.")
        return None

    spec = importlib.util.spec_from_file_location("dcae_ref_eval", str(src_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore

    # Reuse its loader to also get tr_smp (it expects it), but we can call its eval_dcae_fixed directly.
    data2, labels2, tr_smp = mod.load_data(mod.MAT_PATH, n_trans_smp=mod.N_TRANS_SMP, data_divider=mod.DATA_DIVIDER, baseline_n=mod.BASELINE_N)

    # Build SAE exactly like reference does
    sae_cfg_ref = mod.SAEConfig(
        n_visible=data2[0].shape[1],
        n_hidden=mod.DCAE_REF_SAE_HIDDEN,
        act_fun_names=("tanh",),
        corruption_levels=(float(mod.DCAE_REF_SAE_CORRUPT),),
        sparsity_reg=0.0,
        l1_reg=0.0,
        l2_reg=0.0,
        tied_weights=True,
        use_biases=True,
    )
    sae_ref = mod.StackedAE(sae_cfg_ref)
    sae_ref.pretrain(
        data2[0],
        max_iter=mod.DCAE_REF_SAE_PRE_ITERS,
        lr=mod.DCAE_REF_SAE_PRE_LR,
        batch_size=mod.DCAE_REF_SAE_PRE_BS,
        show=False,
    )

    dcae_per, dcae_avg = mod.eval_dcae_fixed(sae_ref, data2, labels2, tr_smp)
    return dcae_per, dcae_avg


# =============================================================================
# Main
# =============================================================================

def main():
    set_seed(SEED)

    print("=== UCI Conditional DANN / CDAN baseline ===")
    print("MAT_PATH      :", MAT_PATH)
    print("USE_CONDITIONAL (CDAN):", USE_CONDITIONAL)
    print("DOMAIN_COND_MODE:", DOMAIN_COND_MODE)
    print("COND_MAP      :", COND_MAP)
    print("LAMBDA_ADV    :", LAMBDA_ADV, "warmup=", ADV_WARMUP_FRAC)
    print("LAMBDA_ENT    :", LAMBDA_ENT, "CONF_GATING=", CONF_GATING)
    print("MAX_ITERS     :", MAX_ITERS)
    print("")

    data, labels, tr_smp = load_data(MAT_PATH, n_trans_smp=N_TRANS_SMP, data_divider=DATA_DIVIDER, baseline_n=BASELINE_N)

    raw_per, raw_avg = eval_raw_lr(data, labels)
    print("[RAW]  per-batch:", fmt_acc_list(raw_per))
    print("[RAW]  avg      :", raw_avg)

    dcae_ref = maybe_run_dcae_reference(data, labels)
    if dcae_ref is not None:
        dcae_per, dcae_avg = dcae_ref
        print("[DCAE] per-batch:", fmt_acc_list(dcae_per))
        print("[DCAE] avg      :", dcae_avg)

    acc_per, acc_avg, _ = train_eval_incremental(data, labels, tr_smp, seed=SEED)
    print(f"[{'CDAN' if USE_CONDITIONAL else 'DANN'}] per-batch:", fmt_acc_list(acc_per))
    print(f"[{'CDAN' if USE_CONDITIONAL else 'DANN'}] avg      :", acc_avg)

    print("\n=== FINAL SUMMARY (easy to copy/paste) ===")
    print("[RAW]  per-batch:", fmt_acc_list(raw_per))
    print("[RAW]  avg      :", raw_avg)
    if dcae_ref is not None:
        print("[DCAE] per-batch:", fmt_acc_list(dcae_per))
        print("[DCAE] avg      :", dcae_avg)
    print(f"[{'CDAN' if USE_CONDITIONAL else 'DANN'}] per-batch:", fmt_acc_list(acc_per))
    print(f"[{'CDAN' if USE_CONDITIONAL else 'DANN'}] avg      :", acc_avg)


class DomainFiLM(nn.Module):
    """Input FiLM: x' = (1+gamma(d))*x + beta(d). Identity-initialized.

    Notes:
      - d is an integer domain id tensor (Long).
      - Uses domain IDs only (NO class labels).
      - Initialized to identity (gamma=0, beta=0).
    """
    def __init__(self, n_domains: int, feat_dim: int):
        super().__init__()
        self.n_domains = int(n_domains)
        self.feat_dim = int(feat_dim)
        self.emb = nn.Embedding(self.n_domains, 2 * self.feat_dim)
        nn.init.zeros_(self.emb.weight)

    def forward(self, x: torch.Tensor, d: Optional[torch.Tensor]) -> torch.Tensor:
        if d is None:
            return x
        if d.dim() == 0:
            d = d.expand(x.size(0))
        d = d.long()
        params = self.emb(d)  # [B, 2D]
        gamma, beta = torch.chunk(params, 2, dim=1)
        return x * (1.0 + gamma) + beta


class DomainSpecificBatchNorm1d(nn.Module):
    """Domain-specific BatchNorm1d with per-sample routing.

    Notes:
      - Uses domain IDs only (NO class labels).
      - For mixed-domain minibatches, routes each domain subset through its own BN.
    """
    def __init__(self, num_features: int, n_domains: int, *, freeze_affine: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.n_domains = int(n_domains)
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.num_features) for _ in range(self.n_domains)])
        if bool(freeze_affine):
            for bn in self.bns:
                if bn.affine:
                    bn.weight.requires_grad_(False)
                    bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor, d: Optional[torch.Tensor]) -> torch.Tensor:
        if d is None:
            return self.bns[0](x)
        if d.dim() == 0:
            d = d.expand(x.size(0))
        d = d.long()
        out = torch.empty_like(x)
        for dom in torch.unique(d):
            dom_i = int(dom.item())
            mask = (d == dom)
            n_m = int(mask.sum().item())
            if self.training and n_m < 2:
                # BatchNorm needs >=2 samples per channel when training.
                # For tiny domain groups (e.g., after confidence gating), fall back to running stats.
                bn = self.bns[dom_i]
                out[mask] = F.batch_norm(
                    x[mask], bn.running_mean, bn.running_var, bn.weight, bn.bias,
                    training=False, momentum=0.0, eps=bn.eps
                )
            else:
                out[mask] = self.bns[dom_i](x[mask])
        return out

if __name__ == "__main__":
    main()
