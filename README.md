# confds-enose-domain-adaptation
# Confidence-Aware Multi-Expert Domain Adaptation for E-Nose Sensor Drift

Reference implementation for:

**Progressive Domain Adaptation under Sensor Drift for Gas Classification with Limited Calibration Labels**

Han Fan and Achim J. Lilienthal  
Munich Institute of Robotics and Machine Intelligence (MIRMI)  
Chair of Perception for Intelligent Systems  
Technical University of Munich, Germany

---

## Overview

Electronic noses based on metal-oxide gas sensors are strongly affected by
**sensor drift**. As the sensor response characteristics change over time,
a classifier trained on an initial calibration dataset may experience a
substantial performance degradation when deployed on later measurements.

This repository implements a domain-adaptation framework designed for the
practical setting where:

- abundant **unlabelled target measurements** are available,
- only a very small number of **labelled calibration samples** can be obtained,
- the drift pattern may vary considerably between deployment periods, and
- no single adaptation strategy is consistently reliable under all drift
  conditions.

The central idea is therefore not to rely on one adaptation model. Instead,
we train a **mechanistically diverse pool of domain-adaptation experts** and
combine their predictions using a **confidence-aware probabilistic consensus
model**.

The framework consists of two main stages:

1. **Mechanistically diverse CDAN experts**
2. **Confidence-aware Dawid--Skene fusion (ConfDS)**

The resulting system combines representation-level domain adaptation with
prediction-level reliability estimation.

---

## Motivation

Sensor drift creates a distribution shift between the initial calibration
domain and later measurements:

\[
p_{\mathrm{source}}(x,y)
\neq
p_{\mathrm{target}}(x,y).
\]

A straightforward solution is to periodically retrain the classifier using
new labelled measurements. In real electronic-nose deployments, however,
obtaining labelled calibration data can be expensive and operationally
inconvenient.

Domain adaptation provides an alternative by exploiting the large amount of
unlabelled target data that is naturally collected during deployment.

However, strong distribution alignment is not always beneficial. Depending
on the type and severity of drift, an adaptation strategy may:

- successfully recover class-discriminative structure,
- under-compensate the drift, or
- over-align source and target distributions and cause negative transfer.

This motivates the use of multiple complementary adaptation experts rather
than relying on a single fixed configuration.

---

## Method

### 1. Diverse domain-adaptation experts

Each expert uses the same basic gas-classification architecture but follows a
different adaptation trajectory.

For expert \(m\), the training objective can be written as

\[
\mathcal{L}^{(m)}
=
\mathcal{L}_{\mathrm{cls}}
+
\lambda_{\mathrm{adv}}^{(m)}
\mathcal{L}_{\mathrm{CDAN}}
+
\lambda_{\mathrm{pair}}^{(m)}
\mathcal{L}_{\mathrm{pair}}
+
\lambda_{\mathrm{pst}}^{(m)}
\mathcal{L}_{\mathrm{pst}}.
\]

The individual terms serve different purposes:

- **Source classification loss**
  \(\mathcal{L}_{\mathrm{cls}}\) preserves gas-discriminative information
  learned from the initial calibration domain.

- **Conditional domain-adversarial loss**
  \(\mathcal{L}_{\mathrm{CDAN}}\) reduces the distribution discrepancy between
  the source measurements and the current drift-affected target batch.

- **Paired alignment loss**
  \(\mathcal{L}_{\mathrm{pair}}\) exploits the small number of available
  source--target calibration pairs.

- **Pseudo-label self-training**
  \(\mathcal{L}_{\mathrm{pst}}\) uses high-confidence target predictions to
  strengthen class-specific adaptation.

Rather than selecting one fixed combination of these mechanisms, the
implementation deliberately varies adaptation strengths, confidence
thresholds, conditioning strategies, domain-normalization settings, and
pair-alignment configurations.

This produces a pool of experts with complementary behaviour under different
drift conditions.

---

### 2. Source-retention filtering

Domain adaptation can occasionally improve target alignment at the expense
of destroying useful source-domain decision structure.

To remove clearly degraded experts without using additional target labels,
the implementation evaluates each adapted expert on the labelled source
domain.

Experts whose mean source classification accuracy falls below a predefined
threshold are removed before fusion:

\[
\bar{A}_{\mathrm{src}}^{(m)}
=
\frac{1}{B}
\sum_{b=1}^{B}
A_{\mathrm{src}}^{(m,b)}.
\]

The default implementation uses

```text
source accuracy threshold = 0.95
minimum number of retained experts = 10
