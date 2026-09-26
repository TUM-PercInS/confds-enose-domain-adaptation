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

Sensor drift causes the statistical characteristics of gas-sensor responses to change over time. As a result, measurements collected during deployment may no longer follow the same distribution as the data used for the initial classifier training.

Conceptually, this can be expressed as:

```text
p_source(x, y) != p_target(x, y)
```

where `source` denotes the initial calibration domain and `target` denotes a later drift-affected operating condition.

A straightforward solution is to periodically retrain the classifier using newly labelled measurements. In practical electronic-nose deployments, however, obtaining labelled calibration data can be expensive and operationally inconvenient.

Domain adaptation provides an alternative by exploiting the large amount of unlabelled target data that is naturally collected during deployment. However, strong distribution alignment is not always beneficial. Depending on the type and severity of drift, an adaptation strategy may:

- successfully recover class-discriminative structure,
- under-compensate the drift, or
- over-align source and target distributions and cause negative transfer.

This motivates the use of multiple complementary adaptation experts rather than relying on a single fixed adaptation configuration.

---

## Method

The proposed framework consists of three main components:

1. a diverse pool of domain-adaptation experts,
2. source-retention filtering, and
3. confidence-aware probabilistic fusion.

### 1. Diverse domain-adaptation experts

Each expert uses the same basic gas-classification architecture but follows a different adaptation trajectory.

For expert `m`, the training objective is:

```text
L^(m)
=
L_cls
+ lambda_adv^(m)  * L_CDAN
+ lambda_pair^(m) * L_pair
+ lambda_pst^(m)  * L_pst
```

The individual terms serve different purposes:

- **Source classification loss (`L_cls`)** preserves gas-discriminative information learned from the initial calibration domain.
- **Conditional domain-adversarial loss (`L_CDAN`)** reduces the distribution discrepancy between the source measurements and the current drift-affected target batch.
- **Paired alignment loss (`L_pair`)** exploits the small number of available source-target calibration pairs.
- **Pseudo-label self-training loss (`L_pst`)** uses high-confidence target predictions to strengthen class-specific adaptation.

Rather than selecting one fixed combination of these mechanisms, the implementation deliberately varies adaptation strengths, pseudo-label settings, confidence gating, conditioning strategies, domain-normalization settings, and pair-alignment configurations.

This produces a pool of experts with complementary behaviour under different drift conditions.

---

### 2. Source-retention filtering

Domain adaptation can occasionally improve target alignment while degrading useful source-domain decision structure.

To remove clearly degraded experts without consuming additional target labels, each adapted expert is evaluated on the labelled source domain.

For expert `m`, its mean source-domain accuracy across the target adaptation stages is:

```text
A_src_mean^(m)
=
(1 / B) * sum_b A_src^(m,b)
```

where `B` is the number of target adaptation stages.

Experts whose mean source accuracy falls below a predefined threshold are removed before fusion.

The default implementation uses:

```text
source accuracy threshold = 0.95
minimum number of retained experts = 10
```

If fewer than 10 experts satisfy the threshold, the implementation retains the 10 experts with the highest source-domain accuracy.

Because this filtering step uses only labelled source-domain data, it does not consume additional target calibration labels.

---

### 3. Confidence-aware probabilistic fusion

Different experts may be reliable under different drift conditions and for different target samples. A simple majority vote or arithmetic average assumes that all experts have comparable reliability, which is not necessarily valid under domain shift.

ConfDS therefore estimates expert reliability using the limited labelled calibration subset.

For each expert `m`, predictions are divided into two confidence channels according to a confidence threshold `tau`:

```text
high-confidence channel : Theta_(m,high)
low-confidence channel  : Theta_(m,low)
```

Each matrix describes the relationship between the expert prediction and the latent true gas class under the corresponding confidence condition.

For target sample `i` and candidate class `c`, the fusion step can be interpreted as:

```text
q_i(c)
proportional to
pi_c * product_m P(prediction_im | true_class_i = c, confidence_channel_im)
```

where:

- `q_i(c)` is the posterior probability that sample `i` belongs to class `c`,
- `pi_c` is the class prior,
- `prediction_im` is the prediction produced by expert `m`, and
- `confidence_channel_im` indicates whether that prediction belongs to the high- or low-confidence channel.

A Dawid--Skene-style Expectation-Maximization procedure is then used to estimate the latent class labels and expert reliability.

During the **E-step**, the current expert reliability matrices are used to estimate posterior class probabilities for the target samples.

During the **M-step**, the high- and low-confidence confusion matrices of each expert are updated using the current posterior estimates.

The small set of labelled target calibration samples acts as anchors during this iterative estimation.

After convergence, ConfDS produces a fused class-posterior distribution:

```text
q(y | x)
```

and the final prediction is:

```text
predicted class = argmax_y q(y | x)
```

This allows the final decision to depend not only on which class each expert predicts, but also on how reliable that expert has been under the corresponding confidence condition.
minimum number of retained experts = 10
