# OMRQ Project Description

## Title
**Oracle-Mediated Red Queen: A Triadic Framework for Co-Evolving Anomaly Detection**


> **SIAM SDM 2026** | [Paper](#) | [Code](https://github.com/apt-rgap/rgap)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview
OMRQ is a research project on anomaly detection in adversarial cybersecurity settings. The core idea is that anomaly detection should not be treated as a static classification problem, because real attackers adapt over time, learn defender blind spots, and exploit limited analyst feedback. Instead, OMRQ models detection as a repeated **triadic game** involving three entities:

- a **defender** that ranks suspicious instances for inspection,
- an **attacker** that generates evasive yet plausible anomalies,
- and a **budgeted oracle** that mediates all access to ground truth through asymmetric feedback.

This oracle-mediated asymmetry is central to the framework. The defender receives exact labels only for inspected top-k alerts, while the attacker receives only compressed summaries of missed positives. This creates a realistic **Red Queen loop** in which both sides continuously adapt under limited information.

## Motivation
Most anomaly detection systems in cybersecurity are trained once and evaluated on fixed test distributions. This setup is mismatched with modern cyber threats such as Advanced Persistent Threats (APTs), which are stealthy, adaptive, and multi-stage. In practice:

- adversaries modify behavior to evade detection,
- defenders receive sparse and delayed analyst feedback,
- and robustness depends on sustained performance over time, not just one-shot accuracy.

OMRQ addresses this mismatch by shifting the problem from static separability to **dynamic resilience under adversarial pressure**.

## Technical Contributions
The project makes five main contributions:

1. **Triadic formalization**  
   Anomaly detection is cast as a repeated game under partial observability, with a budgeted oracle enforcing asymmetric feedback.

2. **Co-evolutionary learning loop**  
   The defender updates via pairwise ranking and similarity-aware reranking, while the attacker generates evasive anomalies guided by oracle summaries.

3. **Trajectory-based evaluation**  
   Instead of relying only on snapshot metrics such as AUROC, the project emphasizes metrics such as **AUC-nDCG**, **Min-nDCG**, and **temporal instability**, which better capture collapse resistance and sustained ranking quality.

4. **Analytical insight**  
   The project studies simplified conditions under which oracle feedback improves local defender ranking quality, and how oracle compression affects attacker adaptation.

5. **Diffusion-based hardening**  
   Among the attacker backbones considered, diffusion provides the most stable and effective adversarial training regime, outperforming GAN-based co-evolution in robustness and stability.

## Method Summary
At each round:

1. The oracle samples an unlabeled batch.
2. The attacker appends synthetic evasive anomalies.
3. The defender ranks the combined pool.
4. The oracle reveals labels only for the top-k inspected items.
5. The defender updates from these labels.
6. The attacker receives a compressed summary of missed positives and adapts.

This repeated loop creates a controlled adversarial curriculum for stress-testing and hardening the defender.

## Datasets
The project is evaluated on heterogeneous cybersecurity benchmarks:

- **DARPA Transparent Computing (TC)**: process-level traces across Android, BSD, Linux, and Windows.
- **NSL-KDD**: KDD-family intrusion detection benchmark, focusing on Probe and U2R attacks.
- **UNSW-NB15**: a more modern network intrusion benchmark with diverse network-flow features.

Together, these datasets test whether OMRQ remains effective across both process-centric and network-centric cyber modalities.

## Main Findings
The experiments show that:

- static defenders often collapse under repeated adversarial pressure,
- feedback-only methods improve somewhat but remain unstable,
- two-player full-information games can induce stalemate or exploitation cycles,
- OMRQ improves **trajectory-level robustness**,
- and **OMRQ-Diffusion** is the strongest variant, achieving the best sustained ranking quality, strongest worst-case behavior, and lowest instability across the main benchmarks.

## Broader Significance
OMRQ suggests a broader shift in how cyber anomaly detection should be designed and evaluated. The key question is no longer only whether a detector performs well on a fixed test set, but whether it can:

- maintain ranking quality over time,
- resist adaptive evasion,
- use limited analyst feedback effectively,
- and avoid catastrophic collapse under evolving attack pressure.

In that sense, OMRQ is both a detection framework and an evaluation framework for **dynamic adversarial robustness**.


## Future Directions
Promising next steps include:

- graph-native or provenance-aware attack generators,
- noisy or delayed oracle models,
- adaptive oracle budgeting,
- streaming or large-scale approximations,
- and deeper theoretical treatment of non-stationary triadic co-evolution.
