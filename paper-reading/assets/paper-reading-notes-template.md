# {{Paper Title}}

## 精简版

### 一句话结论

用一句不沿用作者包装的话说明：它解决什么问题、核心手段是什么、是否成立。

### 核心方法

1.
2.
3.

### 关键结果

- 只保留 2–4 个非常规、反直觉、揭示边界或影响决策的结果，并附 Table/Figure 定位。
- 不写“更大模型更好、更多数据更好、ours 常规涨点”等预期结果。

### 主要限制

- 写出最影响结论可信度或实际应用的 2–4 项限制。

### Takeaway

> 保留一条可迁移、可验证或可推翻的认知。

---

# 完整版

## Metadata

- Authors / venue / year:
- Paper / code:
- Task and setting:
- Reading date:

## One-sentence Verdict

> 用一句不沿用作者包装的话说明：它解决什么问题、核心手段是什么、是否成立。

## Key Figures

![Figure N: short description](<notes-stem>-assets/figure-N-short-name.png)

**读图：** 说明数据流、变量映射或实验趋势；补充该图能支持什么、不能支持什么。

## Problem and Baseline

- **Problem:**
- **Why it matters:**
- **Baseline pipeline:**
- **Exact delta:**

## Method

### Data flow

1.
2.
3.

### Objective and supervision

- Inputs / outputs:
- Main losses:
- Training-only components:
- Inference-time components:

### Assumptions and inductive biases

- Explicit assumptions:
- Hidden assumptions:
- Possible shortcuts or leakage:

## Evidence Audit

| Claim | Direct evidence | Controls / ablations | Assessment |
|---|---|---|---|
|  |  |  | Supported / partial / unverified |

- **Fairness of baselines:**
- **Gain attribution:** method / data / pretraining / compute / post-processing
- **Cross-domain or robustness evidence:**
- **Missing decisive experiment:**

## 关键与非常规结果

- Counterintuitive result or trade-off:
- Dominant ablation or causal evidence:
- Failure, negative result, or applicability boundary:
- Decision-relevant cost or data-efficiency result:

## Independent Assessment

- **Problem value:**
- **Novelty:** new observation / mechanism / integration / engineering
- **Technical soundness:**
- **Experimental strength:**
- **Practicality:**
- **Likely failure modes:**

## Connections

- Closest prior mechanism:
- Reusable idea for other tasks:
- Relation to existing knowledge:

## Open Questions

1.
2.
3.

## Takeaway

保留一条可供以后复用、且能被实验验证或推翻的认知。
