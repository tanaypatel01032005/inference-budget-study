# Inference-Time Computation vs. Reasoning Quality: How Many Sampled Generations Are Enough?

## Abstract

This study investigates the relationship between inference-time computation — quantified as the number of independent model generations — and the accuracy of a language model on moderate mathematical reasoning tasks. Using 1,000 multi-step word problems sampled from the GSM8K test benchmark, we generate 50 independent completions per question and evaluate majority-voted predictions across inference budgets of $N \in \{1, 3, 5, 10, 20, 50\}$. The primary goal is to empirically evaluate where diminishing returns occur in inference-time scaling via majority voting. All experimental results are currently pending execution on the university's Sheshnag batch inference platform.

## 1. Introduction

Sampling multiple solutions and applying majority voting (self-consistency) is a proven test-time compute strategy to enhance reasoning accuracy in Large Language Models (LLMs). However, generating multiple full-length reasoning paths scales computational costs linearly with the number of generations $N$. In practical systems, choosing $N$ represents an engineering trade-off between task performance and serving cost.

This study conducts a controlled, reproducible analysis on GSM8K to quantify marginal accuracy gains across standard generation budgets.

## 2. Research Question

> How does increasing the number of independent generations affect the accuracy of a language model on mathematical reasoning, and at what point do additional generations provide diminishing returns?

## 3. Hypothesis

Increasing the number of independent generations will generally improve mathematical reasoning accuracy through majority voting, but the improvement is expected to diminish as the generation budget increases.

This hypothesis reflects standard theoretical and empirical bounds on ensemble/voting performance. Final empirical verification awaits live execution.

## 4. Methodology

### 4.1 Dataset Selection

The experiment uses the **GSM8K** (Grade School Math 8K) test split (`main` configuration). From the 1,319 test problems, exactly 1,000 questions are sampled without replacement using a fixed random seed of `42`. This ensures reproducible, non-cherrypicked problem selection.

### 4.2 Model Infrastructure

Generations are orchestrated through the university's Sheshnag batch inference platform (an OpenAI-compatible batch execution endpoint). The exact model ID will be selected from the live deployment catalogue prior to execution (`GET /v1/models`).

### 4.3 Prompt Design

A single, fixed prompt template is used across all requests:

```
Solve the following mathematical word problem carefully.

Show your reasoning clearly and then provide the final numerical answer.

At the very end of your response, output the final answer in exactly this format:

FINAL_ANSWER: <answer>

Problem:
{question}
```

No prompt variation, hints, few-shot examples, or budget clues are introduced.

### 4.4 Generation Hyperparameters

- **Temperature**: 0.7 (enabling exploration across diverse reasoning trajectories)
- **Top-p**: 1.0
- **Max tokens**: 512
- **Seed**: `null` (stochastic independent completions)

Because the runtime rejects requests with $n > 1$, all 50 generations per question are dispatched as 50 distinct requests ($1,000 \times 50 = 50,000$ total JSONL requests).

### 4.5 Evaluation & Voting Protocol

1. Completed generations are retrieved and parsed.
2. For each question, the 50 generations are deterministically shuffled using seed `42 + question_id`.
3. For each inference budget $N \in \{1, 3, 5, 10, 20, 50\}$, the prefix of $N$ generations is selected.
4. Plurality voting selects the modal answer.
5. Ties are resolved deterministically by choosing the candidate whose first appearance occurs earliest in the sampled sequence.
6. The winning answer is normalized and matched against the normalized GSM8K ground truth (`#### <answer>`).

## 5. Metrics

- **Exact-Answer Accuracy**:
  $$\text{Accuracy}(N) = \frac{1}{1000} \sum_{i=1}^{1000} \mathbb{I}(\hat{y}_{i, N} = y_i^*)$$
- **Marginal Accuracy Gain**:
  $$\Delta(N_k) = \text{Accuracy}(N_k) - \text{Accuracy}(N_{k-1})$$
- **Gain per Generation**:
  $$\text{Efficiency} = \frac{\Delta(N_k)}{N_k - N_{k-1}}$$
- **Confidence Intervals**: 95% non-parametric bootstrap intervals over 10,000 question-level resamples.

## 6. Results

> [RESULT_PENDING] — Awaiting execution on live Sheshnag batch infrastructure.

### 6.1 Accuracy Summary

| Generations ($N$) | Correct | Accuracy (%) | 95% CI Lower | 95% CI Upper |
|---:|---:|---:|:---|:---|
| 1 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 3 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 5 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 10 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 20 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| 50 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

### 6.2 Marginal Accuracy Improvements

| Transition | $\Delta$ Accuracy | Additional Generations | Gain / Generation |
|:---|---:|---:|---:|
| $1 \rightarrow 3$ | [PENDING] | 2 | [PENDING] |
| $3 \rightarrow 5$ | [PENDING] | 2 | [PENDING] |
| $5 \rightarrow 10$ | [PENDING] | 5 | [PENDING] |
| $10 \rightarrow 20$ | [PENDING] | 10 | [PENDING] |
| $20 \rightarrow 50$ | [PENDING] | 30 | [PENDING] |

### 6.3 Visualizations

Visual artifacts will be compiled by `scripts/analyze.py`:
1. `accuracy_vs_generations.png`: Scaling curve with 95% bootstrap error bars.
2. `marginal_gain.png`: Step-wise accuracy deltas illustrating the onset of diminishing returns.
3. `accuracy_vs_budget.png`: Quality versus total generation volume trade-off.

## 7. Discussion

[PENDING_SHESHNAG] — Will be synthesized following empirical data collection.
Key analytical aspects to examine:
- Initial gain magnitude from single sample ($N=1$) to small plurality ensemble ($N=3$).
- The elbow / inflection point where marginal gains fall below $0.5$ percentage points per generation.
- Impact of parser failures / unparseable formats on effective voter pool sizes.

## 8. Limitations

1. **Benchmark Specificity**: Focuses exclusively on GSM8K grade school arithmetic; findings may not translate directly to symbolic mathematics, formal proofs, or open-ended reasoning.
2. **Single Model**: Results reflect the characteristics and calibration of the chosen Sheshnag deployment model.
3. **Black-Box Cost Proxy**: Number of requests $N$ is utilized as compute proxy. If token accounting is enabled by the endpoint, actual FLOPs/token expenditure can be refined.
4. **Outcome-Supervised Only**: Plurality voting operates over final extracted answers, ignoring intermediate reasoning step validity.

## 9. Conclusion

[PENDING_SHESHNAG] — Final conclusions will be formulated strictly from observed outcomes.

## 10. Reproducibility

Complete reproduction instructions, parameter settings, and environment scripts are detailed in `REPRODUCIBILITY.md`.

## References

1. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.
2. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.

