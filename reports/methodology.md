# Methodology

## Research Design

This study investigates how increasing the number of independent model generations affects mathematical reasoning accuracy when answers are aggregated via majority voting.

## Dataset

- **Source**: GSM8K (Grade School Math 8K)
- **Split**: Test
- **Configuration**: `main`
- **Sample size**: 1,000 questions (randomly selected with seed 42)
- **Selection method**: Random sampling without replacement

## Experimental Procedure

1. **Single generation phase**: Each of the 1,000 questions receives 50 independent model generations, producing 50,000 total responses.
2. **Answer extraction**: The final numerical answer is extracted from each generation using a multi-method parser (`FINAL_ANSWER:`, `####`, `\boxed{}`).
3. **Answer normalization**: Both model and gold answers are normalized (removing commas, dollar signs, trailing units/zeros) for exact mathematical equivalence comparison.
4. **Deterministic shuffling**: The 50 generations per question are deterministically shuffled using a fixed seed to eliminate sequential position bias.
5. **Budget evaluation**: For each inference budget $N \in \{1, 3, 5, 10, 20, 50\}$:
   - The first $N$ responses from the shuffled order are selected.
   - Plurality/majority voting determines the predicted answer.
   - The prediction is compared to the gold answer.

## Majority Voting Rule

- **Rule**: Plurality vote — the answer with the highest frequency wins.
- **Tie-breaking**: Deterministic — on a tie, the candidate answer whose first occurrence appears earliest in the shuffled generation order is selected. No random choices.
- **Unparseable responses**: Excluded from voting candidate sets but tracked in metrics.

## Metrics

- **Primary**: Exact-answer accuracy = $\frac{\text{correct questions}}{1000}$
- **Secondary**: Marginal improvement $\Delta(N_i) = \text{Accuracy}(N_i) - \text{Accuracy}(N_{i-1})$
- **Confidence**: 95% bootstrap confidence intervals (10,000 resamples, question-level)

## Key Design Principles

- The **question** is the experimental unit, not individual generations.
- Temperature is fixed at 0.7 for stochastic exploration across generations.
- No fixed per-generation seed is passed to ensure true stochastic sampling.
- Identical prompt template used across all 50,000 requests.
- All budgets $N$ are computed from the single pool of 50 generations per question.

