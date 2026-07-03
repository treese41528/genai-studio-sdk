---
name: hypothesis-testing
description: Run a statistical hypothesis test correctly and report it honestly. Use when asked whether a difference/association is "significant", to compare groups, or to test a claim about data. Picks the right test, CHECKS its assumptions, uses the hypothesis_test tool (never a p-value from memory), and reports effect size + the honest conclusion.
---

# Hypothesis testing (done right)

Use `hypothesis_test` (or `python_exec` with scipy) for EVERY test — never state a p-value or a verdict
from your head. A test answers a *precise* question; get the question and the assumptions right first.

## The workflow
1. **State H0 / H1 in words** and the unit of analysis. What exactly is being compared, and against what?
2. **Pick the test** by data type + design:
   | question | test |
   |---|---|
   | two group means, numeric | independent t-test (Welch if variances differ) |
   | paired/repeated numeric | paired t-test |
   | 3+ group means | one-way ANOVA (then post-hoc) |
   | association of two categoricals | chi-square |
   | linear association of two numerics | Pearson (or Spearman if non-linear/ordinal) |
3. **CHECK assumptions BEFORE trusting the p-value** — independence, approximate normality (or n large
   enough for CLT; else a non-parametric test: Mann–Whitney / Wilcoxon / Kruskal–Wallis), and roughly
   equal variances (else Welch). Report which held and which didn't.
4. **Run it** with the tool; read the statistic, p-value, and df.
5. **Report effect size, not just significance** — a mean difference / Cohen's d / correlation r / odds
   ratio. A tiny effect can be "significant" with big n and still not matter.
6. **Conclude honestly** — reject or fail-to-reject H0 at the stated alpha; "fail to reject" is NOT
   "proven equal". State the practical meaning and the caveats.

## Rigor rules (the traps)
- **Multiple comparisons**: testing many things inflates false positives — correct (Bonferroni /
  Benjamini–Hochberg) and say you did.
- **No p-hacking**: don't try tests until one is significant; fix the test before looking.
- **Underpowered**: with tiny n, a non-significant result is inconclusive, not evidence of no effect.
- **Significance != importance != causation.** Observational data supports association, not cause.
- If assumptions fail and no valid test applies, **say the data can't answer it** rather than forcing one.

## Example
```python
from scipy import stats
a = df[df.group=="A"].value; b = df[df.group=="B"].value
stats.levene(a, b)                      # equal-variance check -> pick Welch if p<.05
stats.ttest_ind(a, b, equal_var=False) # Welch t-test
(a.mean()-b.mean()), (a.mean()-b.mean())/df.value.std()   # raw diff + Cohen's-d-ish effect size
```
Report: test chosen + why, assumptions, statistic/p, effect size, and the plain-language conclusion.
