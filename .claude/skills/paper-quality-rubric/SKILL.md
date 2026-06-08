---
name: paper-quality-rubric
description: Detailed evaluation criteria and evidence-gathering methodology for scoring a research paper on the 7 ICLR-style quality dimensions (originality, importance, claim support, soundness, clarity, community value, contextualization). Invoke when scoring a paper on any of these dimensions or when justifying a dimension score.
---

# Paper Quality Rubric (7 dimensions)

Use this rubric when assigning an integer 1–10 score to a research paper on any single
quality dimension. **Write the evidence-grounded rationale first, then let the score
follow from it** — never pick a number and rationalize afterward.

## Scoring scale (applies to every dimension)

- **9–10** Paradigm-shifting; flawless execution on this dimension; likely an Oral.
- **7–8**  Strong on this dimension; well-supported; a clear "Accept".
- **5–6**  Interesting but flawed or limited on this dimension; "Borderline".
- **3–4**  Significant problems on this dimension (errors, opacity, negligible novelty).
- **1–2**  Fundamentally deficient on this dimension (incorrect, plagiarized, out of scope).

## Evidence rules

- Ground every judgment in concrete elements of the paper: specific sections, figures,
  tables, equations, claims, datasets, or baselines. Cite them in the rationale.
- You MAY use web search/fetch to verify prior art, novelty, or context.
- You MUST NOT consult this paper's own peer reviews, ratings, rebuttals, or
  accept/reject decision — especially anything hosted on OpenReview. Score only from the
  paper's own content plus general domain knowledge.
- Discuss failure cases and limitations with the same rigor as the strengths.

## Per-dimension criteria

### 1. Originality (`originality`)
- **Gap analysis:** a fundamentally new mechanism (objective, architecture, optimization)
  vs. a "delta" improvement?
- **Creative synthesis:** does it bridge disparate fields in a non-obvious way?
- **Path-clearing:** does it challenge established "folk wisdom" with a fresh perspective?

### 2. Importance of Research Question (`importance_of_research_question`)
- **Problem criticality:** does it address a real bottleneck (latency, data efficiency,
  alignment, etc.)?
- **Breadth of utility:** niche, or generalizable across modalities/tasks?
- **Field evolution:** if the claims hold, would it change how others approach their work?

### 3. Claims Well Supported (`claims_well_supported`)
- **Claim–evidence alignment:** is each central claim directly backed by the presented
  data/proofs (e.g. an efficiency claim backed by a FLOPs-vs-accuracy curve)?
- **Theoretical rigor:** realistic assumptions; intuition backed by analysis.
- **Transparency:** are limitations and failure cases honestly acknowledged?

### 4. Soundness of Experiments (`soundness_of_experiments`)
- **Baseline integrity:** strong, properly-tuned baselines (no straw men)?
- **Statistical significance:** error bars, multiple seeds, sensitivity/ablation studies?
- **Scaling consistency:** does the advantage hold as model/data size grows?

### 5. Clarity of Writing (`clarity_of_writing`)
- **Structure & flow:** can a reader grasp the contribution from abstract + Fig. 1 +
  conclusion?
- **Formalism:** standard, precise, consistent notation.
- **Visual communication:** informative figures/tables with clear axes, labels, captions.

### 6. Value to Research Community (`value_to_research_community`)
- **Resource contribution:** a new dataset, benchmark, or robust codebase?
- **Heuristic value:** transferable "lessons learned" or useful negative results?
- **Ethics & safety:** proactively addresses dual-use concerns or biases?

### 7. Contextualization Relative to Prior Work (`contextualization_relative_to_prior_work`)
- **Historical accuracy:** correctly attributes ideas to original sources, not just the
  most recent famous paper.
- **Critical comparison:** explains how the work differs conceptually, not just a list of
  20 citations.
- **Fairness:** acknowledges contemporaneous work and compares neutrally.
