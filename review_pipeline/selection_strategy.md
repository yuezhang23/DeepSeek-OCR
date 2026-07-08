# Section × Dimension Reading Strategy

Use this table to decide how to process each section of an AI/ML paper before evaluating a review dimension.

> Machine-owned file: `review_pipeline/scorer.py` (`section_control`) parses the table below and
> appends new rows at runtime. The per-dimension judging criteria live in
> `.claude/skills/paper-quality-rubric/SKILL.md` (and `tools._DIMENSION_DESCRIPTIONS`) — not here.
> Column order maps 1:1 onto `tools.DIMENSIONS` via `tools.SELECTION_DIMENSION_COLUMNS`.

## Legend

| Value | Meaning |
|-------|---------|
| `0` | **Raw** — full text required; a summary loses critical detail for this dimension |
| `1` | **Summary** — a summary is sufficient for this dimension |
| `2` | **Omit** — section is of little significance to this dimension |

## Table

| Section | Originality | Research Importance | Claims Supported | Experiment Soundness | Clarity | Value to Community | Prior Work Context |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Abstract | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Introduction | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Related Work | 0 | 1 | 2 | 1 | 1 | 1 | 0 |
| Background / Preliminaries | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Front Matter | 2 | 2 | 2 | 2 | 1 | 2 | 2 |
| 2 FOUNDATIONS: HOW NOISE PRODUCES A NEGATIVE IMPACT | 0 | 1 | 0 | 2 | 0 | 1 | 2 |
| 3 FRAG: FILTERING NOISE USING SNIPPET-LEVEL QUERY RELEVANCE | 0 | 1 | 0 | 1 | 0 | 1 | 2 |
| 4 EXPERIMENTS | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| 5 CONCLUSION | 2 | 1 | 2 | 2 | 1 | 1 | 2 |
| REFERENCES | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| B FRAG DETAILS | 0 | 2 | 0 | 0 | 1 | 1 | 2 |
| C PROOF OF MAIN THEORETICAL ANALYSIS | 0 | 2 | 0 | 2 | 1 | 1 | 2 |
| D ADDITIONAL EXPERIMENT AND DETAILS | 1 | 2 | 0 | 0 | 1 | 0 | 2 |
| E ADDITIONAL DETAILS ON THE ANALYSIS OF THE MAIN RESULTS | 2 | 2 | 0 | 0 | 1 | 1 | 2 |
| F DISCUSSION OF ADDITIONAL LIMITATIONS AND ETHICAL IMPACTS | 2 | 1 | 0 | 1 | 1 | 1 | 2 |
| G DECLARATION OF LLM USAGE | 2 | 2 | 1 | 1 | 2 | 1 | 2 |
| H LICENSE | 2 | 2 | 2 | 2 | 2 | 1 | 2 |
| I PROMPTS AND SAMPLES | 1 | 2 | 0 | 0 | 1 | 0 | 2 |
| 3 METHOD | 0 | 1 | 0 | 0 | 0 | 1 | 2 |
| 6 REPRODUCIBILITY STATEMENT | 2 | 2 | 2 | 1 | 1 | 1 | 2 |
| A APPENDIX | 0 | 2 | 0 | 0 | 1 | 0 | 2 |
| ACKNOWLEDGEMENTS | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| A PROOF OF PROPOSITIONS | 0 | 2 | 0 | 0 | 0 | 0 | 2 |
| B ABLATION STUDY | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| C STATEMENT ON THE USE OF LARGE LANGUAGE MODELS | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 2 PROBLEM FORMULATION AND MOTIVATIONS | 1 | 1 | 2 | 2 | 1 | 1 | 2 |
| 3 PREFERENCE-BASED POLICY OPTIMIZATION | 0 | 1 | 0 | 1 | 1 | 0 | 1 |
| 4 EXPERIMENTS AND EVALUATIONS | 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| 6 DISCUSSION | 2 | 2 | 1 | 1 | 1 | 1 | 2 |
| ACKNOWLEDGMENT | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| A PROOF OF LEMMA 3.1 | 1 | 2 | 0 | 2 | 1 | 1 | 2 |
| B TRAINING CURVES | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| C NOISES IN ACTION-BASED DEGRADATION | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| D ABLATION STUDY OF DEGRADED DATASET SIZE | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| E SENSITIVITY ANALYSIS | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| F COMPARISON AGAINST CPL AND REBRAC ON METAWORLD | 1 | 2 | 1 | 1 | 2 | 1 | 1 |
| G EVALUATION ON ANTMaze ENVIRONMENTS | 1 | 2 | 1 | 1 | 2 | 1 | 1 |
| H EXPERIMENT DETAILS | 2 | 2 | 1 | 0 | 1 | 2 | 2 |
| 2 RELATED WORKS | 1 | 1 | 2 | 2 | 2 | 1 | 0 |
| A THE REASON FOR NORMAL SUPERVISION | 1 | 1 | 0 | 0 | 2 | 1 | 2 |
| B THE REASON FOR SCALE SUPERVISION | 1 | 1 | 0 | 0 | 2 | 1 | 2 |
| C THE USE OF LARGE LANGUAGE MODEL | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 4 MAIN RESULTS | 0 | 0 | 0 | 2 | 0 | 0 | 1 |
| ETHICS STATEMENT | 2 | 2 | 2 | 1 | 1 | 2 | 2 |
| A CONCENTRATION INEQUALITIES | 2 | 2 | 1 | 2 | 1 | 1 | 1 |
| B PROOF SKETCH OF THEOREM 4.1 | 1 | 2 | 1 | 2 | 0 | 1 | 2 |
| D PROOF SKETCH OF THEOREM 4.2 | 1 | 2 | 1 | 2 | 0 | 1 | 2 |
| 2 A DENOISER INDUCED BY MSE TRAINING | 1 | 1 | 2 | 2 | 0 | 1 | 2 |
| 3 DIFFUSION MODELS INDUCED BY THE DENOISER | 1 | 1 | 2 | 2 | 0 | 1 | 2 |
| 4 A DIFFUSION MODEL INDUCED BY THE PENALIZED MAXIMUM LIKELIHOOD | 0 | 1 | 0 | 2 | 0 | 1 | 1 |
| 5 A DIFFUSION MODEL INDUCED BY MSE TRAINING | 1 | 1 | 0 | 2 | 0 | 1 | 2 |
| 6 EXPERIMENT AND DISCUSSION | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| 3 UNCERTAINTY TAGGING | 0 | 1 | 0 | 0 | 2 | 1 | 2 |
| 4 METRICS | 1 | 2 | 0 | 0 | 2 | 2 | 2 |
| 5 METHODS | 0 | 1 | 0 | 0 | 2 | 0 | 2 |
| 6 BLUR-OCR BENCHMARK | 0 | 1 | 0 | 0 | 2 | 0 | 1 |
| 8 CONCLUSION AND FUTURE WORK | 1 | 1 | 2 | 2 | 2 | 1 | 2 |
| ACKNOWLEDGMENTS | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| A USE OF LLMS | 2 | 2 | 1 | 2 | 1 | 2 | 2 |
| B TRAINING HYPERPARAMETERS | 2 | 2 | 2 | 0 | 2 | 1 | 2 |
| C UNC TAGGING EXAMPLES | 1 | 2 | 1 | 1 | 1 | 1 | 2 |
| D PROMPT | 2 | 2 | 2 | 0 | 1 | 1 | 2 |
| E RANDOM-TAG COLD START (FOR EXP2) | 1 | 2 | 0 | 0 | 2 | 1 | 2 |
| F SAMPLE OUTPUT | 2 | 2 | 1 | 1 | 1 | 1 | 2 |
| G DEGRADATION OPERATIONS AND PARAMETERS | 2 | 2 | 2 | 0 | 2 | 1 | 1 |
| 3 PRELIMINARIES AND BASELINES. | 1 | 1 | 2 | 2 | 1 | 1 | 0 |
| 4 OUR METHOD | 0 | 1 | 1 | 2 | 0 | 0 | 2 |
| A IMPLEMENTATION DETAILS | 2 | 2 | 0 | 0 | 1 | 0 | 2 |
| B ADDITIONAL EXPERIMENT RESULTS | 1 | 2 | 0 | 0 | 2 | 1 | 2 |
| C LIMITATION | 2 | 2 | 0 | 0 | 2 | 1 | 2 |
| 2 DATA | 1 | 1 | 0 | 0 | 1 | 1 | 2 |
| 4 RESULTS | 0 | 0 | 0 | 0 | 1 | 0 | 2 |
| 5 CAN LLMs REPLICATE HUMAN CLOSE READING JUDGMENTS OF CREATIVITY? | 1 | 1 | 0 | 0 | 1 | 1 | 2 |
| B LINEAR MODELS | 1 | 2 | 0 | 0 | 2 | 2 | 2 |
| C LLM-JUDGE EVALUATION DETAILS, PROMPTS AND HYPERPARAMETERS | 2 | 2 | 2 | 0 | 2 | 1 | 2 |
| D ADDITIONAL FIGURES | 1 | 2 | 0 | 1 | 1 | 1 | 2 |
