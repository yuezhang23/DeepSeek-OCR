# `scorer.py` — Stage 9b: Dimensional Quality Scoring

> Implementation notes for `review_pipeline/scorer.py`. This is the **scoring** branch of
> Stage 9 (the prose-review branch is `reviewer.py`). It assigns an integer **1–10** score
> to a paper on each of the **7 quality dimensions**, with a short evidence-grounded
> rationale per dimension.

---

## 1. What makes this stage different

Every *other* pipeline stage runs on the cheap DeepSeek/OpenAI-compatible `vendor` through
`tools.py` function-calling. **Stage 9b does not.** Scoring is a **multi-agent orchestration
on the Claude Agent SDK** (`claude_agent_sdk`):

- **7 dimension agents** (default `claude-sonnet-4-6`) run **concurrently**, one per
  dimension, each fully autonomous: it reads its tailored slice of the paper, invokes the
  `paper-quality-rubric` skill, may web-search for prior art, and submits one
  `{rationale, score}` via an in-process MCP tool.

Their 7 `{rationale, score}` results are assembled directly into the final
`DimensionScores` — there is no separate judge/reconciliation agent.

The DeepSeek `vendor` *is* still used here, but only for two **cheap helper** roles:
per-section summarization (done upstream in `paper_prep.py`) and the **unknown-section
importance classification** inside `section_control` (below). Claude is reserved for the
reasoning-heavy scoring + judging.

```
paper_sections (dict)                          summaries (dict)
        │                                            │
        ▼                                            │
  section_control(vendor)  ──►  selections           │
        │  (per section × per dimension: 0/1/2)      │
        ▼                                            ▼
  ┌────────────────────────────────────────────────────────- ─┐
  │  _score_paper_async                                       │
  │   per dim: _build_dimension_text → _run_dimension (Claude)│  ← concurrent, sem-bounded
  │   then:    _assemble(dim_results)                         │
  └─────────────────────────────────────────────────────────--┘
        │
        ▼
   DimensionScores  {dim: {rationale, score 1-10}}
```

---

## 2. Public API & contract

```python
def score_paper(paper_sections: dict, summaries: dict, vendor: Any = None) -> DimensionScores
```

- **`paper_sections`** — the prep-stage (`paper_prep.build_paper_sections`) output:
  `{"sections": [{"title", "raw", "summary"}, ...]}`. **Not** raw markdown — that change is
  intentional (the prep stage was moved out of this file). Every consumer guards
  `isinstance(paper_sections, dict)`, so passing a raw string degrades to *empty sections*
  (and empty papers scored) rather than crashing — watch for that in callers.
- **`summaries`** — related-work summaries (Stage 8 output); optional, may be `{}`.
- **`vendor`** — the DeepSeek/OpenAI-compatible client, used **only** by `section_control`
  for unknown-section importance. Pass `None` to skip the LLM fallback (misses default to
  *summary*).

Returns `DimensionScores`: a dict keyed by the 7 `DIMENSIONS` (from `tools.py`, the single
source of truth), each value `{"rationale": str, "score": int 1-10}`.

`score_paper` is synchronous: it runs `section_control` (sync), then
`asyncio.run(_score_paper_async(...))`.

---

## 3. `section_control` — per-section, per-dimension reading strategy

The heart of the dimension-aware routing. For each section it returns a choice vector
`{dim: 0|1|2}`:

| value | meaning | fed to the dimension agent as |
|:-----:|---------|-------------------------------|
| `0` | **Raw** — full detail required | the section's `raw` text |
| `1` | **Summary** — a summary preserves the key info | the section's `summary` |
| `2` | **Omit** — little significance to this dimension | nothing (skipped) |

### Resolution order (per section)
1. **Normalize** the title (`_normalize_section`): lowercase, strip leading numbering
   (`1`, `3.2`, `A.`, roman numerals via `_LEADING_LABEL_RE`), drop markdown, collapse
   whitespace. `"A RELATED WORK"` → `"related work"`.
2. **Direct table hit** — look the normalized name up in the matrix parsed from
   `selection_strategy.md` (`_load_selection_table`).
3. **Alias hit** — `_ALIASES` maps common headings (`intro`→`Introduction`,
   `setup`→`Experiments / Setup`, …) onto canonical rows, so obvious sections never cost
   an LLM call.
4. **Miss → LLM** — all unresolved sections are batched into **one**
   `_query_section_importance` call (the DeepSeek `vendor`, `SECTION_IMPORTANCE_TOOL`),
   which returns a 0/1/2 vector per section.
5. **Default** — if there's no vendor or the call fails, misses default to all-`1`
   (summary).

### The living table
`selection_strategy.md` is a **machine-owned, dynamically-grown** file:

- `_load_selection_table()` parses its markdown table. The header row is found by locating a
  pipe-row whose first cell is `section` with ≥ `len(DIMENSIONS)` columns; the column order
  maps 1:1 onto `DIMENSIONS` via `tools.SELECTION_DIMENSION_COLUMNS`.
- Newly LLM-classified sections are **appended back** to the table by `_append_table_rows`,
  so the next paper with that section name is a free table hit.
- Writes are serialized with a module-level `threading.Lock` (`_TABLE_LOCK`) and the table
  is **re-read inside the lock** for dedup — safe under batch (multi-paper) parallelism;
  duplicate rows are never written.

### `_query_section_importance` — DeepSeek quirks baked in
- **No forced `tool_choice`.** A forced `tool_choice={"type":"function",...}` returns a 400
  under DeepSeek's *thinking* mode, so we keep the default `"auto"`.
- **`thinking=False` + `max_tokens=4096`.** This is bulk classification; with thinking on,
  the reasoning tokens exhaust the budget before the tool call is ever emitted (observed as
  "no tool call after 3 retries"). Disabling thinking fixes it.
- Result extracted via `clients.get_tool_call(resp).function.arguments`.

---

## 4. Building each agent's input

- **`_build_dimension_text(paper_sections, selections, dim)`** — concatenates, in document
  order, each section's `raw` (0) / `summary` (1) / `""` (2), each under a `## <title>`
  heading. Produces a **different** paper text for each dimension.
- **`_build_related_work_context(summaries)`** — formats Stage 8 summaries into a
  `=== RELATED WORK SUMMARIES ===` block appended to each dimension prompt.

---

## 5. The agents

### Dimension agent — `_run_dimension(dim, paper, related, sem)`
- Options (`_dimension_options`): `model=_DIM_MODEL`, `cwd=_REPO_ROOT`,
  `setting_sources=["project"]` + `skills="all"` (discovers
  `.claude/skills/paper-quality-rubric`), MCP `score` server, and an allow-list of
  `WebSearch`, `WebFetch`, `Skill`, `TodoWrite`, `mcp__score__submit_score`.
- System prompt `_DIMENSION_SYSTEM` pins it to **exactly one** dimension and its
  description (`tools._DIMENSION_DESCRIPTIONS[dim]`); instructs *rationale first, score
  second*; forbids consulting the paper's own peer reviews; ends by calling `submit_score`.
- Concurrency is bounded by an `asyncio.Semaphore(_AGENT_CONCURRENCY)`.
- **Resilience:** the whole `query` loop is wrapped in try/except so one dimension can't
  crash the batch. If the agent forgets `submit_score`, `_maybe_recover_dim` tries to parse
  `{rationale, score}` from its final text. A still-missing score defaults to `5`.

### Result assembly — `_assemble(dim_results)`
- Builds the final `DimensionScores` directly from the 7 per-dimension agent results,
  ensuring all 7 keys are present with valid 1–10 scores (missing/malformed entries default
  to `5`). There is no separate judge/reconciliation agent.

### MCP submit tool (strict structured output)
- `_make_dimension_recorder` → `submit_score(rationale: str, score: int)`, score clamped to
  1–10, stashed in a per-run `holder` dict.
- A fresh per-run `create_sdk_mcp_server`, so concurrent agents don't share state.

---

## 6. Security gate — `_gate_tool` (`can_use_tool`)

A `PreToolUse`-style permission callback applied to every agent:

- **`AskUserQuestion`** → **denied** with a message telling the agent to resolve the
  ambiguity itself (autonomous batch mode has no human), state the assumption, and continue.
- **Filesystem/shell tools** (`_FS_TOOLS`: Read/Write/Edit/MultiEdit/NotebookEdit/Bash/
  Grep/Glob) → **hard-denied.** The repo contains the ground-truth human reviews
  (`metadata_300.json`); agents must never read them.
- Everything else (web/skill/submit) → allowed.

This pairs with the rubric's instruction (and the agents' own prompts) never to consult
**OpenReview** — i.e. the paper's actual peer reviews. (The streaming-input requirement of
`can_use_tool` is why prompts are delivered through `_single_user_msg`, an async generator,
rather than a plain string.)

---

## 7. Configuration (env)

| Env var | Default | Controls |
|---------|---------|----------|
| `SCORE_DIMENSION_MODEL` | `claude-sonnet-4-6` | the 7 dimension agents |
| `SCORE_AGENT_CONCURRENCY` | `4` | max dimension agents in flight |

Auth: `ANTHROPIC_API_KEY` (Claude agents) + the DeepSeek key (for the `vendor` helper).

---

## 8. Running standalone

```bash
python -m review_pipeline.scorer review_pipeline/505.md
```

`__main__` reads the markdown, builds a default `LLMVendor`, runs
`paper_prep.build_paper_sections` to produce the section JSON, then `score_paper`, and
prints the resulting `DimensionScores` as JSON. Via the full pipeline it's reached with
`python -m review_pipeline.main --markdown <file> --score-dimensions` (which caches the
`paper_sections` prep stage first).

---

## 9. Key collaborators

| Module | Role |
|--------|------|
| `paper_prep.py` | Stage 9-prep: splits markdown into major sections + per-section summaries → `paper_sections`. |
| `tools.py` | `DIMENSIONS` (single source of truth), `SELECTION_DIMENSION_COLUMNS`, `SECTION_IMPORTANCE_TOOL`, `_DIMENSION_DESCRIPTIONS`. |
| `selection_strategy.md` | The living section × dimension matrix parsed/grown at runtime. |
| `.claude/skills/paper-quality-rubric/SKILL.md` | The 1–10 rubric + per-dimension criteria each agent applies. |
| `clients.py` | `get_tool_call` (extract DeepSeek tool call), `LLMVendor` (standalone vendor). |
| `main.py` | Wires Stage 9-prep + `score_paper` into the pipeline. |
