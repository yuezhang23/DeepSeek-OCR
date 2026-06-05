"""Train a classifier on stage-9b dimension scores -> ICLR decisions."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).parent / "out_score"
OUT_JSON = Path(__file__).parent / "out" / "human_results.json"  # optional alt source
META_PATH = Path(__file__).parent / "metadata_300.json"

DIMENSIONS = [
    "Originality",
    "Importance of Research Question",
    "Claims Well Supported",
    "Soundness of Experiments",
    "Clarity of Writing",
    "Value to Research Community",
    "Contextualization Relative to Prior Work",
]

# Maps the snake_case keys in alignment-results JSON (e.g. human_results.json)
# to the human-readable labels used throughout this script.
JSON_KEY_TO_LABEL = {
    "originality": "Originality",
    "importance_of_research_question": "Importance of Research Question",
    "claims_well_supported": "Claims Well Supported",
    "soundness_of_experiments": "Soundness of Experiments",
    "clarity_of_writing": "Clarity of Writing",
    "value_to_research_community": "Value to Research Community",
    "contextualization_relative_to_prior_work": "Contextualization Relative to Prior Work",
}

N_SPLITS = 5
SEED = 42

# Matches rows like: | Originality | 6/10 `...` | rationale |
ROW_RE = re.compile(r"\|\s*([^|]+?)\s*\|\s*(\d+)\s*/\s*10")


def parse_scores(md_path: Path) -> dict[str, int] | None:
    text = md_path.read_text(encoding="utf-8")
    found = {label.strip(): int(score) for label, score in ROW_RE.findall(text)}
    scores = {dim: found.get(dim) for dim in DIMENSIONS}
    return None if any(v is None for v in scores.values()) else scores


def parse_scores_json(json_path: Path) -> dict[str, dict[str, int]]:
    """Load a `{stem: DimensionScores}` JSON file (e.g. ``ai_results.json``,
    ``human_results.json``) and return ``{stem: {dimension_label: score}}`` —
    same shape as `parse_scores` output, just keyed by paper stem first.

    Accepts both JSON schemas emitted by ``scorer.score_paper`` over time:

    - Current (nested):  ``{"originality": {"rationale": "...", "score": 7}, ...}``
    - Legacy   (flat):   ``{"originality": 7, ..., "rationale": {"originality": "...", ...}}``

    The nested form is the one produced by the current ``SCORE_TOOL`` schema,
    which enforces rationale-before-score at the decoder level.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for stem, entry in data.items():
        try:
            scores: dict[str, int] = {}
            for key, label in JSON_KEY_TO_LABEL.items():
                value = entry[key]
                # New nested schema: {"rationale": str, "score": int}.
                # Legacy flat schema: bare integer.
                if isinstance(value, dict):
                    scores[label] = int(value["score"])
                else:
                    scores[label] = int(value)
            out[stem] = scores
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  skipping {stem}: missing/invalid dimension fields ({exc})")
    return out


def load_rows_from_md(md_dir: Path, labels: dict[str, str]) -> list[dict]:
    rows = []
    for md in sorted(md_dir.glob("*.md")):
        if md.stem not in labels:
            continue
        s = parse_scores(md)
        if s is None:
            print(f"  skipping {md.name}: missing dimension rows")
            continue
        rows.append({**s, "decision": labels[md.stem], "paper": md.stem})
    return rows


def load_rows_from_json(json_path: Path, labels: dict[str, str]) -> list[dict]:
    rows = []
    for stem, scores in parse_scores_json(json_path).items():
        if stem not in labels:
            continue
        rows.append({**scores, "decision": labels[stem], "paper": stem})
    print(f"Loaded {len(rows)} papers with labels from JSON {json_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-dir", type=Path, default=OUT_DIR,
        help="Directory of stage-9b *.md score files (default).",
    )
    parser.add_argument(
        "--scores-json", type=Path, default=None,
        help=(
            "Optional JSON scores file (e.g. out/human_results.json). "
            "When set, overrides --scores-dir."
        ),
    )
    parser.add_argument(
        "--train-json", type=Path, default=None,
        help=(
            "JSON scores file used as the training set. When set together "
            "with --test-json, runs a single sequential train/test instead "
            "of K-fold CV."
        ),
    )
    parser.add_argument(
        "--test-json", type=Path, default=None,
        help="JSON scores file used as the held-out test set.",
    )
    args = parser.parse_args()

    if (args.train_json is None) != (args.test_json is None):
        parser.error("--train-json and --test-json must be provided together")

    # 1) Labels keyed by paper stem ("4.pdf" -> "4")
    meta = json.loads(META_PATH.read_text())
    labels = {Path(m["pdf_name"]).stem: m["decision"] for m in meta if m.get("decision")}

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "rforest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEED,
            )),
        ]),
    }

    # Sequential train/test mode: fit on train JSON, evaluate on test JSON.
    if args.train_json is not None:
        train_rows = load_rows_from_json(args.train_json, labels)
        test_rows = load_rows_from_json(args.test_json, labels)
        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)
        print(f"Loaded train features from JSON {args.train_json}")
        print(f"Loaded test  features from JSON {args.test_json}")
        print(f"Train: {len(train_df)} papers. Label distribution:\n{train_df['decision'].value_counts()}\n")
        print(f"Test:  {len(test_df)} papers. Label distribution:\n{test_df['decision'].value_counts()}\n")

        X_train = train_df[DIMENSIONS].values.astype(float)
        y_train = (train_df["decision"] != "Reject").astype(int).values
        X_test = test_df[DIMENSIONS].values.astype(float)
        y_test = (test_df["decision"] != "Reject").astype(int).values
        print(f"Train accept rate: {y_train.mean():.3f} ({y_train.sum()}/{len(y_train)})")
        print(f"Test  accept rate: {y_test.mean():.3f} ({y_test.sum()}/{len(y_test)})\n")

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = (preds == y_test).mean()
            print(f"=== {name} (sequential train/test) ===")
            print(f"test accuracy: {acc:.3f}")
            print("classification report on test set:")
            print(classification_report(y_test, preds, target_names=["Reject", "Accept"]))

        # Feature importance from RF fit on the training set.
        full_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ]).fit(X_train, y_train)
        importances = full_pipe.named_steps["clf"].feature_importances_
        print("Feature importance (random forest, train set):")
        for dim, w in sorted(zip(DIMENSIONS, importances), key=lambda x: -x[1]):
            print(f"  {w:.3f}  {dim}")
        return

    # 2) Pull features from either the JSON file or the *.md directory.
    if args.scores_json is not None:
        rows = load_rows_from_json(args.scores_json, labels)
        source = f"JSON {args.scores_json}"
    else:
        rows = load_rows_from_md(args.scores_dir, labels)
        source = f"MD files in {args.scores_dir}"
    print(f"Loaded features from {source}")


    df = pd.DataFrame(rows)
    print(f"Aligned {len(df)} papers. Label distribution:\n{df['decision'].value_counts()}\n")

    # 3) Binary target: accept (poster or oral) vs reject
    X = df[DIMENSIONS].values.astype(float)
    y = (df["decision"] != "Reject").astype(int).values
    print(f"Accept rate: {y.mean():.3f} ({y.sum()}/{len(y)})\n")

    # 4) Stratified K-fold CV. Scaling is inside the Pipeline so the scaler
    # is fit only on each fold's training portion (no leakage). Class
    # imbalance is handled by `class_weight="balanced"` in the estimators.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for name, model in models.items():
        fold_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        fold_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        preds = cross_val_predict(model, X, y, cv=cv)
        print(f"=== {name} ({N_SPLITS}-fold stratified CV) ===")
        print(f"accuracy per fold: {[f'{s:.3f}' for s in fold_acc]}")
        print(f"mean accuracy:     {fold_acc.mean():.3f} ± {fold_acc.std():.3f}")
        print(f"mean macro-F1:     {fold_f1.mean():.3f} ± {fold_f1.std():.3f}")
        print("classification report on out-of-fold predictions:")
        print(classification_report(y, preds, target_names=["Reject", "Accept"]))

    # 5) Feature importance: fit one RF on all data for a stable ranking.
    full_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED)),
    ]).fit(X, y)
    importances = full_pipe.named_steps["clf"].feature_importances_
    print("Feature importance (random forest, full data):")
    for dim, w in sorted(zip(DIMENSIONS, importances), key=lambda x: -x[1]):
        print(f"  {w:.3f}  {dim}")


if __name__ == "__main__":
    main()
