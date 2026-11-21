"""Train, compare and save the ticket-routing model.

Usage:  python src/train.py
Outputs: models/ticket_classifier.joblib, reports/metrics.json,
         reports/model_comparison.csv, reports/figures/confusion_matrix.png
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import clean_text  # noqa: E402

RANDOM_STATE = 42


def main():
    df = pd.read_csv(ROOT / "data" / "tickets.csv").drop_duplicates().dropna()
    df["text"] = df["Document"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["Topic_group"], test_size=0.2, stratify=df["Topic_group"],
        random_state=RANDOM_STATE)

    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)

    candidates = {
        "Complement Naive Bayes": ComplementNB(),
        "Linear SVM": LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(C=5, max_iter=1000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
    }
    rows, fitted = [], {}
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)]).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Macro F1": f1_score(y_test, pred, average="macro"),
                     "Weighted F1": f1_score(y_test, pred, average="weighted")})
        fitted[name] = pipe
        print(rows[-1])
    comp = pd.DataFrame(rows).round(4).set_index("Model")

    # Final model: Logistic Regression (near-identical accuracy to the SVM, but it gives class
    # probabilities, which the app uses for confidence scores and human-triage routing).
    final_name = "Logistic Regression"
    final = fitted[final_name]
    pred = final.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)

    reports = ROOT / "reports"
    comp.to_csv(reports / "model_comparison.csv")
    metrics = {"final_model": final_name,
               "accuracy": round(accuracy_score(y_test, pred), 4),
               "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
               "n_train": len(X_train), "n_test": len(X_test),
               "per_class": {k: {m: round(v, 3) for m, v in d.items()}
                             for k, d in report.items() if isinstance(d, dict)}}
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=45, cmap="Blues",
                                            normalize="true", values_format=".2f", ax=ax,
                                            colorbar=False)
    ax.set_title("Normalised confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(reports / "figures" / "confusion_matrix.png", dpi=130)

    joblib.dump(final, ROOT / "models" / "ticket_classifier.joblib", compress=3)
    print("\nSaved model. Final:", metrics["accuracy"], metrics["macro_f1"])


if __name__ == "__main__":
    main()
