from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")


def train_model(df: pd.DataFrame, output_dir: Path, model_dir: Path) -> dict:
    target_col = "investment_label"
    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    inverse_target_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full)

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_cols), ("cat", categorical_transformer, categorical_cols)])

    baseline_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
    baseline_model.fit(X_train, y_train)
    val_preds_baseline = baseline_model.predict(X_val)

    X_train_processed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    mi_scores = mutual_info_classif(X_train_processed, y_train, random_state=42)
    f_scores, p_values = f_classif(X_train_processed, y_train)
    tree_model = ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    tree_model.fit(X_train_processed, y_train)

    mi_df = pd.DataFrame({"feature": feature_names, "mutual_info": mi_scores}).sort_values("mutual_info", ascending=False)
    anova_df = pd.DataFrame({"feature": feature_names, "f_score": f_scores, "p_value": p_values}).sort_values("f_score", ascending=False)
    tree_df = pd.DataFrame({"feature": feature_names, "importance": tree_model.feature_importances_}).sort_values("importance", ascending=False)

    candidate_k_values = [k for k in [20, 40, 60, 100] if k < len(feature_names)]
    fs_results = []
    for k in candidate_k_values:
        model = Pipeline(steps=[("preprocessor", preprocessor), ("selector", SelectKBest(score_func=mutual_info_classif, k=k)), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        fs_results.append({"model": f"LogReg_SelectKBest_k{k}", "val_accuracy": accuracy_score(y_val, preds), "val_macro_f1": f1_score(y_val, preds, average="macro")})
    results_df = pd.DataFrame(fs_results).sort_values("val_macro_f1", ascending=False)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    comparison_results = []
    for model_name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        comparison_results.append({"model": model_name, "val_accuracy": accuracy_score(y_val, preds), "val_macro_f1": f1_score(y_val, preds, average="macro")})
    comparison_df = pd.DataFrame(comparison_results).sort_values("val_macro_f1", ascending=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for model_name, clf in {k: models[k] for k in ["LogisticRegression", "ExtraTrees", "RandomForest"]}.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_val_score(pipe, X_train_full, y_train_full, cv=cv, scoring="f1_macro")
        cv_results.append({"model": model_name, "cv_macro_f1_mean": scores.mean(), "cv_macro_f1_std": scores.std()})
    cv_results_df = pd.DataFrame(cv_results).sort_values("cv_macro_f1_mean", ascending=False)

    final_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1))])
    final_model.fit(X_train_full, y_train_full)
    test_preds = final_model.predict(X_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    mi_df.to_csv(output_dir / "mutual_information_ranking.csv", index=False)
    anova_df.to_csv(output_dir / "anova_feature_ranking.csv", index=False)
    tree_df.to_csv(output_dir / "tree_feature_importance.csv", index=False)
    results_df.to_csv(output_dir / "feature_selection_results.csv", index=False)
    comparison_df.to_csv(output_dir / "model_comparison_results.csv", index=False)
    cv_results_df.to_csv(output_dir / "cross_validation_results.csv", index=False)

    cm_test = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=label_encoder.classes_)
    disp.plot()
    plt.tight_layout()
    plt.savefig(output_dir / "test_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    test_results = X_test.reset_index(drop=True).copy()
    test_results["actual_label"] = [inverse_target_mapping[v] for v in y_test]
    test_results["predicted_label"] = [inverse_target_mapping[v] for v in test_preds]
    if hasattr(final_model.named_steps["classifier"], "predict_proba"):
        test_proba = final_model.predict_proba(X_test)
        proba_df = pd.DataFrame(test_proba, columns=[f"prob_{cls}" for cls in label_encoder.classes_])
        test_results = pd.concat([test_results.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    test_results.to_csv(output_dir / "investment_model_test_predictions.csv", index=False)

    joblib.dump(final_model, model_dir / "investment_model_pipeline.joblib")
    joblib.dump(X.columns.tolist(), model_dir / "model_features.joblib")
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")

    summary = {
        "baseline_val_accuracy": float(accuracy_score(y_val, val_preds_baseline)),
        "baseline_val_macro_f1": float(f1_score(y_val, val_preds_baseline, average="macro")),
        "final_test_accuracy": float(accuracy_score(y_test, test_preds)),
        "final_test_macro_f1": float(f1_score(y_test, test_preds, average="macro")),
        "classification_report": classification_report(y_test, test_preds, target_names=label_encoder.classes_, output_dict=True),
    }
    with open(output_dir / "model_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature selection and modeling pipeline")
    parser.add_argument("--input-path", default="data/processed/financial_engine_feature_engineered_model_ready.csv")
    parser.add_argument("--output-dir", default="data/processed/model_outputs")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input_path))
    summary = train_model(df, Path(args.output_dir), Path(args.model_dir))
    print("✅ Modeling pipeline complete")
    print(json.dumps({k: v for k, v in summary.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
