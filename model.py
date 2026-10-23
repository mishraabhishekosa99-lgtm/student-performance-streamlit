from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["age", "studytime", "failures", "absences", "G1", "G2"]
MODEL_FEATURES = ["course", *NUMERIC_FEATURES]


def _make_pipeline(seed: int = 42) -> Pipeline:
    prepare = ColumnTransformer(
        transformers=[
            ("course", OneHotEncoder(handle_unknown="ignore"), ["course"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    return Pipeline([("prepare", prepare), ("classifier", classifier)])


def train_and_evaluate(records: pd.DataFrame, random_state: int = 42) -> dict:
    """Train on 80%, evaluate on a held-out 20%, and generate cross-validated row estimates."""
    data = records.reset_index(drop=True).copy()
    features = data[MODEL_FEATURES]
    target = data["G3"].ge(10).astype(int)
    class_counts = target.value_counts()
    if target.nunique() != 2:
        raise ValueError("The dataset must contain both Pass and Fail outcomes.")
    if int(class_counts.min()) < 3:
        raise ValueError("At least three Pass records and three Fail records are required for the stratified holdout and cross-validation.")

    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state,
        stratify=target,
    )
    evaluation_model = _make_pipeline(random_state)
    evaluation_model.fit(features.iloc[train_indices], target.iloc[train_indices])
    test_features = features.iloc[test_indices]
    test_actual = target.iloc[test_indices].to_numpy()
    test_probability = evaluation_model.predict_proba(test_features)[:, 1]
    test_prediction = (test_probability >= 0.5).astype(int)
    test_confidence = np.where(test_prediction == 1, test_probability, 1 - test_probability)

    metrics = {
        "accuracy": float(accuracy_score(test_actual, test_prediction)),
        "precision": float(precision_score(test_actual, test_prediction, zero_division=0)),
        "recall": float(recall_score(test_actual, test_prediction, zero_division=0)),
        "f1": float(f1_score(test_actual, test_prediction, zero_division=0)),
    }
    matrix = confusion_matrix(test_actual, test_prediction, labels=[0, 1])
    confusion = pd.DataFrame(
        matrix,
        index=pd.Index(["Actual Fail", "Actual Pass"], name="Recorded outcome"),
        columns=pd.Index(["Predicted Fail", "Predicted Pass"], name="Model prediction"),
    )

    fold_count = min(3, int(class_counts.min()))
    if fold_count < 2:
        raise ValueError("At least two records in each outcome class are required for cross-validated row predictions.")
    cross_validation = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_state)
    out_of_fold_probabilities = cross_val_predict(
        _make_pipeline(random_state),
        features,
        target,
        cv=cross_validation,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    out_of_fold_predictions = (out_of_fold_probabilities >= 0.5).astype(int)
    out_of_fold_confidence = np.where(
        out_of_fold_predictions == 1,
        out_of_fold_probabilities,
        1 - out_of_fold_probabilities,
    )

    scored_records = data.copy()
    scored_records["actual_outcome"] = np.where(target.to_numpy() == 1, "Pass", "Fail")
    scored_records["model_prediction"] = np.where(out_of_fold_predictions == 1, "Pass", "Fail")
    scored_records["model_confidence"] = out_of_fold_confidence

    holdout_records = data.iloc[test_indices].copy()
    holdout_records["actual_outcome"] = np.where(test_actual == 1, "Pass", "Fail")
    holdout_records["model_prediction"] = np.where(test_prediction == 1, "Pass", "Fail")
    holdout_records["model_confidence"] = test_confidence
    holdout_records = holdout_records.sort_values("record_id").reset_index(drop=True)

    fitted_classifier = evaluation_model.named_steps["classifier"]
    feature_names = evaluation_model.named_steps["prepare"].get_feature_names_out()
    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": fitted_classifier.feature_importances_}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    feature_importance["Feature"] = feature_importance["Feature"].str.replace("course__course_", "Course: ", regex=False)
    feature_importance["Feature"] = feature_importance["Feature"].str.replace("numeric__", "", regex=False)

    return {
        "records": scored_records,
        "holdout_records": holdout_records,
        "metrics": metrics,
        "confusion_matrix": confusion,
        "feature_importance": feature_importance,
        "train_count": int(len(train_indices)),
        "test_count": int(len(test_indices)),
        "pass_threshold": 10,
        "grade_scale_max": 20,
    }
