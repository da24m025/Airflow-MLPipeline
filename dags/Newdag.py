from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    f1_score, make_scorer
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


default_args = {
    "owner": "bhuvan",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

RAW_CSV       = "/opt/airflow/project/data/raw/creditcard.csv"
PROCESSED_DIR = "/opt/airflow/project/data/processed/"
EXPERIMENT    = "FraudDetection"
TEST_SIZE     = 0.3
RANDOM_STATE  = 42


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def preprocess_data():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("CreditRiskPreprocess")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read
             .option("header", True)
             .option("inferSchema", True)
             .csv(RAW_CSV)
             .na.drop()
    )
    features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df_flat = df.select(*features, "Class")

    ensure_dirs()
    df_flat.coalesce(1).write.mode("overwrite").parquet(PROCESSED_DIR)
    spark.stop()


def train_model_advanced():
    # 1) Load & split
    df = pd.read_parquet(PROCESSED_DIR)
    X = df.drop(columns="Class")
    y = df["Class"].astype(int)

    # Visualize & log class distribution
    counts = y.value_counts().sort_index()
    fig_dist, ax_dist = plt.subplots()
    ax_dist.bar(counts.index.astype(str), counts.values)
    ax_dist.set(title="Class Distribution", xlabel="Class", ylabel="Count")
    tmp_dist = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig_dist.savefig(tmp_dist.name, bbox_inches="tight")
    plt.close(fig_dist)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        # log distribution artifact
        mlflow.log_artifact(tmp_dist.name, artifact_path="data_distribution")
        os.remove(tmp_dist.name)

        # log sample input data
        sample_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df.head(10).to_csv(sample_csv.name, index=False)
        mlflow.log_artifact(sample_csv.name, artifact_path="input_examples")
        os.remove(sample_csv.name)

        # proceed with train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # 2) CV pipeline + param grid
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", IsolationForest(random_state=RANDOM_STATE))
        ])
        param_dist = {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_samples": ["auto", 0.8, 0.5],
            "clf__max_features": [1.0, 0.8, 0.5],
            "clf__contamination": [0.001, 0.005, 0.01],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

        def f1_binary_scorer(y_true, y_pred):
            y_bin = np.where(y_pred == -1, 1, 0)
            return f1_score(y_true, y_bin, zero_division=0)
        scorer = make_scorer(f1_binary_scorer)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=20,
            scoring=scorer,
            error_score=0,
            cv=cv,
            n_jobs=-1,
            verbose=2,
            random_state=RANDOM_STATE,
            return_train_score=True
        )

        # 3) Fit and log metrics
        mlflow.log_metric("train_positive_ratio", y.mean())
        search.fit(X_train, y_train)

        # log CV results CSV
        cv_df = pd.DataFrame(search.cv_results_)
        tmpcv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        cv_df.to_csv(tmpcv.name, index=False)
        mlflow.log_artifact(tmpcv.name, artifact_path="cv_results")
        os.remove(tmpcv.name)

        # 4) Evaluate best model
        best = search.best_estimator_
        raw_scores = best.named_steps["clf"].decision_function(X_test)
        preds = best.named_steps["clf"].predict(X_test)
        y_pred = np.where(preds == -1, 1, 0)

        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "roc_auc": roc_auc_score(y_test, -raw_scores),
        })

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4,4))
        ax_cm.matshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm): ax_cm.text(j, i, v, ha="center")
        ax_cm.set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")
        mlflow.log_figure(fig_cm, "confusion_matrix/cm.png")
        plt.close(fig_cm)

        # ROC plot
        fpr, tpr, _ = roc_curve(y_test, -raw_scores)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr)
        ax_roc.set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
        mlflow.log_figure(fig_roc, "roc_curve/roc.png")
        plt.close(fig_roc)

        # final model
        signature = infer_signature(X_test, raw_scores.reshape(-1,1))
        input_example = X_test.head(5)
        mlflow.sklearn.log_model(
            sk_model=best,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        print(f" Best params: {search.best_params_}")

with DAG(
    dag_id="credit_fraud_etl_advanced_visual",
    default_args=default_args,
    start_date=datetime(2025,4,1),
    schedule="@daily",
    catchup=False,
    tags=["etl","ml","visual"]
) as dag:

    t1 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t2 = PythonOperator(
        task_id="train_model_advanced",
        python_callable=train_model_advanced,
    )

    t1 >> t2
