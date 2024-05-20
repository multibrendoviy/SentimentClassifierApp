import os
import json
from ..plotting.charts import plotting_trainer_stats
import requests
import streamlit as st
import torch


def start_training(config: dict, endpoint: object) -> None:

    train_config = config["train"]
    if os.path.exists(train_config["metrics_path"]):
        with open(train_config["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)

    else:
        old_metrics = {"roc_auc": 0, "precision": 0, "recall": 0, "f1": 0, "logloss": 0}

    with st.spinner("Training Bert... :)"):
        st.write(f"Cuda is available: {torch.cuda.is_available()}")
        output = requests.post(endpoint, timeout=8000)
    st.success("Training complete!")

    new_metrics = output.json()["metrics"]

    roc_auc, precision, recall, f1_metric, logloss = st.columns(5)
    roc_auc.metric(
        "ROC-AUC",
        new_metrics["roc_auc"],
        f"{new_metrics['roc_auc'] - old_metrics['roc_auc']:.3f}",
    )
    precision.metric(
        "Precision",
        new_metrics["precision"],
        f"{new_metrics['precision'] - old_metrics['precision']:.3f}",
    )
    recall.metric(
        "Recall",
        new_metrics["recall"],
        f"{new_metrics['recall'] - old_metrics['recall']:.3f}",
    )
    f1_metric.metric(
        "F1 score", new_metrics["f1"], f"{new_metrics['f1'] - old_metrics['f1']:.3f}"
    )
    logloss.metric(
        "Logloss",
        new_metrics["logloss"],
        f"{new_metrics['logloss'] - old_metrics['logloss']:.3f}",
    )

    fig_imp = plotting_trainer_stats(config["train"])

    st.plotly_chart(fig_imp, use_container_width=True)
