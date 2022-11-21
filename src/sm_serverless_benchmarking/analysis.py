import sys
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sm_serverless_benchmarking.cost_constants import (INFERENCE_COST,
                                                       INSTANCE_MAPPING,
                                                       MONTHLY_INSTANCE_COST,
                                                       PROCESSING_COST)


def iqr(ps: pd.Series) -> float:
    p25 = ps.quantile(0.25)
    p75 = ps.quantile(0.75)
    iqr = p75 - p25

    return iqr


def convert_units(ps: pd.Series) -> pd.Series:

    aggregates = ["Average", "Minimum", "Maximum", "p25", "p50", "p75"]
    if ps["Unit"] == "Microseconds":
        ps[aggregates] = ps[aggregates] / 1000
        ps["Unit"] = "Milliseconds"
        return ps
    else:
        return ps


def compute_tps(df: pd.DataFrame) -> pd.DataFrame:

    invocation_end_time_counts = df["end_time"].astype(int).value_counts()

    tps_metrics = (
        invocation_end_time_counts.describe().drop(["count", "std", "min"]).astype(int)
    )
    tps_metrics.rename(
        {
            "25%": "tps_p25",
            "50%": "tps_p50",
            "75%": "tps_p75",
            "max": "tps_max",
            "mean": "tps_avg",
        },
        inplace=True,
    )

    return tps_metrics


def summarize_stability_results(
    df_benchmark_results: pd.DataFrame,
    df_endpoint_metrics: pd.DataFrame,
    result_save_path: str = ".",
) -> Tuple[pd.DataFrame, pd.DataFrame, matplotlib.figure.Figure]:

    save_path = Path(result_save_path) / "stability_benchmark_summary_results"
    save_path.mkdir(exist_ok=True, parents=True)

    df_benchmark_results_success = df_benchmark_results.query(
        "(invocation_latency > 0) & (response_size > 0)"
    )

    df_benchmark_summary = df_benchmark_results_success.groupby("memory_size").agg(
        {
            "invocation_latency": ["count", "min", "mean", "median", "max", iqr],
            # "throttle_exception":["sum"],
            # "insufficient_memory_error":["sum"],
            # "other_model_error":["sum"]
        }
    )

    df_benchmark_summary.columns = [
        f"{x[0]}_{x[1]}" for x in df_benchmark_summary.columns.to_flat_index()
    ]
    df_benchmark_summary.rename(
        columns={"invocation_latency_count": "successful_invocations"}, inplace=True
    )

    df_endpoint_metrics = df_endpoint_metrics.apply(convert_units, axis=1)
    df_metric_summary = df_endpoint_metrics.pivot(
        index="memory_size", columns="metric_name", values="Average"
    ).dropna(thresh=2)

    df_benchmark_summary.to_csv(
        save_path / "invocation_benchmark_summary.csv", index=True
    )
    df_metric_summary.to_csv(save_path / "endpoint_metrics_summary.csv", index=True)

    latency_thresholds = df_benchmark_summary.eval(
        """low = invocation_latency_mean - 5 *invocation_latency_iqr
                                                 high = invocation_latency_mean + 5 *invocation_latency_iqr
                                                """
    )[["low", "high"]]

    df_benchmark_results_no_outliers = df_benchmark_results_success.merge(
        latency_thresholds, on="memory_size"
    ).query("(invocation_latency >= 0) & (invocation_latency <= high)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        data=df_benchmark_results_no_outliers,
        x="invocation_latency",
        hue="memory_size",
        palette="tab10",
        ax=ax,
    ).set_title("Invocation Latency (ms)")

    fig.savefig(save_path / "stability_benchmark_distribution.png")

    return df_benchmark_results_success, df_benchmark_summary, df_metric_summary, fig


def summarize_concurrency_results(
    df_benchmark_results: pd.DataFrame,
    df_endpoint_metrics: pd.DataFrame,
    result_save_path: str = ".",
) -> Tuple[pd.DataFrame, matplotlib.figure.Figure]:

    save_path = Path(result_save_path) / "concurrency_benchmark_summary_results"
    save_path.mkdir(exist_ok=True, parents=True)

    df_endpoint_metrics = df_endpoint_metrics.apply(convert_units, axis=1)
    df_metric_summary = pd.pivot_table(
        df_endpoint_metrics,
        index=["memory_size", "max_concurrency"],
        columns="metric_name",
        values="Average",
    ).dropna(thresh=2)

    df_outlier_thresh = df_benchmark_results.groupby(
        ["max_concurrency", "num_clients", "memory_size"], as_index=False
    ).agg({"invocation_latency": ["mean", iqr]})
    df_outlier_thresh.columns = [
        f"{x[0]}_{x[1]}".strip("_") for x in df_outlier_thresh.columns.to_flat_index()
    ]
    df_outlier_thresh = df_outlier_thresh.eval(
        """low = invocation_latency_mean - 5 *invocation_latency_iqr
                                                     high = invocation_latency_mean + 5 *invocation_latency_iqr
                                                    """
    )[["max_concurrency", "num_clients", "low", "high"]]

    df_benchmark_results_thresh = df_benchmark_results.merge(
        df_outlier_thresh, on=["max_concurrency", "num_clients"]
    )
    df_benchmark_success = df_benchmark_results_thresh.query(
        "(invocation_latency <= high) & (invocation_latency > 0)"
    )

    df_invocation_error_metrics = df_benchmark_results.groupby(
        ["max_concurrency", "num_clients", "memory_size"], as_index=False
    ).agg(
        {
            "throttle_exception": "sum",
            "insufficient_memory_error": "sum",
            "other_model_error": "sum",
            "invocation_latency": "count",
        }
    )
    df_invocation_error_metrics.rename(
        columns={"invocation_latency": "num_invocations"}, inplace=True
    )

    df_invocation_latency_metrics = df_benchmark_success.groupby(
        ["max_concurrency", "num_clients", "memory_size"], as_index=False
    ).agg({"invocation_latency": ["median", "mean", "max", iqr]})

    df_invocation_latency_metrics.columns = [
        f"{x[0]}_{x[1]}".strip("_")
        for x in df_invocation_latency_metrics.columns.to_flat_index()
    ]

    df_invocation_metrics = df_invocation_error_metrics.merge(
        df_invocation_latency_metrics,
        on=["max_concurrency", "num_clients", "memory_size"],
    )

    df_tps_metrics = df_benchmark_success.groupby(
        ["max_concurrency", "num_clients", "memory_size"], as_index=False
    ).apply(compute_tps)
    df_concurrency_metrics = df_invocation_metrics.merge(
        df_tps_metrics, on=["max_concurrency", "num_clients", "memory_size"]
    )

    concurrency_settings = df_concurrency_metrics["max_concurrency"].unique().tolist()
    num_plots = len(concurrency_settings)

    fig, axs = plt.subplots(len(concurrency_settings), 1, figsize=(10, 6 * num_plots))
    for max_conc, ax in zip(concurrency_settings, axs):
        sns.kdeplot(
            data=df_benchmark_success.query(f"max_concurrency=={max_conc}"),
            x="invocation_latency",
            hue="num_clients",
            palette="tab10",
            ax=ax,
        ).set_title(f"Invocation Latency (ms) with max_concurrency={max_conc}")

    df_concurrency_metrics.to_csv(
        save_path / "concurrency_benchmark_summary.csv", index=True
    )
    df_metric_summary.to_csv(save_path / "endpoint_metrics_summary.csv", index=True)
    fig.savefig(save_path / "concurrency_benchmark_distribution.png")

    return df_concurrency_metrics, df_metric_summary, fig


def plot_savings_latency(df_stability_metric_summary: pd.DataFrame):
    sns.set(style="white")
    paper_rc = {"lines.linewidth": 1, "lines.markersize": 10}
    sns.set_context("paper", rc=paper_rc)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.lineplot(
        x=df_stability_metric_summary.index,
        y=df_stability_metric_summary["cost_per_1M_invocations"],
        marker="s",
        ax=ax,
        color="#e76f51",
    )
    ax2 = ax.twinx()
    sns.lineplot(
        x=df_stability_metric_summary.index,
        y=df_stability_metric_summary["average_latency"],
        marker="s",
        ax=ax2,
        color="#588157",
    )
    ax.set_xticks(df_stability_metric_summary.index)
    fig.legend(
        labels=["Cost Per 1M Invocations", "Average Latency"],
        loc="upper center",
        ncol=2,
        fontsize=10,
    )
    for memory, cost, latency in zip(
        df_stability_metric_summary.index,
        df_stability_metric_summary["cost_per_1M_invocations"],
        df_stability_metric_summary["average_latency"],
    ):
        ax.text(
            x=memory - 128, y=cost, s=f"${cost:.2f}", color="white"
        ).set_backgroundcolor("#e76f51")
        ax2.text(
            x=memory - 128, y=latency, s=f"{latency:.2f}", color="white"
        ).set_backgroundcolor("#588157")

    min_latency = df_stability_metric_summary["average_latency"].min()
    max_latency = df_stability_metric_summary["average_latency"].max()
    min_max_diff = max_latency - min_latency

    if min_max_diff < 10:
        yticks = np.linspace(min_latency - 1, max_latency + 1, 5)

    elif min_max_diff < 50:
        yticks = np.linspace(min_latency - 1, max_latency + 1, 10, dtype=np.int32)
    else:
        yticks = np.linspace(min_latency - 1, max_latency + 1, 20, dtype=np.int32)

    ax2.set_yticks(yticks)

    for x in [ax, ax2]:
        x.spines["top"].set_visible(False)
        x.spines["right"].set_visible(False)
        x.spines["bottom"].set_visible(False)
        x.spines["left"].set_visible(False)

    ax.set_xlabel("Memory Size", fontsize=12)
    ax.set_ylabel("Cost Per 1M Invocations", fontsize=12)
    ax2.set_ylabel("Average Latency (ms)", fontsize=12)

    return fig


def compute_cost_savings(
    df_stability_metric_summary: pd.DataFrame,
    invoke_args_list: List[Dict[str, str]],
    average_response_size: int = 1000,
    result_save_path: str = ".",
) -> Tuple[pd.DataFrame, int, str]:

    save_path = Path(result_save_path) / "cost_analysis_summary_results"
    save_path.mkdir(exist_ok=True, parents=True)

    # df_stability_metric_summary.eval("average_latency = ModelLatency + OverheadLatency", inplace=True) Removed overhead latency for now due to variability when there is no cold start
    average_overhead_latency = df_stability_metric_summary["OverheadLatency"].mean()
    df_stability_metric_summary.eval(
        f"average_latency = ModelLatency + {average_overhead_latency}", inplace=True
    )

    #     try:
    #         minimal_successful_config = df_metric_summary["memory_size"].min()
    #     except:
    #         minimal_successful_config = df_metric_summary.index.min()

    #     average_latency = df_metric_summary.query(f"memory_size == {minimal_successful_config}").eval("ModelLatency + OverheadLatency").values[0]
    average_request_size = reduce(
        lambda x, y: x + sys.getsizeof(y["Body"]), invoke_args_list, 0
    ) / len(invoke_args_list)

    # endpoint_inference_cost = INFERENCE_COST[minimal_successful_config]
    endpoint_processing_cost = (
        average_request_size + average_response_size
    ) * PROCESSING_COST

    df_stability_metric_summary["cost_per_invocation"] = (
        df_stability_metric_summary.index.map(INFERENCE_COST)
        * df_stability_metric_summary["average_latency"]
    ) + endpoint_processing_cost
    df_stability_metric_summary["cost_per_1M_invocations"] = (
        df_stability_metric_summary["cost_per_invocation"] * 1_000_000
    )

    discount_factor = np.linspace(1, 0.5, len(INFERENCE_COST))[:df_stability_metric_summary.shape[0]]

    optimal_memory_config = int(
        (
            df_stability_metric_summary.eval(
                "average_latency * cost_per_1M_invocations"
            )
            * discount_factor
        ).idxmin()
    )

    average_cost_per_invocation = df_stability_metric_summary.loc[
        optimal_memory_config, "cost_per_invocation"
    ]

    cost_latency_fig = plot_savings_latency(df_stability_metric_summary)

    # average_cost_per_invocation = (endpoint_inference_cost * average_latency) + endpoint_processing_in_cost

    comparable_sagemaker_instance = INSTANCE_MAPPING[optimal_memory_config]
    instance_monthly_cost = MONTHLY_INSTANCE_COST[comparable_sagemaker_instance]
    break_even_invocations = instance_monthly_cost / average_cost_per_invocation

    if break_even_invocations < 200_000:
        stride = 10_000
    elif break_even_invocations < 1_000_000:
        stride = 50_000
    elif break_even_invocations < 2_000_000:
        stride = 100_000
    elif break_even_invocations < 5_000_000:
        stride = 200_000
    else:
        stride = 500_000

    monthly_invocations = np.arange(
        stride, break_even_invocations, stride, dtype=np.int32
    )
    monthly_cost = monthly_invocations * average_cost_per_invocation
    monthly_percent_savings = np.round(
        100
        * (instance_monthly_cost - (monthly_invocations * average_cost_per_invocation))
        / instance_monthly_cost,
        2,
    )
    df_savings = pd.DataFrame(
        dict(
            monthly_invocations=monthly_invocations,
            serverless_monthly_cost=monthly_cost,
            instance_monthly_cost=instance_monthly_cost,
            monthly_percent_savings=monthly_percent_savings,
        )
    )

    df_stability_metric_summary.to_csv(save_path / "metrics_with_cost.csv", index=False)
    df_savings.to_csv(save_path / "cost_savings_summary.csv", index=False)
    cost_latency_fig.savefig(save_path / "cost_vs_performance.png")

    df_stability_metric_summary.drop(
        ["average_latency", "cost_per_invocation"], axis=1, inplace=True
    )

    return (
        df_savings,
        optimal_memory_config,
        comparable_sagemaker_instance,
        cost_latency_fig,
    )
