import base64
from importlib import resources
from io import BytesIO
from pathlib import Path

import matplotlib
import pandas as pd
from jinja2 import Environment, FileSystemLoader


def b64_png_encode(fig: matplotlib.figure.Figure):
    png = BytesIO()
    fig.savefig(png, bbox_inches="tight")
    png.flush()

    encoded_img = base64.b64encode(png.getvalue())

    return encoded_img


def generate_html_report(
    benchmark_config: pd.DataFrame,
    df_stability_summary: pd.DataFrame,
    df_stability_metric_summary: pd.DataFrame,
    stability_latency_distribution: matplotlib.figure.Figure,
    cost_vs_performance: matplotlib.figure.Figure,
    df_cost_savings: pd.DataFrame,
    optimal_memory_config: int,
    comparable_instance: str,
    df_concurrency_metrics: pd.DataFrame = pd.DataFrame(),
    df_concurrency_metric_summary: pd.DataFrame = pd.DataFrame(),
    concurrency_latency_distribution: matplotlib.figure.Figure = matplotlib.figure.Figure(),
    result_save_path: str = ".",
):

    report_path = Path(result_save_path) / "benchmarking_report"
    report_path.mkdir(exist_ok=True, parents=True)

    with resources.path(
        "sm_serverless_benchmarking.report_templates", "report_template.html"
    ) as p:
        templates_path = p.parent

    environment = Environment(
        loader=FileSystemLoader(templates_path)
    )
    template = environment.get_template("report_template.html")

    stability_latency_distribution_encoded = b64_png_encode(
        stability_latency_distribution
    )
    concurrency_latency_distribution_encoded = b64_png_encode(
        concurrency_latency_distribution
    )
    cost_vs_performance_encoded = b64_png_encode(cost_vs_performance)

    context = {
        "benchmark_configuration": benchmark_config.to_html(
            index=True,
            float_format="%.2f",
            justify="left",
            header=False,
            na_rep="",
            escape=False,
        ),
        "stability_benchmark_summary": df_stability_summary.to_html(
            index=True,
            float_format="%.2f",
            na_rep="",
            justify="center",
            notebook=True,
            escape=False,
        ).replace("<td>", '<td align="center">'),
        "stability_endpoint_metrics": df_stability_metric_summary.to_html(
            index=True,
            float_format="%.2f",
            justify="center",
            na_rep="",
            notebook=True,
            escape=False,
        ).replace("<td>", '<td align="center">'),
        "stability_latency_distribution": stability_latency_distribution_encoded.decode(
            "utf8"
        ),
        "cost_vs_performance": cost_vs_performance_encoded.decode("utf8"),
        "cost_savings_table": df_cost_savings.to_html(
            index=False,
            escape=False,
            formatters={
                "monthly_invocations": lambda x: f"{x:,}",
                "serverless_monthly_cost": lambda x: f"${x:.2f}",
                "instance_monthly_cost": lambda x: f"${x:.2f}",
                "monthly_percent_savings": lambda x: f"{x}%",
            },
        ).replace("<td>", '<td align="center">'),
        "optimal_memory_config": optimal_memory_config,
        "comparable_instance": comparable_instance,
        "concurrency_benchmark_summary": df_concurrency_metrics.to_html(
            index=False,
            escape=False,
            float_format="%.2f",
            na_rep="",
            justify="center",
            notebook=True,
            formatters={
                "insufficient_memory_error": lambda x: f"{x:0.0f}",
                "other_model_error": lambda x: f"{x:0.0f}",
            },
        ).replace("<td>", '<td align="center">'),
        "concurrency_latency_distribution": concurrency_latency_distribution_encoded.decode(
            "utf8"
        ),
        "concurrency_cloudwatch_metrics": df_concurrency_metric_summary.to_html(
            index=True,
            float_format="%.2f",
            justify="center",
            na_rep="",
            notebook=True,
            escape=False,
        ).replace("<td>", '<td align="center">'),
    }

    report = template.render(context=context)

    with (report_path / "benchmarking_report.html").open("w") as f:
        f.write(report)

    return report
