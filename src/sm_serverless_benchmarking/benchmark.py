import datetime as dt
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Union

import botocore.exceptions
import pandas as pd

from sm_serverless_benchmarking.analysis import (compute_cost_savings,
                                                 summarize_concurrency_results,
                                                 summarize_stability_results)
from sm_serverless_benchmarking.endpoint import ServerlessEndpoint
from sm_serverless_benchmarking.report import generate_html_report
from sm_serverless_benchmarking.utils import read_example_args_file


def create_endpoint(
    model_name: str, memory_size: int = 1024, max_concurrency: int = 1
) -> ServerlessEndpoint:
    ep = ServerlessEndpoint(
        model_name=model_name, memory_size=memory_size, max_concurrency=max_concurrency
    )
    ep.create_endpoint()
    return ep


def timed_invocation(endpoint: ServerlessEndpoint, invoke_args: Dict[str, str]):
    time.sleep(random.random())
    t1 = time.perf_counter()
    response_size = 0
    try:
        response = endpoint.invoke_endpoint(invoke_args)
        response_size = int(
            response["ResponseMetadata"]["HTTPHeaders"]["content-length"]
        )
        # body = response["Body"].read()
        # response_size = sys.getsizeof(
        #     body
        # )  # calculate response size to estimate per inference cost

    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "ThrottlingException":
            return {
                "invocation_latency": -1,
                "throttle_exception": 1,
                "start_time": t1,
                "end_time": time.perf_counter(),
            }

        elif error.response["Error"]["Code"] == "ModelError":
            if "insufficient memory" in error.response["Error"]["Message"]:
                return {
                    "invocation_latency": -1,
                    "throttle_exception": 0,
                    "insufficient_memory_error": 1,
                    "other_model_error": 0,
                    "start_time": t1,
                    "end_time": time.perf_counter(),
                }
            else:
                print(
                    f"error in endpoint {endpoint.endpoint_name} ",
                    error.response["Error"]["Message"],
                )
                return {
                    "invocation_latency": -1,
                    "throttle_exception": 0,
                    "insufficient_memory_error": 0,
                    "other_model_error": 1,
                    "start_time": t1,
                    "end_time": time.perf_counter(),
                }

    t2 = time.perf_counter()

    return {
        "invocation_latency": (t2 - t1) * 1000,
        "response_size": response_size,
        "throttle_exception": 0,
        "insufficient_memory_error": 0,
        "other_model_error": 0,
        "start_time": t1,
        "end_time": t2,
    }


def create_endpoint_configs(
    memory_size: Union[int, List[int]], max_concurrency: Union[int, List[int]]
):

    if type(memory_size) == int:
        memory_size = [memory_size]

    if type(max_concurrency) == int:
        max_concurrency = [max_concurrency]

    endpoint_configs = []
    for mem_size in memory_size:
        for max_conc in max_concurrency:
            endpoint_configs.append(
                {"memory_size": mem_size, "max_concurrency": max_conc}
            )

    return endpoint_configs


def setup_endpoints(
    model_name: str,
    memory_size: Union[int, List[int]],
    max_concurrency: Union[int, List[int]],
    sleep: int = 0,
):

    update_configs = create_endpoint_configs(memory_size, max_concurrency)

    endpoint_futures = []

    with ThreadPoolExecutor(max_workers=len(update_configs)) as executor:

        for config_kwargs in update_configs:

            future = executor.submit(create_endpoint, model_name, **config_kwargs)
            endpoint_futures.append(future)
            time.sleep(2)

        endpoints = [future.result() for future in as_completed(endpoint_futures)]


    endpoints = [endpoint for endpoint in endpoints if endpoint._created]
    time.sleep(sleep)  # sleep to increase chance of cold start

    return endpoints


def stability_benchmark(
    endpoint: ServerlessEndpoint, invoke_args_list, num_invocations=1000, error_thresh=3
):

    results = []
    errors = 0

    for _ in range(num_invocations):
        invoke_arg_idx = random.randint(0, len(invoke_args_list)-1)
        invoke_args = invoke_args_list[invoke_arg_idx]
        result = timed_invocation(endpoint, invoke_args)
        result["invoke_arg_index"] = invoke_arg_idx 

        if result["invocation_latency"] == -1:
            errors += 1

        if errors >= error_thresh:
            print(
                f"Terminating benchmark for {endpoint.endpoint_name} due to excessive endpoint errors"
            )

            return results

        results.append(result)

    return results


def run_stability_benchmark(
    endpoints: List[ServerlessEndpoint],
    invoke_args_list: List[Dict[str, str]],
    num_invocations: int = 1000,
    error_thresh: int = 3,
    result_save_path: str = ".",
):

    endpoint_benchmark_futures = {}
    benchmark_results = []
    all_endpoint_metrics = []

    save_path = Path(result_save_path) / "stability_benchmark_raw_results"
    save_path.mkdir(exist_ok=True, parents=True)

    benchmark_start_time = dt.datetime.utcnow()

    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:

        for endpoint in endpoints:
            future = executor.submit(
                stability_benchmark,
                endpoint,
                invoke_args_list,
                num_invocations,
                error_thresh,
            )
            endpoint_benchmark_futures[future] = endpoint

        for future in as_completed(endpoint_benchmark_futures):
            endpoint = endpoint_benchmark_futures[future]

            result = future.result()
            df_result = pd.DataFrame(result)
            df_result["memory_size"] = endpoint.memory_size
            benchmark_results.append(df_result)

            benchmark_end_time = dt.datetime.utcnow()
            lookback_window = (
                benchmark_end_time - benchmark_start_time
            ) + dt.timedelta(hours=1)
        
            endpoint_metrics = endpoint.get_endpoint_metrics(lookback_window)
            df_endpoint_metrics = pd.DataFrame(endpoint_metrics)
            df_endpoint_metrics["memory_size"] = endpoint.memory_size
            all_endpoint_metrics.append(df_endpoint_metrics)

    df_benchmark_results = pd.concat(benchmark_results).reset_index(drop=True)
    df_endpoint_metrics = pd.concat(all_endpoint_metrics).reset_index(drop=True)

    df_benchmark_results.to_csv(
        save_path / "invocation_benchmark_results.csv", index=False
    )
    df_endpoint_metrics.to_csv(save_path / "endpoint_metrics.csv", index=False)

    return df_benchmark_results, df_endpoint_metrics


def concurrency_benchmark(
    endpoint: ServerlessEndpoint, invoke_args_list, num_invocations=50, num_clients=1
):

    futures = []

    with ThreadPoolExecutor(max_workers=num_clients) as executor:

        for _ in range(num_invocations):
            invoke_args = random.choice(invoke_args_list)

            futures.append(executor.submit(timed_invocation, endpoint, invoke_args))

        results = [future.result() for future in futures]

    return results


def run_concurrency_benchmark(
    endpoints: List[ServerlessEndpoint],
    invoke_args_list: List[Dict[str, str]],
    num_invocations: int = 1000,
    num_clients_multipliers: List[float] = [1, 1.5, 1.75, 2],
    result_save_path: str = ".",
):

    endpoint_benchmark_futures = {}
    benchmark_results = []
    all_endpoint_metrics = []

    save_path = Path(result_save_path) / "concurrency_benchmark_raw_results"
    save_path.mkdir(exist_ok=True, parents=True)

    benchmark_start_time = dt.datetime.now()

    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        seen_conc_clients = set()

        for multiplier in num_clients_multipliers:
            futures = []
            for endpoint in endpoints:
                num_clients = math.ceil(endpoint.max_concurrency * multiplier)
                max_conc_num_clients = (endpoint.max_concurrency, num_clients)

                if max_conc_num_clients in seen_conc_clients:
                    continue
                seen_conc_clients.add(max_conc_num_clients)

                future = executor.submit(
                    concurrency_benchmark,
                    endpoint,
                    invoke_args_list,
                    num_invocations,
                    num_clients,
                )
                endpoint_benchmark_futures[future] = (endpoint, num_clients)
                futures.append(future)

            while any([future.running() for future in futures]):
                time.sleep(5)

        for future in as_completed(endpoint_benchmark_futures):
            endpoint = endpoint_benchmark_futures[future][0]

            result = future.result()
            df_result = pd.DataFrame(result)
            df_result["memory_size"] = endpoint.memory_size
            df_result["max_concurrency"] = endpoint.max_concurrency
            df_result["num_clients"] = endpoint_benchmark_futures[future][1]
            benchmark_results.append(df_result)

            benchmark_end_time = dt.datetime.now()
            lookback_window = (
                benchmark_end_time - benchmark_start_time
            ) + dt.timedelta(hours=1)
            endpoint_metrics = endpoint.get_endpoint_metrics(lookback_window)
            df_endpoint_metrics = pd.DataFrame(endpoint_metrics)
            df_endpoint_metrics["memory_size"] = endpoint.memory_size
            df_endpoint_metrics["max_concurrency"] = endpoint.max_concurrency
            df_endpoint_metrics["num_clients"] = endpoint_benchmark_futures[future][1]
            all_endpoint_metrics.append(df_endpoint_metrics)

    df_benchmark_results = pd.concat(benchmark_results).reset_index(drop=True)
    df_endpoint_metrics = pd.concat(all_endpoint_metrics).reset_index(drop=True)

    df_benchmark_results.to_csv(
        save_path / "invocation_benchmark_results.csv", index=False
    )
    df_endpoint_metrics.to_csv(save_path / "endpoint_metrics.csv", index=False)

    return df_benchmark_results, df_endpoint_metrics


def tear_down_endpoints(endpoints: List[ServerlessEndpoint]):
    for endpoint in endpoints:
        try:
            endpoint.clean_up()
        except:
            pass



def run_serverless_benchmarks(
    model_name: str,
    invoke_args_examples_file: Path,
    cold_start_delay: int = 0,
    memory_sizes: List[int] = [1024, 2048, 3072, 4096, 5120, 6144],
    stability_benchmark_invocations: int = 1000,
    stability_benchmark_error_thresh: int = 3,
    include_concurrency_benchmark: bool = True,
    concurrency_benchmark_max_conc: List[int] = [2, 4, 8],
    concurrency_benchmark_invocations: int = 1000,
    concurrency_num_clients_multiplier: List[float] = [1, 1.5, 1.75, 2],
    result_save_path: str = ".",
)->str:
    """Runs a suite of SageMaker Serverless Benchmarks on the specified model_name. 
    Will automatically deploy endpoints for the specified model_name and perform a tear down
    upon completion of the benchmark or an error.

    There are two types of benchmarks that are supported and both are executed by defaults
    - Stability Benchmark: Deploys an endpoint for each of the specified memory configurations and
    max concurrency of 1. Invokes the endpoint the specified number of times and determines the stable
    and most cost effective configuration

    - Concurrency Benchmark: Deploys endpoints with different max_concurrency configurations and performs 
    a load test with a simulated number of concurrent clients 

    Args:
        model_name (str): Name of the SageMaker Model resource
        invoke_args_examples_file (Path: Path to the jsonl file containing the example invocation arcguments
        cold_start_delay (int, optional): Number of seconds to sleep before starting the benchmark. Helps to induce a cold start on initial invocation. Defaults to 0.
        memory_sizes (List[int], optional): List of memory configurations to benchmark Defaults to [1024, 2048, 3072, 4096, 5120, 6144].
        stability_benchmark_invocations (int, optional): Total number of invocations for the stability benchmark. Defaults to 1000.
        stability_benchmark_error_thresh (int, optional): The allowed number of endpoint invocation errors before the benchmark is terminated for a configuration. Defaults to 3.
        include_concurrency_benchmark (bool, optional): Set True to run the concurrency benchmark with the optimal configuration from the stability benchmark. Defaults to True.
        concurrency_benchmark_max_conc (List[int], optional): A list of max_concurency settings to benchmark. Defaults to [2, 4, 8].
        concurrency_benchmark_invocations (int, optional): Total number of invocations for the concurency benchmark. Defaults to 1000.
        concurrency_num_clients_multiplier (List[int], optional): List of multipliers to specify the number of simulated clients which is determined by max_concurency * multiplier. Defaults to [1, 1.5, 1.75, 2].
        result_save_path (str, optional): The location to which the output artifacts will be saved. Defaults to ".".


    Returns:
        str: HTML for the generated benchmarking report
    """

    function_args = locals()
    benchmark_config = pd.Series(function_args).to_frame()

    invoke_args_examples = read_example_args_file(invoke_args_examples_file)

    stability_endpoints = setup_endpoints(
        model_name, memory_size=memory_sizes, max_concurrency=1, sleep=cold_start_delay
    )

    try:
        (
            df_stability_benchmark_results,
            df_stability_endpoint_metrics,
        ) = run_stability_benchmark(
            stability_endpoints,
            invoke_args_list=invoke_args_examples,
            num_invocations=stability_benchmark_invocations,
            error_thresh=stability_benchmark_error_thresh,
            result_save_path=result_save_path,
        )
        (
            df_stability_results,
            df_stability_summary,
            df_stability_metric_summary,
            stability_latency_distribution_fig,
        ) = summarize_stability_results(
            df_stability_benchmark_results,
            df_stability_endpoint_metrics,
            result_save_path=result_save_path,
        )

        avg_response_size = df_stability_results["response_size"].mean()
        (
            df_cost_savings,
            minimal_successful_config,
            comparable_sagemaker_instance,
            cost_vs_performance,
        ) = compute_cost_savings(
            df_stability_metric_summary,
            invoke_args_examples,
            average_response_size=avg_response_size,
            result_save_path=result_save_path
        )

    except Exception as e:
        print(f"Could not complete benchmark due to Exception: {e}")
        raise e

    finally:
        tear_down_endpoints(stability_endpoints)

    if include_concurrency_benchmark:
        concurrency_endpoints = setup_endpoints(
            model_name,
            memory_size=minimal_successful_config,
            max_concurrency=concurrency_benchmark_max_conc,
        )
        try:
            (
                df_conc_benchmark_results,
                df_conc_endpoint_metrics,
            ) = run_concurrency_benchmark(
                concurrency_endpoints,
                invoke_args_examples,
                num_invocations=concurrency_benchmark_invocations,
                num_clients_multipliers=concurrency_num_clients_multiplier,
                result_save_path=result_save_path,
            )
            (
                df_concurrency_metrics,
                df_concurrency_metric_summary,
                concurrency_latency_distribution_fig,
            ) = summarize_concurrency_results(
                df_conc_benchmark_results,
                df_conc_endpoint_metrics,
                result_save_path=result_save_path,
            )

        except Exception as e:
            print(f"Could not complete benchmark due to Exception: {e}")

        finally:
            tear_down_endpoints(concurrency_endpoints)

        report = generate_html_report(
            benchmark_config=benchmark_config,
            df_stability_summary=df_stability_summary,
            df_stability_metric_summary=df_stability_metric_summary,
            stability_latency_distribution=stability_latency_distribution_fig,
            df_cost_savings=df_cost_savings,
            cost_vs_performance=cost_vs_performance,
            optimal_memory_config=minimal_successful_config,
            comparable_instance=comparable_sagemaker_instance,
            df_concurrency_metrics=df_concurrency_metrics,
            df_concurrency_metric_summary=df_concurrency_metric_summary,
            concurrency_latency_distribution=concurrency_latency_distribution_fig,
            result_save_path=result_save_path,
        )
        return report

    else:
        report = generate_html_report(
            benchmark_config=benchmark_config,
            df_stability_summary=df_stability_summary,
            df_stability_metric_summary=df_stability_metric_summary,
            stability_latency_distribution=stability_latency_distribution_fig,
            df_cost_savings=df_cost_savings,
            cost_vs_performance=cost_vs_performance,
            optimal_memory_config=minimal_successful_config,
            comparable_instance=comparable_sagemaker_instance,
            result_save_path=result_save_path,
        )
        return report