
import os

if "SAGEMAKER_TRAINING_MODULE" in os.environ:
    print("Running in a SageMaker Environment")
    import shutil
    shutil.copytree(".", "sm_serverless_benchmarking")


import argparse
from pathlib import Path

from sm_serverless_benchmarking.benchmark import run_serverless_benchmarks


def parse_args():
    parser = argparse.ArgumentParser(description="Run a suite of SageMaker Serverless Benchmarks on the specified model_name")
    parser.add_argument("model_name", type=str, help="Name of the SageMaker Model resource")
    parser.add_argument("invoke_args_examples_file", type=Path, help="Path to the jsonl file containing the example invocation arcguments")
    parser.add_argument("--cold_start_delay", type=int, help="Number of seconds to sleep before starting the benchmark. Helps to induce a cold start on initial invocation. Defaults to 0")
    parser.add_argument("--memory_sizes", type=int, nargs="+", choices=[1024, 2048, 3072, 4096, 5120, 6144], help="List of memory configurations to benchmark Defaults to [1024, 2048, 3072, 4096, 5120, 6144]")
    parser.add_argument("--stability_benchmark_invocations", type=int, help="Total number of invocations for the stability benchmark. Defaults to 1000")
    parser.add_argument("--stability_benchmark_error_thresh", type=int, help="The allowed number of endpoint invocation errors before the benchmark is terminated for that endpoint. Defaults to 3.")
    parser.add_argument("--no_include_concurrency_benchmark", action='store_true', help="Do not run the concurrency benchmark with the optimal configuration from the stability benchmark. Defaults to False")
    parser.add_argument("--concurrency_benchmark_max_conc", type=int, nargs="+",  help="A list of max_concurency settings to benchmark. Defaults to [2, 4, 8]")
    parser.add_argument("--concurrency_benchmark_invocations", type=int, help="Total number of invocations for the concurency benchmark. Defaults to 1000")
    parser.add_argument("--concurrency_num_clients_multiplier", type=float, nargs="+",  help="List of multipliers to specify the number of simulated clients which is determined by max_concurency * multiplier. Defaults to [1, 1.5, 1.75, 2]")
    parser.add_argument("--result_save_path", type=Path, help="The location to which the output artifacts will be saved. Defaults to .")
    args = parser.parse_args()
    arg_dict = vars(args)
    arg_dict = {k:v for k, v in arg_dict.items() if v is not None}
   
    
    if not arg_dict["no_include_concurrency_benchmark"]:
        arg_dict["include_concurrency_benchmark"] = True
    else:
        arg_dict["include_concurrency_benchmark"] = False
    
    arg_dict.pop("no_include_concurrency_benchmark")

    return arg_dict

def main():
    kwargs = parse_args()
    run_serverless_benchmarks(**kwargs)

if __name__ == "__main__":
    main()

