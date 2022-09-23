# SageMaker Serverless Inference Toolkit

Tools to benchmark sagemaker serverless endpoint configurations and help find the most optimal one

## Installation and Prerequisites
To install the toolkit into your environment, first clone this repo. Then inside of the repo directory run
```
pip install sm-serverless-benchmarking
```
In order to run the benchmark, your user profile or execution role would need to have the appropriate IAM Permissions Including:
#### **SageMaker**
- CreateModel
- CreateEndpointConfig / DeleteEndpointConfig
- CreateEndpoint / DeleteEndpoint
- CreateProcessingJob (if using SageMaker Runner) 
#### **SageMaker Runtime**
- InvokeEndpoint
#### **CloudWatch**
- GetMetricStatistics

## Quick Start
To run a benchmark locally, provide your sagemaker [Model](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateModel.html) name and a list of example invocation arguments. Each of these arguments will be passed directly to the SageMaker runtime [InvokeEndpoint](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html#SageMakerRuntime.Client.invoke_endpoint) API
```
from sm_serverless_benchmarking import benchmark
from sm_serverless_benchmarking.utils import convert_invoke_args_to_jsonl

model_name = "<SageMaker Model Name>"

example_invoke_args = [
               {'Body': '1,2,3,4,5', "ContentType": "text/csv"},
               {'Body': '6,7,8,9,10', "ContentType": "text/csv"}
              ]
              
example_args_file = convert_invoke_args_to_jsonl(example_invoke_args, 
                                                 output_path=".")
              
r = benchmark.run_serverless_benchmarks(model_name, example_args_file)
```
Alternativelly, you can run the benchmarks as SageMaker Processing job
```
from sm_serverless_benchmarking.sagemaker_runner import run_as_sagemaker_job

run_as_sagemaker_job(
        role="<execution_role_arn>",
        model_name="<model_name>",
        invoke_args_examples_file="<invoke_args_examples_file>",
    )
```
A utility function `sm_serverless_benchmarking.utils.convert_invoke_args_to_jsonl` is provided to convert a list of invocation argument examples into a JSONLines file. If working with data that cannot be serialized to JSON such as binary data including images, audio, and video, use the `sm_serverless_benchmarking.utils.convert_invoke_args_to_pkl` function which will serilize the examples to a pickle file instead.

Refer to the [sample_notebooks](sample_notebooks) directory for complete examples

## Types of Benchmarks
By default two types of benchmarks will be executed

- **Stability Benchmark** For each memory configuration, and a [MaxConcurency](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints-create.html#serverless-endpoints-create-config) of 1, will invoke the endpoint a specified number of times sequentially. The goal of this benchmark is to determine the most cost effective and stable memory configuration
- **Concurrency Benchmark** Will invoke an endpoint with a simulated number of concurrent clients under different MaxConcurrency configurations 

## Configuring the Benchmarks
For either of the two approaches described above, you can specify a number of parameters to configure the benchmarking job

        cold_start_delay (int, optional): Number of seconds to sleep before starting the benchmark. Helps to induce a cold start on initial invocation. Defaults to 0.
        memory_sizes (List[int], optional): List of memory configurations to benchmark Defaults to [1024, 2048, 3072, 4096, 5120, 6144].
        stability_benchmark_invocations (int, optional): Total number of invocations for the stability benchmark. Defaults to 1000.
        stability_benchmark_error_thresh (int, optional): The allowed number of endpoint invocation errors before the benchmark is terminated for a configuration. Defaults to 3.
        include_concurrency_benchmark (bool, optional): Set True to run the concurrency benchmark with the optimal configuration from the stability benchmark. Defaults to True.
        concurrency_benchmark_max_conc (List[int], optional): A list of max_concurency settings to benchmark. Defaults to [2, 4, 8].
        concurrency_benchmark_invocations (int, optional): Total number of invocations for the concurency benchmark. Defaults to 1000.
        concurrency_num_clients_multiplier (List[int], optional): List of multipliers to specify the number of simulated clients which is determined by max_concurency * multiplier. Defaults to [1, 1.5, 1.75, 2].
        result_save_path (str, optional): The location to which the output artifacts will be saved. Defaults to ".".


