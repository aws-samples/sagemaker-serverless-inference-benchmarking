# SageMaker Serverless Inference Toolkit

Tools to benchmark sagemaker serverless endpoint configurations and help find the most optimal one

## Installation and Prerequisites
To install the toolkit into your environment, first clone this repo. Then inside of the repo directory run
```
python -m pip install .
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
A utility function `sm_serverless_benchmarking.utils.convert_invoke_args_to_jsonl` is provided to convert a list of invocation argument examples into a JSONLines file. If working with data that cannot be serialized to JSON such as binary data including images, audio, and video, use the `sm_serverless_benchmarking.utils.convert_invoke_args_to_pkl` function instead which will serilize the examples to a pickle file instead.

Refer to the [sample_notebooks](sample_notebooks) directory for complete examples

## Configuring the Benchmarks


