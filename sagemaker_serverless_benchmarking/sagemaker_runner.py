import os
from importlib import resources
from pathlib import Path
from typing import List, Union

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.xgboost import XGBoostProcessor


def args_constructor(
    benchmark_args: dict, sm_job_input_dir: str, sm_job_output_dir: str
):

    input_file_name = os.path.basename(benchmark_args["invoke_args_examples_file"])

    args = []
    for k, v in benchmark_args.items():
        if k in {"role", "s3_output_path", "include_concurrency_benchmark", "region"}:
            continue
        if k == "model_name":
            args.extend([v])
        elif k == "invoke_args_examples_file":
            args.extend([os.path.join(sm_job_input_dir, input_file_name)])
        elif k == "result_save_path":
            args.extend([f"--{k}", sm_job_output_dir])
        else:
            args.extend([f"--{k}"])
            if type(v) == list:
                for param in v:
                    args.extend([str(param)])
            else:
                args.extend([str(v)])

    return args


def run_as_sagemaker_job(
    role: str,
    region:str,
    model_name: str,
    invoke_args_examples_file: Union[Path, str],
    s3_output_path: str = None,
    cold_start_delay: int = 0,
    memory_sizes: List[int] = [1024, 2048, 3072, 4096, 5120, 6144],
    stability_benchmark_invocations: int = 1000,
    stability_benchmark_error_thresh: int = 3,
    include_concurrency_benchmark: bool = True,
    concurrency_benchmark_max_conc: List[int] = [2, 4, 8],
    concurrency_benchmark_invocations: int = 1000,
    concurrency_num_clients_multiplier: List[float] = [1, 1.5, 1.75, 2],
    result_save_path: str = ".",
):

    benchmark_args = locals()

    with resources.path("sagemaker_serverless_benchmarking", "__main__.py") as p:
        source_path = str(p.parent)

    sm_job_input_dir = "/opt/ml/processing/input/data"
    sm_job_output_dir = "/opt/ml/processing/output/"
   

    job_args = args_constructor(benchmark_args, sm_job_input_dir, sm_job_output_dir)

    processor = XGBoostProcessor(
        role=role,
        framework_version="1.5-1",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        base_job_name="sagemaker-serverless-inf-bench",
        env={"AWS_DEFAULT_REGION": region}
    )

    processor.run(
        code="__main__.py",
        source_dir=source_path,
        inputs=[
            ProcessingInput(
                source=str(invoke_args_examples_file),
                destination=sm_job_input_dir,
            )
        ],
        outputs=[
            ProcessingOutput(output_name="benchmark_outputs", source=sm_job_output_dir)
        ],
        arguments=job_args,
        wait=False
    )


if __name__ == "__main__":
    t = run_as_sagemaker_job(
        role="arn:aws:iam::152804913371:role/service-role/AmazonSageMaker-ExecutionRole-20200526T152070",
        model_name="huggingface-pytorch-inference-2022-07-18-20-04-43-004",
        invoke_args_examples_file="invoke_args_examples.jsonl",
        region="us-east-1"
    )
