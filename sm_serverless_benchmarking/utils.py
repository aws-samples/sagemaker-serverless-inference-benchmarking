import json
import pickle
from pathlib import Path
from typing import Dict, List, Union


def convert_invoke_args_to_jsonl(
    invoke_args_examples: List[Dict[str, str]], output_path: str = "."
):
    """Converts a list of invokation argument examples to JSON lines format and saves the output to a file

    Args:
        invoke_args_examples (List[Dict[str, str]]): A list of example arguments that will be passed to the InvokeEndpoint SageMaker Runtime API
        output_path (str, optional): The directory to which the output jsonl will be written Defaults to ".".
    """
    output_file = Path(output_path) / "invoke_args_examples.jsonl"

    with output_file.open("w+") as f:
        for example in invoke_args_examples:
            f.write(f"{json.dumps(example)}\n")

    return output_file


def convert_invoke_args_to_pkl(
    invoke_args_examples: List[Dict[str, str]], output_path: str = "."
):
    """Converts a list of invokation argument examples to pickle. For use with binary data such as images, video, and audio

    Args:
        invoke_args_examples (List[Dict[str, str]]): A list of example arguments that will be passed to the InvokeEndpoint SageMaker Runtime API
        output_path (str, optional): The directory to which the output pkl will be written Defaults to ".".
    """
    output_file = Path(output_path) / "invoke_args_examples.pkl"

    with output_file.open("wb+") as f:
        pickle.dump(invoke_args_examples, f)

    return output_file



def read_example_args_file(example_args_file: Union[Path, str]):


    example_args_file = Path(example_args_file)

    if example_args_file.suffix == ".pkl":
        example_args = pickle.loads(example_args_file.open("rb").read())

    else:
        example_args = []
        with example_args_file.open("r") as f:

            for line in f:
                example_args.append(json.loads(line))

    return example_args
