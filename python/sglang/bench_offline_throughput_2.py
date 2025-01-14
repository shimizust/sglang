"""
This is a duplication of bench_offline_throughput to enable pytroch profiler, the main goal is to be able to generate trace file
that could be viewed by Perfetto

Llama-3.1-8B: /shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/
r9c8 model: /shared/public/sharing/fait360brew/dev/qsong/360brew-pipeline/Meta-Llama-3.1-8B-Instruct-r9c8-mini-baseline-SFT/f6f23394a78414c9f827/SFT-saved-model-HF


SGLANG_TORCH_PROFILER_DIR=/shared/user/repos/sglang/python/sglang/profile_traces python -m sglang.bench_offline_throughput_2 --model-path /shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/ --profile

python -m sglang.bench_offline_throughput_2 --model-path /shared/public/sharing/fait360brew/dev/qsong/360brew-pipeline/Meta-Llama-3.1-8B-Instruct-r9c8-mini-baseline-SFT/f6f23394a78414c9f827/SFT-saved-model-HF --dp-size 2 --disable-cuda-graph

To test out star attention parittioning context across 2 GPUs on A100 (require disable-cuda-graph):
python -m sglang.bench_offline_throughput_2 --model-path /shared/public/elr-models/meta-llama/Meta-Llama-3.1-70B-Instruct/846357c7ee5e3f50575fd4294edb3d898c8ea100 --disable-cuda-graph --enable-star-attention --tp 2

"""
import argparse
import dataclasses
import json
import logging
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np

from sglang.bench_serving import (
    get_tokenizer,
    set_ulimit,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.server import Engine, Runtime


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    num_requests: int = 5
    prompts_per_member: int = 4
    output_len: int = 1
    result_filename: str = "result.jsonl"
    dataset_path: str = "/shared/public/sharing/ella/data/inference/jymbii_papply/v2_0_4/llama_benchmark_prompts_icl_8192.json"
    backend: str = "engine"
    skip_warmup: bool = True
    result_filename: str = ""
    profile: bool = False
    seed: int = 1
    disable_ignore_eos: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--backend", type=str, default=BenchArgs.backend)
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--num-requests", type=int, default=BenchArgs.num_requests
        )
        parser.add_argument(
            "--prompts-per-member", type=int, default=BenchArgs.prompts_per_member
        )
        parser.add_argument(
            "--output-len", type=int, default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument(
            "--dataset-path", type=str, default=BenchArgs.dataset_path
        )
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batches.",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with "
            "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")
        parser.add_argument(
            "--disable-ignore-eos",
            type=bool,
            default=BenchArgs.disable_ignore_eos,
            help="Disable ignore EOS token",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to case the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def throughput_test_once(
    backend_name: str,
    backend,
    reqs: List[Tuple[str, int, int]],
    ignore_eos: bool,
    extra_request_body: Dict,
    profile: bool,
):
    measurement_results = {
        "backend": backend_name,
        "successful_requests": len(reqs),
        "total_latency": -1,
        "total_input_tokens": sum(r[1] for r in reqs),
        "total_output_tokens": -1,
        "request_throughput": -1,
        "input_throughput": -1,
        "output_throughput": -1,
        "total_throughput": -1,
    }

    prompt = [r[0] for r in reqs]
    sampling_params = [
        {
            "temperature": 0,
            "max_new_tokens": r[2],
            "ignore_eos": ignore_eos,
            **extra_request_body,
        }
        for r in reqs
    ]

    if profile:
        print("Benchmark script, starting profile")
        backend.start_profile()

    st = time.perf_counter()
    gen_out = backend.generate(prompt=prompt, sampling_params=sampling_params)
    latency = time.perf_counter() - st
    
    if profile:
        print("Benchmark script, stopping profile")
        backend.stop_profile()
        monitor_trace_file(os.getenv("SGLANG_TORCH_PROFILER_DIR"))

    if backend_name == "runtime":
        gen_out = json.loads(gen_out)

    measurement_results["total_latency"] = latency
    measurement_results["total_output_tokens"] = sum(
        o["meta_info"]["completion_tokens"] for o in gen_out
    )
    measurement_results["request_throughput"] = (
        measurement_results["successful_requests"] / latency
    )
    measurement_results["input_throughput"] = (
        measurement_results["total_input_tokens"] / latency
    )
    measurement_results["output_throughput"] = (
        measurement_results["total_output_tokens"] / latency
    )
    measurement_results["total_throughput"] = (
        measurement_results["total_input_tokens"]
        + measurement_results["total_output_tokens"]
    ) / latency

    return measurement_results


def monitor_trace_file(directory, interval=1):

    print(f"Monitoring {directory} for new trace files...")

    known_files = set(os.listdir(directory))

    while True:
        flag = False
        time.sleep(interval)
        current_files = set(os.listdir(directory))

        new_files = current_files - known_files
        for new_file in new_files:
            new_file_path = os.path.join(directory, new_file)
            print(f"New file detected: {new_file}")

            previous_size = 0
            while True:
                try:
                    current_size = os.path.getsize(new_file_path)
                except FileNotFoundError:
                    print(f"File {new_file} is no longer accessible.")
                    break

                if current_size > previous_size:
                    previous_size = current_size
                else:
                    flag = True
                    break

                time.sleep(interval)
        if flag:
            break


def sample_random_requests(data: dict[str, list[str]], num_requests: int, prompts_per_member: int, tokenizer):
    K = 50
    requests = []  # original prompt text, input len
    members = list(data.keys())
    candidates = random.choices(members, k=num_requests)
    for candidate in candidates:
        prompts = data.get(candidate, [])
        prompts = random.choices(prompts, k=prompts_per_member)
        for prompt in prompts:
            prompt_ids = tokenizer(prompt).input_ids
            requests.append((prompt, len(prompt_ids), 16))

    return requests


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    if bench_args.backend == "engine":
        backend = Engine(**dataclasses.asdict(server_args))
        if not backend:
            raise ValueError("Please provide valid engine arguments")
    elif bench_args.backend == "runtime":
        backend = Runtime(**dataclasses.asdict(server_args))
    else:
        raise ValueError('Please set backend to either "engine" or "runtime"')

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # Set global environmnets
    set_ulimit()
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    with open(bench_args.dataset_path, 'r') as file:
        data = json.load(file)

    warmup_requests = sample_random_requests(
        data=data,
        num_requests=bench_args.num_requests,
        prompts_per_member=bench_args.prompts_per_member,
        tokenizer=tokenizer,
    )

    input_requests = sample_random_requests(
        data=data,
        num_requests=bench_args.num_requests,
        prompts_per_member=bench_args.prompts_per_member,
        tokenizer=tokenizer,
    )

    # Warm up
    if not bench_args.skip_warmup:
        logging.info("\nWarmup...")
        throughput_test_once(
            backend_name=bench_args.backend,
            backend=backend,
            reqs=warmup_requests,
            ignore_eos=not bench_args.disable_ignore_eos,
            extra_request_body={},
            profile=False,
        )
        time.sleep(0.5)

    logging.info("\nBenchmark...")
    result = throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=input_requests,
        ignore_eos=not bench_args.disable_ignore_eos,
        extra_request_body={},
        profile=bench_args.profile,
    )
    
    backend.shutdown()

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    print(
        "\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=50, c="=")
    )
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print(
        "{:<40} {:<10}".format("Total generated tokens:", result["total_output_tokens"])
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", result["request_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", result["input_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", result["output_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", result["total_throughput"]
        )
    )
    print("=" * 50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    throughput_test(server_args, bench_args)