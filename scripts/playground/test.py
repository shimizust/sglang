import sglang as sgl
import asyncio
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def run_inference():
    llm = sgl.Engine(model_path="/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/")


    prompts = [
        # "Hello, my name is",
        # "The president of the United States is",
        # "The future of AI is",
        "The capital of France is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("generate_text"):
            outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == "__main__":
    run_inference()