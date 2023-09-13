from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "jkdlsjfowkfkj",
    "write one LLVM IR file",
]

# for i in range(500):
    # prompts.append(f"{i}/2" * 20)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=4)
# llm = LLM(model="bigcode/starcoder")

# llm = LLM(model="lmsys/vicuna-7b-v1.5", tensor_parallel_size=4)
# llm = LLM(model="EleutherAI/gpt-neo-2.7B", tensor_parallel_size=4)
# llm = LLM(model="meta-llama/Llama-2-13b-chat-hf", tensor_parallel_size=4)

# llm = LLM(model="lmsys/vicuna-13b-v1.5", tensor_parallel_size=4)
# llm = LLM(model="lmsys/vicuna-longchat-7b-32k-v1.5", tensor_parallel_size=4)

# import time
# time.sleep(30)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

import torch
from torch.profiler import profile, record_function, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True, with_flops=True, with_stack=True) as prof:
    with record_function("model_inference"):
        outputs = llm.generate(prompts, sampling_params)

print("self_cpu_memory_usage")
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

print("cput_time_total")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("cuda_time_total")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Print aggregated stats
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
prof.export_chrome_trace("trace.json")

# outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# while True:
#     prompts = input("please input one message:")
#     if prompts == "exit":
#         print("exit")
#         break
#     outputs = llm.generate(prompts, sampling_params)
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
