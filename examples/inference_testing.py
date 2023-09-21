from vllm import LLM, SamplingParams

# python examples/inference_testing.py > log.log 2>&1

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# prompts = ["0"*512]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m")
llm = LLM(model="meta-llama/Llama-2-7b-hf", tokenizer='hf-internal-testing/llama-tokenizer')

info_dict = {"model_input_shape":[], "model_output_shape":[], "sampler_output_shape":[], 
             "timer_total_model":[], "timer_total_sampler":[]}

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params, info_dict=info_dict)

print(info_dict)
for k,v in info_dict.items():
    print(k, v)

for i in range(len(info_dict["model_input_shape"])):
    for k in info_dict.keys():
        print(k, info_dict[k][i], end=" ")
    print("\n")

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def profile(llm:LLM, prompts, sampling_params):
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

    # prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
    prof.export_chrome_trace("trace.json")

# profile(llm, prompts, sampling_params)