from vllm import LLM, SamplingParams


def profile(llm:LLM, prompts, sampling_params, info_dict:None, out_json="trace.json", args=None):
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{out_json[:-5]}') if args.tensorboard else None,
                 record_shapes=True, profile_memory=True, use_cuda=True, with_flops=True, with_stack=True) as prof:
        with record_function("generate"):
            outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params, info_dict=info_dict)

    print("self_cpu_memory_usage")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))

    print("cput_time_total")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    print("cuda_time_total")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print aggregated stats
    print("self cuda_time_total")
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=20))

    # prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
    prof.export_chrome_trace(out_json)

    return outputs

def get_model_name(input_string:str):
    import re
    
    match = re.search(r'/(.*)', input_string)
    if match:
        matched_text = match.group(1)
        return matched_text
    return "trace.json"

'''
python examples/inference_testing.py -tp 4 --profile True | tee log.log 2>&1
python examples/inference_testing.py -tp 4 --profile True --tensorboard True | tee log.log 2>&1
 
python examples/inference_testing.py --model "meta-llama/Llama-2-7b-hf" --tokenizer 'hf-internal-testing/llama-tokenizer' | tee log.log 2>&1
python examples/inference_testing.py --model "meta-llama/Llama-2-7b-hf" --tokenizer 'hf-internal-testing/llama-tokenizer' --profile True | tee log.log 2>&1

python examples/inference_testing.py --model "meta-llama/Llama-2-7b-hf" --tokenizer 'hf-internal-testing/llama-tokenizer' --profile True --output-len 2048 --input-len 2048 --batch-size 8 | tee log.log 2>&1 

python examples/inference_testing.py --model "meta-llama/Llama-2-7b-hf" -tp 4 --tokenizer 'hf-internal-testing/llama-tokenizer' | tee log.log 2>&1
python examples/inference_testing.py --model "meta-llama/Llama-2-7b-hf" -tp 4 --tokenizer 'hf-internal-testing/llama-tokenizer' --profile True | tee log.log 2>&1
python examples/inference_testing.py --model "meta-llama/Llama-2-7b-hf" -tp 4 --tokenizer 'hf-internal-testing/llama-tokenizer' --profile True --tensorboard True | tee log.log 2>&1

python examples/inference_testing.py --model "meta-llama/Llama-2-13b-hf" --tokenizer 'hf-internal-testing/llama-tokenizer' | tee log.log 2>&1
python examples/inference_testing.py --model "meta-llama/Llama-2-13b-hf" --tokenizer 'hf-internal-testing/llama-tokenizer' --profile True | tee log.log 2>&1

'''

'''
/mnt/data01/home/lry/envs/miniconda3/envs/vllm/lib/python3.9/site-packages/torch/autograd/profiler_util.py:864,726,708
            name = name[:80]
    src_column_width = 80
    name_column_width = 80
'''

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
                    'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--profile', type=bool, default=False)
    parser.add_argument('--tensorboard', type=bool, default=False)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    args = parser.parse_args()

    # Sample prompts.
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    # prompts = ["0"*32]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=args.output_len)

    std_bsz = 8
    std_inputlen = 8
    std_output_len = 32

    # dummy_prompt_token_ids = [[0] * 32] * 1
    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    # Create an LLM.
    # llm = LLM(model="facebook/opt-125m")
    # llm = LLM(model="meta-llama/Llama-2-7b-hf", tokenizer='hf-internal-testing/llama-tokenizer')
    # llm = LLM(model="meta-llama/Llama-2-7b-hf", tokenizer='hf-internal-testing/llama-tokenizer', tensor_parallel_size=4)
    llm = LLM(model=args.model, tokenizer=args.tokenizer, max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.input_len, tensor_parallel_size=args.tensor_parallel_size
     )

    info_dict = {"model_input_shape":[], "model_output_shape":[], "sampler_output_shape":[], 
                 "timer_total_model":[], "timer_total_sampler":[]}

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    if args.profile:
        outputs = profile(llm, dummy_prompt_token_ids, sampling_params, info_dict=None, out_json=f"{get_model_name(args.model)}.json", args=args)
    else:
        # outputs = llm.generate(prompts, sampling_params, info_dict=info_dict)
        outputs = llm.generate(prompt_token_ids=dummy_prompt_token_ids, sampling_params=sampling_params, info_dict=info_dict)

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
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
