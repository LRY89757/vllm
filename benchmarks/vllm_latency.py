"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams

from vllm.transformers_utils.tokenizer import get_tokenizer

import ray

def main(args: argparse.Namespace):
    print(args)

    # Process all the requests in a single batch if possible.
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.input_len,
        trust_remote_code=args.trust_remote_code,
    )


    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    # dummy_prompt_token_ids = ["pifuj;ajkslsa" * args.input_len] * args.batch_size

    # tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    # prompt_token_ids = tokenizer(dummy_prompt_token_ids).input_ids
    # length = [len(prompt_id) for prompt_id in prompt_token_ids]
    # print(length)
    # print(sum(length))

    def run_to_completion(profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()

        llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        
        end_time = time.time()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warming up...")
    run_to_completion(profile=False)

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile=False))
    print(f'Avg latency: {np.mean(latencies)} seconds')

    return np.mean(latencies)

def profiler(args):
    ## profiler the model

    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True, with_flops=True, with_stack=True) as prof:
        with record_function("model_inference"):
            main(args)

    print("self_cpu_memory_usage")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    print("cput_time_total")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("cuda_time_total")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Print aggregated stats
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

    prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
    prof.export_chrome_trace("benchmark_latency_opt.json")

if __name__ == '__main__':
    '''
    python benchmark_latency.py --model "meta-llama/Llama-2-13b-chat-hf" -tp 4
    python benchmark_latency.py --model "meta-llama/Llama-2-13b-chat-hf" -tp 4

    sudo /usr/local/cuda-11.0/bin/nsys profile /home/lry/envs/miniconda3/envs/vllm/bin/python benchmark_latency.py --model "facebook/opt-125m"
    python benchmark_latency.py --model "facebook/opt-125m" --input-len 32 --num-iters 1 --batch-size 1

    python benchmark_latency.py --model "meta-llama/Llama-2-7b-hf" --tokenizer 'hf-internal-testing/llama-tokenizer'

    python benchmark_latency.py --model "meta-llama/Llama-2-7b-hf" -tp  4 --tokenizer 'hf-internal-testing/llama-tokenizer'
    INFO 08-30 11:41:04 llm_engine.py:196] # GPU blocks: 3278, # CPU blocks: 2048                           │
    SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=1.0, top_p=1.0, │
    top_k=-1, use_beam_search=False, stop=[], ignore_eos=True, max_tokens=128, logprobs=None)               │
    Warming up...                                                                                           │
    Profiling iterations: 100%|███████████████████████████████████████████████| 3/3 [00:14<00:00,  4.78s/it]│
    Avg latency: 4.782389879226685 seconds

    python benchmark_latency.py --model "meta-llama/Llama-2-7b-hf" -tp  4
    INFO 08-30 19:39:25 llm_engine.py:196] # GPU blocks: 3278, # CPU blocks: 2048                           │
    SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=1.0, top_p=1.0, │
    top_k=-1, use_beam_search=False, stop=[], ignore_eos=True, max_tokens=128, logprobs=None)               │
    Warming up...                                                                                           │
    Profiling iterations: 100%|███████████████████████████████████████████████| 3/3 [00:14<00:00,  4.91s/it]│
    Avg latency: 4.913804213205974 seconds
    '''
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
                    'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--out_file', type=str, default="/root/projects/vllm/benchmarks/llama2-7b-latency.json")
    parser.add_argument('--n', type=int, default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters', type=int, default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()

    # import os
    # os.environ['CURL_CA_BUNDLE'] = ''

    std_bsz = 8
    # std_inputlen = 8
    std_inputlen = 32
    std_output_len = 128
    # std_output_len = 256

    # [2, 4, 8, 16, 32, 64]
    all_x_axis = {
                "batch size":[1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 2048], 
                # "batch size":[1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768], 
                # "batch size":[128, 512, 1024, 2048], 
                    "input lens":[32, 64, 128, 256, 512, 1024, 2048, 4096], 
                    "output lens": [128, 256, 512, 1024, 2048, 4096]
                    }

    batch_sizes = [(item, std_inputlen, std_output_len) for item in all_x_axis["batch size"]]
    input_lens = [(std_bsz, item, std_output_len) for item in all_x_axis["input lens"]]
    output_lens = [(std_bsz, std_inputlen, item) for item in all_x_axis["output lens"]]

    latency_dict = {
        "batch size":[],
        "input lens":[],
        "output lens":[]
    }

    metrics = {"batch size":batch_sizes, "input lens":input_lens, "output lens": output_lens}

    # print(batch_sizes)
    # from utils import plot_figure
    import json

    for k, v in metrics.items():

        latencies = []

        for idx, item in enumerate(v):

            bsz, in_len, out_len = item
            args.batch_size = bsz
            args.input_len = in_len
            args.output_len = out_len

            latencies.append(main(args))

            if args.tensor_parallel_size != 1:
                ray.shutdown()

            # plot_figure(all_x_axis[k][0:idx+1], latencies, k, "latency")

            latency_dict[k].append((bsz, in_len, out_len, latencies[-1]))

            # path = "/root/projects/vllm/benchmarks/latency.json"
            path = args.out_file
            with open(path, 'w') as f:
                f.write(json.dumps(latency_dict, indent=4) + '\n')