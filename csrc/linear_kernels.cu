#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace vllm {

template <typename scalar_t>
__global__ void gemm_kernel(scalar_t *__restrict__ out,         // [M, N]
                            const scalar_t *__restrict__ input, // [M, K]
                            const scalar_t *__restrict__ weight, // [K, N]
                            int M,
                            int N, int K) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

	int row = bidx / N;
	int col = bidx % N; 

  int bdimx = blockDim.x;

	static __shared__ scalar_t mem[1024];

  scalar_t tmp = 0;
  for (int i = tidx; i < K; i += bdimx) {
		tmp += input[row * K + i] * weight[i * N + col];
  }

	mem[tidx] = tmp;

	__syncthreads();
}

} // namespace vllm

void linear(torch::Tensor &out,   // [M, N]
            torch::Tensor &input, // [M, K]
            torch::Tensor &weight // [K, N]
) {
  int M = input.size(0);
  int K = input.size(1);
  int N = weight.size(1);

  dim3 grid(M * N);
  dim3 block(std::min(K, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "linear",
      [&] {
        vllm::gemm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            M, N, K);
      });
}
