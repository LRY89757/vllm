#include <torch/extension.h>

void linear(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"linear",
		&linear,
		"Apply Linear to the input tensor");
}