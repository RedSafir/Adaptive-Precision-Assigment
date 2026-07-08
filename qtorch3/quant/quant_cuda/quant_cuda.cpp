#include <torch/torch.h>
#include "quant_cuda.h"
#include <tuple>

using namespace at;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl, bool use_clamp, bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_nearest_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl,
                                                             bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_nearest_mask_cuda(a, wl, fl, symmetric);
}

Tensor block_quantize_nearest(Tensor a, int wl, int dim)
{
  CHECK_INPUT(a);
  return block_quantize_nearest_cuda(a, wl, dim);
}

Tensor block_quantize_sim_nearest(Tensor a, int wl)
{
  CHECK_INPUT(a);
  return block_quantize_sim_nearest_cuda(a, wl);
}

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits)
{
  CHECK_INPUT(a);
  return float_quantize_nearest_cuda(a, man_bits, exp_bits);
}

//----- 
// NOTE: return +-inf if overflow occurs, unlike float_quantize_nearest().
Tensor float_quantize_nearest_inf(Tensor a, int man_bits, int exp_bits)
{
  CHECK_INPUT(a);
  //----- 
  return float_quantize_nearest_inf_cuda(a, man_bits, exp_bits);
  //-----
}

Tensor float_quantize_nearest_custom(Tensor a, int man_bits, int exp_bits, float exp_bias_pow, bool allow_inf)
{
  CHECK_INPUT(a);
  return float_quantize_nearest_custom_cuda(a, man_bits, exp_bits, exp_bias_pow, allow_inf);
}

Tensor ratio_abs_leq_thrs(Tensor a, float thrs)
{
  CHECK_INPUT(a);
  return ratio_abs_leq_thrs_cuda(a, thrs);
}

Tensor ratio_abs_geq_thrs(Tensor a, float thrs)
{
  CHECK_INPUT(a);
  return ratio_abs_geq_thrs_cuda(a, thrs);
}

Tensor ratio_abs_leq_geq_thrs(Tensor a, float thrs_l, float thrs_g)
{
  CHECK_INPUT(a);
  return ratio_abs_leq_geq_thrs_cuda(a, thrs_l, thrs_g);
}
//-----

Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl, bool use_clamp, bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_stochastic_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl,
                                                                bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_stochastic_mask_cuda(a, wl, fl, symmetric);
}

Tensor block_quantize_stochastic(Tensor a, int wl, int dim)
{
  CHECK_INPUT(a);
  return block_quantize_stochastic_cuda(a, wl, dim);
}

Tensor block_quantize_sim_stochastic(Tensor a, int wl)
{
  CHECK_INPUT(a);
  return block_quantize_sim_stochastic_cuda(a, wl);
}

Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits)
{
  CHECK_INPUT(a);
  return float_quantize_stochastic_cuda(a, man_bits, exp_bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic, "Fixed Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_stochastic_mask", &fixed_point_quantize_stochastic_mask, "Fixed Point Number Stochastic Quantization (CUDA)");
  m.def("block_quantize_stochastic", &block_quantize_stochastic, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("block_quantize_sim_stochastic", &block_quantize_sim_stochastic, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("float_quantize_stochastic", &float_quantize_stochastic, "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest, "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest_mask", &fixed_point_quantize_nearest_mask, "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("block_quantize_nearest", &block_quantize_nearest, "Block Floating Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("block_quantize_sim_nearest", &block_quantize_sim_nearest, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("float_quantize_nearest", &float_quantize_nearest, "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CUDA)");
  //----- 
  m.def("float_quantize_nearest_inf", &float_quantize_nearest_inf, "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CUDA, inf)");
  m.def("float_quantize_nearest_custom", &float_quantize_nearest_custom,
        "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CUDA, with {exp_bias_pow,allow_inf}, compute {under,over}flow_ratio)");
  m.def("ratio_abs_leq_thrs", &ratio_abs_leq_thrs, "#{i : |t_i| <= thrs and t_i != 0}");
  m.def("ratio_abs_geq_thrs", &ratio_abs_geq_thrs, "#{i : |t_i| >= thrs}");
  m.def("ratio_abs_leq_geq_thrs", &ratio_abs_leq_geq_thrs, "#{i : |t_i| <= thrs and t_i != 0}, #{i : |t_i| >= thrs}");
  //-----
}
