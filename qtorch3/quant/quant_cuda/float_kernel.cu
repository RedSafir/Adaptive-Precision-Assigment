#include "quant_kernel.h"
#include "bit_helper.cu"
//----- 
#include <cstdio>
//-----

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        int man_bits,
                                        int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) -127; 
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal){
      float shift_float,val;
      int shift_bits = ((127+min_exp)<<23) | (target >> 31 <<31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val=a[index]+shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else{
      quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float* __restrict__ a,
                                     float* o, int size,
                                     int man_bits,
                                     int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) -127; 
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal){
      float shift_float,val;
      int shift_bits = ((127+min_exp)<<23) | (target >> 31 <<31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val=a[index]+shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else{
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

//----- 
// NOTE: return +-inf if overflow occurs, unlike float_kernel_nearest(...).
// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest_inf(float* __restrict__ a,
                                         float* o, int size,
                                         int man_bits,
                                         int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) -127; 
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal){
      float shift_float,val;
      int shift_bits = ((127+min_exp)<<23) | (target >> 31 <<31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val=a[index]+shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else{
      quantize_bits = round_bitwise_nearest(target, man_bits);
      //----- 
      quantize_bits = clip_exponent_inf(exp_bits, man_bits, target, quantize_bits);
      //-----
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

__global__ void float_kernel_nearest_custom_infno(float* __restrict__ a, float* o, int size,
                                                  int man_bits, int exp_bits, float exp_bias_pow, float exp_bias_pow_inv) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    // set: target_bit = a[index] * exp_bias_pow.
    float        target_flt = a[index] * exp_bias_pow;
    unsigned int target_bit = FLOAT_TO_BITS(&target_flt);

    // set: min_exp, subnormal.
    unsigned int target_abs_bit = target_bit & 0x7fffffff;
    int  target_exp = (target_abs_bit >> 23) - 127; //= target's exponent.
    int  min_exp    = -((1 << (exp_bits - 1)) - 2); //= min exponent.
    bool subnormal  = (target_exp < min_exp);       //= if target is subnormal.

    // set: quantize_flt = round(a[index] * exp_bias_pow).
    float quantize_flt;

    if (!subnormal) {
      // quantize = clip(round(target)).
      unsigned int quantize_bit = round_bitwise_nearest(target_bit, man_bits);
      quantize_bit = clip_exponent(exp_bits, man_bits, target_bit, quantize_bit); // ***DON'T USE INFS*** //
      quantize_flt = BITS_TO_FLOAT(&quantize_bit);
    } else {
      // target_shift = target + shift.
      unsigned int shift_bit = ((min_exp + 127) << 23) | (target_bit & 0x80000000); //= min positive non-subnormal num.
      float        shift_flt = BITS_TO_FLOAT(&shift_bit);
      float        target_shift_flt = target_flt + shift_flt;
      unsigned int target_shift_bit = FLOAT_TO_BITS(&target_shift_flt);

      // quantize = round(target + shift) - shift.
      unsigned int quantize_bit = round_bitwise_nearest(target_shift_bit, man_bits);
      quantize_flt = BITS_TO_FLOAT(&quantize_bit) - shift_flt;
    }

    // set: o[index] = round(a[index] * exp_bias_pow) / exp_bias_pow.
    o[index] = quantize_flt * exp_bias_pow_inv;
  }
}

__global__ void float_kernel_nearest_custom_infok(float* __restrict__ a, float* o, int size,
                                                  int man_bits, int exp_bits, float exp_bias_pow, float exp_bias_pow_inv) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    // set: target_bit = a[index] * exp_bias_pow.
    float        target_flt = a[index] * exp_bias_pow;
    unsigned int target_bit = FLOAT_TO_BITS(&target_flt);

    // set: min_exp, subnormal.
    unsigned int target_abs_bit = target_bit & 0x7fffffff;
    int  target_exp = (target_abs_bit >> 23) - 127; //= target's exponent.
    int  min_exp    = -((1 << (exp_bits - 1)) - 2); //= min exponent.
    bool subnormal  = (target_exp < min_exp);       //= if target is subnormal.

    // set: quantize_flt = round(a[index] * exp_bias_pow).
    float quantize_flt;

    if (!subnormal) {
      // quantize = clip(round(target)).
      unsigned int quantize_bit = round_bitwise_nearest(target_bit, man_bits);
      quantize_bit = clip_exponent_inf(exp_bits, man_bits, target_bit, quantize_bit); // ***DO USE INFS*** //
      quantize_flt = BITS_TO_FLOAT(&quantize_bit);
    } else {
      // target_shift = target + shift.
      unsigned int shift_bit = ((min_exp + 127) << 23) | (target_bit & 0x80000000); //= min positive non-subnormal num.
      float        shift_flt = BITS_TO_FLOAT(&shift_bit);
      float        target_shift_flt = target_flt + shift_flt;
      unsigned int target_shift_bit = FLOAT_TO_BITS(&target_shift_flt);

      // quantize = round(target + shift) - shift.
      unsigned int quantize_bit = round_bitwise_nearest(target_shift_bit, man_bits);
      quantize_flt = BITS_TO_FLOAT(&quantize_bit) - shift_flt;
    }

    // set: o[index] = round(a[index] * exp_bias_pow) / exp_bias_pow.
    o[index] = quantize_flt * exp_bias_pow_inv;
  }
}

__global__ void ratio_abs_leq_thrs_cuda_kernel(float* __restrict__ a, float* o, int size, unsigned int thrs_bit) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    // set: target_abs = |a[index]|.
    float        target_flt = a[index];
    unsigned int target_abs_bit = FLOAT_TO_BITS(&target_flt) & 0x7fffffff;

    // set: o.
    bool flag = (target_abs_bit <= thrs_bit) && (target_abs_bit != 0x0);
    unsigned int res_bit = (-flag) & 0x3f800000;
    o[index] = BITS_TO_FLOAT(&res_bit);
  }
}
__global__ void ratio_abs_geq_thrs_cuda_kernel(float* __restrict__ a, float* o, int size, unsigned int thrs_bit) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    // set: target_abs = |a[index]|.
    float        target_flt = a[index];
    unsigned int target_abs_bit = FLOAT_TO_BITS(&target_flt) & 0x7fffffff;

    // set: o.
    bool flag = (target_abs_bit >= thrs_bit);
    unsigned int res_bit = (-flag) & 0x3f800000;
    o[index] = BITS_TO_FLOAT(&res_bit);
  }
}

__global__ void ratio_abs_leq_geq_thrs_cuda_kernel(float* __restrict__ a, float* o, int size, 
                                                   unsigned int thrs_l_bit, unsigned int thrs_g_bit) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    // set: target_abs = |a[index]|.
    float        target_flt = a[index];
    unsigned int target_abs_bit = FLOAT_TO_BITS(&target_flt) & 0x7fffffff;

    // set: o.
    bool         flag_l     = (target_abs_bit <= thrs_l_bit) && (target_abs_bit != 0x0);
    bool         flag_g     = (target_abs_bit >= thrs_g_bit);
    unsigned int flag_l_bit = (-flag_l) & 0x3f800000; //= 1.0 if flag_l else 0.0.
    unsigned int flag_g_bit = (-flag_g) & 0x3f800000; //= 1.0 if flag_g else 0.0.
    // o[index     ] = BITS_TO_FLOAT(&flag_l_bit);
    // o[index+size] = BITS_TO_FLOAT(&flag_g_bit);
    o[index*2  ] = BITS_TO_FLOAT(&flag_l_bit); // o[index*2  ] = float(flag_l);
    o[index*2+1] = BITS_TO_FLOAT(&flag_g_bit); // o[index*2+1] = float(flag_g);
  }
}
//-----
