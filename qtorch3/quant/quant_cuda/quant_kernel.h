#include <stdint.h>

__global__ void fixed_point_quantize_kernel_stochastic(float *__restrict__ a,
                                                       float *__restrict__ r,
                                                       float *o, int size,
                                                       int sigma, bool clamp,
                                                       float t_min, float t_max);

__global__ void fixed_point_quantize_kernel_nearest(float *__restrict__ a,
                                                    float *o, int size,
                                                    int sigma, bool clamp,
                                                    float t_min, float t_max);

__global__ void fixed_point_quantize_kernel_mask_stochastic(float *__restrict__ a,
                                                            float *__restrict__ r,
                                                            float *o, uint8_t *mask,
                                                            int size, int sigma,
                                                            float t_min, float t_max);

__global__ void fixed_point_quantize_kernel_mask_nearest(float *__restrict__ a,
                                                         float *o, uint8_t *mask,
                                                         int size, int sigma,
                                                         float t_min, float t_max);

__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r,
                                        float *o, int size,
                                        int man_bits, int exp_bits);

__global__ void float_kernel_nearest(float *__restrict__ a,
                                     float *o, int size,
                                     int man_bits, int exp_bits);

//----- 
__global__ void float_kernel_nearest_inf(float *__restrict__ a,
                                         float *o, int size,
                                         int man_bits, int exp_bits);
__global__ void float_kernel_nearest_custom_infno(float* __restrict__ a, float* o, int size,
                                                  int man_bits, int exp_bits, float exp_bias_pow, float exp_bias_pow_inv);
__global__ void float_kernel_nearest_custom_infok(float* __restrict__ a, float* o, int size,
                                                  int man_bits, int exp_bits, float exp_bias_pow, float exp_bias_pow_inv);
__global__ void ratio_abs_leq_thrs_cuda_kernel(float* __restrict__ a, float* o, int size, unsigned int thrs_bit);
__global__ void ratio_abs_geq_thrs_cuda_kernel(float* __restrict__ a, float* o, int size, unsigned int thrs_bit);
__global__ void ratio_abs_leq_geq_thrs_cuda_kernel(float* __restrict__ a, float* o, int size, 
                                                   unsigned int thrs_l_bit, unsigned int thrs_g_bit);
//-----

__global__ void block_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r,
                                        float *o, int size,
                                        float *max_entry,
                                        int man_bits);

__global__ void block_kernel_nearest(float *__restrict__ a,
                                     float *o, int size,
                                     float *max_entry,
                                     int man_bits);

__global__ void block_kernel_sim_stochastic(float *__restrict__ a,
                                            float *__restrict__ r,
                                            float *o, int size,
                                            float *max_entry,
                                            int wl);

__global__ void block_kernel_sim_nearest(float *__restrict__ a,
                                         float *o, int size,
                                         float *max_entry,
                                         int wl);
