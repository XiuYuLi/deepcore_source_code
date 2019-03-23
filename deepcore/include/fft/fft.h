#ifndef __fft_h__
#define __fft_h__

#include"../../include/cuda/cuda_context.h"

__local_func void idc_create_fft_kernel_r2c( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int );
__local_func void idc_create_fft_kernel_r2c_opt( cuda_kernel_t*, const cuda_context_t*, int, int, int, int );
__local_func void idc_create_fft_kernel_c2r( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int );
__local_func void idc_create_fft_kernel_c2r_grad( cuda_kernel_t*, const cuda_context_t*, int, int );
__local_func void idc_create_fft_kernel_c2r_grad_opt( cuda_kernel_t*, const cuda_context_t*, int, int, int );
__local_func void idc_create_cellfft_kernel_r2c( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int );
__local_func void idc_create_cellfft_kernel_r2c_opt( cuda_kernel_t*, const cuda_context_t*, int, int, int, int );
__local_func void idc_create_cellfft_kernel_c2r( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int );
__local_func void idc_create_cellfft_kernel_c2r_grad( cuda_kernel_t*, const cuda_context_t*, int, int );
__local_func void idc_create_cellfft_kernel_c2r_grad_opt( cuda_kernel_t*, const cuda_context_t*, int, int, int );

INLINE void idc_fft2d_r2c( cuda_kernel_t* p_kernel, CUdeviceptr d_dst, CUdeviceptr d_src, CUstream s )
{
    cuda_kernel_sep_ptr( p_kernel, 0, d_dst );
    cuda_kernel_sep_ptr( p_kernel, 1, d_src );
    cuda_kernel_launch( p_kernel, s );
}
INLINE void idc_fft2d_c2r( cuda_kernel_t* p_kernel, CUdeviceptr d_dst, CUdeviceptr d_src, CUdeviceptr d_x, float alpha, CUstream s )
{
    cuda_kernel_sep_ptr( p_kernel, 0, d_dst );
    cuda_kernel_sep_ptr( p_kernel, 1, d_src );
    cuda_kernel_sep_ptr( p_kernel, 3, d_x   );
    cuda_kernel_sep_f32( p_kernel, 4, alpha );
    cuda_kernel_launch( p_kernel, s );
}
INLINE void idc_fft2d_c2r_grad( cuda_kernel_t* p_kernel, CUdeviceptr d_dst, CUdeviceptr d_src, float scale, CUstream s )
{
    cuda_kernel_sep_ptr( p_kernel, 0, d_dst );
    cuda_kernel_sep_ptr( p_kernel, 1, d_src );
    cuda_kernel_sep_f32( p_kernel, 3, scale );
    cuda_kernel_launch( p_kernel, s );
}

#endif
