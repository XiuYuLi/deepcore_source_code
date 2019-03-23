#ifndef __fftconv_h__
#define __fftconv_h__

#include"../../include/fft/fft.h"
#include"../../include/blas/cgemm.h"

typedef struct idc_fftconvOp{
    cuda_kernel_t kfft[3];
    cuda_kernel_t kcgemm;
    size_t        divpt[2];
    uint32_t      ng;
    uint32_t      ags;
    uint32_t      bgs;
    uint32_t      cgs;
} idc_fftconvOp_t;

__local_func int    idc_cellconv_choose_optimal_size( int, int, int, int );
__local_func int    idc_fftconv_choose_optimal_size( int, int, int, int, int, int );

__local_func size_t idc_fftconv_createOp( idc_fftconvOp_t*, const cuda_context_t*, uint32_t, int, int, int, int, int, int, int, int, int, int, int, int, int, int );
__local_func size_t idc_fftconv_createOp_grad( idc_fftconvOp_t*, const cuda_context_t*, int, int, int, int, int, int, int, int, int, int, int, int, int );

__local_func size_t idc_cellconv_createOp( idc_fftconvOp_t*, const cuda_context_t*, uint32_t, int, int, int, int, int, int, int, int, int, int, int, int, int, int );
__local_func size_t idc_cellconv_createOp_grad( idc_fftconvOp_t*, const cuda_context_t*, int, int, int, int, int, int, int, int, int, int, int, int, int );

__local_func void   idc_fftconv( idc_fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
__local_func void   idc_fftconv_grad( idc_fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif
