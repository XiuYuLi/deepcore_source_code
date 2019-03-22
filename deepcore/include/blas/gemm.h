#ifndef __gemm_h__
#define __gemm_h__

#include"../idc_string.h"
#include"../../include/cuda/cuda_context.h"

typedef struct idc_gemmOp{
    cuda_kernel_t kernel;
    uint32_t      ldx;
    uint32_t      dix;
    uint32_t      diy;
} idc_gemmOp_t;


__local_func void idc_gemm_createOp( idc_gemmOp_t*, const cuda_context_t*, uint32_t, int, int, int, int, int, int, int );
__local_func void idc_gemm_createOp_grad( idc_gemmOp_t*, const cuda_context_t*, uint32_t, int, int, int, int, int, int, int );
__local_func void idc_gemm( idc_gemmOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
__local_func void idc_gemm_grad( idc_gemmOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif