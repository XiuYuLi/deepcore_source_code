#ifndef __cgemm_h__
#define __cgemm_h__

#include"../../include/cuda/cuda_context.h"
#include"../idc_string.h"
#include"../idc_bop.h"

__local_func void idc_cgemm_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int );
__local_func void idc_cgemv_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int );
__local_func void idc_cgevv_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int );
__local_func void idc_flatcgemm_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int );
__local_func void idc_flatcgevv_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int );
__local_func void idc_cgemm( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUstream );

#endif