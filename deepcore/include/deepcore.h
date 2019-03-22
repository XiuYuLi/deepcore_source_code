#ifndef __deepcore_h__
#define __deepcore_h__

#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
  #ifdef _MSC_VER
    #ifdef _DLL
      #define DEEPCOREAPIENTRY __declspec(dllexport)
    #else
      #define DEEPCOREAPIENTRY __declspec(dllimport)
    #endif
  #elif defined(__GNUC__)
    #if __GNUC__>=4
      #define DEEPCOREAPIENTRY __attribute__((visibility("default")))
    #else
      #define DEEPCOREAPIENTRY
    #endif
  #else
    #define DEEPCOREAPIENTRY
  #endif
#else
  #error "Only 64bits-OS supported!"
#endif

#include<stdint.h>
#include<cuda.h>

#define dcMaskDirectionForward  0x00000000
#define dcMaskDirectionBackward 0x00000001
#define dcMaskPrecisionFloat    0x00000000
#define dcMaskPrecisionMixed    0x00000002
#define dcMaskAddBias           0x00000008
#define dcMaskMulDrv            0x00000008
#define dcMaskActivationRelu    0x01000000

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct __dc_fftconvOp  * dc_fftconvOp;
typedef struct __dc_cellconvOp * dc_cellconvOp;
typedef struct __dc_gemmOp     * dc_gemmOp;
#ifdef _MSC_VER
  typedef unsigned __int64         dc_tensorshape_t;
#else
  typedef unsigned long long int   dc_tensorshape_t;
#endif

typedef enum dc_status{
    dc_success=0                 ,
    dc_error_invalid_device      ,
    dc_error_invalid_driver      ,
    dc_error_invalid_value       ,
    dc_error_no_active_device    ,
    dc_error_out_of_memory       ,
    dc_error_out_of_device_memory,
    dc_error_out_of_range        ,
    dc_error_mutually_exclusive  ,
    dc_error_mismatch            ,
    dc_error_unsupported
} dc_status_t;

DEEPCOREAPIENTRY dc_status_t      dc_init();
DEEPCOREAPIENTRY int              dc_get_device_count();
DEEPCOREAPIENTRY dc_status_t      dc_set_device( int );
DEEPCOREAPIENTRY dc_tensorshape_t dc_create_tensor_shape( int, uint32_t, uint32_t );
DEEPCOREAPIENTRY dc_tensorshape_t dc_create_tensor_shape_filter( int, uint32_t, uint32_t );
DEEPCOREAPIENTRY dc_tensorshape_t dc_create_tensor_shape_linear( size_t );
DEEPCOREAPIENTRY dc_status_t      dc_create_tensor( void**, dc_tensorshape_t );
DEEPCOREAPIENTRY dc_status_t      dc_release_tensor( void* );
DEEPCOREAPIENTRY dc_status_t      dc_tensor_zero( void*, dc_tensorshape_t, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_tensor_subzero( void*, dc_tensorshape_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_tensor_copy( void*, dc_tensorshape_t, const void*, dc_tensorshape_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_tensor_subcopy( void*, dc_tensorshape_t, const void*, dc_tensorshape_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_tensor_store( void*, dc_tensorshape_t, const void*, size_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_tensor_load( void*, size_t, const void*, dc_tensorshape_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_create_fftconvOp( dc_fftconvOp*, size_t*, uint32_t, int, dc_tensorshape_t, dc_tensorshape_t, dc_tensorshape_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t      dc_create_fftconvOp_grad( dc_fftconvOp*, size_t*, uint32_t, int, dc_tensorshape_t, dc_tensorshape_t, dc_tensorshape_t );
DEEPCOREAPIENTRY dc_status_t      dc_create_cellconvOp( dc_cellconvOp*, size_t*, uint32_t, int, dc_tensorshape_t, dc_tensorshape_t, dc_tensorshape_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t      dc_create_cellconvOp_grad( dc_cellconvOp*, size_t*, uint32_t, int, dc_tensorshape_t, dc_tensorshape_t, dc_tensorshape_t );
DEEPCOREAPIENTRY dc_status_t      dc_create_gemmOp( dc_gemmOp*, uint32_t, int, dc_tensorshape_t, dc_tensorshape_t, dc_tensorshape_t );
DEEPCOREAPIENTRY dc_status_t      dc_create_gemmOp_grad( dc_gemmOp*, uint32_t, int, dc_tensorshape_t, dc_tensorshape_t, dc_tensorshape_t );
DEEPCOREAPIENTRY dc_status_t      dc_fftconv( dc_fftconvOp, void*, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_fftconv_grad( dc_fftconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_cellconv( dc_cellconvOp, void*, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_cellconv_grad( dc_cellconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_gemm( dc_gemmOp, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_gemm_grad( dc_gemmOp, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t      dc_destroy_fftconvOp( dc_fftconvOp );
DEEPCOREAPIENTRY dc_status_t      dc_destroy_cellconvOp( dc_cellconvOp );
DEEPCOREAPIENTRY dc_status_t      dc_destroy_gemmOp( dc_gemmOp );
DEEPCOREAPIENTRY dc_status_t      dc_exit();

#ifdef __cplusplus
}
#endif

#endif
