#pragma warning( disable:4996 )
#include"../../include/cuda/cuda_context.h"

static const unsigned int long long kbin_fftconv_sm50[]=
{
#include"../../include/dev/fftconv/kbin_sm50.h"
};
static const unsigned int long long kbin_fftconv_sm52[]=
{
#include"../../include/dev/fftconv/kbin_sm52.h"
};
static const unsigned int long long kbin_fftconv_sm60[]=
{
#include"../../include/dev/fftconv/kbin_sm60.h"
};
static const unsigned int long long kbin_fftconv_sm61[]=
{
#include"../../include/dev/fftconv/kbin_sm61.h"
};
static const unsigned int long long kbin_fftconv_sm70[]=
{
#include"../../include/dev/fftconv/kbin_sm70.h"
};

static const unsigned int long long kbin_blas_sm50[]=
{
#include"../../include/dev/blas/kbin_sm50.h"
};
static const unsigned int long long kbin_blas_sm52[]=
{
#include"../../include/dev/blas/kbin_sm52.h"
};
static const unsigned int long long kbin_blas_sm60[]=
{
#include"../../include/dev/blas/kbin_sm60.h"
};
static const unsigned int long long kbin_blas_sm61[]=
{
#include"../../include/dev/blas/kbin_sm61.h"
};
static const unsigned int long long kbin_blas_sm70[]=
{
#include"../../include/dev/blas/kbin_sm70.h"
};

static const unsigned int long long* p_devbin[][4]=
{
    { kbin_fftconv_sm50, kbin_blas_sm50 },
    { kbin_fftconv_sm52, kbin_blas_sm52 },
    { kbin_fftconv_sm61, kbin_blas_sm61 },
    { kbin_fftconv_sm60, kbin_blas_sm60 },
    { kbin_fftconv_sm70, kbin_blas_sm70 }
};

/****************************************************************************************************************************************************************
=================================================================================================================================================================
****************************************************************************************************************************************************************/

__local_func int cuda_context_create( cuda_context_t* p_ctx, char* p_temp )
{
    int i, n, p, q; 
    cuDriverGetVersion(&i);
    if(i<9010)
        return idc_error_invalid_driver;
    cuCtxGetCurrent( &p_ctx->ctx );
    p_ctx->status=0;
    if(p_ctx!= NULL){ p_ctx->status=1; }
    cuDevicePrimaryCtxRetain( &p_ctx->ctx, p_ctx->dev );
    cuDevicePrimaryCtxSetFlags( p_ctx->dev, CU_CTX_SCHED_AUTO|CU_CTX_MAP_HOST|CU_CTX_LMEM_RESIZE_TO_MAX );
    cuCtxPushCurrent( p_ctx->ctx );
    switch(p_ctx->arch)
    {
    case 50: i=0; break;
    case 52: i=1; break;
    case 61: i=2; break;
    case 60: i=3; break;
    case 70: i=4; break;
    }    
    cuModuleLoadFatBinary( &p_ctx->module_fftconv, p_devbin[i][0] );
    cuModuleLoadFatBinary( &p_ctx->module_blas   , p_devbin[i][1] );
    n=sizeof(g_fftco)/sizeof(g_fftco[0])-1;
    if(cuMemAlloc(&p_ctx->d_RF, g_fftco[n]*8)!=CUDA_SUCCESS){
        cuModuleUnload( p_ctx->module_fftconv );
        cuModuleUnload( p_ctx->module_blas    );
        cuDevicePrimaryCtxRelease(p_ctx->dev);
        p_ctx->ctx=NULL;
        return idc_error_out_of_memory;
    }
    for( p=g_fftco[(i=0)]; i<n; p=q ){
        q=g_fftco[++i];
        idc_fft_calcRF( ((float2*)p_temp)+p, q-p, 1.0/(q-p) );
    }
    cuMemcpyHtoD( p_ctx->d_RF, p_temp, g_fftco[n]*8 );
    cuDeviceGetAttribute( &p_ctx->n_sm                , CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT       , p_ctx->dev );
    cuDeviceGetAttribute( &p_ctx->cmemnb              , CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY      , p_ctx->dev );
    cuDeviceGetAttribute( &p_ctx->max_smemnb_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, p_ctx->dev );
    cuCtxPopCurrent(NULL);
    return idc_success;
}
__local_func int cuda_context_get_current( const cuda_context_t* p_ctx, int n_devices )
{
    CUcontext ctx;
    int i=n_devices-1;
    do{
        cuCtxGetCurrent( &ctx );
        if(p_ctx[i].ctx==ctx) break;
    }while((--i)>=0);
    return (i<n_devices?i:-1);
}
__local_func void cuda_context_release( cuda_context_t* p_ctx )
{
    if(p_ctx->ctx!=NULL){
        cuda_context_bind( p_ctx );
        cuMemFree( p_ctx->d_RF ); 
        cuModuleUnload( p_ctx->module_fftconv );
        cuModuleUnload( p_ctx->module_blas    );
        if(p_ctx->status==0){
            cuDevicePrimaryCtxRelease( p_ctx->dev );
        }
        p_ctx->ctx=NULL;
    }
}
