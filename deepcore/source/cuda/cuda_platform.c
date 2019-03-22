#include"../../include/cuda/cuda_platform.h"
#include"../../include/idc_status.h"

static int __get_devices( CUdevice* const p_device, int* const p_arch )
{   
    struct{ int x, y; } cc;
    CUdevice device;
    int i, n_devices, sm, n_valided;    
    cuDeviceGetCount( &n_devices );
    if( n_devices<=0 ) return -1;
    for( i=0, n_valided=0; i<n_devices; ++i ){
        cuDeviceGet( &device, i );
        cuDeviceGetAttribute( &cc.x, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device );
        cuDeviceGetAttribute( &cc.y, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device );
        sm=10*cc.x+cc.y;
        if((sm<50)||(sm==53)||(sm==62)) continue;
        p_device[n_valided]=device;
        p_arch[n_valided]=sm;
        ++n_valided;
    }
    return n_valided;
}
__local_func int cuda_platform_init( cuda_platform_t* p_platform )
{
    int i;
    if(cuInit(0)!=CUDA_SUCCESS) return idc_error_invalid_driver;
    p_platform->n_devices=__get_devices( &p_platform->devices[0], &p_platform->arch[0] );
    if(p_platform->n_devices<=0) return idc_error_invalid_device;
    for( i=0; i<p_platform->n_devices; ++i ){   
        cuDeviceGetAttribute( &p_platform->nSM[i], CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, p_platform->devices[i] );
    }
    return idc_success;
}
