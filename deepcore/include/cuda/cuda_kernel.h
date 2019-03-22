#ifndef __cuda_kernel_h__
#define __cuda_kernel_h__

#include<stdint.h>
#include<cuda.h>
#include<vector_types.h>
#include"../idc_macro.h"
#include"../idc_argmask.h"

typedef struct cuda_kernel{
    CUfunction id;
    uint32_t   gdx;
    uint32_t   gdy;
    uint32_t   gdz;
    ushort2    block;
    uint32_t   smemnb;
    uint32_t   arg_size;
    void*      extra[5];
    uint8_t    arg_ofs[16];
    uint8_t    args[128];
} cuda_kernel_t;

INLINE void cuda_kernel_sao( cuda_kernel_t* p, uint32_t mask )
{   
    uint32_t i=0, ofs=0, k=mask;
    do{
        p->arg_ofs[i++]=(uint8_t)ofs;
        if((k&0x3)==PA){
            ofs=IDC_AFFIS(ofs,__alignof(CUdeviceptr)); ofs+=sizeof(CUdeviceptr); 
        } else {
            ofs=IDC_AFFIS(ofs,__alignof(int32_t)); ofs+=__alignof(int32_t);
        }
    }while((k>>=2)!=0);
    p->arg_size=ofs;
}
INLINE void cuda_kernel_set_smemnb( cuda_kernel_t* p_kernel, uint32_t nb )
{
    p_kernel->smemnb=nb;
}
INLINE void cuda_kernel_sgl( cuda_kernel_t* p_kernel, uint32_t gdx, uint32_t gdy, uint32_t gdz )
{
    p_kernel->gdx=gdx; p_kernel->gdy=gdy; p_kernel->gdz=gdz;
}
INLINE void cuda_kernel_sbl( cuda_kernel_t* p_kernel, uint32_t bdx, uint32_t bdy )
{
    p_kernel->block.x=bdx; p_kernel->block.y=bdy;
}
INLINE void cuda_kernel_sep_ptr( cuda_kernel_t* p_kernel, int i, CUdeviceptr p )
{
    *((CUdeviceptr*)&p_kernel->args[p_kernel->arg_ofs[i]])=p;
}
INLINE void cuda_kernel_sep_i32( cuda_kernel_t* p_kernel, int i, int p )
{
    *((int*)&p_kernel->args[p_kernel->arg_ofs[i]])=p;
}
INLINE void cuda_kernel_sep_f32( cuda_kernel_t* p_kernel, int i, float p )
{
    *((float*)&p_kernel->args[p_kernel->arg_ofs[i]])=p;
}
INLINE void cuda_kernel_launch( cuda_kernel_t* p, CUstream s )
{
    cuLaunchKernel( p->id, p->gdx, p->gdy, p->gdz, p->block.x, p->block.y, 1, p->smemnb, s, NULL, (void**)p->extra );
}

#endif
