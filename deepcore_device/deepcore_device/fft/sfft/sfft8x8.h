__device__ __forceinline__ void s_vfft8( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[4];
    FFT4(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<4; ++i )
    {
        temp=s_RF[i];
        d[i].x = c[brev[  i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[  i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[  i]].y*temp.x;
        d[i].y+=-c[brev[  i]].x*temp.x;
        d[i].x+= c[brev[4-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[4-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[4-i]].x*temp.x;
        d[i].x+= c[brev[4-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE4(sst,d,10,.x) sync
    LOAD4(c,sld,2,.x)   sync
    STORE4(sst,d,10,.y) sync
    LOAD4(c,sld,2,.y)
}
__device__ __forceinline__ void s_hfft8( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev, int x, int xu )
{
    float2 temp;
    FFT4(c,)
#pragma unroll
    for( int i=1; i<4; ++i ){
        c[i]=s_cmul(c[i],s_RF[xu*brev[i]]);
    }
    float sign=xu?-1.f:1.f;
#pragma unroll
    for( int i=0; i<4; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,8);
        temp.y=SHFL_XOR(c[i].y,1,8);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    PERMUTE(4,sst,sld,c,1,10,perm_mask)
    if(x>0){
        temp.x=0.5f*SHFL(c[0].x,8-x,8);
        temp.y=0.5f*SHFL(c[0].y,8-x,8);
        c[4].x=( 0.5f)*c[0].y+temp.y;
        c[4].y=(-0.5f)*c[0].x+temp.x;
        c[0].x=0.5f*c[0].x+( temp.x);
        c[0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[4].x=c[0].y;
        c[4].y=c[0].y=0.f;
    }
}
__device__ __forceinline__ void s_store5( float2* d_c, float* sst, float* sld, float2* c, unsigned int ldc )
{
    __syncthreads();
#pragma unroll
    for( int i=0; i<5; ++i ){ sst[i*8]=c[i].x; } __syncthreads();
#pragma unroll
    for( int i=0; i<5; ++i ){ c[i].x=sld[i*8]; } __syncthreads();
#pragma unroll
    for( int i=0; i<5; ++i ){ sst[i*8]=c[i].y; } __syncthreads();
#pragma unroll
    for( int i=0; i<5; ++i ){ c[i].y=sld[i*8]; }
#pragma unroll
    for( int i=0; i<5; ++i ){ d_c[0]=c[i]; d_c+=ldc; }
}
__device__ __forceinline__ void s_load5( float2* c, float* sst, float* sld, const float2* d_c, unsigned int ldc )
{
#pragma unroll
    for( int i=0; i<5; ++i ){ c[i]=d_c[0]; d_c+=ldc; }
#pragma unroll
    for( int i=0; i<5; ++i ){ sst[i*8]=c[i].x; } __syncthreads();
#pragma unroll
    for( int i=0; i<5; ++i ){ c[i].x=sld[i*8]; } __syncthreads();
#pragma unroll
    for( int i=0; i<5; ++i ){ sst[i*8]=c[i].y; } __syncthreads();
#pragma unroll
    for( int i=0; i<5; ++i ){ c[i].y=sld[i*8]; } __syncthreads();
}
__device__ __forceinline__ void s_hifft8( float2* c, float2* d, float* sptr, const float2* s_RF, const int* brev, unsigned int x )
{
    float2 temp;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    float* spx=&sptr[x];
    float* spy=&sptr[v*10+u];
    float* spz=&sptr[v*11+u*4];
    if(x>0){
        d[0].x-=d[4].y; 
        d[0].y+=d[4].x;
    } else {
        d[0].y=d[4].x;
    }
    STORE4(spx,d,10,.x) sync
    LOAD4(c,spy,2,.x)   sync
    STORE4(spx,d,10,.y) sync
    LOAD4(c,spy,2,.y)   sync
    FFT4(c,i)
#pragma unroll
    for( int i=1; i<4; ++i ){
        c[i]=s_icmul(c[i],s_RF[u*brev[i]]);
    }
    float sign=u?-1.f:1.f;
#pragma unroll
    for( int i=0; i<4; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,8);
        temp.y=SHFL_XOR(c[i].y,1,8);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    PERMUTE(4,spz,spx,c,1,11,perm_mask)
}
__device__ __forceinline__ void s_vifft8( float2* d, float2* c, const float2* s_RF, const int* brev, int x )
{
    float2 a, b, temp=c[0];
    float sign=((x&3)>0)?1.f:-1.f;
    c[0].x=-sign*temp.y+temp.x;
    c[0].y= sign*temp.y+temp.x;
#pragma unroll
    for( int i=1; i<2; ++i ){
        temp=s_RF[i];
        a.x=c[i].x+c[4-i].x;
        a.y=c[i].y-c[4-i].y;
        b.x=c[i].y+c[4-i].y;
        b.y=c[i].x-c[4-i].x;
        c[  i].x=(-b.x)*temp.x+(( b.y)*temp.y+( a.x));
        c[  i].y=( b.y)*temp.x+(( b.x)*temp.y+( a.y));
        c[4-i].x=( b.x)*temp.x+((-b.y)*temp.y+( a.x));
        c[4-i].y=( b.y)*temp.x+(( b.x)*temp.y+(-a.y));
    }
    c[2].x*= 2.0f;
    c[2].y*=-2.0f;
    FFT4(c,i)
#pragma unroll
    for( int i=0; i<4; ++i ){ d[i]=c[brev[i]]; }
    if((x&3)==0){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            d[i].x=c[brev[(4-i)%4]].x;
            d[i].y=c[brev[(3-i)%4]].y;
        }
    }
}
__device__ __forceinline__ void sfft8x8_c2r_store( float* dst, float2* c, const float* null, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((2*i+0)<ny){ *dst=alpha*c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=alpha*c[i].y; dst+=nx; }
        }
    }
}
__device__ __forceinline__ void sfft8x8_c2r_store_relu( float* dst, float2* c, const float* null, float alpha, bool bc, int nx, int ny )
{
#pragma unroll
    for( int i=0; i<4; ++i ){
        c[i].x=s_relu(alpha*c[i].x); 
        c[i].y=s_relu(alpha*c[i].y); 
    }
    if(bc){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft8x8_c2r_store_bias( float* dst, float2* c, const float* bias, float alpha, bool bc, int nx, int ny )
{
    float b=*bias;
#pragma unroll
    for( int i=0; i<4; ++i ){
        c[i].x=alpha*c[i].x+b; 
        c[i].y=alpha*c[i].y+b;
    }
    if(bc){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft8x8_c2r_store_bias_relu( float* dst, float2* c, const float* bias, float alpha, bool bc, int nx, int ny )
{
    float b=*bias;
#pragma unroll
    for( int i=0; i<4; ++i ){
        c[i].x=s_relu(alpha*c[i].x+b); 
        c[i].y=s_relu(alpha*c[i].y+b); 
    }
    if(bc){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft8x8_c2r_store_drelu( float* dst, float2* c, const float* a, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((2*i+0)<ny){ *dst=alpha*c[i].x*s_drelu(a[0]); } a+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=alpha*c[i].y*s_drelu(a[0]); } a+=nx; dst+=nx; 
        }
    }
}
__device__ __forceinline__ void sfft8x8_c2r_store_xdrv( float* dst, float2* c, const float* da, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((2*i+0)<ny){ *dst=alpha*c[i].x*da[0]; } da+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=alpha*c[i].y*da[0]; } da+=nx; dst+=nx; 
        }
    }
}

#include"sfft8x8_r2c.h"
#include"sfft8x8_c2r.h"
#include"sfft8x8_c2r_grad.h"
#include"sfft8x8_r2c_perm2d.h"
#include"sfft8x8_c2r_perm2d.h"
#include"sfft8x8_r2c_perm3d.h"
#include"sfft8x8_c2r_perm3d.h"
#include"sfft8x8_r2c_split.h"
#include"sfft8x8_c2r_splice.h"
#include"sfft8x8_r2c_split_perm.h"
#include"sfft8x8_c2r_splice_perm.h"
#include"sfft8x8_c2r_grad_perm.h"
