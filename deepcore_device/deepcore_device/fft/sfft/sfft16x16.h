__device__ __forceinline__ void s_vfft16( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[8];
    FFT8(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<8; ++i )
    {
        temp=s_RF[i];
        d[i].x = c[brev[  i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[  i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[  i]].y*temp.x;
        d[i].y+=-c[brev[  i]].x*temp.x;
        d[i].x+= c[brev[8-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[8-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[8-i]].x*temp.x;
        d[i].x+= c[brev[8-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE8(sst,d,18,.x) sync
    LOAD8(c,sld,2,.x)   sync
    STORE8(sst,d,18,.y) sync
    LOAD8(c,sld,2,.y)
}
__device__ __forceinline__ void s_hfft16( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev, int x, int xu )
{
    float2 temp;
    FFT8(c,)
#pragma unroll
    for( int i=1; i<8; ++i ){
        c[i]=s_cmul(c[i],s_RF[xu*brev[i]]);
    }
    float sign=xu?-1.f:1.f;
#pragma unroll
    for( int i=0; i<8; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,16);
        temp.y=SHFL_XOR(c[i].y,1,16);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    PERMUTE(8,sst,sld,c,1,18,perm_mask)
    if(x>0){
        temp.x=0.5f*SHFL(c[0].x,16-x,16);
        temp.y=0.5f*SHFL(c[0].y,16-x,16);
        c[8].x=( 0.5f)*c[0].y+temp.y;
        c[8].y=(-0.5f)*c[0].x+temp.x;
        c[0].x=0.5f*c[0].x+( temp.x);
        c[0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[8].x=c[0].y;
        c[8].y=c[0].y=0.f;
    }
}
__device__ __forceinline__ void s_store9( float2* d_c, float* sst, float* sld, float2* c, unsigned int ldc )
{
    __syncthreads();
#pragma unroll
    for( int i=0; i<9; ++i ){
        sst[i*16]=c[i].x;
    } __syncthreads();
#pragma unroll
    for( int i=0; i<9; ++i ){
        c[i].x=sld[i*16];
    } __syncthreads();
#pragma unroll
    for( int i=0; i<9; ++i ){
        sst[i*16]=c[i].y;
    } __syncthreads();
#pragma unroll
    for( int i=0; i<9; ++i ){
        c[i].y=sld[i*16];
    }
#pragma unroll
    for( int i=0; i<9; ++i ){
        d_c[0]=c[i]; d_c+=ldc;
    }
}
__device__ __forceinline__ void s_load9( float2* c, float* sst, float* sld, const float2* d_c, unsigned int ldc )
{
#pragma unroll
    for( int i=0; i<9; ++i ){ c[i]=d_c[0]; d_c+=ldc; }
#pragma unroll
    for( int i=0; i<9; ++i ){ sst[i*16]=c[i].x; }
    __syncthreads();
#pragma unroll
    for( int i=0; i<9; ++i ){ c[i].x=sld[i*16]; }
    __syncthreads();
#pragma unroll
    for( int i=0; i<9; ++i ){ sst[i*16]=c[i].y; }
    __syncthreads();
#pragma unroll
    for( int i=0; i<9; ++i ){ c[i].y=sld[i*16]; }
    __syncthreads();
}
__device__ __forceinline__ void s_hifft16( float2* c, float2* d, float* sptr, const float2* s_RF, const int* brev, unsigned int x )
{
    float2 temp;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    float* spx=&sptr[x];
    float* spy=&sptr[v*18+u];
    float* spz=&sptr[v*18+u*8];
    if(x>0){
        d[0].x-=d[8].y; 
        d[0].y+=d[8].x;
    } else {
        d[0].y=d[8].x;
    }
    STORE8(spx,d,18,.x) sync
    LOAD8(c,spy,2,.x)   sync
    STORE8(spx,d,18,.y) sync
    LOAD8(c,spy,2,.y)   sync
    FFT8(c,i)
#pragma unroll
    for( int i=1; i<8; ++i ){
        c[i]=s_icmul(c[i],s_RF[u*brev[i]]);
    }
    float sign=u?-1.f:1.f;
#pragma unroll
    for( int i=0; i<8; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,16);
        temp.y=SHFL_XOR(c[i].y,1,16);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    PERMUTE(8,spz,spx,c,1,18,perm_mask)
}
__device__ __forceinline__ void s_vifft16( float2* d, float2* c, const float2* s_RF, const int* brev, int x )
{
    float2 a, b, temp=c[0];
    float sign=((x&7)>0)?1.f:-1.f;
    c[0].x=-sign*temp.y+temp.x;
    c[0].y= sign*temp.y+temp.x;
#pragma unroll
    for( int i=1; i<4; ++i )
    {
        temp=s_RF[i];
        a.x=c[i].x+c[8-i].x;
        a.y=c[i].y-c[8-i].y;
        b.x=c[i].y+c[8-i].y;
        b.y=c[i].x-c[8-i].x;
        c[  i].x=(-b.x)*temp.x+(( b.y)*temp.y+( a.x));
        c[  i].y=( b.y)*temp.x+(( b.x)*temp.y+( a.y));
        c[8-i].x=( b.x)*temp.x+((-b.y)*temp.y+( a.x));
        c[8-i].y=( b.y)*temp.x+(( b.x)*temp.y+(-a.y));
    }
    c[4].x*= 2.0f;
    c[4].y*=-2.0f;
    FFT8(c,i)
#pragma unroll
    for( int i=0; i<8; ++i ){ d[i]=c[brev[i]]; }
    if((x&7)==0)
    {
    #pragma unroll
        for( int i=0; i<8; ++i ){
            d[i].x=c[brev[(8-i)%8]].x;
            d[i].y=c[brev[(7-i)%8]].y;
        }
    }
}
__device__ __forceinline__ void s_vfft16_s3( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[8];
    FFT8_M2(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<8; ++i )
    {
        temp=s_RF[i];
        d[i].x = c[brev[  i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[  i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[  i]].y*temp.x;
        d[i].y+=-c[brev[  i]].x*temp.x;
        d[i].x+= c[brev[8-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[8-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[8-i]].x*temp.x;
        d[i].x+= c[brev[8-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE8(sst,d,18,.x)
    sync
    c[0].x=sld[0];
    c[1].x=sld[2];
    sync
    STORE8(sst,d,18,.y)
    sync
    c[0].y=sld[0];
    c[1].y=sld[2];
}
__device__ __forceinline__ void s_hfft16_s3( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev, int x, int xu )
{
    float2 temp;
    FFT8_M2(c,)
#pragma unroll
    for( int i=1; i<8; ++i ){
        c[i]=s_cmul(c[i],s_RF[xu*brev[i]]);
    }
    float sign=xu?-1.f:1.f;
#pragma unroll
    for( int i=0; i<8; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,16);
        temp.y=SHFL_XOR(c[i].y,1,16);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    PERMUTE(8,sst,sld,c,1,18,perm_mask)
    if(x>0){
        temp.x=0.5f*SHFL(c[0].x,16-x,16);
        temp.y=0.5f*SHFL(c[0].y,16-x,16);
        c[8].x=( 0.5f)*c[0].y+temp.y;
        c[8].y=(-0.5f)*c[0].x+temp.x;
        c[0].x=0.5f*c[0].x+( temp.x);
        c[0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[8].x=c[0].y;
        c[8].y=c[0].y=0.f;
    }
}
__device__ __forceinline__ void s_vfft16_s5( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[8];
    FFT8_M3(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<8; ++i )
    {
        temp=s_RF[i];
        d[i].x = c[brev[  i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[  i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[  i]].y*temp.x;
        d[i].y+=-c[brev[  i]].x*temp.x;
        d[i].x+= c[brev[8-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[8-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[8-i]].x*temp.x;
        d[i].x+= c[brev[8-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE8(sst,d,18,.x)
    sync
    c[0].x=sld[0];
    c[1].x=sld[2];
    c[2].x=sld[4];
    sync
    STORE8(sst,d,18,.y)
    sync
    c[0].y=sld[0];
    c[1].y=sld[2];
    c[2].y=sld[4];
}
__device__ __forceinline__ void s_hfft16_s5( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev, int x, int xu )
{
    float2 temp;
    FFT8_M3(c,)
#pragma unroll
    for( int i=1; i<8; ++i ){
        c[i]=s_cmul(c[i],s_RF[xu*brev[i]]);
    }
    float sign=xu?-1.f:1.f;
#pragma unroll
    for( int i=0; i<8; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,16);
        temp.y=SHFL_XOR(c[i].y,1,16);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    PERMUTE(8,sst,sld,c,1,18,perm_mask)
    if(x>0){
        temp.x=0.5f*SHFL(c[0].x,16-x,16);
        temp.y=0.5f*SHFL(c[0].y,16-x,16);
        c[8].x=( 0.5f)*c[0].y+temp.y;
        c[8].y=(-0.5f)*c[0].x+temp.x;
        c[0].x=0.5f*c[0].x+( temp.x);
        c[0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[8].x=c[0].y;
        c[8].y=c[0].y=0.f;
    }
}
__device__ __forceinline__ void s_vfft16_s7( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[8];
    FFT8_M4(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<8; ++i )
    {
        temp=s_RF[i];
        d[i].x = c[brev[  i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[  i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[  i]].y*temp.x;
        d[i].y+=-c[brev[  i]].x*temp.x;
        d[i].x+= c[brev[8-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[8-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[8-i]].x*temp.x;
        d[i].x+= c[brev[8-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE8(sst,d,18,.x)
    sync
    c[0].x=sld[0];
    c[1].x=sld[2];
    c[2].x=sld[4];
    c[3].x=sld[6];
    sync
    STORE8(sst,d,18,.y)
    sync
    c[0].y=sld[0];
    c[1].y=sld[2];
    c[2].y=sld[4];
    c[3].y=sld[6];
}
__device__ __forceinline__ void s_hfft16_s7( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev, int x, int xu )
{
    float2 temp;
    FFT8_M4(c,)
#pragma unroll
    for( int i=1; i<8; ++i ){
        c[i]=s_cmul(c[i],s_RF[xu*brev[i]]);
    }
    float sign=xu?-1.f:1.f;
#pragma unroll
    for( int i=0; i<8; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,16);
        temp.y=SHFL_XOR(c[i].y,1,16);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    PERMUTE(8,sst,sld,c,1,18,perm_mask)
    if(x>0){
        temp.x=0.5f*SHFL(c[0].x,16-x,16);
        temp.y=0.5f*SHFL(c[0].y,16-x,16);
        c[8].x=( 0.5f)*c[0].y+temp.y;
        c[8].y=(-0.5f)*c[0].x+temp.x;
        c[0].x=0.5f*c[0].x+( temp.x);
        c[0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[8].x=c[0].y;
        c[8].y=c[0].y=0.f;
    }
}
__device__ __forceinline__ void sfft16x16_c2r_store( float* dst, float2* c, const float* null, float alpha, bool bc, int nx, int ny )
{
#pragma unroll
    for( int i=0; i<8; ++i ){
        c[i].x*=alpha; 
        c[i].y*=alpha; 
    }
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft16x16_c2r_store_relu( float* dst, float2* c, const float* null, float alpha, bool bc, int nx, int ny )
{
#pragma unroll
    for( int i=0; i<8; ++i ){
        c[i].x=s_relu(alpha*c[i].x); 
        c[i].y=s_relu(alpha*c[i].y); 
    }
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft16x16_c2r_store_bias( float* dst, float2* c, const float* bias, float alpha, bool bc, int nx, int ny )
{
    float b=*bias;
#pragma unroll
    for( int i=0; i<8; ++i ){
        c[i].x=alpha*c[i].x+b; 
        c[i].y=alpha*c[i].y+b; 
    }
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft16x16_c2r_store_bias_relu( float* dst, float2* c, const float* bias, float alpha, bool bc, int nx, int ny )
{
    float b=*bias;
#pragma unroll
    for( int i=0; i<8; ++i ){
        c[i].x=s_relu(alpha*c[i].x+b); 
        c[i].y=s_relu(alpha*c[i].y+b); 
    }
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft16x16_c2r_store_drelu( float* dst, float2* c, const float* a, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=alpha*c[i].x*s_drelu(a[0]); } a+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=alpha*c[i].y*s_drelu(a[0]); } a+=nx; dst+=nx; 
        }
    }
}
__device__ __forceinline__ void sfft16x16_c2r_store_xdrv( float* dst, float2* c, const float* da, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=alpha*c[i].x*da[0]; } da+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=alpha*c[i].y*da[0]; } da+=nx; dst+=nx; 
        }
    }
}

#include"sfft16x16_r2c.h"
#include"sfft16x16_c2r.h"
#include"sfft16x16_r2c_perm2d.h"
#include"sfft16x16_r2c_perm3d.h"
#include"sfft16x16_c2r_perm2d.h"
#include"sfft16x16_c2r_perm3d.h"
#include"sfft16x16_r2c_split.h"
#include"sfft16x16_c2r_splice.h"
#include"sfft16x16_r2c_split_perm.h"
#include"sfft16x16_c2r_splice_perm.h"
#include"sfft16x16_r2c_opt.h"
#include"sfft16x16_r2c_opt_perm.h"
#include"sfft16x16_c2r_grad.h"
#include"sfft16x16_c2r_grad_perm.h"