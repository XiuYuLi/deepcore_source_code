__device__ __forceinline__ void s_vfft64( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[32];
    FFT32(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<32; ++i ){
        temp=s_RF[i];
        d[i].x = c[brev[   i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[   i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[   i]].y*temp.x;
        d[i].y+=-c[brev[   i]].x*temp.x;
        d[i].x+= c[brev[32-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[32-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[32-i]].x*temp.x;
        d[i].x+= c[brev[32-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE32(sst,d,66,.x) __syncthreads();
    LOAD32(c,sld,2,.x)   __syncthreads();
    STORE32(sst,d,66,.y) __syncthreads();
    LOAD32(c,sld,2,.y)   __syncthreads();
}
__device__ __forceinline__ void s_hfft64( float2* c, float* smem, const float2* s_RF, const int* brev, int tid, int u, int v )
{
    float2 temp;
    unsigned int flip=tid?(64-tid):tid;
    float* spx=&smem[66*v+33*u];
    float* spy=&smem[33*( tid>>5)+( tid&31)];
    float* spz=&smem[33*(flip>>5)+(flip&31)];
    FFT32(c,)
#pragma unroll
    for( int i=1; i<32; ++i ){
        c[i]=s_cmul(c[i],s_RF[u*brev[i]]);
    }
    float sign=u?-1.f:1.f;
#pragma unroll
    for( int i=0; i<32; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,32);
        temp.y=SHFL_XOR(c[i].y,1,32);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    ISTORE32(spx,c,1,.x)
    __syncthreads();
    LOAD32(c,spy,66,.x) 
    __syncthreads();
    ISTORE32(&spx[66],c,1,.y)
    __syncthreads();
    LOAD32(c,&spy[66],66,.y)
    if(tid>0){
        temp.x=0.5f*spz[ 0];
        temp.y=0.5f*spz[66];
        c[32].x=( 0.5f)*c[0].y+temp.y;
        c[32].y=(-0.5f)*c[0].x+temp.x;
        c[ 0].x=0.5f*c[0].x+( temp.x);
        c[ 0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[32].x=c[0].y;
        c[32].y=c[0].y=0.f;
    }
}
__device__ __forceinline__ void s_hifft64( float2* c, float2* d, float* smem, const float2* s_RF, const int* brev, unsigned int tid )
{
    float2 temp;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    float* spx=&smem[tid];
    float* spy=&smem[v*66+u];
    float* spp=&smem[v*66+u*33];
    float* spq=&smem[(tid>>5)*33+(tid&31)];
    if(tid>0){
        d[0].x-=d[32].y; 
        d[0].y+=d[32].x;
    } else {
        d[0].y=d[32].x;
    }
    STORE32(spx,d,66,.x) __syncthreads();
    LOAD32(c,spy,2,.x)   __syncthreads();
    STORE32(spx,d,66,.y) __syncthreads();
    LOAD32(c,spy,2,.y)   __syncthreads();
    FFT32(c,i)
#pragma unroll
    for( int i=1; i<32; ++i ){
        c[i]=s_icmul(c[i],s_RF[u*brev[i]]);
    }
    float sign=u?-1.f:1.f;
#pragma unroll
    for( int i=0; i<32; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,32);
        temp.y=SHFL_XOR(c[i].y,1,32);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    PERMUTE(32,spp,spq,c,1,66,0x7)
}
__device__ __forceinline__ void s_vifft64( float2* d, float2* c, const float2* s_RF, const int* brev, unsigned int tid )
{
    float2 a, b, temp=c[0];
    float sign=((tid&31)>0)?1.f:-1.f;
    c[0].x=-sign*temp.y+temp.x;
    c[0].y= sign*temp.y+temp.x;
#pragma unroll
    for( int i=1; i<16; ++i ){
        temp=s_RF[i];
        a.x=c[i].x+c[32-i].x;
        a.y=c[i].y-c[32-i].y;
        b.x=c[i].y+c[32-i].y;
        b.y=c[i].x-c[32-i].x;
        c[   i].x=(-b.x)*temp.x+(( b.y)*temp.y+( a.x));
        c[   i].y=( b.y)*temp.x+(( b.x)*temp.y+( a.y));
        c[32-i].x=( b.x)*temp.x+((-b.y)*temp.y+( a.x));
        c[32-i].y=( b.y)*temp.x+(( b.x)*temp.y+(-a.y));
    }
    c[16].x*= 2.0f;
    c[16].y*=-2.0f;
    FFT32(c,i)
#pragma unroll
    for( int i=0; i<32; ++i ){ d[i]=c[brev[i]]; }
    if((tid&31)==0){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            d[i].x=c[brev[(32-i)%32]].x;
            d[i].y=c[brev[(31-i)%32]].y;
        }
    }
}
__device__ __forceinline__ void s_vfft64_s3( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[32];
    FFT32_M2(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<32; ++i ){
        temp=s_RF[i];
        d[i].x = c[brev[   i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[   i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[   i]].y*temp.x;
        d[i].y+=-c[brev[   i]].x*temp.x;
        d[i].x+= c[brev[32-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[32-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[32-i]].x*temp.x;
        d[i].x+= c[brev[32-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE32(sst,d,66,.x)
    __syncthreads();
    c[0].x=sld[0];
    c[1].x=sld[2];
    __syncthreads();
    STORE32(sst,d,66,.y)
    __syncthreads();
    c[0].y=sld[0];
    c[1].y=sld[2];
    __syncthreads();
}
__device__ __forceinline__ void s_hfft64_s3( float2* c, float* smem, const float2* s_RF, const int* brev, unsigned int tid, unsigned int u, unsigned int v )
{
    float2 temp;    
    unsigned int flip=tid?(64-tid):tid;
    float* spx=&smem[66*v+33*u];
    float* spy=&smem[33*( tid>>5)+( tid&31)];
    float* spz=&smem[33*(flip>>5)+(flip&31)];
    FFT32_M2(c,)
#pragma unroll
    for( int i=1; i<32; ++i ){
        c[i]=s_cmul(c[i],s_RF[u*brev[i]]);
    }
    float sign=u?-1.f:1.f;
#pragma unroll
    for( int i=0; i<32; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,32);
        temp.y=SHFL_XOR(c[i].y,1,32);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    ISTORE32(spx,c,1,.x)
    __syncthreads();
    LOAD32(c,spy,66,.x) 
    __syncthreads();
    ISTORE32(&spx[66],c,1,.y)
    __syncthreads();
    LOAD32(c,&spy[66],66,.y)
    if(tid>0){
        temp.x=0.5f*spz[ 0];
        temp.y=0.5f*spz[66];
        c[32].x=( 0.5f)*c[0].y+temp.y;
        c[32].y=(-0.5f)*c[0].x+temp.x;
        c[ 0].x=0.5f*c[0].x+( temp.x);
        c[ 0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[32].x=c[0].y;
        c[32].y=c[0].y=0.f;
    }
}
__device__ __forceinline__ void s_vfft64_s5( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[32];
    FFT32_M3(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<32; ++i )
    {
        temp=s_RF[i];
        d[i].x = c[brev[   i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[   i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[   i]].y*temp.x;
        d[i].y+=-c[brev[   i]].x*temp.x;
        d[i].x+= c[brev[32-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[32-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[32-i]].x*temp.x;
        d[i].x+= c[brev[32-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE32(sst,d,66,.x)
    __syncthreads();
    c[0].x=sld[0];
    c[1].x=sld[2];
    c[2].x=sld[4];
    __syncthreads();
    STORE32(sst,d,66,.y)
    __syncthreads();
    c[0].y=sld[0];
    c[1].y=sld[2];
    c[2].y=sld[4];
    __syncthreads();
}
__device__ __forceinline__ void s_hfft64_s5( float2* c, float* smem, const float2* s_RF, const int* brev, unsigned int tid, unsigned int u, unsigned int v )
{
    float2 temp;    
    unsigned int flip=tid?(64-tid):tid;
    float* spx=&smem[66*v+33*u];
    float* spy=&smem[33*( tid>>5)+( tid&31)];
    float* spz=&smem[33*(flip>>5)+(flip&31)];
    FFT32_M3(c,)
#pragma unroll
    for( int i=1; i<32; ++i ){
        c[i]=s_cmul(c[i],s_RF[u*brev[i]]);
    }
    float sign=u?-1.f:1.f;
#pragma unroll
    for( int i=0; i<32; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,32);
        temp.y=SHFL_XOR(c[i].y,1,32);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    ISTORE32(spx,c,1,.x)
    __syncthreads();
    LOAD32(c,spy,66,.x) 
    __syncthreads();
    ISTORE32(&spx[66],c,1,.y)
    __syncthreads();
    LOAD32(c,&spy[66],66,.y)
    if(tid>0){
        temp.x=0.5f*spz[ 0];
        temp.y=0.5f*spz[66];
        c[32].x=( 0.5f)*c[0].y+temp.y;
        c[32].y=(-0.5f)*c[0].x+temp.x;
        c[ 0].x=0.5f*c[0].x+( temp.x);
        c[ 0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[32].x=c[0].y;
        c[32].y=c[0].y=0.f;
    }
}
__device__ __forceinline__ void s_vfft64_s7( float2* c, float* sst, float* sld, const float2* s_RF, const int* brev )
{
    float2 temp, d[32];
    FFT32_M4(c,)
    d[0].x=c[0].x+c[0].y;
    d[0].y=c[0].x-c[0].y;
#pragma unroll
    for( int i=1; i<32; ++i )
    {
        temp=s_RF[i];
        d[i].x = c[brev[   i]].x*temp.y+c[brev[i]].x;
        d[i].y = c[brev[   i]].y*temp.y+c[brev[i]].y;
        d[i].x+= c[brev[   i]].y*temp.x;
        d[i].y+=-c[brev[   i]].x*temp.x;
        d[i].x+= c[brev[32-i]].x*(1.f-temp.y);
        d[i].y+=-c[brev[32-i]].y*(1.f-temp.y);
        d[i].y+= c[brev[32-i]].x*temp.x;
        d[i].x+= c[brev[32-i]].y*temp.x;
        d[i].x*=0.5f;
        d[i].y*=0.5f;
    }
    STORE32(sst,d,66,.x)
    __syncthreads();
    c[0].x=sld[0];
    c[1].x=sld[2];
    c[2].x=sld[4];
    c[3].x=sld[6];
    __syncthreads();
    STORE32(sst,d,66,.y)
    __syncthreads();
    c[0].y=sld[0];
    c[1].y=sld[2];
    c[2].y=sld[4];
    c[3].y=sld[6];
    __syncthreads();
}
__device__ __forceinline__ void s_hfft64_s7( float2* c, float* smem, const float2* s_RF, const int* brev, unsigned int tid, unsigned int u, unsigned int v )
{
    float2 temp;
    unsigned int flip=tid?(64-tid):tid;
    float* spx=&smem[66*v+33*u];
    float* spy=&smem[33*( tid>>5)+( tid&31)];
    float* spz=&smem[33*(flip>>5)+(flip&31)];
    FFT32_M4(c,)
#pragma unroll
    for( int i=1; i<32; ++i ){
        c[i]=s_cmul(c[i],s_RF[u*brev[i]]);
    }
    float sign=u?-1.f:1.f;
#pragma unroll
    for( int i=0; i<32; ++i ){
        temp.x=SHFL_XOR(c[i].x,1,32);
        temp.y=SHFL_XOR(c[i].y,1,32);
        c[i].x=sign*c[i].x+temp.x;
        c[i].y=sign*c[i].y+temp.y;
    }
    ISTORE32(spx,c,1,.x)
    __syncthreads();
    LOAD32(c,spy,66,.x) 
    __syncthreads();
    ISTORE32(&spx[66],c,1,.y)
    __syncthreads();
    LOAD32(c,&spy[66],66,.y)
    if(tid>0){
        temp.x=0.5f*spz[ 0];
        temp.y=0.5f*spz[66];
        c[32].x=( 0.5f)*c[0].y+temp.y;
        c[32].y=(-0.5f)*c[0].x+temp.x;
        c[ 0].x=0.5f*c[0].x+( temp.x);
        c[ 0].y=0.5f*c[0].y+(-temp.y);
    } else {
        c[32].x=c[0].y;
        c[32].y=c[0].y=0.f;
    }
}

__device__ __forceinline__ void sfft64x64_c2r_store( float* dst, float2* c, const float* null, float alpha, int flip_x, int nx, int ny )
{
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x*=alpha; 
        c[i].y*=alpha; 
    }
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft64x64_c2r_store_relu( float* dst, float2* c, const float* null, float alpha, int flip_x, int nx, int ny )
{
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x=s_relu(alpha*c[i].x); 
        c[i].y=s_relu(alpha*c[i].y); 
    }
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft64x64_c2r_store_bias( float* dst, float2* c, const float* bias, float alpha, int flip_x, int nx, int ny )
{
    float b=*bias;
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x=alpha*c[i].x+b; 
        c[i].y=alpha*c[i].y+b; 
    }
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft64x64_c2r_store_bias_relu( float* dst, float2* c, const float* bias, float alpha, int flip_x, int nx, int ny )
{
    float b=*bias;
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x=s_relu(alpha*c[i].x+b); 
        c[i].y=s_relu(alpha*c[i].y+b); 
    }
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=c[i].x; dst+=nx; } 
            if((2*i+1)<ny){ *dst=c[i].y; dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void sfft64x64_c2r_store_drelu( float* dst, float2* c, const float* a, float alpha, int flip_x, int nx, int ny )
{
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=alpha*c[i].x*s_drelu(a[0]); } a+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=alpha*c[i].y*s_drelu(a[0]); } a+=nx; dst+=nx; 
        }
    }
}
__device__ __forceinline__ void sfft64x64_c2r_store_xdrv( float* dst, float2* c, const float* da, float alpha, int flip_x, int nx, int ny )
{
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=alpha*c[i].x*da[0]; } da+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=alpha*c[i].y*da[0]; } da+=nx; dst+=nx; 
        }
    }
}

#include"sfft64x64_r2c.h"
#include"sfft64x64_c2r.h"
#include"sfft64x64_r2c_opt.h"
#include"sfft64x64_c2r_grad.h"