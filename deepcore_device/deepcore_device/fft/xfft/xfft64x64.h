__device__ __forceinline__ void xfft64x64_c2r_store( __half* dst, float2* c, const __half* null, float alpha, int flip_x, int nx, int ny )
{
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(alpha*c[i].x); dst+=nx; } 
            if((2*i+1)<ny){ *dst=__float2half(alpha*c[i].y); dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void xfft64x64_c2r_store_relu( __half* dst, float2* c, const __half* null, float alpha, int flip_x, int nx, int ny )
{
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(s_relu(alpha*c[i].x)); dst+=nx; } 
            if((2*i+1)<ny){ *dst=__float2half(s_relu(alpha*c[i].y)); dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void xfft64x64_c2r_store_bias( __half* dst, float2* c, const __half* bias, float alpha, int flip_x, int nx, int ny )
{
    float b=__half2float(*bias);
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(alpha*c[i].x+b); dst+=nx; } 
            if((2*i+1)<ny){ *dst=__float2half(alpha*c[i].y+b); dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void xfft64x64_c2r_store_bias_relu( __half* dst, float2* c, const __half* bias, float alpha, int flip_x, int nx, int ny )
{
    float b=__half2float(*bias);
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(s_relu(alpha*c[i].x+b)); dst+=nx; } 
            if((2*i+1)<ny){ *dst=__float2half(s_relu(alpha*c[i].y+b)); dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void xfft64x64_c2r_store_drelu( __half* dst, float2* c, const __half* a, float alpha, int flip_x, int nx, int ny )
{
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(alpha*c[i].x*s_drelu(__half2float(a[0]))); } a+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=__float2half(alpha*c[i].y*s_drelu(__half2float(a[0]))); } a+=nx; dst+=nx; 
        }
    }
}
__device__ __forceinline__ void xfft64x64_c2r_store_xdrv( __half* dst, float2* c, const __half* da, float alpha, int flip_x, int nx, int ny )
{
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(alpha*c[i].x*__half2float(da[0])); } da+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=__float2half(alpha*c[i].y*__half2float(da[0])); } da+=nx; dst+=nx; 
        }
    }
}

#include"xfft64x64_c2r.h"
#include"xfft64x64_r2c.h"
#include"xfft64x64_r2c_opt.h"
#include"xfft64x64_c2r_grad.h"
