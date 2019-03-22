__device__ __forceinline__ void xfft16x16_c2r_store( __half* dst, float2* c, const __half* null, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(alpha*c[i].x); dst+=nx; } 
            if((2*i+1)<ny){ *dst=__float2half(alpha*c[i].y); dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void xfft16x16_c2r_store_relu( __half* dst, float2* c, const __half* null, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(s_relu(alpha*c[i].x)); dst+=nx; } 
            if((2*i+1)<ny){ *dst=__float2half(s_relu(alpha*c[i].y)); dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void xfft16x16_c2r_store_bias( __half* dst, float2* c, const __half* bias, float alpha, bool bc, int nx, int ny )
{
    float b=__half2float(*bias);
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(alpha*c[i].x+b); dst+=nx; } 
            if((2*i+1)<ny){ *dst=__float2half(alpha*c[i].y+b); dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void xfft16x16_c2r_store_bias_relu( __half* dst, float2* c, const __half* bias, float alpha, bool bc, int nx, int ny )
{
    float b=__half2float(*bias);
#pragma unroll
    for( int i=0; i<8; ++i ){
        c[i].x=s_relu(alpha*c[i].x+b); 
        c[i].y=s_relu(alpha*c[i].y+b); 
    }
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(c[i].x); dst+=nx; } 
            if((2*i+1)<ny){ *dst=__float2half(c[i].y); dst+=nx; } 
        }
    }
}
__device__ __forceinline__ void xfft16x16_c2r_store_drelu( __half* dst, float2* c, const __half* a, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(alpha*c[i].x*s_drelu(__half2float(a[0]))); } a+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=__float2half(alpha*c[i].y*s_drelu(__half2float(a[0]))); } a+=nx; dst+=nx; 
        }
    }
}
__device__ __forceinline__ void xfft16x16_c2r_store_xdrv( __half* dst, float2* c, const __half* da, float alpha, bool bc, int nx, int ny )
{
    if(bc){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ *dst=__float2half(alpha*c[i].x*__half2float(da[0])); } da+=nx; dst+=nx; 
            if((2*i+1)<ny){ *dst=__float2half(alpha*c[i].y*__half2float(da[0])); } da+=nx; dst+=nx; 
        }
    }
}


#include"xfft16x16_r2c.h"
#include"xfft16x16_c2r.h"
#include"xfft16x16_r2c_perm2d.h"
#include"xfft16x16_r2c_perm3d.h"
#include"xfft16x16_c2r_perm2d.h"
#include"xfft16x16_c2r_perm3d.h"
#include"xfft16x16_r2c_split.h"
#include"xfft16x16_c2r_splice.h"
#include"xfft16x16_r2c_split_perm.h"
#include"xfft16x16_c2r_splice_perm.h"
#include"xfft16x16_r2c_opt.h"
#include"xfft16x16_r2c_opt_perm.h"
#include"xfft16x16_c2r_grad.h"
#include"xfft16x16_c2r_grad_perm.h"