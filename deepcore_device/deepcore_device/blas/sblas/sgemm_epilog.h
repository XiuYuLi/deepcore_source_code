#ifndef __sgemm_epilog_h__
#define __sgemm_epilog_h__

__device__ __forceinline__ void sgemm_epilog64x32( char* d_c, const char* p_null, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane&0x10)<<1)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(alpha*c[k*32+i+ 0],alpha*c[k*32+i+ 1],alpha*c[k*32+i+ 2],alpha*c[k*32+i+ 3]);
            sst[8]=make_float4(alpha*c[k*32+i+16],alpha*c[k*32+i+17],alpha*c[k*32+i+18],alpha*c[k*32+i+19]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float2*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((float2*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((float2*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((float2*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            } 
        }
    }
}
__device__ __forceinline__ void sgemm_epilog64x32_relu( char* d_c, const char* p_null, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane&0x10)<<1)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<64; ++i ){ c[i]=s_relu(alpha*c[i]); }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(c[k*32+i+ 0],c[k*32+i+ 1],c[k*32+i+ 2],c[k*32+i+ 3]);
            sst[8]=make_float4(c[k*32+i+16],c[k*32+i+17],c[k*32+i+18],c[k*32+i+19]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float2*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((float2*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((float2*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((float2*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            } 
        }
    }
}
__device__ __forceinline__ void sgemm_epilog64x32_bias( char* d_c, const char* s_bias, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane&0x10)<<1)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<64; ++i ){
        c[i]=alpha*c[i]+((const float*)s_bias)[16*(i/32)+(i%32)/8];
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        { 
            __syncthreads();
            sst[0]=make_float4(c[k*32+i+ 0],c[k*32+i+ 1],c[k*32+i+ 2],c[k*32+i+ 3]);
            sst[8]=make_float4(c[k*32+i+16],c[k*32+i+17],c[k*32+i+18],c[k*32+i+19]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float2*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((float2*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((float2*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((float2*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            }
        }
    }
}
__device__ __forceinline__ void sgemm_epilog64x32_bias_relu( char* d_c, const char* s_bias, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane&0x10)<<1)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<64; ++i ){
        c[i]=s_relu(alpha*c[i]+((const float*)s_bias)[16*(i/32)+(i%32)/8]);
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(c[k*32+i+ 0],c[k*32+i+ 1],c[k*32+i+ 2],c[k*32+i+ 3]);
            sst[8]=make_float4(c[k*32+i+16],c[k*32+i+17],c[k*32+i+18],c[k*32+i+19]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float2*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((float2*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((float2*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((float2*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            }
        }
    }
}
__device__ __forceinline__ void sgemm_epilog64x32_drelu( char* d_c, const char* d_x, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane&0x10)<<1)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(alpha*c[k*32+i+ 0],alpha*c[k*32+i+ 1],alpha*c[k*32+i+ 2],alpha*c[k*32+i+ 3]);
            sst[8]=make_float4(alpha*c[k*32+i+16],alpha*c[k*32+i+17],alpha*c[k*32+i+18],alpha*c[k*32+i+19]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float2*)&d_c[(k*16+0x0+i/4)*ldc])=s_drelu_x2(sld[0*32],*((const float2*)&d_x[(k*16+0x0+i/4)*ldc])); }
                if((k*16+0x4+i/4)<ny){ *((float2*)&d_c[(k*16+0x4+i/4)*ldc])=s_drelu_x2(sld[1*32],*((const float2*)&d_x[(k*16+0x4+i/4)*ldc])); }
                if((k*16+0x8+i/4)<ny){ *((float2*)&d_c[(k*16+0x8+i/4)*ldc])=s_drelu_x2(sld[2*32],*((const float2*)&d_x[(k*16+0x8+i/4)*ldc])); }
                if((k*16+0xc+i/4)<ny){ *((float2*)&d_c[(k*16+0xc+i/4)*ldc])=s_drelu_x2(sld[3*32],*((const float2*)&d_x[(k*16+0xc+i/4)*ldc])); }
            } 
        }
    }
}
__device__ __forceinline__ void sgemm_epilog64x32_xdrv( char* d_c, const char* d_x, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane&0x10)<<1)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(alpha*c[k*32+i+ 0],alpha*c[k*32+i+ 1],alpha*c[k*32+i+ 2],alpha*c[k*32+i+ 3]);
            sst[8]=make_float4(alpha*c[k*32+i+16],alpha*c[k*32+i+17],alpha*c[k*32+i+18],alpha*c[k*32+i+19]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float2*)&d_c[(k*16+0x0+i/4)*ldc])=*((const float2*)&d_x[(k*16+0x0+i/4)*ldc])*sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((float2*)&d_c[(k*16+0x4+i/4)*ldc])=*((const float2*)&d_x[(k*16+0x4+i/4)*ldc])*sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((float2*)&d_c[(k*16+0x8+i/4)*ldc])=*((const float2*)&d_x[(k*16+0x8+i/4)*ldc])*sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((float2*)&d_c[(k*16+0xc+i/4)*ldc])=*((const float2*)&d_x[(k*16+0xc+i/4)*ldc])*sld[3*32]; }
            } 
        }
    }
}
__device__ __forceinline__ void sgemm_epilog32x32( char* d_c, const char* p_null, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[(lane&0x10)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    float* sld=&((float*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i]*=alpha; }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<4; ++i )
        {
            __syncthreads();
            sst[0]=make_float4(c[k*16+i*4],c[k*16+i*4+1],c[k*16+i*4+2],c[k*16+i*4+3]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+ 0+i)<ny){ *((float*)&d_c[(k*16+ 0+i)*ldc])=sld[0*32]; }
                if((k*16+ 4+i)<ny){ *((float*)&d_c[(k*16+ 4+i)*ldc])=sld[1*32]; }
                if((k*16+ 8+i)<ny){ *((float*)&d_c[(k*16+ 8+i)*ldc])=sld[2*32]; }
                if((k*16+12+i)<ny){ *((float*)&d_c[(k*16+12+i)*ldc])=sld[3*32]; }
            }
        }
    }
}
__device__ __forceinline__ void sgemm_epilog32x32_relu( char* d_c, const char* p_null, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[(lane&0x10)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    float* sld=&((float*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i]=s_relu(alpha*c[i]); }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(c[k*16+i],c[k*16+i+1],c[k*16+i+2],c[k*16+i+3]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((float*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((float*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((float*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            } 
        }
    }
}
__device__ __forceinline__ void sgemm_epilog32x32_bias( char* d_c, const char* s_bias, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[(lane&0x10)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    float* sld=&((float*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i]=alpha*c[i]+((const float*)s_bias)[16*(i/16)+(i%16)/4];
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(c[k*16+i],c[k*16+i+1],c[k*16+i+2],c[k*16+i+3]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((float*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((float*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((float*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            } 
        }
    }
}
__device__ __forceinline__ void sgemm_epilog32x32_bias_relu( char* d_c, const char* s_bias, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[(lane&0x10)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    float* sld=&((float*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i]=s_relu(alpha*c[i]+((const float*)s_bias)[16*(i/16)+(i%16)/4]);
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(c[k*16+i],c[k*16+i+1],c[k*16+i+2],c[k*16+i+3]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((float*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((float*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((float*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((float*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            } 
        }
    }
}
__device__ __forceinline__ void sgemm_epilog32x32_drelu( char* d_c, const char* d_x, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[(lane&0x10)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    float* sld=&((float*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i]*=alpha; }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<4; ++i )
        {
            __syncthreads();
            sst[0]=make_float4(c[k*16+i*4],c[k*16+i*4+1],c[k*16+i*4+2],c[k*16+i*4+3]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+ 0+i)<ny){ *((float*)&d_c[(k*16+ 0+i)*ldc])=s_drelu(*((const float*)&d_x[(k*16+ 0+i)*ldc]))*sld[0*32]; }
                if((k*16+ 4+i)<ny){ *((float*)&d_c[(k*16+ 4+i)*ldc])=s_drelu(*((const float*)&d_x[(k*16+ 4+i)*ldc]))*sld[1*32]; }
                if((k*16+ 8+i)<ny){ *((float*)&d_c[(k*16+ 8+i)*ldc])=s_drelu(*((const float*)&d_x[(k*16+ 8+i)*ldc]))*sld[2*32]; }
                if((k*16+12+i)<ny){ *((float*)&d_c[(k*16+12+i)*ldc])=s_drelu(*((const float*)&d_x[(k*16+12+i)*ldc]))*sld[3*32]; }
            }
        }
    }
}
__device__ __forceinline__ void sgemm_epilog32x32_xdrv( char* d_c, const char* d_x, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[(lane&0x10)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    float* sld=&((float*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i]*=alpha; }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<4; ++i )
        {
            __syncthreads();
            sst[0]=make_float4(c[k*16+i*4],c[k*16+i*4+1],c[k*16+i*4+2],c[k*16+i*4+3]);
            __syncthreads();
            if(x<nx)
            {
                if((k*16+ 0+i)<ny){ *((float*)&d_c[(k*16+ 0+i)*ldc])=(*((const float*)&d_x[(k*16+ 0+i)*ldc]))*sld[0*32]; }
                if((k*16+ 4+i)<ny){ *((float*)&d_c[(k*16+ 4+i)*ldc])=(*((const float*)&d_x[(k*16+ 4+i)*ldc]))*sld[1*32]; }
                if((k*16+ 8+i)<ny){ *((float*)&d_c[(k*16+ 8+i)*ldc])=(*((const float*)&d_x[(k*16+ 8+i)*ldc]))*sld[2*32]; }
                if((k*16+12+i)<ny){ *((float*)&d_c[(k*16+12+i)*ldc])=(*((const float*)&d_x[(k*16+12+i)*ldc]))*sld[3*32]; }
            }
        }
    }
}
#endif