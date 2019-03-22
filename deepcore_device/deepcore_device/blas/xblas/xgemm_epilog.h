#ifndef __xgemm_epilog_h__
#define __xgemm_epilog_h__

__device__ __forceinline__ void xgemm_epilog64x32( char* d_c, const char* p_null, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    __half4* sst=&((__half4*)sptr)[((lane>>4)<<5)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    __half2* sld=&((__half2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<64; ++i ){ c[i]*=alpha; }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_half4(__float2half(c[k*32+i+ 0]),__float2half(c[k*32+i+ 1]),__float2half(c[k*32+i+ 2]),__float2half(c[k*32+i+ 3]));
            sst[8]=make_half4(__float2half(c[k*32+i+16]),__float2half(c[k*32+i+17]),__float2half(c[k*32+i+18]),__float2half(c[k*32+i+19]));
            __syncthreads();
            if(x<nx){
                if((k*16+0x0+i/4)<ny){ *((__half2*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((__half2*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((__half2*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((__half2*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            } 
        }
    }
}
__device__ __forceinline__ void xgemm_epilog64x32_relu( char* d_c, const char* p_null, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    __half4* sst=&((__half4*)sptr)[((lane>>4)<<5)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    __half2* sld=&((__half2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<64; ++i ){ c[i]=s_relu(alpha*c[i]); }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_half4(__float2half(c[k*32+i+ 0]),__float2half(c[k*32+i+ 1]),__float2half(c[k*32+i+ 2]),__float2half(c[k*32+i+ 3]));
            sst[8]=make_half4(__float2half(c[k*32+i+16]),__float2half(c[k*32+i+17]),__float2half(c[k*32+i+18]),__float2half(c[k*32+i+19]));
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((__half2*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((__half2*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((__half2*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((__half2*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            } 
        }
    }
}
__device__ __forceinline__ void xgemm_epilog64x32_bias( char* d_c, const char* bias, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    __half4* sst=&((__half4*)sptr)[((lane>>4)<<5)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    __half2* sld=&((__half2*)sptr)[lane];
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<32; i+=8 ){
            float b=((const char*)bias)[k*16+i/8];
            c[i+0]=alpha*c[i+0]+b;
            c[i+1]=alpha*c[i+1]+b;
            c[i+2]=alpha*c[i+2]+b;
            c[i+3]=alpha*c[i+3]+b;
            c[i+4]=alpha*c[i+4]+b;
            c[i+5]=alpha*c[i+5]+b;
            c[i+6]=alpha*c[i+6]+b;
            c[i+7]=alpha*c[i+7]+b;
        }
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        { 
            __syncthreads();
            sst[0]=make_half4(__float2half(c[k*32+i+ 0]),__float2half(c[k*32+i+ 1]),__float2half(c[k*32+i+ 2]),__float2half(c[k*32+i+ 3]));
            sst[8]=make_half4(__float2half(c[k*32+i+16]),__float2half(c[k*32+i+17]),__float2half(c[k*32+i+18]),__float2half(c[k*32+i+19]));
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((__half2*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((__half2*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((__half2*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((__half2*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            }
        }
    }
}
__device__ __forceinline__ void xgemm_epilog64x32_bias_relu( char* d_c, const char* bias, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    __half4* sst=&((__half4*)sptr)[((lane>>4)<<5)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    __half2* sld=&((__half2*)sptr)[lane];
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<32; i+=8 ){
            float b=((const char*)bias)[k*16+i/8];
            c[i+0]=s_relu(alpha*c[i+0]+b);
            c[i+1]=s_relu(alpha*c[i+1]+b);
            c[i+2]=s_relu(alpha*c[i+2]+b);
            c[i+3]=s_relu(alpha*c[i+3]+b);
            c[i+4]=s_relu(alpha*c[i+4]+b);
            c[i+5]=s_relu(alpha*c[i+5]+b);
            c[i+6]=s_relu(alpha*c[i+6]+b);
            c[i+7]=s_relu(alpha*c[i+7]+b);
        }
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_half4(__float2half(c[k*32+i+ 0]),__float2half(c[k*32+i+ 1]),__float2half(c[k*32+i+ 2]),__float2half(c[k*32+i+ 3]));
            sst[8]=make_half4(__float2half(c[k*32+i+16]),__float2half(c[k*32+i+17]),__float2half(c[k*32+i+18]),__float2half(c[k*32+i+19]));
            __syncthreads();
            if(x<nx)
            {
                if((k*16+0x0+i/4)<ny){ *((__half2*)&d_c[(k*16+0x0+i/4)*ldc])=sld[0*32]; }
                if((k*16+0x4+i/4)<ny){ *((__half2*)&d_c[(k*16+0x4+i/4)*ldc])=sld[1*32]; }
                if((k*16+0x8+i/4)<ny){ *((__half2*)&d_c[(k*16+0x8+i/4)*ldc])=sld[2*32]; }
                if((k*16+0xc+i/4)<ny){ *((__half2*)&d_c[(k*16+0xc+i/4)*ldc])=sld[3*32]; }
            }
        }
    }
}
__device__ __forceinline__ void xgemm_epilog64x32_drelu( char* d_c, const char* d_x, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane>>4)<<5)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<64; ++i ){ c[i]*=alpha; }
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
            if(x<nx){
                if((k*16+0x0+i/4)<ny){ *((__half2*)&d_c[(k*16+0x0+i/4)*ldc])=x_drelu_x2(sld[0*32],*((const __half2*)&d_x[(k*16+0x0+i/4)*ldc])); }
                if((k*16+0x4+i/4)<ny){ *((__half2*)&d_c[(k*16+0x4+i/4)*ldc])=x_drelu_x2(sld[1*32],*((const __half2*)&d_x[(k*16+0x4+i/4)*ldc])); }
                if((k*16+0x8+i/4)<ny){ *((__half2*)&d_c[(k*16+0x8+i/4)*ldc])=x_drelu_x2(sld[2*32],*((const __half2*)&d_x[(k*16+0x8+i/4)*ldc])); }
                if((k*16+0xc+i/4)<ny){ *((__half2*)&d_c[(k*16+0xc+i/4)*ldc])=x_drelu_x2(sld[3*32],*((const __half2*)&d_x[(k*16+0xc+i/4)*ldc])); }
            } 
        }
    }
}
__device__ __forceinline__ void xgemm_epilog64x32_xdrv( char* d_c, const char* d_x, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane>>4)<<5)+((lane&0x1)<<4)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<64; ++i ){ c[i]*=alpha; }
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
            if(x<nx){
                if((k*16+0x0+i/4)<ny){ *((__half2*)&d_c[(k*16+0x0+i/4)*ldc])=__float22half2_rn(sld[0*32]*__half22float2(*((const __half2*)&d_x[(k*16+0x0+i/4)*ldc]))); }
                if((k*16+0x4+i/4)<ny){ *((__half2*)&d_c[(k*16+0x4+i/4)*ldc])=__float22half2_rn(sld[1*32]*__half22float2(*((const __half2*)&d_x[(k*16+0x4+i/4)*ldc]))); }
                if((k*16+0x8+i/4)<ny){ *((__half2*)&d_c[(k*16+0x8+i/4)*ldc])=__float22half2_rn(sld[2*32]*__half22float2(*((const __half2*)&d_x[(k*16+0x8+i/4)*ldc]))); }
                if((k*16+0xc+i/4)<ny){ *((__half2*)&d_c[(k*16+0xc+i/4)*ldc])=__float22half2_rn(sld[3*32]*__half22float2(*((const __half2*)&d_x[(k*16+0xc+i/4)*ldc]))); }
            } 
        }
    }
}
__device__ __forceinline__ void xgemm_epilog32x32( char* d_c, const char* p_null, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    __half4* sst=&((__half4*)sptr)[((lane>>4)<<4)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    __half2* sld=&((__half2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i]*=alpha; }
#pragma unroll
    for( int k=0; k<32; k+=16 )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_half4(__float2half(c[k+i]),__float2half(c[k+i+1]),__float2half(c[k+i+2]),__float2half(c[k+i+3]));
            __syncthreads();
            if(x<nx){
                if((k+0+i/4)<ny){ *((__half2*)&d_c[(k+0+i/4)*ldc])=sld[0*32]; }
                if((k+4+i/4)<ny){ *((__half2*)&d_c[(k+4+i/4)*ldc])=sld[1*32]; }
            }
        }
    }
}
__device__ __forceinline__ void xgemm_epilog32x32_bias( char* d_c, const char* s_bias, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    __half4* sst=&((__half4*)sptr)[((lane>>4)<<4)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    __half2* sld=&((__half2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i]=alpha*c[i]+((const char*)s_bias)[16*(i/16)+(i%16)/4];
    }
#pragma unroll
    for( int k=0; k<32; k+=16 )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_half4(__float2half(c[k+i]),__float2half(c[k+i+1]),__float2half(c[k+i+2]),__float2half(c[k+i+3]));
            __syncthreads();
            if(x<nx){
                if((k+0+i/4)<ny){ *((__half2*)&d_c[(k+0+i/4)*ldc])=sld[0*32]; }
                if((k+4+i/4)<ny){ *((__half2*)&d_c[(k+4+i/4)*ldc])=sld[1*32]; }
            }
        }
    }
}
__device__ __forceinline__ void xgemm_epilog32x32_relu( char* d_c, const char* p_null, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    __half4* sst=&((__half4*)sptr)[((lane>>4)<<4)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    __half2* sld=&((__half2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i]=s_relu(alpha*c[i]);
    }
#pragma unroll
    for( int k=0; k<32; k+=16 )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_half4(__float2half(c[k+i]),__float2half(c[k+i+1]),__float2half(c[k+i+2]),__float2half(c[k+i+3]));
            __syncthreads();
            if(x<nx){
                if((k+0+i/4)<ny){ *((__half2*)&d_c[(k+0+i/4)*ldc])=sld[0*32]; }
                if((k+4+i/4)<ny){ *((__half2*)&d_c[(k+4+i/4)*ldc])=sld[1*32]; }
            }
        }
    }
}
__device__ __forceinline__ void xgemm_epilog32x32_bias_relu( char* d_c, const char* s_bias, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    __half4* sst=&((__half4*)sptr)[((lane>>4)<<4)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    __half2* sld=&((__half2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i]=s_relu(alpha*c[i]+((const char*)s_bias)[16*(i/16)+(i%16)/4]);
    }
#pragma unroll
    for( int k=0; k<32; k+=16 )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_half4(__float2half(c[k+i]),__float2half(c[k+i+1]),__float2half(c[k+i+2]),__float2half(c[k+i+3]));
            __syncthreads();
            if(x<nx){
                if((k+0+i/4)<ny){ *((__half2*)&d_c[(k+0+i/4)*ldc])=sld[0*32]; }
                if((k+4+i/4)<ny){ *((__half2*)&d_c[(k+4+i/4)*ldc])=sld[1*32]; }
            }
        }
    }
}
__device__ __forceinline__ void xgemm_epilog32x32_drelu( char* d_c, const char* d_x, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane>>4)<<4)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i]*=alpha; }
#pragma unroll
    for( int k=0; k<32; k+=16 )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(c[k+i],c[k+i+1],c[k+i+2],c[k+i+3]);
            __syncthreads();
            if(x<nx){
                if((k+0+i/4)<ny){ *((__half2*)&d_c[(k+0+i/4)*ldc])=x_drelu_x2(sld[0*32],*((const __half2*)&d_x[(k+0+i/4)*ldc])); }
                if((k+4+i/4)<ny){ *((__half2*)&d_c[(k+4+i/4)*ldc])=x_drelu_x2(sld[1*32],*((const __half2*)&d_x[(k+4+i/4)*ldc])); }
            }
        }
    }
}
__device__ __forceinline__ void xgemm_epilog32x32_xdrv( char* d_c, const char* d_x, char* sptr, float* c, int lane, int ldc, int x, int nx, int ny, float alpha )
{
    float4* sst=&((float4*)sptr)[((lane>>4)<<4)+((lane&0x1)<<3)+((lane&0xf)>>1)];
    float2* sld=&((float2*)sptr)[lane];
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i]*=alpha; }
#pragma unroll
    for( int k=0; k<32; k+=16 )
    {
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            __syncthreads();
            sst[0]=make_float4(c[k+i],c[k+i+1],c[k+i+2],c[k+i+3]);
            __syncthreads();
            if(x<nx){
                if((k+0+i/4)<ny){ *((__half2*)&d_c[(k+0+i/4)*ldc])=__float22half2_rn(sld[0*32]*__half22float2(*((const __half2*)&d_x[(k+0+i/4)*ldc]))); }
                if((k+4+i/4)<ny){ *((__half2*)&d_c[(k+4+i/4)*ldc])=__float22half2_rn(sld[1*32]*__half22float2(*((const __half2*)&d_x[(k+4+i/4)*ldc]))); }
            }
        }
    }
}

#endif