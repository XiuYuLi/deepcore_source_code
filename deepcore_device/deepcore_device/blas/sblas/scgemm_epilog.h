#ifndef __scgemm_epilog_h__
#define __scgemm_epilog_h__

__forceinline__ __device__ void scgemm_epilog32x16( char* d_c, char* sptr, float2* c, int lane, int nx, int ny, int ldc, float alpha )
{
    unsigned int x=lane&15;
    unsigned int y=lane>>4;
    float2* sst=(float2*)&sptr[(y<<9)|((x&1)<<8)|((x>>1)<<4)];
    float2* sld=(float2*)&sptr[lane<<3];
#pragma unroll
    for( int i=0; i<16; ++i ){ c[i].x*=alpha; c[i].y*=alpha; }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<8; i+=4 )
        {
            __syncthreads();
            sst[ 0]=c[k*8+i+0];
            sst[ 1]=c[k*8+i+1];
            sst[16]=c[k*8+i+2];
            sst[17]=c[k*8+i+3];
            __syncthreads();
            if(nx>0)
            {
                if((k*8+0+i/4)<ny){ *((float2*)&d_c[(k*8+0+i/4)*ldc])=sld[0*32]; }
                if((k*8+2+i/4)<ny){ *((float2*)&d_c[(k*8+2+i/4)*ldc])=sld[1*32]; }
                if((k*8+4+i/4)<ny){ *((float2*)&d_c[(k*8+4+i/4)*ldc])=sld[2*32]; }
                if((k*8+6+i/4)<ny){ *((float2*)&d_c[(k*8+6+i/4)*ldc])=sld[3*32]; }
            }
        }
    }
}
__forceinline__ __device__ void scgemm_epilog16x32( char* d_c, char* sptr, float2* c, int lane, int u, int v, int nx, int ny, int ldc, float alpha )
{
    float2* sst=(float2*)&sptr[(v<<8)|(u<<4)];
    float2* sld=(float2*)&sptr[lane<<3];
#pragma unroll
    for( int i=0; i<16; ++i ){ c[i].x*=alpha; c[i].y*=alpha; }
#pragma unroll
    for( int i=0; i<2; ++i )
    {
        __syncthreads();
        sst[   0]=c[i*8+0];
        sst[   1]=c[i*8+1];
        sst[   8]=c[i*8+2];
        sst[   9]=c[i*8+3];
        sst[16+0]=c[i*8+4];
        sst[16+1]=c[i*8+5];
        sst[16+8]=c[i*8+6];
        sst[16+9]=c[i*8+7];
        __syncthreads();
        if(nx>0){
            if((i*16+ 0)<ny){ *((float2*)&d_c[(i*16+ 0)*ldc])=sld[0*32]; }
            if((i*16+ 2)<ny){ *((float2*)&d_c[(i*16+ 2)*ldc])=sld[1*32]; }
            if((i*16+ 4)<ny){ *((float2*)&d_c[(i*16+ 4)*ldc])=sld[2*32]; }
            if((i*16+ 6)<ny){ *((float2*)&d_c[(i*16+ 6)*ldc])=sld[3*32]; }
            if((i*16+ 8)<ny){ *((float2*)&d_c[(i*16+ 8)*ldc])=sld[4*32]; }
            if((i*16+10)<ny){ *((float2*)&d_c[(i*16+10)*ldc])=sld[5*32]; }
            if((i*16+12)<ny){ *((float2*)&d_c[(i*16+12)*ldc])=sld[6*32]; }
            if((i*16+14)<ny){ *((float2*)&d_c[(i*16+14)*ldc])=sld[7*32]; }
        }
    }
}

#endif