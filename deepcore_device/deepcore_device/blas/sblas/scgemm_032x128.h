__global__ void dk_scgemm_32x128( 
          char*              d_c, 
    const char* __restrict__ d_a, 
    const char* __restrict__ d_b, 
    float scale, unsigned int nbx, 
    int anr, int bnr, int cnc, 
    int lda, int ldb )
{
    __shared__ char smem[2048+8192];
    float2 c[16]={{0.f,0.f}};
    float4 a[2], b[2];
    unsigned int bx=blockIdx.x;
    unsigned int z=blockIdx.y;
    unsigned int x=bx%nbx;
    unsigned int y=bx/nbx;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&31;
    unsigned int slot=tid>>5;
    unsigned int hcnc=(cnc+1)>>1;
    unsigned int u=__imad(x, 32,lane);
    unsigned int h=__imad(y, 64,tid&63);
    unsigned int v=__imad(y,128,slot<<4);   
    unsigned int elda=lda<<3;
    unsigned int qldb=ldb<<2;
    d_a+=(z*bnr+slot    )*lda+(u< anr?u:( anr-1))* 8;
    d_b+=(z*bnr+(tid>>6))*ldb+(h<hcnc?h:(hcnc-1))*16;
    d_c+=(z*cnc+v)*lda+u*8;     
    float2* __restrict__ asst=(float2*)&smem[tid<<3];
    float4* __restrict__ bsst=(float4*)&smem[2048+(tid<<4)];
    float4* __restrict__ asld=(float4*)&smem[(lane&0xe)<<3];
    float4* __restrict__ bsld=(float4*)&smem[0x800|(slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];    
    float2 p0=*((const float2* __restrict__)d_a);
    float4 p1=*((const float4* __restrict__)d_b); d_b+=qldb;
    float4 p2=*((const float4* __restrict__)d_b);

    for( int k=bnr-8; k>0; k-=8 )
    {
        asst[0x000]=p0;
        bsst[0x000]=p1;
        bsst[0x100]=p2;
        float2 q0=*((const float2* __restrict__)(d_a+=elda));
        float4 q1=*((const float4* __restrict__)(d_b+=qldb));
        float4 q2=*((const float4* __restrict__)(d_b+=qldb));
        __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){
            a[0]=asld[i*16+0];
            a[1]=asld[i*16+8];
            b[0]=bsld[i*64+0];
            b[1]=bsld[i*64+4];
            CFMA4x4(c,a,b)
        } __syncthreads();
        p0=q0;
        p1=q1;
        p2=q2;
    }
    asst[0x000]=p0;
    bsst[0x000]=p1;
    bsst[0x100]=p2;
    __syncthreads();
#pragma unroll
    for( int i=0; i<8; ++i ){
        a[0]=asld[i*16+0];
        a[1]=asld[i*16+8];
        b[0]=bsld[i*64+0];
        b[1]=bsld[i*64+4];
        CFMA4x4(c,a,b)
    }
    scgemm_epilog32x16( d_c, &smem[slot<<10], c, lane, anr-u, cnc-v, lda, scale );
}