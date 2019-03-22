__global__ void dk_scgemm_32x32( 
          char*              d_c, 
    const char* __restrict__ d_a, 
    const char* __restrict__ d_b, 
    float scale, unsigned int nbx, 
    int anr, int bnr, int cnc, 
    int lda, int ldb )
{
    __shared__ char smem[4096];
    float2 c[16]={{0.f,0.f}};
    float4 a[2], b[2];
    unsigned int bx=blockIdx.x;
    unsigned int z=blockIdx.y;
    unsigned int x=bx%nbx;
    unsigned int y=bx/nbx;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&31;
    unsigned int slot=tid>>5;
    unsigned int u=__imad(x,32,lane);
    unsigned int v=__imad(y,32,slot<<4);
    unsigned int hlane=tid&15;
    unsigned int hslot=tid>>4;
    unsigned int hanr=(anr+1)>>1;
    unsigned int hcnc=(cnc+1)>>1;
    unsigned int vi=__imad(x,16,hlane);
    unsigned int hi=__imad(y,16,hlane);
    unsigned int qlda=lda<<2;
    unsigned int qldb=ldb<<2;
    d_a+=(z*bnr+hslot)*lda+(vi<hanr?vi:(hanr-1))*16;
    d_b+=(z*bnr+hslot)*ldb+(hi<hcnc?hi:(hcnc-1))*16;
    d_c+=(z*cnc+v)*lda+(u<<3);      
    float4* __restrict__ sst =(float4*)&smem[tid<<4];
    float4* __restrict__ asld=(float4*)&smem[(lane&0xe)<<3];
    float4* __restrict__ bsld=(float4*)&smem[0x800|(slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];
    float4 p0=*((const float4* __restrict__)d_a); d_a+=qlda;
    float4 p1=*((const float4* __restrict__)d_a);
    float4 p2=*((const float4* __restrict__)d_b); d_b+=qldb;
    float4 p3=*((const float4* __restrict__)d_b);

    for( int k=bnr-8; k>0; k-=8 )
    {
        sst[0*64]=p0;
        sst[1*64]=p1;
        sst[2*64]=p2;
        sst[3*64]=p3;
        float4 q0=*((const float4* __restrict__)(d_a+=qlda));
        float4 q1=*((const float4* __restrict__)(d_a+=qlda));
        float4 q2=*((const float4* __restrict__)(d_b+=qldb));
        float4 q3=*((const float4* __restrict__)(d_b+=qldb));
        __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){
            a[0]=asld[i*16+0];
            a[1]=asld[i*16+8];
            b[0]=bsld[i*16+0];
            b[1]=bsld[i*16+4];
            CFMA4x4(c,a,b)
        } __syncthreads();
        p0=q0;
        p1=q1;
        p2=q2;
        p3=q3;
    }
    sst[0*64]=p0;
    sst[1*64]=p1;
    sst[2*64]=p2;
    sst[3*64]=p3;
    __syncthreads();
#pragma unroll
    for( int i=0; i<8; ++i ){
        a[0]=asld[i*16+0];
        a[1]=asld[i*16+8];
        b[0]=bsld[i*16+0];
        b[1]=bsld[i*16+4];
        CFMA4x4(c,a,b)
    }
    scgemm_epilog32x16( d_c, &smem[slot<<10], c, lane, anr-u, cnc-v, lda, scale );
}