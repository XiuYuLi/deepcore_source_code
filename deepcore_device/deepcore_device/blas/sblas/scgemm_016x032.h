__global__ void dk_scgemm_16x32( 
          char*              d_c, 
    const char* __restrict__ d_a, 
    const char* __restrict__ d_b, 
    float scale, unsigned int nbx, 
    int anr, int bnr, int cnc, int lda, int ldb )
{
    __shared__ char smem[1024+2048];
    float2 c[16]={{0.f,0.f}};
    float4 a[2], b[2];
    unsigned int bx=blockIdx.x;
    unsigned int z=blockIdx.y;
    unsigned int x=bx%nbx;
    unsigned int y=bx/nbx;
    unsigned int tid=threadIdx.x;
    unsigned int hanr=(anr+1)>>1;
    unsigned int hcnc=(cnc+1)>>1;
    unsigned int u=tid&3;
    unsigned int v=tid>>2;
    unsigned int su=__imad(x, 8,tid& 7);
    unsigned int sv=__imad(y,16,tid&15);
    unsigned int ox=__imad(x,16,tid&15);
    unsigned int oy=__imad(y,32,tid>>4);
    unsigned int qlda=lda<<2;
    unsigned int dldb=ldb<<1;
    d_a+=(z*bnr+(tid>>3))*lda+((su<hanr?su:(hanr-1))<<4);
    d_b+=(z*bnr+(tid>>4))*ldb+((sv<hcnc?sv:(hcnc-1))<<4);
    d_c+=(z*cnc+oy)*lda+(ox<<3);    
    float4* __restrict__ sst =(float4*)&smem[tid<<4];
    float4* __restrict__ asld=(float4*)&smem[u<<4];
    float4* __restrict__ bsld=(float4*)&smem[0x400+(v<<4)];
    float4 p0=*((const float4* __restrict__)d_a); d_a+=qlda;
    float4 p1=*((const float4* __restrict__)d_a);
    float4 p2=*((const float4* __restrict__)d_b); d_b+=dldb;
    float4 p3=*((const float4* __restrict__)d_b); d_b+=dldb;
    float4 p4=*((const float4* __restrict__)d_b); d_b+=dldb;
    float4 p5=*((const float4* __restrict__)d_b);

    for( int k=bnr-8; k>0; k-=8 )
    {
        sst[0*32]=p0;
        sst[1*32]=p1;
        sst[2*32]=p2;
        sst[3*32]=p3;
        sst[4*32]=p4;
        sst[5*32]=p5;
        float4 q0=*((const float4* __restrict__)(d_a+=qlda));
        float4 q1=*((const float4* __restrict__)(d_a+=qlda));
        float4 q2=*((const float4* __restrict__)(d_b+=dldb));
        float4 q3=*((const float4* __restrict__)(d_b+=dldb));
        float4 q4=*((const float4* __restrict__)(d_b+=dldb));
        float4 q5=*((const float4* __restrict__)(d_b+=dldb));
        __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){
            a[0]=asld[i* 8+0];
            a[1]=asld[i* 8+4];
            b[0]=bsld[i*16+0];
            b[1]=bsld[i*16+8];
            CFMA4x4(c,a,b)
        } __syncthreads();
        p0=q0;
        p1=q1;
        p2=q2;
        p3=q3;
        p4=q4;
        p5=q5;
    }
    sst[0*32]=p0;
    sst[1*32]=p1;
    sst[2*32]=p2;
    sst[3*32]=p3;
    sst[4*32]=p4;
    sst[5*32]=p5;
    __syncthreads();
#pragma unroll
    for( int i=0; i<8; ++i ){
        a[0]=asld[i* 8+0];
        a[1]=asld[i* 8+4];
        b[0]=bsld[i*16+0];
        b[1]=bsld[i*16+8];
        CFMA4x4(c,a,b)
    }
    scgemm_epilog16x32( d_c, smem, c, tid, u, v, anr-ox, cnc-oy, lda, scale );
}