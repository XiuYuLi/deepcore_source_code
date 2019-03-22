__global__ void dk_scgemm_64x64( 
          char*              d_c, 
    const char* __restrict__ d_a, 
    const char* __restrict__ d_b, 
    float scale, unsigned int nbx, 
    int anr, int bnr, int cnc, 
    int lda, int ldb )
{
    __shared__ char smem[8192];
    float2 c[16]={{0.f,0.f}};
    float4 a[2], b[2];
    unsigned int bx=blockIdx.x;
    unsigned int z=blockIdx.y;
    unsigned int x=bx%nbx;
    unsigned int y=bx/nbx;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&31;
    unsigned int slot=tid>>5;
    unsigned int slot_x=slot&1;
    unsigned int slot_y=slot>>1;
    unsigned int u=(x<<6)|(slot_x<<5)|lane;
    unsigned int v=(y<<6)|(slot_y<<4);
    unsigned int hanr=(anr+1)>>1;
    unsigned int hcnc=(cnc+1)>>1;
    unsigned int vi=__imad(x,32,lane);
    unsigned int hi=__imad(y,32,lane);  
    unsigned int elda=lda<<3;
    unsigned int eldb=ldb<<3;
    d_a+=(z*bnr+slot)*lda+(vi<hanr?vi:(hanr-1))*16;
    d_b+=(z*bnr+slot)*ldb+(hi<hcnc?hi:(hcnc-1))*16;
    d_c+=(z*cnc+v)*lda+u*8;     
    float4* __restrict__ sst =(float4*)&smem[tid<<4];
    float4* __restrict__ asld=(float4*)&smem[(slot_x<<8)|((lane&0xe)<<3)];
    float4* __restrict__ bsld=(float4*)&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];
    float4 p0=*((const float4* __restrict__)d_a);
    float4 p1=*((const float4* __restrict__)d_b);

    for( int k=bnr-8; k>0; k-=8 )
    {    
        sst[0x000]=p0;
        sst[0x100]=p1;
        float4 q0=*((const float4* __restrict__)(d_a+=elda));
        float4 q1=*((const float4* __restrict__)(d_b+=eldb));
        __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){
            a[0]=asld[i*32+0];
            a[1]=asld[i*32+8];
            b[0]=bsld[i*32+0];
            b[1]=bsld[i*32+4];
            CFMA4x4(c,a,b)
        } __syncthreads();
        p0=q0;
        p1=q1;
    }
    sst[0x000]=p0;
    sst[0x100]=p1;
    __syncthreads();
#pragma unroll
    for( int i=0; i<8; ++i ){
        a[0]=asld[i*32+0];
        a[1]=asld[i*32+8];
        b[0]=bsld[i*32+0];
        b[1]=bsld[i*32+4];
        CFMA4x4(c,a,b)
    }
    scgemm_epilog32x16( d_c, &smem[slot<<10], c, lane, anr-u, cnc-v, lda, scale );
}