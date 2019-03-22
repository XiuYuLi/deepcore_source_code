__global__ void dk_xgemvn( 
          char*              d_c, 
    const char* __restrict__ d_a, 
    const char* __restrict__ d_b,
    float alpha, int nx, int ny, int lda )
{
    __shared__ char smem[512];
    float c[16]={0.f};
    float2 a[16];
    __half2 x;
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int lane=tid&15;
    int slot=tid>>4;
    int start=bid*64+slot;
    int ia=lane;
    int ib=tid;
    int hnx=(nx+1)>>1;
    int hny=(ny+1)>>1;
    d_a+=start*lda;
    if(ib<hnx){ x=((const __half2* __restrict__)d_b)[ib]; }
    int pnx=(hnx+63)&~63;
    float2* spx=&((float2*)smem)[tid];
    float2* spy=&((float2*)smem)[lane];
    for( int s=0; s<pnx; s+=64 )
    {
        *spx=ib<hnx?__half22float2(x):HZERO;
        __syncthreads();
        ib+=64;
        if(ib<hnx){ x=((const __half2* __restrict__)d_b)[ib]; }
    #pragma unroll
        for( int k=0; k<4; ++k ){    
        #pragma unroll
            for( int i=0; i<16; ++i ){ if(((start+4*i)<ny)&(ia<hnx)){ a[i]=__half22float2(*((const __half2* __restrict__)&d_a[4*i*lda+ia*4])); } }
            ia+=16;
            float2 b=spy[k*16];
        #pragma unroll
            for( int i=0; i<16; ++i ){
                c[i]+=a[i].x*b.x;
                c[i]+=a[i].y*b.y;
            }
        } __syncthreads();
    }
#pragma unroll
    for( int i=0; i<16; ++i ){ ReduceUnit(add,c[i],16) }
    if(lane==0){
        float* spz=&smem[slot];
    #pragma unroll
        for( int i=0; i<16; ++i ){ spz[i*4]=c[i]; }
    } __syncthreads();
    int col=bid*32+(tid&31);
    if((col<hny)&(tid<32)){
        x=spx[0];
        ((__half2*)d_c)[col]=__halves2half2(__float2half(alpha*x.x),__float2half(alpha*x.y)); 
    }
}