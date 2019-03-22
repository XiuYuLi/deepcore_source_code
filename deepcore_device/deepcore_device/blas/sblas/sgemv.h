__global__ void dk_sgemvn( 
          char*              d_c, 
    const char* __restrict__ d_a, 
    const char* __restrict__ d_b,
    float alpha, int nx, int ny, int lda )
{
    __shared__ float smem[128];
    float c[16]={0.f}, a[16], x;
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int lane=tid&31;
    int slot=tid>>5;
    int start=bid*64+slot;
    int ia=lane;
    int ib=tid;
    d_a+=start*lda;
    if(ib<nx){ x=((const float* __restrict__)d_b)[ib]; }
    int pnx=(nx+127)&~127;
    float* spx=&smem[tid];
    float* spy=&smem[lane];
    for( int s=0; s<pnx; s+=128 )
    {
        *spx=ib<nx?x:0.f;
        __syncthreads();
        ib+=128;
        if(ib<nx){ x=((const float* __restrict__)d_b)[ib]; }
    #pragma unroll
        for( int k=0; k<4; ++k ){    
        #pragma unroll
            for( int i=0; i<16; ++i ){ if(((start+4*i)<ny)&(ia<nx)){ a[i]=d_a[4*i*lda+ia*4]; } }
            ia+=32;
            float b=spy[k*32];
        #pragma unroll
            for( int i=0; i<16; ++i ){ c[i]+=a[i]*b; }
        } __syncthreads();
    }
#pragma unroll
    for( int i=0; i<16; ++i ){ ReduceUnit(add,c[i]) }
    if(lane==0){
        spy=&smem[slot];
    #pragma unroll
        for( int i=0; i<16; ++i ){ spy[i*4]=c[i]; }
    } __syncthreads();
    int col=bid*64+(tid&63);
    if((col<ny)&(tid<64)){ ((float*)d_c)[col]=alpha*spx[0]; }
}
__global__ void dk_sgemvt( 
          char*              d_c, 
    const char* __restrict__ d_a, 
    const char* __restrict__ d_b,
    float alpha, int nr, int nc, int lda )
{
    __shared__ float smem[512];
    float c[16]={0.f}, a[16], b[4], x;
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int lane=tid&31;
    int slot=tid>>5;
    int start=bid*128+lane;
    int ia=lane;
    int ib=tid;
    d_a+=slot*lda+start;
    d_b+=tid;
    x=ib<nc?((const float* __restrict__)d_b)[ib]:0.f;
    int pnc=(nc+127)&~127;
    float* spx=&smem[tid];
    float* spy=&smem[slot];
    for( int s=pnc-128; s>=0; s-=128 )
    {
        spx[0]=x;
        __syncthreads();
        ib+=128;
        x=ib<nc?((const float* __restrict__)d_b)[ib]:0.f;
    #pragma unroll
        for( int i=0; i<8; ++i ){
            a[ 0]=((const float* __restrict__)d_a)[0*4*lda+0*32];
            a[ 1]=((const float* __restrict__)d_a)[0*4*lda+1*32];
            a[ 2]=((const float* __restrict__)d_a)[0*4*lda+2*32];
            a[ 3]=((const float* __restrict__)d_a)[0*4*lda+3*32];
            a[ 4]=((const float* __restrict__)d_a)[1*4*lda+0*32];
            a[ 5]=((const float* __restrict__)d_a)[1*4*lda+1*32];
            a[ 6]=((const float* __restrict__)d_a)[1*4*lda+2*32];
            a[ 7]=((const float* __restrict__)d_a)[1*4*lda+3*32];
            a[ 8]=((const float* __restrict__)d_a)[2*4*lda+0*32];
            a[ 9]=((const float* __restrict__)d_a)[2*4*lda+1*32];
            a[10]=((const float* __restrict__)d_a)[2*4*lda+2*32];
            a[11]=((const float* __restrict__)d_a)[2*4*lda+3*32];
            a[12]=((const float* __restrict__)d_a)[3*4*lda+0*32];
            a[13]=((const float* __restrict__)d_a)[3*4*lda+1*32];
            a[14]=((const float* __restrict__)d_a)[3*4*lda+2*32];
            a[15]=((const float* __restrict__)d_a)[3*4*lda+3*32];
            b[0]=spy[i*16+0*4];
            b[1]=spy[i*16+1*4];
            b[2]=spy[i*16+2*4];
            b[3]=spy[i*16+3*4];
            c[ 0]+=a[ 0]*b[0];
            c[ 1]+=a[ 1]*b[0];
            c[ 2]+=a[ 2]*b[0];
            c[ 3]+=a[ 3]*b[0];
            c[ 4]+=a[ 4]*b[1];
            c[ 5]+=a[ 5]*b[1];
            c[ 6]+=a[ 6]*b[1];
            c[ 7]+=a[ 7]*b[1];
            c[ 8]+=a[ 8]*b[2];
            c[ 9]+=a[ 9]*b[2];
            c[10]+=a[10]*b[2];
            c[11]+=a[11]*b[2];
            c[12]+=a[12]*b[3];
            c[13]+=a[13]*b[3];
            c[14]+=a[14]*b[3];
            c[15]+=a[15]*b[3];
            d_a+=16*lda;
        }
    }
    c[ 0]+=c[ 4];
    c[ 1]+=c[ 5];
    c[ 2]+=c[ 6];
    c[ 3]+=c[ 7];
    c[ 8]+=c[12];
    c[ 9]+=c[13];
    c[10]+=c[14];
    c[11]+=c[15];
    c[ 0]+=c[ 8];
    c[ 1]+=c[ 9];
    c[ 2]+=c[10];
    c[ 3]+=c[11];
    spy=&smem[slot*128+lane];
    spy[0*32]=c[0];
    spy[1*32]=c[1];
    spy[2*32]=c[2];
    spy[3*32]=c[3];
    __syncthreads();
    if((bid*128+tid)<nr){ ((float*)d_c)[bid*128+tid]=alpha*(spx[0]+spx[128]+spx[256]+spx[384]);}
}