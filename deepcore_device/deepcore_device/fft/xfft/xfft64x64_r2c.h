__global__ void dk_xfft64x64_r2c( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF, unsigned int nx, unsigned int ny, unsigned int ldr )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=(by*gridDim.x+bx)*33*64+tid;
    d_r+=by*ldr+(bx<<12)+tid;
    ((float*)s_RF)[tid]=d_RF[tid];
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x=__half2float(d_r[(2*i+0)*64]);
        c[i].y=__half2float(d_r[(2*i+1)*64]);
    } __syncthreads();
    s_vfft64( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}
__global__ void dk_xfft64x64_r2c_ext( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF, unsigned int nx, unsigned int ny, unsigned int ldr )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=(by*gridDim.x+bx)*33*64+tid;
    d_r+=by*ldr+bx*ny*nx+tid;
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i].x=c[i].y=0.f; }
    ((float*)s_RF)[tid]=d_RF[tid];
    if(tid<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((i*2+0)<ny){ c[i].x=__half2float(*d_r); } d_r+=nx;
            if((i*2+1)<ny){ c[i].y=__half2float(*d_r); } d_r+=nx;
        }
    } __syncthreads();
    s_vfft64( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}
__global__ void dk_xfft64x64_r2c_pad( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldr, int pad_x, int pad_y )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    int ox=(int)tid-pad_x;
    int oy=-pad_y;
    d_c+=(by*gridDim.x+bx)*33*64+tid;
    d_r+=by*ldr+bx*ny*nx+ox;
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i].x=c[i].y=0.f; }
    ((float*)s_RF)[tid]=d_RF[tid];
    if((ox>=0)&(ox<nx)){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((oy>=0)&(oy<ny)){ c[i].x=__half2float(*d_r); d_r+=nx; } ++oy;
            if((oy>=0)&(oy<ny)){ c[i].y=__half2float(*d_r); d_r+=nx; } ++oy;
        }
    } __syncthreads();
    s_vfft64( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}
__global__ void dk_xfft64x64_r2c_flip( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF, unsigned int nx, unsigned int ny, unsigned int ldr )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=(by*gridDim.x+bx)*33*64+tid;
    d_r+=by*ldr+(bx+1)*ny*nx-tid-1;
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i].x=c[i].y=0.f; }
    ((float*)s_RF)[tid]=d_RF[tid];
    if(tid<nx){
        c[0].x=__half2float(*d_r);
        c[0].y=__half2float(*(d_r-=nx));
    #pragma unroll
        for( int i=1; i<32; ++i ){
            if((i*2+0)<ny){ c[i].x=__half2float(*(d_r-=nx)); }
            if((i*2+1)<ny){ c[i].y=__half2float(*(d_r-=nx)); }
        }
    } __syncthreads();
    s_vfft64( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}