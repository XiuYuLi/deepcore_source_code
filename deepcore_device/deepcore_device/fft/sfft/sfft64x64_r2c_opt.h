__global__ void dk_sfft64x64_r2c_s3( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=bid*33*64+tid;
    d_r+=bid*9+tid;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    ((float*)s_RF)[tid]=d_RF[tid];
    if(tid<3){
        c[0].x=d_r[0];
        c[0].y=d_r[3];
        c[1].x=d_r[6];
    } __syncthreads();
    s_vfft64_s3( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64_s3( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}
__global__ void dk_sfft64x64_r2c_s5( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=bid*33*64+tid;
    d_r+=bid*25+tid;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    ((float*)s_RF)[tid]=d_RF[tid];
    if(tid<5){
        c[0].x=d_r[0*5];
        c[0].y=d_r[1*5];
        c[1].x=d_r[2*5];
        c[1].y=d_r[3*5];
        c[2].x=d_r[4*5];
    } __syncthreads();
    s_vfft64_s5( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64_s5( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}
__global__ void dk_sfft64x64_r2c_s7( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF, unsigned int ldr )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=bid*33*64+tid;
    d_r+=bid*49+tid;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    c[3].x=0.f;
    c[3].y=0.f;
    ((float*)s_RF)[tid]=d_RF[tid];
    if(tid<7){
        c[0].x=d_r[0*7];
        c[0].y=d_r[1*7];
        c[1].x=d_r[2*7];
        c[1].y=d_r[3*7];
        c[2].x=d_r[4*7];
        c[2].y=d_r[5*7];
        c[3].x=d_r[6*7];
    } __syncthreads();
    s_vfft64_s7( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64_s7( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}
__global__ void dk_sfft64x64_r2c_flip_s3( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=bid*33*64+tid;
    d_r+=bid*9+2-tid;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    ((float*)s_RF)[tid]=d_RF[tid];
    if(tid<3){
        c[0].x=d_r[6];
        c[0].y=d_r[3];
        c[1].x=d_r[0];
    } __syncthreads();
    s_vfft64_s3( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64_s3( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}
__global__ void dk_sfft64x64_r2c_flip_s5( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=bid*33*64+tid;
    d_r+=bid*25+4-tid;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    ((float*)s_RF)[tid]=d_RF[tid];
    if(tid<5){
        c[0].x=d_r[20];
        c[0].y=d_r[15];
        c[1].x=d_r[10];
        c[1].y=d_r[ 5];
        c[2].x=d_r[ 0];
    } __syncthreads();
    s_vfft64_s5( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64_s5( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}
__global__ void dk_sfft64x64_r2c_flip_s7( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*33];
    __shared__ float2 s_RF[32];
    float2 c[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int u=tid&1;
    unsigned int v=tid>>1;
    d_c+=bid*33*64+tid;
    d_r+=bid*49+6-tid;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    c[3].x=0.f;
    c[3].y=0.f;
    ((float*)s_RF)[tid]=d_RF[tid];
    if(tid<7){
        c[0].x=d_r[42];
        c[0].y=d_r[35];
        c[1].x=d_r[28];
        c[1].y=d_r[21];
        c[2].x=d_r[14];
        c[2].y=d_r[ 7];
        c[3].x=d_r[ 0];
    } __syncthreads();
    s_vfft64_s7( c, &smem[tid], &smem[v*66+u], s_RF, brev );
    s_hfft64_s7( c, smem, s_RF, brev, tid, u, v );
#pragma unroll
    for( int i=0; i<33; ++i ){ d_c[i*64]=c[i]; }
}