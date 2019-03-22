__global__ void dk_xfft16x16_r2c_perm_s3( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int ldc, unsigned int ldr )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int inc=gridDim.y;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*9+icell*ldr+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    if(x<3){
        c[0].x=__half2float(d_r[0]);
        c[0].y=__half2float(d_r[3]);
        c[1].x=__half2float(d_r[6]);
    } __syncthreads();
    s_vfft16_s3( c, spx, spy, s_RF, brev );
    s_hfft16_s3( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}
__global__ void dk_xfft16x16_r2c_perm_s5( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int ldc, unsigned int ldr )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int inc=gridDim.y;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*25+icell*ldr+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    if(x<5){
        c[0].x=__half2float(d_r[0*5]);
        c[0].y=__half2float(d_r[1*5]);
        c[1].x=__half2float(d_r[2*5]);
        c[1].y=__half2float(d_r[3*5]);
        c[2].x=__half2float(d_r[4*5]);
    } __syncthreads();
    s_vfft16_s5( c, spx, spy, s_RF, brev );
    s_hfft16_s5( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}
__global__ void dk_xfft16x16_r2c_perm_s7( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int ldc, unsigned int ldr )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int inc=gridDim.y;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*49+icell*ldr+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    c[3].x=0.f;
    c[3].y=0.f;
    if(x<7){
        c[0].x=__half2float(d_r[0*7]);
        c[0].y=__half2float(d_r[1*7]);
        c[1].x=__half2float(d_r[2*7]);
        c[1].y=__half2float(d_r[3*7]);
        c[2].x=__half2float(d_r[4*7]);
        c[2].y=__half2float(d_r[5*7]);
        c[3].x=__half2float(d_r[6*7]);
    } __syncthreads();
    s_vfft16_s7( c, spx, spy, s_RF, brev );
    s_hfft16_s7( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}
__global__ void dk_xfft16x16_r2c_perm_flip_s3( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int ldc, unsigned int ldr )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int inc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    int icell=(bx<<4)+y;
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*ldr+icell*9+2-x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    if(x<3){
        c[0].x=__half2float(d_r[6]);
        c[0].y=__half2float(d_r[3]);
        c[1].x=__half2float(d_r[0]);
    } __syncthreads();
    s_vfft16_s3( c, spx, spy, s_RF, brev );
    s_hfft16_s3( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}
__global__ void dk_xfft16x16_r2c_perm_flip_s5( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int ldc, unsigned int ldr )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int inc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    int icell=(bx<<4)+y;
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*ldr+icell*25+4-x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    if(x<5){
        c[0].x=__half2float(d_r[20]);
        c[0].y=__half2float(d_r[15]);
        c[1].x=__half2float(d_r[10]);
        c[1].y=__half2float(d_r[ 5]);
        c[2].x=__half2float(d_r[ 0]);
    } __syncthreads();
    s_vfft16_s5( c, spx, spy, s_RF, brev );
    s_hfft16_s5( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}
__global__ void dk_xfft16x16_r2c_perm_flip_s7( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int ldc, unsigned int ldr )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int inc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    int icell=(bx<<4)+y;
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*ldr+icell*49+6-x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    c[3].x=0.f;
    c[3].y=0.f;
    if(x<7){
        c[0].x=__half2float(d_r[42]);
        c[0].y=__half2float(d_r[35]);
        c[1].x=__half2float(d_r[28]);
        c[1].y=__half2float(d_r[21]);
        c[2].x=__half2float(d_r[14]);
        c[2].y=__half2float(d_r[ 7]);
        c[3].x=__half2float(d_r[ 0]);
    } __syncthreads();
    s_vfft16_s7( c, spx, spy, s_RF, brev );
    s_hfft16_s7( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}