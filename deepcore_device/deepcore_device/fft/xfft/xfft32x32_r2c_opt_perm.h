__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm_s3( 
    float2* d_c, const __half* __restrict__ d_r, 
    const float* __restrict__ d_RF, int ldc, int ldr )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    int tid=threadIdx.x;    
    int x=tid&31;
    int y=tid>>5;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int gdy=gridDim.y;
    int p=tid&15;
    int q=tid>>4;
    int u=x&1;
    int v=x>>1;
    int icell=(bx<<4)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*9+icell*ldr+x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<3){
        c[0].x=__half2float(d_r[0]);
        c[0].y=__half2float(d_r[3]);
        c[1].x=__half2float(d_r[6]);
    } __syncthreads();
    s_vfft_s3( c, spx, spy, s_RF, brev );
    s_hfft_s3( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*gdy*ldc );
}
__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm_s5( 
    float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    int ldc, int ldr )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    int tid=threadIdx.x;
    int x=tid&31;   
    int y=tid>>5;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int gdy=gridDim.y;
    int p=tid&15;
    int q=tid>>4;
    int u=x&1;
    int v=x>>1;
    int icell=(bx<<4)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*25+icell*ldr+x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<5){
        c[0].x=__half2float(d_r[0*5]);
        c[0].y=__half2float(d_r[1*5]);
        c[1].x=__half2float(d_r[2*5]);
        c[1].y=__half2float(d_r[3*5]);
        c[2].x=__half2float(d_r[4*5]);
    } __syncthreads();
    s_vfft_s5( c, spx, spy, s_RF, brev );
    s_hfft_s5( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*gdy*ldc );
}
__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm_s7( 
    float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    int ldc, int ldr )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    int tid=threadIdx.x;
    int x=tid&31;   
    int y=tid>>5;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int gdy=gridDim.y;
    int p=tid&15;
    int q=tid>>4;
    int u=x&1;
    int v=x>>1;
    int icell=(bx<<4)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*49+icell*ldr+x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    c[3].x=0.f;
    c[3].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<7){
        c[0].x=__half2float(d_r[0*7]);
        c[0].y=__half2float(d_r[1*7]);
        c[1].x=__half2float(d_r[2*7]);
        c[1].y=__half2float(d_r[3*7]);
        c[2].x=__half2float(d_r[4*7]);
        c[2].y=__half2float(d_r[5*7]);
        c[3].x=__half2float(d_r[6*7]);
    } __syncthreads();
    s_vfft_s7( c, spx, spy, s_RF, brev );
    s_hfft_s7( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*gdy*ldc );
}
__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm_flip_s3( 
    float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    int ldc, int ldr )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int gdy=gridDim.y;
    int tid=threadIdx.x;
    int x=tid&31;
    int y=tid>>5;
    int p=tid&15;
    int q=tid>>4;
    int u=x&1;
    int v=x>>1;
    int icell=(bx<<4)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldr+icell*9+2-x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<3){
        c[0].x=__half2float(d_r[6]);
        c[0].y=__half2float(d_r[3]);
        c[1].x=__half2float(d_r[0]);
    } __syncthreads();
    s_vfft_s3( c, spx, spy, s_RF, brev );
    s_hfft_s3( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*gdy*ldc );
}
__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm_flip_s5( 
    float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    int ldc, int ldr )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int gdy=gridDim.y;
    int tid=threadIdx.x;
    int x=tid&31;
    int y=tid>>5;
    int p=tid&15;
    int q=tid>>4;
    int u=x&1;
    int v=x>>1;
    int icell=(bx<<4)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldr+icell*25+4-x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<5){
        c[0].x=__half2float(d_r[20]);
        c[0].y=__half2float(d_r[15]);
        c[1].x=__half2float(d_r[10]);
        c[1].y=__half2float(d_r[ 5]);
        c[2].x=__half2float(d_r[ 0]);
    } __syncthreads();
    s_vfft_s5( c, spx, spy, s_RF, brev );
    s_hfft_s5( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*gdy*ldc );
}
__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm_flip_s7( 
    float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    int ldc, int ldr )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int gdy=gridDim.y;
    int tid=threadIdx.x;
    int x=tid&31;
    int y=tid>>5;
    int p=tid&15;
    int q=tid>>4;
    int u=x&1;
    int v=x>>1;
    int icell=(bx<<4)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldr+icell*49+6-x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    c[3].x=0.f;
    c[3].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<7){
        c[0].x=__half2float(d_r[42]);
        c[0].y=__half2float(d_r[35]);
        c[1].x=__half2float(d_r[28]);
        c[1].y=__half2float(d_r[21]);
        c[2].x=__half2float(d_r[14]);
        c[2].y=__half2float(d_r[ 7]);
        c[3].x=__half2float(d_r[ 0]);
    } __syncthreads();
    s_vfft_s7( c, spx, spy, s_RF, brev );
    s_hfft_s7( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*gdy*ldc );
}