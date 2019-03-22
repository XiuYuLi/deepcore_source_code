__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c( __half2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldr, unsigned int n, unsigned int n_cells )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;    
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=(icell/n)*ldr+((icell%n)<<10)+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(icell<n_cells){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            c[i].x=__half2float(d_r[(2*i+0)*32]);
            c[i].y=__half2float(d_r[(2*i+1)*32]);
        }
    } __syncthreads();
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=__float22half2_rn(c[i]); }
}
__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_ext( __half2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldr, unsigned int n, unsigned int n_cells )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;    
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=(icell/n)*ldr+(icell%n)*ny*nx+x;
    CLEAR16C(c)
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if((icell<n_cells)&(x<nx)){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((2*i+0)<ny){ c[i].x=__half2float(*d_r); } d_r+=nx;
            if((2*i+1)<ny){ c[i].y=__half2float(*d_r); } d_r+=nx;
        } 
    } __syncthreads();
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=__float22half2_rn(c[i]); }
}
__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_pad( __half2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldr, unsigned int n, unsigned int n_cells, int pad_x, int pad_y )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;    
    unsigned int icell=(blockIdx.x<<3)+y;
    int ox=(int)x-pad_x;
    int oy=-pad_y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=(icell/n)*ldr+(icell%n)*ny*nx+ox;
    CLEAR16C(c)
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if((icell<n_cells)&(ox>=0)&(ox<nx)){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((oy>=0)&(oy<ny)){ c[i].x=__half2float(*d_r); d_r+=nx; } ++oy;
            if((oy>=0)&(oy<ny)){ c[i].y=__half2float(*d_r); d_r+=nx; } ++oy;
        }
    } __syncthreads();
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=__float22half2_rn(c[i]); }
}
__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_flip( __half2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldr, unsigned int n, unsigned int n_cells )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;    
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=(icell/n)*ldr+((icell%n)+1)*ny*nx-x-1;
    CLEAR16C(c)
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if((icell<n_cells)&(x<nx)){
        c[0].x=__half2float(*d_r);
        c[0].y=__half2float(*(d_r-=nx));
    #pragma unroll
        for( int i=1; i<16; ++i ){
            if((i*2+0)<ny){ c[i].x=__half2float(*(d_r-=nx)); }
            if((i*2+1)<ny){ c[i].y=__half2float(*(d_r-=nx)); }
        }
    } __syncthreads();
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=__float22half2_rn(c[i]); }
}