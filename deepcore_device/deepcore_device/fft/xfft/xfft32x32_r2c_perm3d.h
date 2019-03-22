__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm3d( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, unsigned int n_cells, int dir )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int bx=blockIdx.x; 
    unsigned int by=blockIdx.y;
    unsigned int gdy=gridDim.y;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldy+icell*ldx+x;
    if(icell<n_cells){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            c[i].x=__half2float(d_r[(2*i+0)*32]);
            c[i].y=__half2float(d_r[(2*i+1)*32]);
        } 
    } __syncthreads();
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*ldc*gdy );
}
__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm3d_ext( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, unsigned int n_cells, int dir )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int gdy=gridDim.y;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldy+icell*ldx+x;
    CLEAR16C(c)
    if((icell<n_cells)&(x<nx)){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((2*i+0)<ny){ c[i].x=__half2float(*d_r); } d_r+=nx;
            if((2*i+1)<ny){ c[i].y=__half2float(*d_r); } d_r+=nx;
        }
    } __syncthreads();
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*gdy*ldc );
}
__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm3d_pad( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, unsigned int n_cells, int pad_x, int pad_y )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int inc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    int ox=(int)x-pad_x;
    int oy=-pad_y;
    CLEAR16C(c)
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*inc+by)*ldc+(bx<<4)+p;
    d_r+=by*ldr+icell*ny*nx+ox;
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
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*ldc*inc );
}
__global__ void __launch_bounds__(512,1) dk_xfft32x32_r2c_perm3d_flip( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, unsigned int n_cells, int dir )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[16*545];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int gdy=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldy+icell*ldx+ny*nx-x-1;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    CLEAR16C(c)
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
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*gdy*ldc );
}