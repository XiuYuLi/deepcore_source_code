__global__ void dk_xfft16x16_r2c_perm3d( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, unsigned int n_cells, int dir )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int inc=gridDim.y;
    unsigned int y=tid>>4;
    unsigned int x=tid&15;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*ldy+icell*ldx+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(icell<n_cells){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            c[i].x=__half2float(d_r[(i*2+0)*16]);
            c[i].y=__half2float(d_r[(i*2+1)*16]);
        } 
    } __syncthreads();
    s_vfft16( c, spx, spy, s_RF, brev );
    s_hfft16( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}
__global__ void dk_xfft16x16_r2c_perm3d_ext( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, unsigned int n_cells, int dir )
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
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*ldy+icell*ldx+x;
    CLEAR8C(c)
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if((icell<n_cells)&(x<nx)){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ c[i].x=__half2float(*d_r); } d_r+=nx;
            if((2*i+1)<ny){ c[i].y=__half2float(*d_r); } d_r+=nx;
        } 
    } __syncthreads();
    s_vfft16( c, spx, spy, s_RF, brev );
    s_hfft16( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}
__global__ void dk_xfft16x16_r2c_perm3d_pad( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, unsigned int n_cells, int pad_x, int pad_y )
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
    unsigned int icell=(bx<<4)+y;
    int ox=(int)x-pad_x;
    int oy=-pad_y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*ldr+icell*ny*nx+ox;
    CLEAR8C(c)
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if((icell<n_cells)&(ox>=0)&(ox<nx)){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((oy>=0)&(oy<ny)){ c[i].x=__half2float(*d_r); d_r+=nx; } ++oy;
            if((oy>=0)&(oy<ny)){ c[i].y=__half2float(*d_r); d_r+=nx; } ++oy;
        }
    } __syncthreads();
    s_vfft16( c, spx, spy, s_RF, brev );
    s_hfft16( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}
__global__ void dk_xfft16x16_r2c_perm3d_flip( float2* d_c, 
    const __half* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, unsigned int n_cells, int dir )
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
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    unsigned int icell=(bx<<4)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=(y*inc+by)*ldc+(bx<<4)+x;
    d_r+=by*ldy+icell*ldx+ny*nx-x-1;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    CLEAR8C(c)
    if((icell<n_cells)&(x<nx)){
        c[0].x=__half2float(*d_r);
        c[0].y=__half2float(*(d_r-=nx));
    #pragma unroll
        for( int i=1; i<8; ++i ){
            if((2*i+0)<ny){ c[i].x=__half2float(*(d_r-=nx)); }
            if((2*i+1)<ny){ c[i].y=__half2float(*(d_r-=nx)); }
        }
    } __syncthreads();
    s_vfft16( c, spx, spy, s_RF, brev );
    s_hfft16( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
    s_store9( d_c, &smem[y*145+x], &smem[x*145+y], c, 16*ldc*inc );
}