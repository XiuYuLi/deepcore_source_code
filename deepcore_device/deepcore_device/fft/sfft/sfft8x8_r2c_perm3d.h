__global__ void dk_sfft8x8_r2c_perm3d( float2* d_c, 
    const float* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, int n_cells, int dir )
{
    const int brev[]={0,2,1,3};
    __shared__ float  smem[16*41];
    __shared__ float2 s_RF[4];
    float2 c[5];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int gdy=gridDim.y;
    unsigned int x=tid&7;
    unsigned int y=tid>>3;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    float* spx=&smem[y*40+x];
    float* spy=&smem[y*40+v*10+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldy+icell*ldx+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(icell<n_cells){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            c[i].x=d_r[(i*2+0)*8];
            c[i].y=d_r[(i*2+1)*8];
        } 
    } __syncthreads();
    s_vfft8( c, spx, spy, s_RF, brev );
    s_hfft8( c, &smem[y*40+v*10+u*4], spx, s_RF, brev, x, u );
    s_store5( d_c, &smem[y*41+x], &smem[p*41+q], c, 8*ldc*gdy );
}
__global__ void dk_sfft8x8_r2c_perm3d_ext( float2* d_c, 
    const float* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, int n_cells, int dir )
{
    const int brev[]={0,2,1,3};
    __shared__ float  smem[16*41];
    __shared__ float2 s_RF[4];
    float2 c[5];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int gdy=gridDim.y;
    unsigned int x=tid&7;
    unsigned int y=tid>>3;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    float* spx=&smem[y*40+x];
    float* spy=&smem[y*40+v*10+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldy+icell*ldx+x;
    CLEAR4C(c)
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if((icell<n_cells)&(x<nx)){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((2*i+0)<ny){ c[i].x=d_r[0]; d_r+=nx; }
            if((2*i+1)<ny){ c[i].y=d_r[0]; d_r+=nx; }
        }
    } __syncthreads();
    s_vfft8( c, spx, spy, s_RF, brev );
    s_hfft8( c, &smem[y*40+v*10+u*4], spx, s_RF, brev, x, u );
    s_store5( d_c, &smem[y*41+x], &smem[p*41+q], c, 8*ldc*gdy );
}
__global__ void dk_sfft8x8_r2c_perm3d_pad( float2* d_c, 
    const float* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, int n_cells, int pad_x, int pad_y )
{
    const int brev[]={0,2,1,3};
    __shared__ float  smem[16*41];
    __shared__ float2 s_RF[4];
    float2 c[5];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int gdy=gridDim.y;
    unsigned int x=tid&7;
    unsigned int y=tid>>3;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    float* spx=&smem[y*40+x];
    float* spy=&smem[y*40+v*10+u];
    int ox=(int)x-pad_x;
    int oy=-pad_y;
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldr+icell*ny*nx+ox;
    CLEAR4C(c)
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if((icell<n_cells)&(ox>=0)&(ox<nx)){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((oy>=0)&(oy<ny)){ c[i].x=*d_r; d_r+=nx; } ++oy;
            if((oy>=0)&(oy<ny)){ c[i].y=*d_r; d_r+=nx; } ++oy;
        }
    } __syncthreads();
    s_vfft8( c, spx, spy, s_RF, brev );
    s_hfft8( c, &smem[y*40+v*10+u*4], spx, s_RF, brev, x, u );
    s_store5( d_c, &smem[y*41+x], &smem[p*41+q], c, 8*ldc*gdy );
}
__global__ void dk_sfft8x8_r2c_perm3d_flip( float2* d_c, 
    const float* __restrict__ d_r, const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldc, unsigned int ldr, int n_cells, int dir )
{
    const int brev[]={0,2,1,3};
    __shared__ float  smem[16*41];
    __shared__ float2 s_RF[4];
    float2 c[5];
    unsigned int tid=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int gdy=gridDim.y;
    unsigned int x=tid&7;
    unsigned int y=tid>>3;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bx<<4)+y;
    unsigned int nxy=nx*ny;
    unsigned int ldx=dir==0?nxy:ldr;
    unsigned int ldy=dir==0?ldr:nxy;
    float* spx=&smem[y*40+x];
    float* spy=&smem[y*40+v*10+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=by*ldy+icell*ldx+nx*ny-x-1;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    CLEAR4C(c)
    if((icell<n_cells)&(x<nx)){
        c[0].x=*d_r;
        c[0].y=*(d_r-=nx);
    #pragma unroll
        for( int i=1; i<4; ++i ){
            if((2*i+0)<ny){ c[i].x=*(d_r-=nx); }
            if((2*i+1)<ny){ c[i].y=*(d_r-=nx); }
        }
    } __syncthreads();
    s_vfft8( c, spx, spy, s_RF, brev );
    s_hfft8( c, &smem[y*40+v*10+u*4], spx, s_RF, brev, x, u );
    s_store5( d_c, &smem[y*41+x], &smem[p*41+q], c, 8*ldc*gdy );
}