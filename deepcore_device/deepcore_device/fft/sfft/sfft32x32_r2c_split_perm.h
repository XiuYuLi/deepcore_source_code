/*data:n=bat*grid*grid£¬ grad:n=pnc */
__global__ void __launch_bounds__(512,1) 
dk_sfft32x32_r2c_split_perm( 
      float2*              d_c, 
const float * __restrict__ d_r, 
const float * __restrict__ d_RF, 
      int                  nx, 
      int                  ny, 
      int                  ldc, 
      int                  ldr, 
      int                  n, 
      int                  grid_x, 
      int                  grid_y, 
      int                  sx, 
      int                  sy, 
      int                  is_grad )
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
    int n_cells_per_map=grid_x*grid_y;
    int idx=is_grad?by:icell;
    int idy=is_grad?icell:by;
    int map_id=idx/n_cells_per_map;
    int map_cell_id=idx%n_cells_per_map;
    int ox=(map_cell_id%grid_x)*sx;
    int oy=(map_cell_id/grid_x)*sy;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=idy*ldr+(map_id*ny+oy)*nx+(ox+=x);
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    CLEAR16C(c)
    int vay=ny-oy;
    if((icell<n)&(ox<nx)){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((2*i+0)<vay){ c[i].x=d_r[0]; } d_r+=nx;
            if((2*i+1)<vay){ c[i].y=d_r[0]; } d_r+=nx;
        }
    } __syncthreads();
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*ldc*gdy );
}
__global__ void __launch_bounds__(512,1) dk_sfft32x32_r2c_split_perm_pad( 
      float2*              d_c, 
const float * __restrict__ d_r, 
const float * __restrict__ d_RF, 
      int                  nx, 
      int                  ny, 
      int                  ldc, 
      int                  ldr, 
      int                  n, 
      int                  grid_x, 
      int                  grid_y, 
      int                  sx, 
      int                  sy, 
      int                  is_grad,
      int                  pad_x, 
      int                  pad_y )
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
    int idx=is_grad?by:icell;
    int idy=is_grad?icell:by;
    int n_cells_per_map=grid_x*grid_y;
    int map_id=idx/n_cells_per_map;
    int map_cell_id=idx%n_cells_per_map;
    int cell_x=map_cell_id%grid_x;
    int cell_y=map_cell_id/grid_x;
    int ox=cell_x*sx-pad_x+x;
    int oy=cell_y*sy-pad_y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=(q*gdy+by)*ldc+(bx<<4)+p;
    d_r+=idy*ldr+(map_id*ny+(oy+((cell_y==0)?pad_y:0)))*nx+ox;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    CLEAR16C(c)
    if((icell<n)&(ox>=0)&(ox<nx)){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((oy>=0)&(oy<ny)){ c[i].x=d_r[0]; d_r+=nx; } ++oy;
            if((oy>=0)&(oy<ny)){ c[i].y=d_r[0]; d_r+=nx; } ++oy;
        }
    } __syncthreads();
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
    s_store( d_c, &smem[y*545+x], &smem[p*545+q], c, 32*ldc*gdy );
}