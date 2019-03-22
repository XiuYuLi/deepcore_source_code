__global__ void dk_sfft32x32_r2c_split( 
    float2* d_c, 
    const float* __restrict__ d_r, 
    const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldr, 
    unsigned int n_cells_per_chl, unsigned int n_cells,    
    unsigned int grid_x, unsigned int grid_y, 
    int sx, int sy )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bid<<3)+y;
    unsigned int n_cells_per_map=grid_x*grid_y;
    unsigned int channel=icell/n_cells_per_chl;
    unsigned int map_id=(icell%n_cells_per_chl)/n_cells_per_map;
    unsigned int map_cell_id=icell%n_cells_per_map;
    unsigned int ox=(map_cell_id%grid_x)*sx;
    unsigned int oy=(map_cell_id/grid_x)*sy;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=channel*ldr+(map_id*ny+oy)*nx+(ox+=x);
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; } 
    __syncthreads();
    if(icell>=n_cells) return;
    CLEAR16C(c)
    int valid_y=ny-oy;
    if(ox<nx){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((2*i+0)<valid_y){ c[i].x=d_r[0]; d_r+=nx; }
            if((2*i+1)<valid_y){ c[i].y=d_r[0]; d_r+=nx; }
        }
    }
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=c[i]; }
}
__global__ void dk_sfft32x32_r2c_split_pad( 
    float2* d_c, 
    const float* __restrict__ d_r, 
    const float* __restrict__ d_RF, 
    unsigned int nx, unsigned int ny, unsigned int ldr, 
    unsigned int n_cells_per_chl, unsigned int n_cells, 
    unsigned int grid_x, unsigned int grid_y, 
    int sx, int sy, int pad_x, int pad_y )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bid<<3)+y;
    unsigned int n_cells_per_map=grid_x*grid_y;
    unsigned int channel=icell/n_cells_per_chl;
    unsigned int map_id=(icell%n_cells_per_chl)/n_cells_per_map;
    unsigned int map_cell_id=icell%n_cells_per_map;
    int cell_x=(int)(map_cell_id%grid_x);
    int cell_y=(int)(map_cell_id/grid_x);
    int ox=cell_x*sx-pad_x+(int)x;
    int oy=cell_y*sy-pad_y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=channel*ldr+(map_id*ny+(oy+((cell_y==0)?pad_y:0)))*nx+ox;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; } 
    __syncthreads();
    if(icell>=n_cells) return;
    CLEAR16C(c)
    if((ox>=0)&(ox<nx)){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((oy>=0)&(oy<ny)){ c[i].x=d_r[0]; d_r+=nx; } ++oy;
            if((oy>=0)&(oy<ny)){ c[i].y=d_r[0]; d_r+=nx; } ++oy;
        }
    }
    s_vfft( c, spx, spy, s_RF, brev );
    s_hfft( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=c[i]; }
}