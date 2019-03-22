#define sfft8x8_c2r_splice(dir,suffix)        \
__global__ void dk_sfft8x8_c2r_splice##suffix(\
          float *              d_r,    \
    const float2* __restrict__ d_c,    \
    const float * __restrict__ d_RF,   \
    const float * __restrict__ d_x,    \
    float                      alpha,  \
    unsigned int               nx,     \
    unsigned int               ny,     \
    unsigned int               ldr,    \
    unsigned int               n,      \
    unsigned int               n_cells,\
    unsigned int               grid_x, \
    unsigned int               grid_y, \
    unsigned int               sx,     \
    unsigned int               sy )    \
{                                      \
    __shared__ float smem[16*44]; \
    __shared__ float2 s_RF[4];    \
    float2 c[4], d[5];            \
    const int brev[]={0,2,1,3};   \
    unsigned int tid=threadIdx.x; \
    unsigned int bid=blockIdx.x;  \
    unsigned int x=tid&7;         \
    unsigned int y=tid>>3;        \
    unsigned int icell=(bid<<4)+y;\
    unsigned int n_cells_per_map=grid_x*grid_y;    \
    unsigned int map_id=(icell%n)/n_cells_per_map; \
    unsigned int channel=icell/n;                  \
    unsigned int map_cell_id=icell%n_cells_per_map;\
    unsigned int cell_x=map_cell_id%grid_x;        \
    unsigned int cell_y=map_cell_id/grid_x;        \
    unsigned int ox=cell_x*sx;                     \
    unsigned int oy=cell_y*sy;                     \
    unsigned int flip_x=(8-x)&7;                   \
    unsigned int vax=(cell_x<grid_x-1)?sx:(nx-ox); \
    unsigned int vay=(cell_y<grid_y-1)?sy:(ny-oy); \
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }         \
    __syncthreads();                               \
    d_c+=icell*40+x;                               \
    d_r+=channel*ldr+(map_id*ny+oy)*nx+ox+flip_x;  \
    if(dir==0){ d_x+=channel; } else               \
    if(dir==1){ d_x+=channel*ldr+(map_id*ny+oy)*nx+ox+flip_x; }\
    d[0]=d_c[0*8];\
    d[1]=d_c[1*8];\
    d[2]=d_c[2*8];\
    d[3]=d_c[3*8];\
    d[4]=d_c[4*8];\
    s_hifft8( c, d, &smem[y*44], s_RF, brev, x );\
    s_vifft8( d, c, s_RF, brev, x );\
    bool bc=(icell<n)&&(flip_x<vax);\
    sfft8x8_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, vay );\
}

sfft8x8_c2r_splice(-1,)
sfft8x8_c2r_splice(-1,_relu)
sfft8x8_c2r_splice( 0,_bias)
sfft8x8_c2r_splice( 0,_bias_relu)
sfft8x8_c2r_splice( 1,_drelu)
sfft8x8_c2r_splice( 1,_xdrv)