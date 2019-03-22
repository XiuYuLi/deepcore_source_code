#define sfft16x16_c2r_splice_perm(dir,suffix)        \
__global__ void dk_sfft16x16_c2r_splice_perm##suffix(\
          float *              d_r   ,  \
    const float2* __restrict__ d_c   ,  \
    const float * __restrict__ d_RF  ,  \
    const float * __restrict__ d_x   ,  \
    float                      alpha ,  \
    unsigned int               nx    ,  \
    unsigned int               ny    ,  \
    unsigned int               ldr   ,  \
    unsigned int               ldc   ,  \
    unsigned int               n     ,  \
    unsigned int               grid_x,  \
    unsigned int               grid_y,  \
    unsigned int               sx    ,  \
    unsigned int               sy )     \
{                                       \
    __shared__ float smem[16*145];      \
    __shared__ float2 s_RF[8];          \
    float2 c[8], d[9];                  \
    const int brev[]={0,4,2,6,1,5,3,7}; \
    unsigned int tid=threadIdx.x;\
    unsigned int bx=blockIdx.x;  \
    unsigned int by=blockIdx.y;  \
    unsigned int onc=gridDim.y;  \
    unsigned int x=tid&15;       \
    unsigned int y=tid>>4;       \
    unsigned int icell=(bx<<4)+y;\
    unsigned int n_cells_per_map=grid_x*grid_y;    \
    unsigned int map_id=icell/n_cells_per_map;     \
    unsigned int map_cell_id=icell%n_cells_per_map;\
    unsigned int cell_x=map_cell_id%grid_x;        \
    unsigned int cell_y=map_cell_id/grid_x;        \
    unsigned int ox=cell_x*sx;                     \
    unsigned int oy=cell_y*sy;                     \
    unsigned int flip_x=(16-x)&15;                 \
    unsigned int vax=(cell_x<grid_x-1)?sx:(nx-ox); \
    unsigned int vay=(cell_y<grid_y-1)?sy:(ny-oy); \
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }         \
    d_c+=(y*onc+by)*ldc+(bx<<4)+x;                 \
    d_r+=by*ldr+(map_id*ny+oy)*nx+ox+flip_x;       \
    if(dir==0){ d_x+=by; } else                    \
    if(dir==1){ d_x+=by*ldr+(map_id*ny+oy)*nx+ox+flip_x; }\
    s_load9( d, &smem[x*145+y], &smem[y*145+x], d_c, onc*ldc*16 );\
    s_hifft16( c, d, &smem[y*144], s_RF, brev, x );\
    s_vifft16( d, c, s_RF, brev, x );\
    bool bc=(icell<n)&&(flip_x<vax); \
    sfft16x16_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, vay );\
}

sfft16x16_c2r_splice_perm(-1,)
sfft16x16_c2r_splice_perm(-1,_relu)
sfft16x16_c2r_splice_perm( 0,_bias)
sfft16x16_c2r_splice_perm( 0,_bias_relu)
sfft16x16_c2r_splice_perm( 1,_drelu)
sfft16x16_c2r_splice_perm( 1,_xdrv)