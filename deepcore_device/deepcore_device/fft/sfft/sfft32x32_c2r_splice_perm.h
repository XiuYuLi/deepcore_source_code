#define sfft32x32_c2r_splice_perm(dir,suffix)\
__global__ void __launch_bounds__(256,2)\
dk_sfft32x32_c2r_splice_perm##suffix(   \
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
    __shared__ float smem[8*560];       \
    __shared__ float2 s_RF[16];         \
    float2 c[16], d[17];                \
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};\
    unsigned int tid=threadIdx.x;\
    unsigned int bx=blockIdx.x;  \
    unsigned int by=blockIdx.y;  \
    unsigned int onc=gridDim.y;  \
    unsigned int x=tid&31;       \
    unsigned int y=tid>>5;       \
    unsigned int p=tid&7;        \
    unsigned int q=tid>>3;       \
    unsigned int icell=(bx<<3)+y;\
    unsigned int n_cells_per_map=grid_x*grid_y;    \
    unsigned int map_id=icell/n_cells_per_map;     \
    unsigned int map_cell_id=icell%n_cells_per_map;\
    unsigned int cell_x=map_cell_id%grid_x;        \
    unsigned int cell_y=map_cell_id/grid_x;        \
    unsigned int ox=cell_x*sx;                     \
    unsigned int oy=cell_y*sy;                     \
    unsigned int flip_x=(32-x)&31;                 \
    unsigned int vax=(cell_x<grid_x-1)?sx:(nx-ox); \
    unsigned int vay=(cell_y<grid_y-1)?sy:(ny-oy); \
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }         \
    d_c+=(q*onc+by)*ldc+(bx<<3)+p;                 \
    d_r+=by*ldr+(map_id*ny+oy)*nx+ox+flip_x;       \
    if(dir==0){ d_x+=by; } else                    \
    if(dir==1){ d_x+=by*ldr+(map_id*ny+oy)*nx+ox+flip_x; }       \
    s_load( d, &smem[p*547+q], &smem[y*547+x], d_c, onc*ldc*32 );\
    s_hifft( c, d, &smem[y*560], s_RF, brev, x );\
    s_vifft( d, c, s_RF, brev, x );              \
    bool bc=(icell<n)&&(flip_x<vax);             \
    sfft32x32_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, vay );\
}

sfft32x32_c2r_splice_perm(-1,)
sfft32x32_c2r_splice_perm(-1,_relu)
sfft32x32_c2r_splice_perm( 0,_bias)
sfft32x32_c2r_splice_perm( 0,_bias_relu)
sfft32x32_c2r_splice_perm( 1,_drelu)
sfft32x32_c2r_splice_perm( 1,_xdrv)