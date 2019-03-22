#define xfft16x16_c2r_splice(dir,suffix)        \
__global__ void dk_xfft16x16_c2r_splice##suffix(\
          __half *              d_r,    \
    const __half2* __restrict__ d_c,    \
    const __half2* __restrict__ d_RF,   \
    const __half * __restrict__ d_x,    \
    float                       alpha,  \
    unsigned int                nx,     \
    unsigned int                ny,     \
    unsigned int                ldr,    \
    unsigned int                n,      \
    unsigned int                n_cells,\
    unsigned int                grid_x, \
    unsigned int                grid_y, \
    unsigned int                sx,     \
    unsigned int                sy )    \
{                                       \
    __shared__ float smem[8*144];       \
    __shared__ float2 s_RF[8];          \
    float2 c[8], d[9];                  \
    const int brev[]={0,4,2,6,1,5,3,7}; \
    unsigned int tid=threadIdx.x;                  \
    unsigned int bid=blockIdx.x;                   \
    unsigned int x=tid&15;                         \
    unsigned int y=tid>>4;                         \
    unsigned int icell=(bid<<3)+y;                 \
    unsigned int n_cells_per_map=grid_x*grid_y;    \
    unsigned int map_id=(icell%n)/n_cells_per_map; \
    unsigned int channel=icell/n;                  \
    unsigned int map_cell_id=icell%n_cells_per_map;\
    unsigned int cell_x=map_cell_id%grid_x;        \
    unsigned int cell_y=map_cell_id/grid_x;        \
    unsigned int ox=cell_x*sx;                     \
    unsigned int oy=cell_y*sy;                     \
    unsigned int flip_x=(16-x)&15;                 \
    unsigned int vax=(cell_x<grid_x-1)?sx:(nx-ox); \
    unsigned int vay=(cell_y<grid_y-1)?sy:(ny-oy); \
    if(tid<8){ ((float2*)s_RF)[tid]=__half22float2(d_RF[tid]); }\
    __syncthreads(); \
    d_c+=icell*144+x;\
    d_r+=channel*ldr+(map_id*ny+oy)*nx+ox+flip_x;\
    if(dir==0){ d_x+=channel; } else             \
    if(dir==1){ d_x+=channel*ldr+(map_id*ny+oy)*nx+ox+flip_x; }\
    d[0]=__half22float2(d_c[0*16]);\
    d[1]=__half22float2(d_c[1*16]);\
    d[2]=__half22float2(d_c[2*16]);\
    d[3]=__half22float2(d_c[3*16]);\
    d[4]=__half22float2(d_c[4*16]);\
    d[5]=__half22float2(d_c[5*16]);\
    d[6]=__half22float2(d_c[6*16]);\
    d[7]=__half22float2(d_c[7*16]);\
    d[8]=__half22float2(d_c[8*16]);\
    s_hifft16( c, d, &smem[y*144], s_RF, brev, x );\
    s_vifft16( d, c, s_RF, brev, x );\
    bool bc=(icell<n_cells)&&(flip_x<vax);\
    xfft16x16_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, vay );\
}

xfft16x16_c2r_splice(-1,)
xfft16x16_c2r_splice(-1,_relu)
xfft16x16_c2r_splice( 0,_bias)
xfft16x16_c2r_splice( 0,_bias_relu)
xfft16x16_c2r_splice( 1,_drelu)
xfft16x16_c2r_splice( 1,_xdrv)