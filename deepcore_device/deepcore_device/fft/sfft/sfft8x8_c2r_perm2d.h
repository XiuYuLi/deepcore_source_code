#define sfft8x8_c2r_perm2d(dir,suffix)        \
__global__ void dk_sfft8x8_c2r_perm2d##suffix(\
          float *              d_r  ,     \
    const float2* __restrict__ d_c  ,     \
    const float * __restrict__ d_RF ,     \
    const float * __restrict__ d_x  ,     \
    float                      alpha,     \
    unsigned int               nx   ,     \
    unsigned int               ny   ,     \
    unsigned int               ldr  ,     \
    unsigned int               ldc  ,     \
    unsigned int               n_cells )  \
{                                         \
    __shared__ float smem[16*44];         \
    __shared__ float2 s_RF[4];            \
    float2 c[4], d[5];                    \
    const int brev[]={0,2,1,3};           \
    unsigned int bid=blockIdx.x;          \
    unsigned int tid=threadIdx.x;         \
    unsigned int x=tid&7;                 \
    unsigned int y=tid>>3;                \
    unsigned int p=tid&15;                \
    unsigned int q=tid>>4;                \
    unsigned int icell=(bid<<4)+y;        \
    unsigned int flip_x=(8-x)&7;          \
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }\
    d_c+=q*ldc+(bid<<4)+p;                \
    d_r+=icell*ldr+flip_x;                \
    if(dir==0){ d_x+=icell; } else        \
    if(dir==1){ d_x+=icell*ldr+flip_x; }  \
    s_load5( d, &smem[p*41+q], &smem[y*41+x], d_c, ldc<<3 );\
    s_hifft8( c, d, &smem[y*44], s_RF, brev, x );\
    s_vifft8( d, c, s_RF, brev, x );\
    bool bc=(icell<n_cells)&&(flip_x<nx);\
    sfft8x8_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, ny ); \
}

sfft8x8_c2r_perm2d(-1,)
sfft8x8_c2r_perm2d(-1,_relu)
sfft8x8_c2r_perm2d( 0,_bias)
sfft8x8_c2r_perm2d( 0,_bias_relu)
sfft8x8_c2r_perm2d( 1,_drelu)
sfft8x8_c2r_perm2d( 1,_xdrv)