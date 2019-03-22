#define sfft32x32_c2r_perm3d(dir,suffix)\
__global__ void __launch_bounds__(256,2)\
dk_sfft32x32_c2r_perm3d##suffix(        \
          float *              d_r  ,   \
    const float2* __restrict__ d_c  ,   \
    const float * __restrict__ d_RF ,   \
    const float * __restrict__ d_x  ,   \
    float                      alpha,   \
    unsigned int               nx   ,   \
    unsigned int               ny   ,   \
    unsigned int               ldr  ,   \
    unsigned int               ldc  ,   \
    unsigned int               n_cells )\
{                                       \
    __shared__ float smem[8*560];       \
    __shared__ float2 s_RF[16];         \
    float2 c[16], d[17];                \
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};\
    unsigned int tid=threadIdx.x; \
    unsigned int bx=blockIdx.x;   \
    unsigned int by=blockIdx.y;   \
    unsigned int onc=gridDim.y;   \
    unsigned int x=tid&31;        \
    unsigned int y=tid>>5;        \
    unsigned int p=tid&7;         \
    unsigned int q=tid>>3;        \
    unsigned int icell=(bx<<3)+y; \
    unsigned int flip_x=(32-x)&31;\
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }\
    d_c+=(q*onc+by)*ldc+(bx<<3)+p; \
    d_r+=by*ldr+icell*ny*nx+flip_x;\
    if(dir==0){ d_x+=by; } else    \
    if(dir==1){ d_x+=by*ldr+icell*ny*nx+flip_x; }\
    s_load( d, &smem[p*547+q], &smem[y*547+x], d_c, onc*ldc<<5 );\
    s_hifft( c, d, &smem[y*560], s_RF, brev, x );\
    s_vifft( d, c, s_RF, brev, x );\
    bool bc=(icell<n_cells)&&(flip_x<nx);\
    sfft32x32_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, ny );\
}

sfft32x32_c2r_perm3d(-1,)
sfft32x32_c2r_perm3d(-1,_relu)
sfft32x32_c2r_perm3d( 0,_bias)
sfft32x32_c2r_perm3d( 0,_bias_relu)
sfft32x32_c2r_perm3d( 1,_drelu)
sfft32x32_c2r_perm3d( 1,_xdrv)