#define xfft16x16_c2r_perm3d(dir,suffix)        \
__global__ void dk_xfft16x16_c2r_perm3d##suffix(\
          __half*              d_r  ,   \
    const float2* __restrict__ d_c  ,   \
    const float * __restrict__ d_RF ,   \
    const __half* __restrict__ d_x  ,   \
    float                      alpha,   \
    unsigned int               nx   ,   \
    unsigned int               ny   ,   \
    unsigned int               ldr  ,   \
    unsigned int               ldc  ,   \
    unsigned int               n_cells )\
{                                       \
    __shared__ float smem[16*145];      \
    __shared__ float2 s_RF[8];          \
    float2 c[8], d[9];                  \
    const int brev[]={0,4,2,6,1,5,3,7}; \
    unsigned int bx=blockIdx.x;   \
    unsigned int by=blockIdx.y;   \
    unsigned int onc=gridDim.y;   \
    unsigned int tid=threadIdx.x; \
    unsigned int x=tid&15;        \
    unsigned int y=tid>>4;        \
    unsigned int icell=(bx<<4)+y; \
    unsigned int flip_x=(16-x)&15;\
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }\
    d_c+=(y*onc+by)*ldc+(bx<<4)+x; \
    d_r+=by*ldr+icell*ny*nx+flip_x;\
    if(dir==0){ d_x+=by; } else    \
    if(dir==1){ d_x+=by*ldr+icell*ny*nx+flip_x; }\
    s_load9( d, &smem[x*145+y], &smem[y*145+x], d_c, 16*onc*ldc );\
    s_hifft16( c, d, &smem[y*144], s_RF, brev, x );\
    s_vifft16( d, c, s_RF, brev, x );\
    bool bc=(icell<n_cells)&&(flip_x<nx);\
    xfft16x16_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, ny );\
}

xfft16x16_c2r_perm3d(-1,)
xfft16x16_c2r_perm3d(-1,_relu)
xfft16x16_c2r_perm3d( 0,_bias)
xfft16x16_c2r_perm3d( 0,_bias_relu)
xfft16x16_c2r_perm3d( 1,_drelu)
xfft16x16_c2r_perm3d( 1,_xdrv)
