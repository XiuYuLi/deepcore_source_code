#define xfft16x16_c2r(dir,suffix)        \
__global__ void dk_xfft16x16_c2r##suffix(\
          __half*              d_r  ,    \
    const float2* __restrict__ d_c  ,    \
    const float * __restrict__ d_RF ,    \
    const __half* __restrict__ d_x  ,    \
    float                      alpha,    \
    unsigned int               nx   ,    \
    unsigned int               ny   ,    \
    unsigned int               ldr  ,    \
    unsigned int               ldc  ,    \
    unsigned int               n    ,    \
    unsigned int               n_cells ) \
{                                        \
    __shared__ float smem[8*144];        \
    __shared__ float2 s_RF[8];           \
    float2 c[8], d[9];                   \
    const int brev[]={0,4,2,6,1,5,3,7};  \
    unsigned int tid=threadIdx.x;        \
    unsigned int x=tid&15;               \
    unsigned int y=tid>>4;               \
    unsigned int icell=(blockIdx.x<<3)+y;\
    unsigned int channel=icell/n;        \
    unsigned int flip_x=(16-x)&15;       \
    d_r+=channel*ldr+(icell%n)*ny*nx+flip_x;\
    if(dir==0){ d_x+=channel; } else        \
    if(dir==1){ d_x+=channel*ldr+(icell%n)*ny*nx+flip_x; }\
    d_c+=icell*144+x;\
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }\
    d[0]=d_c[0*16]; \
    d[1]=d_c[1*16]; \
    d[2]=d_c[2*16]; \
    d[3]=d_c[3*16]; \
    d[4]=d_c[4*16]; \
    d[5]=d_c[5*16]; \
    d[6]=d_c[6*16]; \
    d[7]=d_c[7*16]; \
    d[8]=d_c[8*16]; \
    __syncthreads();\
    s_hifft16( c, d, &smem[y*144], s_RF, brev, x );\
    s_vifft16( d, c, s_RF, brev, x );    \
    bool bc=(icell<n_cells)&&(flip_x<nx);\
    xfft16x16_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, ny );\
}

xfft16x16_c2r(-1,)
xfft16x16_c2r(-1,_relu)
xfft16x16_c2r( 0,_bias)
xfft16x16_c2r( 0,_bias_relu)
xfft16x16_c2r( 1,_drelu)
xfft16x16_c2r( 1,_xdrv)