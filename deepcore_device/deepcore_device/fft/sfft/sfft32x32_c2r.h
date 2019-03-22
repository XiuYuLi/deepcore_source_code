#define sfft32x32_c2r(dir,suffix)        \
__global__ void dk_sfft32x32_c2r##suffix(\
          float *              d_r  ,    \
    const float2* __restrict__ d_c  ,    \
    const float * __restrict__ d_RF ,    \
    const float * __restrict__ d_x  ,    \
    float                      alpha,    \
    unsigned int               nx   ,    \
    unsigned int               ny   ,    \
    unsigned int               ldr  ,    \
    unsigned int               n    ,    \
    unsigned int               n_cells ) \
{                                        \
    __shared__ float smem[8*560];        \
    __shared__ float2 s_RF[16];          \
    float2 c[16], d[17];                 \
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};\
    unsigned int tid=threadIdx.x;           \
    unsigned int x=tid&31;                  \
    unsigned int y=tid>>5;                  \
    unsigned int icell=(blockIdx.x<<3)+y;   \
    unsigned int flip_x=(32-x)&31;          \
    unsigned int channel=icell/n;           \
    d_c+=icell*544+x;                       \
    d_r+=channel*ldr+(icell%n)*ny*nx+flip_x;\
    if(dir==0){ d_x+=channel; } else        \
    if(dir==1){ d_x+=channel*ldr+(icell%n)*ny*nx+flip_x; }\
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }\
    d[ 0]=d_c[ 0*32];\
    d[ 1]=d_c[ 1*32];\
    d[ 2]=d_c[ 2*32];\
    d[ 3]=d_c[ 3*32];\
    d[ 4]=d_c[ 4*32];\
    d[ 5]=d_c[ 5*32];\
    d[ 6]=d_c[ 6*32];\
    d[ 7]=d_c[ 7*32];\
    d[ 8]=d_c[ 8*32];\
    d[ 9]=d_c[ 9*32];\
    d[10]=d_c[10*32];\
    d[11]=d_c[11*32];\
    d[12]=d_c[12*32];\
    d[13]=d_c[13*32];\
    d[14]=d_c[14*32];\
    d[15]=d_c[15*32];\
    d[16]=d_c[16*32];\
    __syncthreads(); \
    s_hifft( c, d, &smem[y*560], s_RF, brev, x );\
    s_vifft( d, c, s_RF, brev, x );              \
    bool bc=(icell<n_cells)&&(flip_x<nx);        \
    sfft32x32_c2r_store##suffix( d_r, d, d_x, alpha, bc, nx, ny );\
}

sfft32x32_c2r(-1,)
sfft32x32_c2r(-1,_relu)
sfft32x32_c2r(0,_bias)
sfft32x32_c2r(0,_bias_relu)
sfft32x32_c2r(1,_drelu)
sfft32x32_c2r(1,_xdrv)