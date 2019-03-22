#define sfft64x64_c2r(dir,suffix)        \
__global__ void dk_sfft64x64_c2r##suffix(\
          float *              d_r  ,    \
    const float2* __restrict__ d_c  ,    \
    const float * __restrict__ d_RF ,    \
    const float * __restrict__ d_x  ,    \
    float                      alpha,    \
    unsigned int               nx   ,    \
    unsigned int               ny   ,    \
    unsigned int               ldr )     \
{                                        \
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30, \
                      1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};\
    __shared__ float smem[66*32];             \
    __shared__ float2 s_RF[32];               \
    float2 c[32],d[33];                       \
    unsigned int bx=blockIdx.x;               \
    unsigned int by=blockIdx.y;               \
    unsigned int tid=threadIdx.x;             \
    unsigned int flip_x=tid?(64-tid):tid;     \
    d_c+=(by*gridDim.x+bx)*33*64+tid;         \
    d_r+=by*ldr+bx*ny*nx+flip_x;              \
    if(dir==0){ d_x+=by; } else               \
    if(dir==1){ d_x+=by*ldr+bx*ny*nx+flip_x; }\
    ((float*)s_RF)[tid]=d_RF[tid];            \
    d[ 0]=d_c[ 0*64];\
    d[ 1]=d_c[ 1*64];\
    d[ 2]=d_c[ 2*64];\
    d[ 3]=d_c[ 3*64];\
    d[ 4]=d_c[ 4*64];\
    d[ 5]=d_c[ 5*64];\
    d[ 6]=d_c[ 6*64];\
    d[ 7]=d_c[ 7*64];\
    d[ 8]=d_c[ 8*64];\
    d[ 9]=d_c[ 9*64];\
    d[10]=d_c[10*64];\
    d[11]=d_c[11*64];\
    d[12]=d_c[12*64];\
    d[13]=d_c[13*64];\
    d[14]=d_c[14*64];\
    d[15]=d_c[15*64];\
    d[16]=d_c[16*64];\
    d[17]=d_c[17*64];\
    d[18]=d_c[18*64];\
    d[19]=d_c[19*64];\
    d[20]=d_c[20*64];\
    d[21]=d_c[21*64];\
    d[22]=d_c[22*64];\
    d[23]=d_c[23*64];\
    d[24]=d_c[24*64];\
    d[25]=d_c[25*64];\
    d[26]=d_c[26*64];\
    d[27]=d_c[27*64];\
    d[28]=d_c[28*64];\
    d[29]=d_c[29*64];\
    d[30]=d_c[30*64];\
    d[31]=d_c[31*64];\
    d[32]=d_c[32*64];\
    s_hifft64(c,d,smem,s_RF,brev,tid);\
    s_vifft64(d,c,s_RF,brev,tid);     \
    sfft64x64_c2r_store##suffix( d_r, d, d_x, alpha, flip_x, nx, ny );\
}

sfft64x64_c2r(-1,)
sfft64x64_c2r(-1,_relu)
sfft64x64_c2r(0,_bias)
sfft64x64_c2r(0,_bias_relu)
sfft64x64_c2r(1,_drelu)
sfft64x64_c2r(1,_xdrv)