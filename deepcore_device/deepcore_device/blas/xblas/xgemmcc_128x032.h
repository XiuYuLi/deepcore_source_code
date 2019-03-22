#define xgemmcc_128x32(add_bias,suffix)   \
__global__ void dk_xgemmcc_128x32##suffix(\
          char *              d_c   ,\
    const char * __restrict__ d_a   ,\
    const char * __restrict__ d_b   ,\
    const float* __restrict__ d_bias,\
    float                     alpha ,\
    int                       anr   ,\
    int                       pnc   ,\
    int                       qnc   ,\
    int                       lda   ,\
    int                       ldb   ,\
    int                       ldc ){ \
    __shared__ char smem[2*10240];\
    __shared__ float s_bias[32];  \
    float c[32];                  \
    float4 a[2], b[4];            \
    unsigned int bx=blockIdx.x;   \
    unsigned int by=blockIdx.y;   \
    unsigned int tid=threadIdx.x; \
    unsigned int lane=tid&31;     \
    unsigned int slot=tid>>5;     \
    unsigned int u=tid&15;        \
    unsigned int v=tid>>4;        \
    unsigned int ai=bx*16+u;      \
    unsigned int enr=(anr+7)>>3;  \
    unsigned int sai=ai<enr?ai:(enr-1);\
    unsigned int x=(bx<<6)|(slot<<4)|u;\
    unsigned int y=(lane>>4)<<3;       \
    unsigned int elda=lda<<3;          \
    d_a+=(by*pnc+v   )*lda+(sai<<4);   \
    d_b+=(by*qnc+lane)*ldb+(slot<<3);  \
    d_c+=(by*qnc+y   )*ldc+(x<<2);     \
    __half8 p0=*((const __half8*)d_a); d_a+=elda;\
    __half8 p1=*((const __half8*)d_a);\
    __half4 p2=*((const __half4*)d_b);\
    char* __restrict__ asst_base=&smem[tid<<5];\
    char* __restrict__ bsst_base=&smem[(slot<<9)|(lane<<2)];\
    char* __restrict__ asld_base=&smem[(slot<<7)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x2000|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)&asst_base[0x0000])=__half42float4(p0.lo);\
    *((float4*)&asst_base[0x0010])=__half42float4(p0.hi);\
    *((float4*)&asst_base[0x1000])=__half42float4(p1.lo);\
    *((float4*)&asst_base[0x1010])=__half42float4(p1.hi);\
    *((float *)&bsst_base[0x2000])=__half2float(HLO(p2.x));\
    *((float *)&bsst_base[0x2080])=__half2float(HHI(p2.x));\
    *((float *)&bsst_base[0x2100])=__half2float(HLO(p2.y));\
    *((float *)&bsst_base[0x2180])=__half2float(HHI(p2.y));\
    __syncthreads();\
    if(add_bias){ if(tid<qnc){ s_bias[tid]=d_bias[by*qnc+tid]; } }\
    SZERO32(c)\
    b[0]=*((float4*)&bsld[0x00]);   \
    a[0]=*((float4*)&asld[0x00]);   \
    b[1]=*((float4*)&bsld[0x40]);   \
    unsigned int ofs=0x2800;        \
    for( int k=pnc-16; k>0; k-=16 ){\
        p0=*((const __half8*)(d_a+=elda));  \
        p1=*((const __half8*)(d_a+=elda));  \
        p2=*((const __half4*)(d_b+=32  ));  \
        b[2]=*((float4*)&bsld[ 1*128+0x00]);\
        a[1]=*((float4*)&asld[ 1*512+0x00]);\
        b[3]=*((float4*)&bsld[ 1*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[ 2*128+0x00]);\
        a[0]=*((float4*)&asld[ 2*512+0x00]);\
        b[1]=*((float4*)&bsld[ 2*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[ 3*128+0x00]);\
        a[1]=*((float4*)&asld[ 3*512+0x00]);\
        b[3]=*((float4*)&bsld[ 3*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[ 4*128+0x00]);\
        a[0]=*((float4*)&asld[ 4*512+0x00]);\
        b[1]=*((float4*)&bsld[ 4*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[ 5*128+0x00]);\
        a[1]=*((float4*)&asld[ 5*512+0x00]);\
        b[3]=*((float4*)&bsld[ 5*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[ 6*128+0x00]);\
        a[0]=*((float4*)&asld[ 6*512+0x00]);\
        b[1]=*((float4*)&bsld[ 6*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[ 7*128+0x00]);\
        a[1]=*((float4*)&asld[ 7*512+0x00]);\
        b[3]=*((float4*)&bsld[ 7*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[ 8*128+0x00]);\
        a[0]=*((float4*)&asld[ 8*512+0x00]);\
        b[1]=*((float4*)&bsld[ 8*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[ 9*128+0x00]);\
        a[1]=*((float4*)&asld[ 9*512+0x00]);\
        b[3]=*((float4*)&bsld[ 9*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[10*128+0x00]);\
        a[0]=*((float4*)&asld[10*512+0x00]);\
        b[1]=*((float4*)&bsld[10*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[11*128+0x00]);\
        a[1]=*((float4*)&asld[11*512+0x00]);\
        b[3]=*((float4*)&bsld[11*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[12*128+0x00]);\
        a[0]=*((float4*)&asld[12*512+0x00]);\
        b[1]=*((float4*)&bsld[12*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[13*128+0x00]);\
        a[1]=*((float4*)&asld[13*512+0x00]);\
        b[3]=*((float4*)&bsld[13*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[14*128+0x00]);\
        a[0]=*((float4*)&asld[14*512+0x00]);\
        b[1]=*((float4*)&bsld[14*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[15*128+0x00]);\
        a[1]=*((float4*)&asld[15*512+0x00]);\
        b[3]=*((float4*)&bsld[15*128+0x40]);\
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP4x8(c,&a[0],&b[0])\
        *((float4*)&asst[0x0000])=__half42float4(p0.lo);  \
        *((float4*)&asst[0x0010])=__half42float4(p0.hi);  \
        *((float4*)&asst[0x1000])=__half42float4(p1.lo);  \
        *((float4*)&asst[0x1010])=__half42float4(p1.hi);  \
        *((float *)&bsst[0x2000])=__half2float(HLO(p2.x));\
        *((float *)&bsst[0x2080])=__half2float(HHI(p2.x));\
        *((float *)&bsst[0x2100])=__half2float(HLO(p2.y));\
        *((float *)&bsst[0x2180])=__half2float(HHI(p2.y));\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        BOP4x8(c,&a[1],&b[2])\
        ofs^=0x2800;\
    }\
    b[2]=*((float4*)&bsld[ 1*128+0x00]);\
    a[1]=*((float4*)&asld[ 1*512+0x00]);\
    b[3]=*((float4*)&bsld[ 1*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[ 2*128+0x00]);\
    a[0]=*((float4*)&asld[ 2*512+0x00]);\
    b[1]=*((float4*)&bsld[ 2*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[ 3*128+0x00]);\
    a[1]=*((float4*)&asld[ 3*512+0x00]);\
    b[3]=*((float4*)&bsld[ 3*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[ 4*128+0x00]);\
    a[0]=*((float4*)&asld[ 4*512+0x00]);\
    b[1]=*((float4*)&bsld[ 4*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[ 5*128+0x00]);\
    a[1]=*((float4*)&asld[ 5*512+0x00]);\
    b[3]=*((float4*)&bsld[ 5*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[ 6*128+0x00]);\
    a[0]=*((float4*)&asld[ 6*512+0x00]);\
    b[1]=*((float4*)&bsld[ 6*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[ 7*128+0x00]);\
    a[1]=*((float4*)&asld[ 7*512+0x00]);\
    b[3]=*((float4*)&bsld[ 7*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[ 8*128+0x00]);\
    a[0]=*((float4*)&asld[ 8*512+0x00]);\
    b[1]=*((float4*)&bsld[ 8*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[ 9*128+0x00]);\
    a[1]=*((float4*)&asld[ 9*512+0x00]);\
    b[3]=*((float4*)&bsld[ 9*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[10*128+0x00]);\
    a[0]=*((float4*)&asld[10*512+0x00]);\
    b[1]=*((float4*)&bsld[10*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[11*128+0x00]);\
    a[1]=*((float4*)&asld[11*512+0x00]);\
    b[3]=*((float4*)&bsld[11*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[12*128+0x00]);\
    a[0]=*((float4*)&asld[12*512+0x00]);\
    b[1]=*((float4*)&bsld[12*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[13*128+0x00]);\
    a[1]=*((float4*)&asld[13*512+0x00]);\
    b[3]=*((float4*)&bsld[13*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[14*128+0x00]);\
    a[0]=*((float4*)&asld[14*512+0x00]);\
    b[1]=*((float4*)&bsld[14*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[15*128+0x00]);\
    a[1]=*((float4*)&asld[15*512+0x00]);\
    b[3]=*((float4*)&bsld[15*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])\
    BOP4x8(c,&a[1],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[((lane&0x10)>>1)|((lane&1)<<2)]; }\
    xgemm_epilog32x32##suffix( d_c, (const char*)bias, &smem[slot<<8], c, lane, ldc, x, (anr+1)>>1, qnc-y, alpha );\
}

xgemmcc_128x32(0,)
xgemmcc_128x32(0,_relu)
xgemmcc_128x32(1,_bias)
xgemmcc_128x32(1,_bias_relu)