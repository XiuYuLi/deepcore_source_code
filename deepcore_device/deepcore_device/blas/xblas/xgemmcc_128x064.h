#define xgemmcc_128x64(add_bias,suffix)   \
__global__ void dk_xgemmcc_128x64##suffix(\
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
    __shared__ char smem[12288]; \
    __shared__ float s_bias[64]; \
    float c[64];                 \
    float4 a[4], b[4];           \
    unsigned int bx=blockIdx.x;  \
    unsigned int by=blockIdx.y;  \
    unsigned int tid=threadIdx.x;\
    unsigned int lane=tid&31;    \
    unsigned int slot=tid>>5;    \
    unsigned int slot_x=slot&1;  \
    unsigned int slot_y=slot>>1; \
    unsigned int u=tid&63;       \
    unsigned int v=tid>>6;       \
    unsigned int x=(bx<<6)|(slot_x<<5)|lane;\
    unsigned int y=slot_y<<5;           \
    unsigned int ai=(bx<<4)|(tid&15);   \
    unsigned int enr=(anr+7)>>3;        \
    unsigned int sai=ai<enr?ai:(enr-1); \
    unsigned int sbi=u <qnc?u :(qnc-1); \
    unsigned int elda=lda<<3;           \
    d_a+=(by*pnc+(tid>>4))*lda+(sai<<4);\
    d_b+=(by*qnc+sbi     )*ldb+(v<<3);  \
    d_c+=(by*qnc+y       )*ldc+(x<<2);  \
    __half8 p=*((const __half8*)d_a);\
    __half4 q=*((const __half4*)d_b);\
    char* __restrict__ asst_base=&smem[tid<<5];\
    char* __restrict__ bsst_base=&smem[(v<<10)|(u<<2)];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)&asst_base[0x00])=__half42float4(p.lo);\
    *((float4*)&asst_base[0x10])=__half42float4(p.hi);\
    *((float*)&bsst_base[4096+0*256])=__half2float(HLO(q.x));\
    *((float*)&bsst_base[4096+1*256])=__half2float(HHI(q.x));\
    *((float*)&bsst_base[4096+2*256])=__half2float(HLO(q.y));\
    *((float*)&bsst_base[4096+3*256])=__half2float(HHI(q.y));\
    __syncthreads();\
    if(add_bias){ if(tid<qnc){ s_bias[tid]=d_bias[by*qnc+tid]; } }\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=0x1800;     \
    for( int k=pnc-8; k>0; k-=8 )\
    {\
        q=*((const __half4*)(d_b+=16  ));  \
        p=*((const __half8*)(d_a+=elda));  \
        b[2]=*((float4*)&bsld[1*256+0x00]);\
        a[2]=*((float4*)&asld[1*512+0x00]);\
        b[3]=*((float4*)&bsld[1*256+0x40]);\
        a[3]=*((float4*)&asld[1*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[2*256+0x00]);\
        a[0]=*((float4*)&asld[2*512+0x00]);\
        b[1]=*((float4*)&bsld[2*256+0x40]);\
        a[1]=*((float4*)&asld[2*512+0x80]);\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[3*256+0x00]);\
        a[2]=*((float4*)&asld[3*512+0x00]);\
        b[3]=*((float4*)&bsld[3*256+0x40]);\
        a[3]=*((float4*)&asld[3*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[4*256+0x00]);\
        a[0]=*((float4*)&asld[4*512+0x00]);\
        b[1]=*((float4*)&bsld[4*256+0x40]);\
        a[1]=*((float4*)&asld[4*512+0x80]);\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[5*256+0x00]);\
        a[2]=*((float4*)&asld[5*512+0x00]);\
        b[3]=*((float4*)&bsld[5*256+0x40]);\
        a[3]=*((float4*)&asld[5*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[6*256+0x00]);\
        a[0]=*((float4*)&asld[6*512+0x00]);\
        b[1]=*((float4*)&bsld[6*256+0x40]);\
        a[1]=*((float4*)&asld[6*512+0x80]);\
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[7*256+0x00]);\
        a[2]=*((float4*)&asld[7*512+0x00]);\
        b[3]=*((float4*)&bsld[7*256+0x40]);\
        a[3]=*((float4*)&asld[7*512+0x80]);\
        *((float4*)&asst[0x00])=__half42float4(p.lo);\
        *((float4*)&asst[0x10])=__half42float4(p.hi);\
        BOP8x8(c,&a[0],&b[0])\
        *((float*)&bsst[4096+0*256])=__half2float(HLO(q.x));\
        *((float*)&bsst[4096+1*256])=__half2float(HHI(q.x));\
        *((float*)&bsst[4096+2*256])=__half2float(HLO(q.y));\
        *((float*)&bsst[4096+3*256])=__half2float(HHI(q.y));\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])\
        ofs^=0x1800;\
    }\
    b[2]=*((float4*)&bsld[1*256+0x00]);\
    a[2]=*((float4*)&asld[1*512+0x00]);\
    b[3]=*((float4*)&bsld[1*256+0x40]);\
    a[3]=*((float4*)&asld[1*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[2*256+0x00]);\
    a[0]=*((float4*)&asld[2*512+0x00]);\
    b[1]=*((float4*)&bsld[2*256+0x40]);\
    a[1]=*((float4*)&asld[2*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[3*256+0x00]);\
    a[2]=*((float4*)&asld[3*512+0x00]);\
    b[3]=*((float4*)&bsld[3*256+0x40]);\
    a[3]=*((float4*)&asld[3*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[4*256+0x00]);\
    a[0]=*((float4*)&asld[4*512+0x00]);\
    b[1]=*((float4*)&bsld[4*256+0x40]);\
    a[1]=*((float4*)&asld[4*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[5*256+0x00]);\
    a[2]=*((float4*)&asld[5*512+0x00]);\
    b[3]=*((float4*)&bsld[5*256+0x40]);\
    a[3]=*((float4*)&asld[5*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[6*256+0x00]);\
    a[0]=*((float4*)&asld[6*512+0x00]);\
    b[1]=*((float4*)&bsld[6*256+0x40]);\
    a[1]=*((float4*)&asld[6*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[7*256+0x00]);\
    a[2]=*((float4*)&asld[7*512+0x00]);\
    b[3]=*((float4*)&bsld[7*256+0x40]);\
    a[3]=*((float4*)&asld[7*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[(slot_y<<5)|((lane&0x10)>>1)|((lane&1)<<2)]; }\
    xgemm_epilog64x32##suffix( d_c, (const char*)bias, &smem[slot<<10], c, lane, ldc, x, (anr+1)>>1, qnc-y, alpha );\
}

xgemmcc_128x64(0,)
xgemmcc_128x64(0,_relu)
xgemmcc_128x64(1,_bias)
xgemmcc_128x64(1,_bias_relu)