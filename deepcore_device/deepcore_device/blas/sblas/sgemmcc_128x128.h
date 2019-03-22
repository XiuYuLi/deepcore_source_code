#define sgemmcc_128x128(add_bias,suffix)\
__global__ void dk_sgemmcc_128x128##suffix(\
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
    __shared__ char smem[16384]; \
    __shared__ float s_bias[128];\
    float c[64];                 \
    float4 a[4], b[4];           \
    unsigned int bx=blockIdx.x;  \
    unsigned int by=blockIdx.y;  \
    unsigned int tid=threadIdx.x;\
    unsigned int lane=tid&31;    \
    unsigned int slot=tid>>5;    \
    unsigned int slot_x=slot&1;  \
    unsigned int slot_y=slot>>1; \
    unsigned int u=tid&127;      \
    unsigned int v=tid>>7;       \
    unsigned int qnr=(anr+3)>>2; \
    unsigned int x=(bx<<6)+((slot_x<<5)+lane);\
    unsigned int y=(by<<7)+(slot_y<<5);\
    unsigned int ai=(bx<<5)+lane;\
    unsigned int bi=(by<<7)+u;   \
    unsigned int sai=ai<qnr?ai:(qnr-1);\
    unsigned int sbi=bi<qnc?bi:(qnc-1);\
    d_a+=slot*lda+(sai<<4);\
    d_b+=sbi *ldb+(v<<4);  \
    d_c+=y   *ldc+(x<<3);  \
    float4 q=*((const float4*)d_b);\
    float4 p=*((const float4*)d_a);\
    char* __restrict__ asst_base=&smem[tid<<4];\
    char* __restrict__ bsst_base=&smem[(v<<11)|(u<<2)];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)asst_base)=p;          \
    *((float*)&bsst_base[0x1000])=q.x;\
    *((float*)&bsst_base[0x1200])=q.y;\
    *((float*)&bsst_base[0x1400])=q.z;\
    *((float*)&bsst_base[0x1600])=q.w;\
    SZERO64(c)\
    __syncthreads();\
    if(add_bias){ if((v==0)&(bi<qnc)){ s_bias[u]=d_bias[bi]; } }\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=0x2000; \
    unsigned int elda=lda<<3;\
    for( int k=pnc-8; k>0; k-=8 )\
    {\
        q=*((const float4*)(d_b+=32  ));\
        p=*((const float4*)(d_a+=elda));\
        b[2]=*((float4*)&bsld[1*512+0x00]);\
        a[2]=*((float4*)&asld[1*512+0x00]);\
        b[3]=*((float4*)&bsld[1*512+0x40]);\
        a[3]=*((float4*)&asld[1*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[2*512+0x00]);\
        a[0]=*((float4*)&asld[2*512+0x00]);\
        b[1]=*((float4*)&bsld[2*512+0x40]);\
        a[1]=*((float4*)&asld[2*512+0x80]);\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[3*512+0x00]);\
        a[2]=*((float4*)&asld[3*512+0x00]);\
        b[3]=*((float4*)&bsld[3*512+0x40]);\
        a[3]=*((float4*)&asld[3*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[4*512+0x00]);\
        a[0]=*((float4*)&asld[4*512+0x00]);\
        b[1]=*((float4*)&bsld[4*512+0x40]);\
        a[1]=*((float4*)&asld[4*512+0x80]);\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[5*512+0x00]);\
        a[2]=*((float4*)&asld[5*512+0x00]);\
        b[3]=*((float4*)&bsld[5*512+0x40]);\
        a[3]=*((float4*)&asld[5*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[6*512+0x00]);\
        a[0]=*((float4*)&asld[6*512+0x00]);\
        b[1]=*((float4*)&bsld[6*512+0x40]);\
        a[1]=*((float4*)&asld[6*512+0x80]);\
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[7*512+0x00]);\
        a[2]=*((float4*)&asld[7*512+0x00]);\
        b[3]=*((float4*)&bsld[7*512+0x40]);\
        a[3]=*((float4*)&asld[7*512+0x80]);\
        *((float4*)asst)=p;\
        BOP8x8(c,&a[0],&b[0])        \
        *((float*)&bsst[0x1000])=q.x;\
        *((float*)&bsst[0x1200])=q.y;\
        *((float*)&bsst[0x1400])=q.z;\
        *((float*)&bsst[0x1600])=q.w;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])\
        ofs^=0x2000;\
    }\
    b[2]=*((float4*)&bsld[1*512+0x00]);\
    a[2]=*((float4*)&asld[1*512+0x00]);\
    b[3]=*((float4*)&bsld[1*512+0x40]);\
    a[3]=*((float4*)&asld[1*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[2*512+0x00]);\
    a[0]=*((float4*)&asld[2*512+0x00]);\
    b[1]=*((float4*)&bsld[2*512+0x40]);\
    a[1]=*((float4*)&asld[2*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[3*512+0x00]);\
    a[2]=*((float4*)&asld[3*512+0x00]);\
    b[3]=*((float4*)&bsld[3*512+0x40]);\
    a[3]=*((float4*)&asld[3*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[4*512+0x00]);\
    a[0]=*((float4*)&asld[4*512+0x00]);\
    b[1]=*((float4*)&bsld[4*512+0x40]);\
    a[1]=*((float4*)&asld[4*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[5*512+0x00]);\
    a[2]=*((float4*)&asld[5*512+0x00]);\
    b[3]=*((float4*)&bsld[5*512+0x40]);\
    a[3]=*((float4*)&asld[5*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[6*512+0x00]);\
    a[0]=*((float4*)&asld[6*512+0x00]);\
    b[1]=*((float4*)&bsld[6*512+0x40]);\
    a[1]=*((float4*)&asld[6*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[7*512+0x00]);\
    a[2]=*((float4*)&asld[7*512+0x00]);\
    b[3]=*((float4*)&bsld[7*512+0x40]);\
    a[3]=*((float4*)&asld[7*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[(slot_y<<7)|((lane&0x10)>>1)|((lane&1)<<2)]; }\
    sgemm_epilog64x32##suffix( d_c, (const char*)bias, &smem[slot<<10], c, lane, ldc, x, (anr+1)>>1, qnc-y, alpha );\
}

sgemmcc_128x128(0,)
sgemmcc_128x128(0,_relu)
sgemmcc_128x128(1,_bias)
sgemmcc_128x128(1,_bias_relu)

#define sgemmccx_128x128(add_bias,suffix)   \
__global__ void dk_sgemmccx_128x128##suffix(\
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
    __shared__ char smem[16384]; \
    __shared__ float s_bias[128];\
    float c[64];                 \
    float4 a[4], b[4];           \
    unsigned int bx=blockIdx.x;  \
    unsigned int by=blockIdx.y;  \
    unsigned int bz=blockIdx.z;  \
    unsigned int tid=threadIdx.x;\
    unsigned int lane=tid&31;    \
    unsigned int slot=tid>>5;    \
    unsigned int slot_x=slot&1;  \
    unsigned int slot_y=slot>>1; \
    unsigned int u=tid&127;      \
    unsigned int v=tid>>7;       \
    unsigned int qnr=(anr+3)>>2; \
    unsigned int x=(bx<<6)+((slot_x<<5)+lane);\
    unsigned int y=(by<<7)+(slot_y<<5);\
    unsigned int ai=(bx<<5)+lane;      \
    unsigned int bi=(by<<7)+u;         \
    unsigned int sai=ai<qnr?ai:(qnr-1);\
    unsigned int sbi=bi<qnc?bi:(qnc-1);\
    d_a+=(bz*pnc+slot)*lda+(sai<<4);\
    d_b+=(bz*qnc+sbi )*ldb+(v<<4);  \
    d_c+=(bz*qnc+y   )*ldc+(x<<3);  \
    float4 q=*((const float4* __restrict__)d_b);\
    float4 p=*((const float4* __restrict__)d_a);\
    char* __restrict__ asst_base=&smem[tid<<4]; \
    char* __restrict__ bsst_base=&smem[(v<<11)|(u<<2)];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)asst_base)=p;          \
    *((float*)&bsst_base[0x1000])=q.x;\
    *((float*)&bsst_base[0x1200])=q.y;\
    *((float*)&bsst_base[0x1400])=q.z;\
    *((float*)&bsst_base[0x1600])=q.w;\
    SZERO64(c)\
    __syncthreads();\
    if(add_bias){ if((v==0)&(bi<qnc)){ s_bias[u]=d_bias[bz*qnc+bi]; } }\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=0x2000; \
    unsigned int elda=lda<<3;\
    for( int k=pnc-8; k>0; k-=8 )\
    {\
        q=*((const float4* __restrict__)(d_b+=32  ));\
        p=*((const float4* __restrict__)(d_a+=elda));\
        b[2]=*((float4*)&bsld[1*512+0x00]);\
        a[2]=*((float4*)&asld[1*512+0x00]);\
        b[3]=*((float4*)&bsld[1*512+0x40]);\
        a[3]=*((float4*)&asld[1*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[2*512+0x00]);\
        a[0]=*((float4*)&asld[2*512+0x00]);\
        b[1]=*((float4*)&bsld[2*512+0x40]);\
        a[1]=*((float4*)&asld[2*512+0x80]);\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[3*512+0x00]);\
        a[2]=*((float4*)&asld[3*512+0x00]);\
        b[3]=*((float4*)&bsld[3*512+0x40]);\
        a[3]=*((float4*)&asld[3*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[4*512+0x00]);\
        a[0]=*((float4*)&asld[4*512+0x00]);\
        b[1]=*((float4*)&bsld[4*512+0x40]);\
        a[1]=*((float4*)&asld[4*512+0x80]);\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[5*512+0x00]);\
        a[2]=*((float4*)&asld[5*512+0x00]);\
        b[3]=*((float4*)&bsld[5*512+0x40]);\
        a[3]=*((float4*)&asld[5*512+0x80]);\
        BOP8x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[6*512+0x00]);\
        a[0]=*((float4*)&asld[6*512+0x00]);\
        b[1]=*((float4*)&bsld[6*512+0x40]);\
        a[1]=*((float4*)&asld[6*512+0x80]);\
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])              \
        b[2]=*((float4*)&bsld[7*512+0x00]);\
        a[2]=*((float4*)&asld[7*512+0x00]);\
        b[3]=*((float4*)&bsld[7*512+0x40]);\
        a[3]=*((float4*)&asld[7*512+0x80]);\
        *((float4*)asst)=p;\
        BOP8x8(c,&a[0],&b[0])        \
        *((float*)&bsst[0x1000])=q.x;\
        *((float*)&bsst[0x1200])=q.y;\
        *((float*)&bsst[0x1400])=q.z;\
        *((float*)&bsst[0x1600])=q.w;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])\
        ofs^=0x2000;\
    }\
    b[2]=*((float4*)&bsld[1*512+0x00]);\
    a[2]=*((float4*)&asld[1*512+0x00]);\
    b[3]=*((float4*)&bsld[1*512+0x40]);\
    a[3]=*((float4*)&asld[1*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[2*512+0x00]);\
    a[0]=*((float4*)&asld[2*512+0x00]);\
    b[1]=*((float4*)&bsld[2*512+0x40]);\
    a[1]=*((float4*)&asld[2*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[3*512+0x00]);\
    a[2]=*((float4*)&asld[3*512+0x00]);\
    b[3]=*((float4*)&bsld[3*512+0x40]);\
    a[3]=*((float4*)&asld[3*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[4*512+0x00]);\
    a[0]=*((float4*)&asld[4*512+0x00]);\
    b[1]=*((float4*)&bsld[4*512+0x40]);\
    a[1]=*((float4*)&asld[4*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[5*512+0x00]);\
    a[2]=*((float4*)&asld[5*512+0x00]);\
    b[3]=*((float4*)&bsld[5*512+0x40]);\
    a[3]=*((float4*)&asld[5*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[6*512+0x00]);\
    a[0]=*((float4*)&asld[6*512+0x00]);\
    b[1]=*((float4*)&bsld[6*512+0x40]);\
    a[1]=*((float4*)&asld[6*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[7*512+0x00]);\
    a[2]=*((float4*)&asld[7*512+0x00]);\
    b[3]=*((float4*)&bsld[7*512+0x40]);\
    a[3]=*((float4*)&asld[7*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[(slot_y<<7)|((lane&0x10)>>1)|((lane&1)<<2)]; }\
    sgemm_epilog64x32##suffix( d_c, (const char*)bias, &smem[slot<<10], c, lane, ldc, x, (anr+1)>>1, qnc-y, alpha );\
}

sgemmccx_128x128(0,)
sgemmccx_128x128(0,_relu)
sgemmccx_128x128(1,_bias)
sgemmccx_128x128(1,_bias_relu)