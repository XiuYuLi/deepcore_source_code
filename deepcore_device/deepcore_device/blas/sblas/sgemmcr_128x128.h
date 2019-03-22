#define sgemmcr_128x128(grad,suffix)\
__global__ void dk_sgemmcr_128x128##suffix( \
          char*              d_c  ,\
    const char* __restrict__ d_a  ,\
    const char* __restrict__ d_b  ,\
    const char* __restrict__ d_x  ,\
    float                    alpha,\
    int                      anr  ,\
    int                      pnc  ,\
    int                      qnc  ,\
    int                      lda  ,\
    int                      ldb  ,\
    int                      ldc ){\
    __shared__ char smem[16384];   \
    float c[64];                   \
    float4 a[4], b[4];             \
    unsigned int bx=blockIdx.x;    \
    unsigned int by=blockIdx.y;    \
    unsigned int tid=threadIdx.x;  \
    unsigned int lane=tid&31;      \
    unsigned int slot=tid>>5;      \
    unsigned int slot_x=slot&1;    \
    unsigned int slot_y=slot>>1;   \
    unsigned int x=(bx<<6)+(slot_x<<5)+lane;\
    unsigned int y=(by<<7)+(slot_y<<5);\
    unsigned int ai=(bx<<5)+lane;\
    unsigned int bi=(by<<5)+lane;\
    unsigned int qnr=(anr+3)>>2;\
    unsigned int cnc=(pnc+3)>>2;\
    unsigned int sai=ai<qnr?ai:(qnr-1);\
    unsigned int sbi=bi<cnc?bi:(cnc-1);\
    d_a+=slot*lda+(sai<<4);\
    d_b+=slot*ldb+(sbi<<4);\
    d_c+=y   *ldc+(x<<3);  \
    if(grad){ d_x+=y*ldc+(x<<3); } \
    float4 p=*((const float4*)d_a);\
    float4 q=*((const float4*)d_b);\
    char* __restrict__ sst_base=&smem[tid<<4];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)&sst_base[0x0000])=p;  \
    *((float4*)&sst_base[0x1000])=q;  \
    __syncthreads();\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]); \
    a[0]=*((float4*)&asld[0x00]); \
    b[1]=*((float4*)&bsld[0x40]); \
    a[1]=*((float4*)&asld[0x80]); \
    unsigned int ofs=0x2000;      \
    unsigned int elda=lda<<3;     \
    unsigned int eldb=ldb<<3;     \
    for( int k=qnc-8; k>0; k-=8 ){\
        q=*((const float4*)(d_b+=eldb));\
        p=*((const float4*)(d_a+=elda));\
        b[2]=*((float4*)&bsld[1*512+0x00]); \
        a[2]=*((float4*)&asld[1*512+0x00]); \
        b[3]=*((float4*)&bsld[1*512+0x40]); \
        a[3]=*((float4*)&asld[1*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[2*512+0x00]); \
        a[0]=*((float4*)&asld[2*512+0x00]); \
        b[1]=*((float4*)&bsld[2*512+0x40]); \
        a[1]=*((float4*)&asld[2*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[3*512+0x00]); \
        a[2]=*((float4*)&asld[3*512+0x00]); \
        b[3]=*((float4*)&bsld[3*512+0x40]); \
        a[3]=*((float4*)&asld[3*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[4*512+0x00]); \
        a[0]=*((float4*)&asld[4*512+0x00]); \
        b[1]=*((float4*)&bsld[4*512+0x40]); \
        a[1]=*((float4*)&asld[4*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[5*512+0x00]); \
        a[2]=*((float4*)&asld[5*512+0x00]); \
        b[3]=*((float4*)&bsld[5*512+0x40]); \
        a[3]=*((float4*)&asld[5*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[6*512+0x00]); \
        a[0]=*((float4*)&asld[6*512+0x00]); \
        b[1]=*((float4*)&bsld[6*512+0x40]); \
        a[1]=*((float4*)&asld[6*512+0x80]); \
        char* __restrict__ sst=sst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[7*512+0x00]); \
        a[2]=*((float4*)&asld[7*512+0x00]); \
        b[3]=*((float4*)&bsld[7*512+0x40]); \
        a[3]=*((float4*)&asld[7*512+0x80]); \
        *((float4*)&sst[0x0000])=p;\
        BOP8x8(c,&a[0],&b[0])      \
        *((float4*)&sst[0x1000])=q;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])        \
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
    sgemm_epilog64x32##suffix( d_c, d_x, &smem[slot<<10], c, lane, ldc, x, (anr+1)>>1, qnc-y, alpha );\
}

sgemmcr_128x128(0,)
sgemmcr_128x128(1,_drelu)
sgemmcr_128x128(1,_xdrv)

#define sgemmcrx_128x128(grad,suffix)       \
__global__ void dk_sgemmcrx_128x128##suffix(\
          char*              d_c  ,\
    const char* __restrict__ d_a  ,\
    const char* __restrict__ d_b  ,\
    const char* __restrict__ d_x  ,\
    float                    alpha,\
    int                      anr  ,\
    int                      pnc  ,\
    int                      qnc  ,\
    int                      lda  ,\
    int                      ldb  ,\
    int                      ldc ){\
    __shared__ char smem[16384];   \
    float c[64];                   \
    float4 a[4], b[4];             \
    unsigned int bx=blockIdx.x;    \
    unsigned int by=blockIdx.y;    \
    unsigned int bz=blockIdx.z;    \
    unsigned int tid=threadIdx.x;  \
    unsigned int lane=tid&31;      \
    unsigned int slot=tid>>5;      \
    unsigned int slot_x=slot&1;    \
    unsigned int slot_y=slot>>1;   \
    unsigned int x=(bx<<6)+(slot_x<<5)+lane;\
    unsigned int y=(by<<7)+(slot_y<<5);\
    unsigned int ai=(bx<<5)+lane;\
    unsigned int bi=(by<<5)+lane;\
    unsigned int qnr=(anr+3)>>2;\
    unsigned int cnc=(pnc+3)>>2;\
    unsigned int sai=ai<qnr?ai:(qnr-1);\
    unsigned int sbi=bi<cnc?bi:(cnc-1);\
    d_a+=(bz*qnc+slot)*lda+(sai<<4);\
    d_b+=(bz*qnc+slot)*ldb+(sbi<<4);\
    d_c+=(bz*pnc+y   )*ldc+(x<<3);  \
    if(grad){ d_x+=(bz*pnc+y)*ldc+(x<<3); }     \
    float4 p=*((const float4* __restrict__)d_a);\
    float4 q=*((const float4* __restrict__)d_b);\
    char* __restrict__ sst_base=&smem[tid<<4];  \
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)&sst_base[0x0000])=p;  \
    *((float4*)&sst_base[0x1000])=q;  \
    __syncthreads();\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]); \
    a[0]=*((float4*)&asld[0x00]); \
    b[1]=*((float4*)&bsld[0x40]); \
    a[1]=*((float4*)&asld[0x80]); \
    unsigned int ofs=0x2000;      \
    unsigned int elda=lda<<3;     \
    unsigned int eldb=ldb<<3;     \
    for( int k=qnc-8; k>0; k-=8 ){\
        q=*((const float4* __restrict__)(d_b+=eldb));\
        p=*((const float4* __restrict__)(d_a+=elda));\
        b[2]=*((float4*)&bsld[1*512+0x00]); \
        a[2]=*((float4*)&asld[1*512+0x00]); \
        b[3]=*((float4*)&bsld[1*512+0x40]); \
        a[3]=*((float4*)&asld[1*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[2*512+0x00]); \
        a[0]=*((float4*)&asld[2*512+0x00]); \
        b[1]=*((float4*)&bsld[2*512+0x40]); \
        a[1]=*((float4*)&asld[2*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[3*512+0x00]); \
        a[2]=*((float4*)&asld[3*512+0x00]); \
        b[3]=*((float4*)&bsld[3*512+0x40]); \
        a[3]=*((float4*)&asld[3*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[4*512+0x00]); \
        a[0]=*((float4*)&asld[4*512+0x00]); \
        b[1]=*((float4*)&bsld[4*512+0x40]); \
        a[1]=*((float4*)&asld[4*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[5*512+0x00]); \
        a[2]=*((float4*)&asld[5*512+0x00]); \
        b[3]=*((float4*)&bsld[5*512+0x40]); \
        a[3]=*((float4*)&asld[5*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[6*512+0x00]); \
        a[0]=*((float4*)&asld[6*512+0x00]); \
        b[1]=*((float4*)&bsld[6*512+0x40]); \
        a[1]=*((float4*)&asld[6*512+0x80]); \
        char* __restrict__ sst=sst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[7*512+0x00]); \
        a[2]=*((float4*)&asld[7*512+0x00]); \
        b[3]=*((float4*)&bsld[7*512+0x40]); \
        a[3]=*((float4*)&asld[7*512+0x80]); \
        *((float4*)&sst[0x0000])=p;\
        BOP8x8(c,&a[0],&b[0])      \
        *((float4*)&sst[0x1000])=q;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])        \
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
    sgemm_epilog64x32##suffix( d_c, d_x, &smem[slot<<10], c, lane, ldc, x, (anr+1)>>1, pnc-y, alpha );\
}

sgemmcrx_128x128(0,)
sgemmcrx_128x128(1,_drelu)
sgemmcrx_128x128(1,_xdrv)