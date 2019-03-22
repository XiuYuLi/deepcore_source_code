#define xgemmcr_128x64(grad,suffix)       \
__global__ void dk_xgemmcr_128x64##suffix(\
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
    __shared__ char smem[12288];   \
    float c[64];                   \
    float4 a[4], b[4];             \
    unsigned int bx=blockIdx.x;    \
    unsigned int by=blockIdx.y;    \
    unsigned int tid=threadIdx.x;  \
    unsigned int lane=tid&31;      \
    unsigned int slot=tid>>5;      \
    unsigned int slot_x=slot&1;    \
    unsigned int slot_y=slot>>1;   \
    unsigned int u=tid&15;         \
    unsigned int v=tid>>4;         \
    unsigned int x=(bx<<6)|(slot_x<<5)|lane;\
    unsigned int y=slot_y<<5;    \
    unsigned int ai=(bx<<5)|lane;\
    unsigned int enr=(anr+7)>>3;\
    unsigned int cnc=(pnc+3)>>2;\
    unsigned int sai=ai<enr?ai:(enr-1);\
    unsigned int sbi=u <cnc?u :(cnc-1);\
    unsigned int elda=lda<<3;\
    unsigned int eldb=ldb<<3;\
    d_a+=(by*qnc+slot)*lda+(sai<<4);\
    d_b+=(by*qnc+v   )*ldb+(sbi<<3);\
    d_c+=(by*pnc+y   )*ldc+(x<<2);  \
    if(grad){ d_x+=(by*pnc+y)*ldc+(x<<2); } \
    __half8 p=*((const __half8*)d_a);\
    __half4 q=*((const __half4*)d_b);\
    char* __restrict__ asst_base=&smem[tid<<5];\
    char* __restrict__ bsst_base=&smem[tid<<4];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)&asst_base[0x0000])=__half42float4(p.lo);\
    *((float4*)&asst_base[0x0010])=__half42float4(p.hi);\
    *((float4*)&bsst_base[0x1000])=__half42float4(q);   \
    __syncthreads();\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]); \
    a[0]=*((float4*)&asld[0x00]); \
    b[1]=*((float4*)&bsld[0x40]); \
    a[1]=*((float4*)&asld[0x80]); \
    unsigned int ofs=0x1800;      \
    for( int k=qnc-8; k>0; k-=8 ){\
        p=*((const __half8*)(d_a+=elda));\
        q=*((const __half4*)(d_b+=eldb));\
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
        *((float4*)&asst[0x0000])=__half42float4(p.lo);\
        *((float4*)&asst[0x0010])=__half42float4(p.hi);\
        *((float4*)&bsst[0x1000])=__half42float4(q);   \
        asld=asld_base+ofs;  \
        bsld=bsld_base+ofs;  \
        BOP8x8(c,&a[0],&b[0])\
        __syncthreads();     \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])        \
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
    xgemm_epilog64x32##suffix( d_c, d_x, &smem[slot<<10], c, lane, ldc, x, anr>>1, pnc-y, alpha );\
}

xgemmcr_128x64(0,)
xgemmcr_128x64(1,_drelu)
xgemmcr_128x64(1,_xdrv)
