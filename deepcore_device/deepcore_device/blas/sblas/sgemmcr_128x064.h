#define sgemmcr_128x64(grad,suffix)\
__global__ void __launch_bounds__(128,4)\
dk_sgemmcr_128x64##suffix(         \
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
    unsigned int x=(bx<<6)+(slot_x<<5)+lane;\
    unsigned int y=slot_y<<5;    \
    unsigned int ai=(bx<<5)+lane;\
    unsigned int qnr=(anr+3)>>2;\
    unsigned int cnc=(pnc+3)>>2;\
    unsigned int sai=ai<qnr?ai:(qnr-1);\
    unsigned int sbi=u <cnc?u :(cnc-1);\
    unsigned int qlda=lda<<2;\
    unsigned int eldb=ldb<<3;\
    d_a+=(by*qnc+slot)*lda+(sai<<4);\
    d_b+=(by*qnc+v   )*ldb+(sbi<<4);\
    d_c+=(by*pnc+y   )*ldc+(x<<3);  \
    if(grad){ d_x+=(by*pnc+y)*ldc+(x<<3); }\
    float4 p0=*((const float4*)d_a); d_a+=qlda;\
    float4 p1=*((const float4*)d_a);\
    float4 p2=*((const float4*)d_b);\
    char* __restrict__ sst_base=&smem[tid<<4];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)&sst_base[0x0000])=p0; \
    *((float4*)&sst_base[0x0800])=p1; \
    *((float4*)&sst_base[0x1000])=p2; \
    __syncthreads();\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]); \
    a[0]=*((float4*)&asld[0x00]); \
    b[1]=*((float4*)&bsld[0x40]); \
    a[1]=*((float4*)&asld[0x80]); \
    unsigned int ofs=0x1800;      \
    for( int k=qnc-8; k>0; k-=8 ){\
        p2=*((const float4*)(d_b+=eldb));\
        p0=*((const float4*)(d_a+=qlda));\
        p1=*((const float4*)(d_a+=qlda));\
        b[2]=*((float4*)&bsld[1*256+0x00]); \
        a[2]=*((float4*)&asld[1*512+0x00]); \
        b[3]=*((float4*)&bsld[1*256+0x40]); \
        a[3]=*((float4*)&asld[1*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[2*256+0x00]); \
        a[0]=*((float4*)&asld[2*512+0x00]); \
        b[1]=*((float4*)&bsld[2*256+0x40]); \
        a[1]=*((float4*)&asld[2*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[3*256+0x00]); \
        a[2]=*((float4*)&asld[3*512+0x00]); \
        b[3]=*((float4*)&bsld[3*256+0x40]); \
        a[3]=*((float4*)&asld[3*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[4*256+0x00]); \
        a[0]=*((float4*)&asld[4*512+0x00]); \
        b[1]=*((float4*)&bsld[4*256+0x40]); \
        a[1]=*((float4*)&asld[4*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[5*256+0x00]); \
        a[2]=*((float4*)&asld[5*512+0x00]); \
        b[3]=*((float4*)&bsld[5*256+0x40]); \
        a[3]=*((float4*)&asld[5*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[6*256+0x00]); \
        a[0]=*((float4*)&asld[6*512+0x00]); \
        b[1]=*((float4*)&bsld[6*256+0x40]); \
        a[1]=*((float4*)&asld[6*512+0x80]); \
        char* __restrict__ sst=sst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[7*256+0x00]); \
        a[2]=*((float4*)&asld[7*512+0x00]); \
        b[3]=*((float4*)&bsld[7*256+0x40]); \
        a[3]=*((float4*)&asld[7*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        *((float4*)&sst[0x0000])=p0;\
        *((float4*)&sst[0x0800])=p1;\
        *((float4*)&sst[0x1000])=p2;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
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
    sgemm_epilog64x32##suffix( d_c, d_x, &smem[slot<<10], c, lane, ldc, x, (anr+1)>>1, pnc-y, alpha );\
}

sgemmcr_128x64(0,)
sgemmcr_128x64(1,_drelu)
sgemmcr_128x64(1,_xdrv)
