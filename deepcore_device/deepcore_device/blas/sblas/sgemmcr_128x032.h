#define sgemmcr_128x32(grad,suffix)\
__global__ void dk_sgemmcr_128x32##suffix(\
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
    __shared__ char smem[2*10240]; \
    float c[32];                   \
    float4 a[2], b[4];             \
    unsigned int bx=blockIdx.x;    \
    unsigned int by=blockIdx.y;    \
    unsigned int tid=threadIdx.x;  \
    unsigned int lane=tid&31;      \
    unsigned int slot=tid>>5;      \
    unsigned int ai=(bx<<5)+lane;  \
    unsigned int bi=tid&7;         \
    unsigned int qnr=(anr+3)>>2;\
    unsigned int cnc=(qnc+3)>>2;\
    unsigned int sai=ai<qnr?ai:(qnr-1);\
    unsigned int sbi=bi<cnc?bi:(cnc-1);\
    unsigned int qlda=lda<<2; \
    unsigned int xldb=ldb<<4; \
    unsigned int x=bx*128+tid;\
    d_a+=(by*qnc+slot    )*lda+(sai<<4);\
    d_b+=(by*qnc+(tid>>3))*ldb+(sbi<<4);\
    d_c+=by*pnc*ldc+(x<<2);             \
    if(grad){ d_x+=by*pnc*ldc+(x<<2); } \
    float4 p0=*((const float4*)d_a); d_a+=qlda;\
    float4 p1=*((const float4*)d_a); d_a+=qlda;\
    float4 p2=*((const float4*)d_a); d_a+=qlda;\
    float4 p3=*((const float4*)d_a);\
    float4 p4=*((const float4*)d_b);\
    char* __restrict__ sst_base =&smem[tid<<4];\
    char* __restrict__ asld_base=&smem[(slot<<7)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x2000|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float4*)&sst_base[0x0000])=p0; \
    *((float4*)&sst_base[0x0800])=p1; \
    *((float4*)&sst_base[0x1000])=p2; \
    *((float4*)&sst_base[0x1800])=p3; \
    *((float4*)&sst_base[0x2000])=p4; \
    __syncthreads();\
    SZERO32(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    unsigned int ofs=0x2800;     \
    for( int k=qnc-16; k>0; k-=16 ){\
        p0=*((const float4*)(d_a+=qlda));\
        p1=*((const float4*)(d_a+=qlda));\
        p2=*((const float4*)(d_a+=qlda));\
        p3=*((const float4*)(d_a+=qlda));\
        p4=*((const float4*)(d_b+=xldb));\
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
        char* __restrict__ sst=sst_base+ofs;\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[15*128+0x00]);\
        a[1]=*((float4*)&asld[15*512+0x00]);\
        b[3]=*((float4*)&bsld[15*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        *((float4*)&sst[0x0000])=p0;\
        *((float4*)&sst[0x0800])=p1;\
        *((float4*)&sst[0x1000])=p2;\
        *((float4*)&sst[0x1800])=p3;\
        *((float4*)&sst[0x2000])=p4;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        BOP4x8(c,&a[1],&b[2])        \
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
    sgemm_epilog32x32##suffix( d_c, d_x, &smem[slot<<9], c, lane, ldc, x, anr, pnc, alpha );\
}

sgemmcr_128x32(0,)
sgemmcr_128x32(1,_drelu)
sgemmcr_128x32(1,_xdrv)