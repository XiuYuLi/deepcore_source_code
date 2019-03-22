__global__ void dk_xgemmrc_128x128( 
    char*                       d_c, 
    const char*  __restrict__   d_a, 
    const char*  __restrict__   d_b, 
    float                       alpha, 
    int                         anr, 
    int                         bnr, 
    int                         cnc,
    int                         lda,
    int                         ldb,
    int                         ldc )
{
    __shared__ char smem[16384];
    float c[64];
    float4 a[4], b[4];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int bz=blockIdx.z;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&31;
    unsigned int slot=tid>>5;   
    unsigned int slot_x=slot&1;
    unsigned int slot_y=slot>>1;
    unsigned int u=tid&127;
    unsigned int v=tid>>7;
    unsigned int x=(bx<<6)+(slot_x<<5)+lane;
    unsigned int y=(by<<7)+(slot_y<<5);
    unsigned int ai=(bx<<7)+u;
    unsigned int bi=(by<<7)+u;
    unsigned int sai=ai<anr?ai:(anr-1);
    unsigned int sbi=bi<cnc?bi:(cnc-1);
    d_a+=(bz*anr+sai)*lda;
    d_b+=(bz*cnc+sbi)*ldb;
    d_c+=(bz*cnc+y  )*ldc+(x<<3);
    const char* d_x=v==0?d_a:d_b;
    __half8 p=*((const __half8*)d_x);
    char* __restrict__ sst_base =&smem[(v<<11)|(u<<2)];
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];
    char* __restrict__ asld=asld_base;
    char* __restrict__ bsld=bsld_base;
    *((float*)&sst_base[0x000])=__half2float(HLO(p.lo.x));
    *((float*)&sst_base[0x200])=__half2float(HHI(p.lo.x));
    *((float*)&sst_base[0x400])=__half2float(HLO(p.lo.y));
    *((float*)&sst_base[0x600])=__half2float(HHI(p.lo.y));
    *((float*)&sst_base[0x800])=__half2float(HLO(p.hi.x));
    *((float*)&sst_base[0xa00])=__half2float(HHI(p.hi.x));
    *((float*)&sst_base[0xc00])=__half2float(HLO(p.hi.y));
    *((float*)&sst_base[0xe00])=__half2float(HHI(p.hi.y));
    __syncthreads();
#pragma unroll
    for( int i=0; i<64; ++i ){ c[i]=0.f; }
    b[0]=*((float4*)&bsld[0x00]); 
    a[0]=*((float4*)&asld[0x00]);
    b[1]=*((float4*)&bsld[0x40]); 
    a[1]=*((float4*)&asld[0x80]); 
    unsigned int ofs=0x2000;
    for( int k=bnr-8; k>0; k-=8 )
    {
        p=*((const __half8*)(d_x+=16));
        b[2]=*((float4*)&bsld[1*512+0x00]);
        a[2]=*((float4*)&asld[1*512+0x00]);
        b[3]=*((float4*)&bsld[1*512+0x40]);
        a[3]=*((float4*)&asld[1*512+0x80]);
        BOP8x8(c,&a[0],&b[0])
        b[0]=*((float4*)&bsld[2*512+0x00]);
        a[0]=*((float4*)&asld[2*512+0x00]);
        b[1]=*((float4*)&bsld[2*512+0x40]);
        a[1]=*((float4*)&asld[2*512+0x80]);
        BOP8x8(c,&a[2],&b[2])
        b[2]=*((float4*)&bsld[3*512+0x00]);
        a[2]=*((float4*)&asld[3*512+0x00]);
        b[3]=*((float4*)&bsld[3*512+0x40]);
        a[3]=*((float4*)&asld[3*512+0x80]);
        BOP8x8(c,&a[0],&b[0])
        b[0]=*((float4*)&bsld[4*512+0x00]);
        a[0]=*((float4*)&asld[4*512+0x00]);
        b[1]=*((float4*)&bsld[4*512+0x40]);
        a[1]=*((float4*)&asld[4*512+0x80]);
        BOP8x8(c,&a[2],&b[2])
        b[2]=*((float4*)&bsld[5*512+0x00]);
        a[2]=*((float4*)&asld[5*512+0x00]);
        b[3]=*((float4*)&bsld[5*512+0x40]);
        a[3]=*((float4*)&asld[5*512+0x80]);
        BOP8x8(c,&a[0],&b[0])
        b[0]=*((float4*)&bsld[6*512+0x00]);
        a[0]=*((float4*)&asld[6*512+0x00]);
        b[1]=*((float4*)&bsld[6*512+0x40]);
        a[1]=*((float4*)&asld[6*512+0x80]);
        char* __restrict__ sst=sst_base+ofs;
        BOP8x8(c,&a[2],&b[2])
        b[2]=*((float4*)&bsld[7*512+0x00]);
        a[2]=*((float4*)&asld[7*512+0x00]);
        b[3]=*((float4*)&bsld[7*512+0x40]);
        a[3]=*((float4*)&asld[7*512+0x80]);
        *((float*)&sst[0x000])=__half2float(HLO(p.lo.x));
        *((float*)&sst[0x200])=__half2float(HHI(p.lo.x));
        *((float*)&sst[0x400])=__half2float(HLO(p.lo.y));
        *((float*)&sst[0x600])=__half2float(HHI(p.lo.y));
        BOP8x8(c,&a[0],&b[0])
        *((float*)&sst[0x800])=__half2float(HLO(p.hi.x));
        *((float*)&sst[0xa00])=__half2float(HHI(p.hi.x));
        *((float*)&sst[0xc00])=__half2float(HLO(p.hi.y));
        *((float*)&sst[0xe00])=__half2float(HHI(p.hi.y));
        asld=asld_base+ofs;
        bsld=bsld_base+ofs;
        __syncthreads();
        b[0]=*((float4*)&bsld[0x00]);
        a[0]=*((float4*)&asld[0x00]);
        b[1]=*((float4*)&bsld[0x40]);
        a[1]=*((float4*)&asld[0x80]);
        BOP8x8(c,&a[2],&b[2])
        ofs^=0x2000;
    }
    b[2]=*((float4*)&bsld[1*512+0x00]); 
    a[2]=*((float4*)&asld[1*512+0x00]);
    b[3]=*((float4*)&bsld[1*512+0x40]); 
    a[3]=*((float4*)&asld[1*512+0x80]); 
    BOP8x8(c,&a[0],&b[0])
    b[0]=*((float4*)&bsld[2*512+0x00]); 
    a[0]=*((float4*)&asld[2*512+0x00]);
    b[1]=*((float4*)&bsld[2*512+0x40]); 
    a[1]=*((float4*)&asld[2*512+0x80]); 
    BOP8x8(c,&a[2],&b[2])
    b[2]=*((float4*)&bsld[3*512+0x00]); 
    a[2]=*((float4*)&asld[3*512+0x00]);
    b[3]=*((float4*)&bsld[3*512+0x40]); 
    a[3]=*((float4*)&asld[3*512+0x80]); 
    BOP8x8(c,&a[0],&b[0])
    b[0]=*((float4*)&bsld[4*512+0x00]); 
    a[0]=*((float4*)&asld[4*512+0x00]);
    b[1]=*((float4*)&bsld[4*512+0x40]); 
    a[1]=*((float4*)&asld[4*512+0x80]); 
    BOP8x8(c,&a[2],&b[2])
    b[2]=*((float4*)&bsld[5*512+0x00]); 
    a[2]=*((float4*)&asld[5*512+0x00]);
    b[3]=*((float4*)&bsld[5*512+0x40]); 
    a[3]=*((float4*)&asld[5*512+0x80]); 
    BOP8x8(c,&a[0],&b[0])
    b[0]=*((float4*)&bsld[6*512+0x00]); 
    a[0]=*((float4*)&asld[6*512+0x00]);
    b[1]=*((float4*)&bsld[6*512+0x40]); 
    a[1]=*((float4*)&asld[6*512+0x80]); 
    BOP8x8(c,&a[2],&b[2])
    b[2]=*((float4*)&bsld[7*512+0x00]); 
    a[2]=*((float4*)&asld[7*512+0x00]);
    b[3]=*((float4*)&bsld[7*512+0x40]); 
    a[3]=*((float4*)&asld[7*512+0x80]);
    BOP8x8(c,&a[0],&b[0])
    BOP8x8(c,&a[2],&b[2])
    xgemm_epilog64x32( d_c, NULL, &smem[slot<<10], c, lane, ldc, x, anr>>1, cnc-y, alpha );
}