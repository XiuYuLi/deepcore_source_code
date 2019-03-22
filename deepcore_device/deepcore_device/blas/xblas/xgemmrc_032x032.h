__global__ void dk_xgemmrc_32x32( 
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
    __shared__ char smem[4096];
    float c[32];
    float4 a[2], b[4];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int bz=blockIdx.z;
    unsigned int tid=threadIdx.x;
    unsigned int sai=tid<anr?tid:(anr-1);
    unsigned int sbi=tid<cnc?tid:(cnc-1);
    unsigned int x=(bx<<5)+tid;
    unsigned int y=(by<<5)+((tid>>4)<<3);
    d_a+=(bz*anr+sai)*lda;
    d_b+=(bz*cnc+sbi)*ldb;
    d_c+=(bz*cnc+y)*ldc+(x<<2);
    __half8 p=*((const __half8*)d_a);
    __half8 q=*((const __half8*)d_b);
    char* __restrict__ sst_base=&smem[tid<<2];
    char* __restrict__ asld_base=&smem[((tid&0xe)<<3)];
    char* __restrict__ bsld_base=&smem[0x400|((tid&0x10)<<1)|((tid&0x1)<<4)];
    char* __restrict__ asld=asld_base;
    char* __restrict__ bsld=bsld_base;
    *((float*)&sst_base[0x000])=__half2float(HLO(p.lo.x));
    *((float*)&sst_base[0x080])=__half2float(HHI(p.lo.x));
    *((float*)&sst_base[0x100])=__half2float(HLO(p.lo.y));
    *((float*)&sst_base[0x180])=__half2float(HHI(p.lo.y));
    *((float*)&sst_base[0x200])=__half2float(HLO(p.hi.x));
    *((float*)&sst_base[0x280])=__half2float(HHI(p.hi.x));
    *((float*)&sst_base[0x300])=__half2float(HLO(p.hi.y));
    *((float*)&sst_base[0x380])=__half2float(HHI(p.hi.y));
    *((float*)&sst_base[0x400])=__half2float(HLO(q.lo.x));
    *((float*)&sst_base[0x480])=__half2float(HHI(q.lo.x));
    *((float*)&sst_base[0x500])=__half2float(HLO(q.lo.y));
    *((float*)&sst_base[0x580])=__half2float(HHI(q.lo.y));
    *((float*)&sst_base[0x600])=__half2float(HLO(q.hi.x));
    *((float*)&sst_base[0x680])=__half2float(HHI(q.hi.x));
    *((float*)&sst_base[0x700])=__half2float(HLO(q.hi.y));
    *((float*)&sst_base[0x780])=__half2float(HHI(q.hi.y));
    __syncthreads();
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i]=0.f; }
    b[0]=*((float4*)&bsld[0x00]); 
    a[0]=*((float4*)&asld[0x00]);
    b[1]=*((float4*)&bsld[0x40]);  
    unsigned int ofs=0x800;
    for( int k=bnr-8; k>0; k-=8 )
    {
        d_a+=16; d_b+=16;
        p=*((const __half8*)d_a);
        q=*((const __half8*)d_b);
        b[2]=*((float4*)&bsld[1*128+0x00]);
        a[1]=*((float4*)&asld[1*128+0x00]);
        b[3]=*((float4*)&bsld[1*128+0x40]);
        BOP4x8(c,&a[0],&b[0]) 
        b[0]=*((float4*)&bsld[2*128+0x00]);
        a[0]=*((float4*)&asld[2*128+0x00]);
        b[1]=*((float4*)&bsld[2*128+0x40]);
        BOP4x8(c,&a[1],&b[2]) 
        b[2]=*((float4*)&bsld[3*128+0x00]);
        a[1]=*((float4*)&asld[3*128+0x00]);
        b[3]=*((float4*)&bsld[3*128+0x40]);
        BOP4x8(c,&a[0],&b[0]) 
        b[0]=*((float4*)&bsld[4*128+0x00]);
        a[0]=*((float4*)&asld[4*128+0x00]);
        b[1]=*((float4*)&bsld[4*128+0x40]);
        BOP4x8(c,&a[1],&b[2]) 
        b[2]=*((float4*)&bsld[5*128+0x00]);
        a[1]=*((float4*)&asld[5*128+0x00]);
        b[3]=*((float4*)&bsld[5*128+0x40]);
        char* __restrict__ sst=sst_base+ofs;    
        BOP4x8(c,&a[0],&b[0]) 
        b[0]=*((float4*)&bsld[6*128+0x00]);
        a[0]=*((float4*)&asld[6*128+0x00]);
        b[1]=*((float4*)&bsld[6*128+0x40]);
        *((float*)&sst[0x000])=__half2float(HLO(p.lo.x));
        *((float*)&sst[0x080])=__half2float(HHI(p.lo.x));
        *((float*)&sst[0x100])=__half2float(HLO(p.lo.y));
        *((float*)&sst[0x180])=__half2float(HHI(p.lo.y));
        *((float*)&sst[0x200])=__half2float(HLO(p.hi.x));
        *((float*)&sst[0x280])=__half2float(HHI(p.hi.x));
        *((float*)&sst[0x300])=__half2float(HLO(p.hi.y));
        *((float*)&sst[0x380])=__half2float(HHI(p.hi.y));
        BOP4x8(c,&a[1],&b[2]) 
        b[2]=*((float4*)&bsld[7*128+0x00]);
        a[1]=*((float4*)&asld[7*128+0x00]);
        b[3]=*((float4*)&bsld[7*128+0x40]);
        BOP4x8(c,&a[0],&b[0])
        *((float*)&sst[0x400])=__half2float(HLO(q.lo.x));
        *((float*)&sst[0x480])=__half2float(HHI(q.lo.x));
        *((float*)&sst[0x500])=__half2float(HLO(q.lo.y));
        *((float*)&sst[0x580])=__half2float(HHI(q.lo.y));
        *((float*)&sst[0x600])=__half2float(HLO(q.hi.x));
        *((float*)&sst[0x680])=__half2float(HHI(q.hi.x));
        *((float*)&sst[0x700])=__half2float(HLO(q.hi.y));
        *((float*)&sst[0x780])=__half2float(HHI(q.hi.y));
        asld=asld_base+ofs;
        bsld=bsld_base+ofs;
        __syncthreads();
        b[0]=*((float4*)&bsld[0x00]);
        a[0]=*((float4*)&asld[0x00]);
        b[1]=*((float4*)&bsld[0x40]);
        BOP4x8(c,&a[1],&b[2])
        ofs^=0x800;
    }
    b[2]=*((float4*)&bsld[1*128+0x00]);
    a[1]=*((float4*)&asld[1*128+0x00]);
    b[3]=*((float4*)&bsld[1*128+0x40]);
    BOP4x8(c,&a[0],&b[0]) 
    b[0]=*((float4*)&bsld[2*128+0x00]);
    a[0]=*((float4*)&asld[2*128+0x00]);
    b[1]=*((float4*)&bsld[2*128+0x40]);
    BOP4x8(c,&a[1],&b[2]) 
    b[2]=*((float4*)&bsld[3*128+0x00]);
    a[1]=*((float4*)&asld[3*128+0x00]);
    b[3]=*((float4*)&bsld[3*128+0x40]);
    BOP4x8(c,&a[0],&b[0]) 
    b[0]=*((float4*)&bsld[4*128+0x00]);
    a[0]=*((float4*)&asld[4*128+0x00]);
    b[1]=*((float4*)&bsld[4*128+0x40]);
    BOP4x8(c,&a[1],&b[2]) 
    b[2]=*((float4*)&bsld[5*128+0x00]);
    a[1]=*((float4*)&asld[5*128+0x00]);
    b[3]=*((float4*)&bsld[5*128+0x40]);
    BOP4x8(c,&a[0],&b[0]) 
    b[0]=*((float4*)&bsld[6*128+0x00]);
    a[0]=*((float4*)&asld[6*128+0x00]);
    b[1]=*((float4*)&bsld[6*128+0x40]);
    BOP4x8(c,&a[1],&b[2]) 
    b[2]=*((float4*)&bsld[7*128+0x00]);
    a[1]=*((float4*)&asld[7*128+0x00]);
    b[3]=*((float4*)&bsld[7*128+0x40]);
    BOP4x8(c,&a[0],&b[0])
    BOP4x8(c,&a[1],&b[2])
    xgemm_epilog32x32( d_c, NULL, smem, c, tid, ldc, x, (anr+1)>>1, cnc-y, alpha );
}