__global__ void dk_sgemmrc_64x64( 
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
    __shared__ char smem[8192];
    float c[64];
    float4 a[4], b[4];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int bz=blockIdx.z;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&31;
    unsigned int slot=tid>>5;
    unsigned int x=(bx<<5)+lane;
    unsigned int y=(by<<6)+(slot<<5);
    unsigned int sai=tid<anr?tid:(anr-1);
    unsigned int sbi=tid<cnc?tid:(cnc-1);
    d_a+=(bz*anr+sai)*lda;
    d_b+=(bz*cnc+sbi)*ldb;
    d_c+=(bz*cnc+y  )*ldc+(x<<3);
    float4 p0=*((const float4*)(d_a+0x00));
    float4 p1=*((const float4*)(d_a+0x10));
    float4 p2=*((const float4*)(d_b+0x00));
    float4 p3=*((const float4*)(d_b+0x10));
    char* __restrict__ sst_base =&smem[tid<<2];
    char* __restrict__ asld_base=&smem[((lane&0xe)<<3)];
    char* __restrict__ bsld_base=&smem[0x800|(slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];
    char* __restrict__ asld=asld_base;
    char* __restrict__ bsld=bsld_base;
    *((float*)&sst_base[0x000])=p0.x;
    *((float*)&sst_base[0x100])=p0.y;
    *((float*)&sst_base[0x200])=p0.z;
    *((float*)&sst_base[0x300])=p0.w;
    *((float*)&sst_base[0x400])=p1.x;
    *((float*)&sst_base[0x500])=p1.y;
    *((float*)&sst_base[0x600])=p1.z;
    *((float*)&sst_base[0x700])=p1.w;
    *((float*)&sst_base[0x800])=p2.x;
    *((float*)&sst_base[0x900])=p2.y;
    *((float*)&sst_base[0xa00])=p2.z;
    *((float*)&sst_base[0xb00])=p2.w;
    *((float*)&sst_base[0xc00])=p3.x;
    *((float*)&sst_base[0xd00])=p3.y;
    *((float*)&sst_base[0xe00])=p3.z;
    *((float*)&sst_base[0xf00])=p3.w;
    __syncthreads();
#pragma unroll
    for( int i=0; i<64; ++i ){ c[i]=0.f; }
    b[0]=*((float4*)&bsld[0x00]); 
    a[0]=*((float4*)&asld[0x00]);
    b[1]=*((float4*)&bsld[0x40]); 
    a[1]=*((float4*)&asld[0x80]); 
    unsigned int ofs=0x1000;
    for( int k=bnr-8; k>0; k-=8 )
    {
        d_a+=32; d_b+=32;
        p0=*((const float4*)(d_a+0x00));
        p1=*((const float4*)(d_a+0x10));        
        p2=*((const float4*)(d_b+0x00));
        p3=*((const float4*)(d_b+0x10));
        b[2]=*((float4*)&bsld[1*256+0x00]); 
        a[2]=*((float4*)&asld[1*256+0x00]);
        b[3]=*((float4*)&bsld[1*256+0x40]); 
        a[3]=*((float4*)&asld[1*256+0x80]); 
        BOP8x8(c,&a[0],&b[0])
        b[0]=*((float4*)&bsld[2*256+0x00]); 
        a[0]=*((float4*)&asld[2*256+0x00]);
        b[1]=*((float4*)&bsld[2*256+0x40]); 
        a[1]=*((float4*)&asld[2*256+0x80]); 
        BOP8x8(c,&a[2],&b[2])
        b[2]=*((float4*)&bsld[3*256+0x00]); 
        a[2]=*((float4*)&asld[3*256+0x00]);
        b[3]=*((float4*)&bsld[3*256+0x40]); 
        a[3]=*((float4*)&asld[3*256+0x80]); 
        BOP8x8(c,&a[0],&b[0])
        b[0]=*((float4*)&bsld[4*256+0x00]); 
        a[0]=*((float4*)&asld[4*256+0x00]);
        b[1]=*((float4*)&bsld[4*256+0x40]); 
        a[1]=*((float4*)&asld[4*256+0x80]); 
        BOP8x8(c,&a[2],&b[2])
        b[2]=*((float4*)&bsld[5*256+0x00]); 
        a[2]=*((float4*)&asld[5*256+0x00]);
        b[3]=*((float4*)&bsld[5*256+0x40]); 
        a[3]=*((float4*)&asld[5*256+0x80]);
        BOP8x8(c,&a[0],&b[0])
        b[0]=*((float4*)&bsld[6*256+0x00]); 
        a[0]=*((float4*)&asld[6*256+0x00]);
        b[1]=*((float4*)&bsld[6*256+0x40]); 
        a[1]=*((float4*)&asld[6*256+0x80]);     
        char* __restrict__ sst=sst_base+ofs;
        BOP8x8(c,&a[2],&b[2])
        b[2]=*((float4*)&bsld[7*256+0x00]); 
        a[2]=*((float4*)&asld[7*256+0x00]);
        b[3]=*((float4*)&bsld[7*256+0x40]);
        a[3]=*((float4*)&asld[7*256+0x80]);
        *((float*)&sst[0x000])=p0.x;
        *((float*)&sst[0x100])=p0.y;
        *((float*)&sst[0x200])=p0.z;
        *((float*)&sst[0x300])=p0.w;
        *((float*)&sst[0x400])=p1.x;
        *((float*)&sst[0x500])=p1.y;
        *((float*)&sst[0x600])=p1.z;
        *((float*)&sst[0x700])=p1.w;
        BOP8x8(c,&a[0],&b[0])   
        *((float*)&sst[0x800])=p2.x;
        *((float*)&sst[0x900])=p2.y;
        *((float*)&sst[0xa00])=p2.z;
        *((float*)&sst[0xb00])=p2.w;
        *((float*)&sst[0xc00])=p3.x;
        *((float*)&sst[0xd00])=p3.y;
        *((float*)&sst[0xe00])=p3.z;
        *((float*)&sst[0xf00])=p3.w;
        asld=asld_base+ofs;
        bsld=bsld_base+ofs;     
        __syncthreads();
        b[0]=*((float4*)&bsld[0x00]); 
        a[0]=*((float4*)&asld[0x00]);
        b[1]=*((float4*)&bsld[0x40]); 
        a[1]=*((float4*)&asld[0x80]); 
        BOP8x8(c,&a[2],&b[2])
        ofs^=0x1000;
    }
    b[2]=*((float4*)&bsld[1*256+0x00]); 
    a[2]=*((float4*)&asld[1*256+0x00]);
    b[3]=*((float4*)&bsld[1*256+0x40]); 
    a[3]=*((float4*)&asld[1*256+0x80]); 
    BOP8x8(c,&a[0],&b[0])
    b[0]=*((float4*)&bsld[2*256+0x00]); 
    a[0]=*((float4*)&asld[2*256+0x00]);
    b[1]=*((float4*)&bsld[2*256+0x40]); 
    a[1]=*((float4*)&asld[2*256+0x80]); 
    BOP8x8(c,&a[2],&b[2])
    b[2]=*((float4*)&bsld[3*256+0x00]); 
    a[2]=*((float4*)&asld[3*256+0x00]);
    b[3]=*((float4*)&bsld[3*256+0x40]); 
    a[3]=*((float4*)&asld[3*256+0x80]); 
    BOP8x8(c,&a[0],&b[0])
    b[0]=*((float4*)&bsld[4*256+0x00]); 
    a[0]=*((float4*)&asld[4*256+0x00]);
    b[1]=*((float4*)&bsld[4*256+0x40]); 
    a[1]=*((float4*)&asld[4*256+0x80]); 
    BOP8x8(c,&a[2],&b[2])
    b[2]=*((float4*)&bsld[5*256+0x00]); 
    a[2]=*((float4*)&asld[5*256+0x00]);
    b[3]=*((float4*)&bsld[5*256+0x40]); 
    a[3]=*((float4*)&asld[5*256+0x80]); 
    BOP8x8(c,&a[0],&b[0])
    b[0]=*((float4*)&bsld[6*256+0x00]); 
    a[0]=*((float4*)&asld[6*256+0x00]);
    b[1]=*((float4*)&bsld[6*256+0x40]); 
    a[1]=*((float4*)&asld[6*256+0x80]); 
    BOP8x8(c,&a[2],&b[2])
    b[2]=*((float4*)&bsld[7*256+0x00]); 
    a[2]=*((float4*)&asld[7*256+0x00]);
    b[3]=*((float4*)&bsld[7*256+0x40]); 
    a[3]=*((float4*)&asld[7*256+0x80]);
    BOP8x8(c,&a[0],&b[0])
    BOP8x8(c,&a[2],&b[2])
    sgemm_epilog64x32( d_c, NULL, &smem[slot<<10], c, lane, ldc, x, anr>>1, cnc-y, alpha );
}