__global__ void dk_sflatcgevv_16x32( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int pnc, int qnc )
{
    __shared__ float2 smem[256+512];
    float2 p[6], q[6], a[8], b[8], c[64]={{0.f,0.f}};
    unsigned int bx=blockIdx.x; 
    unsigned int n_tiles=(pnc+15)>>4;
    unsigned int x=blockIdx.y;
    unsigned int y=bx%n_tiles;
    unsigned int z=bx/n_tiles;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int u=slot&1;
    unsigned int v=slot>>1;
    unsigned int ox=(x<<4)|lane;    
    unsigned int oy=(y<<4)|u;
    unsigned int oz=(z<<5)|v;
    unsigned int n=bat*slice_size;
    unsigned int m=pnc*slice_size;
    unsigned int ai=(y<<4)|slot;
    unsigned int bi=(z<<5)|slot;
    unsigned int ia0=n*((ai+0* 8)<pnc?(ai+0* 8):(pnc-1));
    unsigned int ia1=n*((ai+1* 8)<pnc?(ai+1* 8):(pnc-1));
    unsigned int ib0=n*((bi+0*16)<qnc?(bi+0*16):(qnc-1));
    unsigned int ib1=n*((bi+1*16)<qnc?(bi+1*16):(qnc-1));
    unsigned int ib2=n*((bi+2*16)<qnc?(bi+2*16):(qnc-1));
    unsigned int ib3=n*((bi+3*16)<qnc?(bi+3*16):(qnc-1));
    d_a+=ox;
    d_b+=ox;
    d_c+=oz*m+oy*slice_size+ox;
    p[0]=d_a[ia0];
    p[1]=d_a[ia1];
    p[2]=d_b[ib0];
    p[3]=d_b[ib1];
    p[4]=d_b[ib2];
    p[5]=d_b[ib3];
    float2* sst=&smem[tid];
    float2* asld=&smem[u*16+lane];
    float2* bsld=&smem[v*16+lane];
    for( int k=bat-1; k>0; --k )
    {
    #pragma unroll
        for( int i=0; i<6; ++i ){ sst[i*128]=p[i]; }
        d_a+=slice_size; 
        d_b+=slice_size;
        q[0]=d_a[ia0];
        q[1]=d_a[ia1];
        q[2]=d_b[ib0];
        q[3]=d_b[ib1];
        q[4]=d_b[ib2];
        q[5]=d_b[ib3];
        __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){ a[i]=asld[i*32]; b[i]=bsld[256+i*64]; }
    #pragma unroll
        for( int i=0; i<8; ++i ){
            CFMA(c[i*8+0],a[0],b[i])
            CFMA(c[i*8+1],a[1],b[i])
            CFMA(c[i*8+2],a[2],b[i])
            CFMA(c[i*8+3],a[3],b[i])
            CFMA(c[i*8+4],a[4],b[i])
            CFMA(c[i*8+5],a[5],b[i])
            CFMA(c[i*8+6],a[6],b[i])
            CFMA(c[i*8+7],a[7],b[i])
        } __syncthreads();
    #pragma unroll
        for( int i=0; i<6; ++i ){ p[i]=q[i]; }
    }
#pragma unroll
    for( int i=0; i<6; ++i ){ sst[i*128]=p[i]; }
    __syncthreads();
#pragma unroll
    for( int i=0; i<8; ++i ){ a[i]=asld[i*32]; b[i]=bsld[256+i*64]; }
#pragma unroll
    for( int i=0; i<8; ++i ){
        CFMA(c[i*8+0],a[0],b[i])
        CFMA(c[i*8+1],a[1],b[i])
        CFMA(c[i*8+2],a[2],b[i])
        CFMA(c[i*8+3],a[3],b[i])
        CFMA(c[i*8+4],a[4],b[i])
        CFMA(c[i*8+5],a[5],b[i])
        CFMA(c[i*8+6],a[6],b[i])
        CFMA(c[i*8+7],a[7],b[i])
    }
#pragma unroll
    for( int i=0; i<64; ++i ){ 
        c[i].x*=scale;
        c[i].y*=scale;
    }
    pnc-=oy;
    qnc-=oz;
#pragma unroll
    for( int i=0; i<8; ++i )
    {
        float2* temp=d_c;
        if(i*4<qnc)
        {
            if(0*2<pnc){ *temp=c[i*8+0]; temp+=2*slice_size; }
            if(1*2<pnc){ *temp=c[i*8+1]; temp+=2*slice_size; }
            if(2*2<pnc){ *temp=c[i*8+2]; temp+=2*slice_size; }
            if(3*2<pnc){ *temp=c[i*8+3]; temp+=2*slice_size; }
            if(4*2<pnc){ *temp=c[i*8+4]; temp+=2*slice_size; }
            if(5*2<pnc){ *temp=c[i*8+5]; temp+=2*slice_size; }
            if(6*2<pnc){ *temp=c[i*8+6]; temp+=2*slice_size; }
            if(7*2<pnc){ *temp=c[i*8+7]; temp+=2*slice_size; }
        }
        d_c+=4*m;
    }
}