__global__ void dk_sflatcgevv_32x32( float2* d_c, 
    float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int pnc, int qnc )
{
    __shared__ float2 smem[1024];
    float2 p[4], q[4], a[8], b[8], c[64]={{0.f,0.f}};
    unsigned int bx=blockIdx.x; 
    unsigned int n_tiles=(pnc+31)>>5;
    unsigned int x=blockIdx.y;
    unsigned int y=bx%n_tiles;
    unsigned int z=bx/n_tiles;
    unsigned int tid=threadIdx.x;   
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int u=slot&3;
    unsigned int v=slot>>2;
    unsigned int ox=__imad(x,16,lane);  
    unsigned int oy=__imad(y,32,u);
    unsigned int oz=__imad(z,32,v);
    unsigned int m=pnc*slice_size;
    unsigned int n=bat*slice_size;
    unsigned int ai=__imad(y,32,slot);
    unsigned int bi=__imad(z,32,slot);
    unsigned int al=n*(ai<pnc?ai:(pnc-1));
    unsigned int bl=n*(bi<qnc?bi:(qnc-1));
    unsigned int ah=n*((ai+16)<pnc?(ai+16):(pnc-1));
    unsigned int bh=n*((bi+16)<qnc?(bi+16):(qnc-1));
    d_a+=ox;
    d_b+=ox;
    d_c+=oz*pnc*slice_size+oy*slice_size+ox;
    p[0]=d_a[al];
    p[1]=d_a[ah];
    p[2]=d_b[bl];
    p[3]=d_b[bh];
    float2* sst=&smem[tid];
    float2* asld=&smem[__imad(u,16,lane)];
    float2* bsld=&smem[__imad(v,16,lane)];
    for( int k=bat-1; k>0; --k )
    {
        sst[0*256]=p[0]; 
        sst[1*256]=p[1];
        sst[2*256]=p[2];
        sst[3*256]=p[3];
        d_a+=slice_size; 
        d_b+=slice_size;
        q[0]=d_a[al]; 
        q[1]=d_a[ah];
        q[2]=d_b[bl];
        q[3]=d_b[bh];
        __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){ a[i]=asld[i*64]; b[i]=bsld[512+i*64]; }
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
        p[0]=q[0];
        p[1]=q[1];
        p[2]=q[2];
        p[3]=q[3];
    }
    sst[0*256]=p[0]; 
    sst[1*256]=p[1];
    sst[2*256]=p[2];
    sst[3*256]=p[3];
    __syncthreads();
#pragma unroll
    for( int i=0; i<8; ++i ){ a[i]=asld[i*64]; b[i]=bsld[512+i*64]; }
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
            if(pnc>0*4){ *temp=c[i*8+0]; temp+=4*slice_size; }
            if(pnc>1*4){ *temp=c[i*8+1]; temp+=4*slice_size; }
            if(pnc>2*4){ *temp=c[i*8+2]; temp+=4*slice_size; }
            if(pnc>3*4){ *temp=c[i*8+3]; temp+=4*slice_size; }
            if(pnc>4*4){ *temp=c[i*8+4]; temp+=4*slice_size; }
            if(pnc>5*4){ *temp=c[i*8+5]; temp+=4*slice_size; }
            if(pnc>6*4){ *temp=c[i*8+6]; temp+=4*slice_size; }
            if(pnc>7*4){ *temp=c[i*8+7]; temp+=4*slice_size; }
        }
        d_c+=4*m;
    }
}
