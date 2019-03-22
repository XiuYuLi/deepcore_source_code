__global__ void dk_sflatcgemm_8x32( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[128];
    float2 p[2], a[8], b[8], c[32]={{0.f,0.f}};
    unsigned int bx=blockIdx.x;
    unsigned int n_tiles=(bat+7)>>3;
    unsigned int x=blockIdx.y;
    unsigned int y=bx%n_tiles;
    unsigned int z=bx/n_tiles;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int ox=(x<<4)|lane;
    unsigned int oy=y<<3;
    unsigned int oz=(z<<5)|slot;
    unsigned int l=bat*slice_size;
    unsigned int ai=oy|slot;
    unsigned int ib0=n*((oz+0*8)<onc?(oz+0*8):(onc-1));
    unsigned int ib1=n*((oz+1*8)<onc?(oz+1*8):(onc-1));
    unsigned int ib2=n*((oz+2*8)<onc?(oz+2*8):(onc-1));
    unsigned int ib3=n*((oz+3*8)<onc?(oz+3*8):(onc-1));
    d_a+=(ai<bat?ai:(bat-1))*slice_size+ox;
    d_b+=ox;
    d_c+=oz*l+oy*slice_size+ox;
    p[0]=*d_a;
    b[0]=d_b[ib0];
    b[1]=d_b[ib1];
    b[2]=d_b[ib2];
    b[3]=d_b[ib3];
    float2* sst=&smem[tid];
    float2* sld=&smem[lane];
    for( int s=inc-2; s>0; s-=2 )
    {       
    #pragma unroll
        for( int k=0; k<2; ++k )
        {
            *sst=p[k];
            p[(k+1)%2]=*(d_a+=l);
            d_b+=m;
            b[((k+1)%2)*4+0]=d_b[ib0];
            b[((k+1)%2)*4+1]=d_b[ib1];
            b[((k+1)%2)*4+2]=d_b[ib2];
            b[((k+1)%2)*4+3]=d_b[ib3];
            __syncthreads();
        #pragma unroll
            for( int i=0; i<8; ++i ){ a[i]=sld[i*16]; }
        #pragma unroll
            for( int i=0; i<4; ++i ){
                CFMA(c[i*8+0],a[0],b[k*4+i])
                CFMA(c[i*8+1],a[1],b[k*4+i])
                CFMA(c[i*8+2],a[2],b[k*4+i])
                CFMA(c[i*8+3],a[3],b[k*4+i])
                CFMA(c[i*8+4],a[4],b[k*4+i])
                CFMA(c[i*8+5],a[5],b[k*4+i])
                CFMA(c[i*8+6],a[6],b[k*4+i])
                CFMA(c[i*8+7],a[7],b[k*4+i])
            } __syncthreads();
        }
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
        *sst=p[k];
        if(k==0)
        {
            p[1]=*(d_a+=l);
            d_b+=m;
            b[4]=d_b[ib0];
            b[5]=d_b[ib1];
            b[6]=d_b[ib2];
            b[7]=d_b[ib3];
        } __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){ a[i]=sld[i*16]; }
    #pragma unroll
        for( int i=0; i<4; ++i ){
            CFMA(c[i*8+0],a[0],b[k*4+i])
            CFMA(c[i*8+1],a[1],b[k*4+i])
            CFMA(c[i*8+2],a[2],b[k*4+i])
            CFMA(c[i*8+3],a[3],b[k*4+i])
            CFMA(c[i*8+4],a[4],b[k*4+i])
            CFMA(c[i*8+5],a[5],b[k*4+i])
            CFMA(c[i*8+6],a[6],b[k*4+i])
            CFMA(c[i*8+7],a[7],b[k*4+i])
        } 
        if(k==0){ __syncthreads(); }
    }
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    bat-=oy;
    onc-=oz;
#pragma unroll
    for( int i=0; i<4; ++i )
    {
        float2* temp=d_c+i*8*l;
        if(i*8<onc)
        {
            if(bat>0){ *temp=c[8*i+0]; } temp+=slice_size;
            if(bat>1){ *temp=c[8*i+1]; } temp+=slice_size;
            if(bat>2){ *temp=c[8*i+2]; } temp+=slice_size;
            if(bat>3){ *temp=c[8*i+3]; } temp+=slice_size;
            if(bat>4){ *temp=c[8*i+4]; } temp+=slice_size;
            if(bat>5){ *temp=c[8*i+5]; } temp+=slice_size;
            if(bat>6){ *temp=c[8*i+6]; } temp+=slice_size;
            if(bat>7){ *temp=c[8*i+7]; }
        }
    }
}
__global__ void dk_sflatcgemm_8x64( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[128];
    float2 p[2], a[8], b[16], c[64]={{0.f,0.f}};
    unsigned int bx=blockIdx.x;
    unsigned int n_tiles=(bat+7)>>3;
    unsigned int x=blockIdx.y;
    unsigned int y=bx%n_tiles;
    unsigned int z=bx/n_tiles;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int ox=(x<<4)|lane;
    unsigned int oy=y<<3;
    unsigned int oz=(z<<6)|slot;
    unsigned int l=bat*slice_size;
    unsigned int ai=oy|slot;
    unsigned int ib0=n*((oz+0*8)<onc?(oz+0*8):(onc-1));
    unsigned int ib1=n*((oz+1*8)<onc?(oz+1*8):(onc-1));
    unsigned int ib2=n*((oz+2*8)<onc?(oz+2*8):(onc-1));
    unsigned int ib3=n*((oz+3*8)<onc?(oz+3*8):(onc-1));
    unsigned int ib4=n*((oz+4*8)<onc?(oz+4*8):(onc-1));
    unsigned int ib5=n*((oz+5*8)<onc?(oz+5*8):(onc-1));
    unsigned int ib6=n*((oz+6*8)<onc?(oz+6*8):(onc-1));
    unsigned int ib7=n*((oz+7*8)<onc?(oz+7*8):(onc-1));
    d_a+=(ai<bat?ai:(bat-1))*slice_size+ox;
    d_b+=ox;    
    d_c+=oz*l+oy*slice_size+ox;
    p[0]=d_a[0  ];
    b[0]=d_b[ib0];
    b[1]=d_b[ib1];
    b[2]=d_b[ib2];
    b[3]=d_b[ib3];
    b[4]=d_b[ib4];
    b[5]=d_b[ib5];
    b[6]=d_b[ib6];
    b[7]=d_b[ib7];
    float2* sst=&smem[tid];
    float2* sld=&smem[lane];

    for( int s=inc-2; s>0; s-=2 )
    {   
    #pragma unroll
        for( int k=0; k<2; ++k )
        {
            *sst=p[k];
            d_b+=m;
            p[(k+1)%2]=*(d_a+=l);
            b[((k+1)%2)*8+0]=d_b[ib0];
            b[((k+1)%2)*8+1]=d_b[ib1];
            b[((k+1)%2)*8+2]=d_b[ib2];
            b[((k+1)%2)*8+3]=d_b[ib3];
            b[((k+1)%2)*8+4]=d_b[ib4];
            b[((k+1)%2)*8+5]=d_b[ib5];
            b[((k+1)%2)*8+6]=d_b[ib6];
            b[((k+1)%2)*8+7]=d_b[ib7];
            __syncthreads();
        #pragma unroll
            for( int i=0; i<8; ++i ){ a[i]=sld[i*16]; }
        #pragma unroll
            for( int i=0; i<8; ++i ){
                CFMA(c[i*8+0],a[0],b[k*8+i])
                CFMA(c[i*8+1],a[1],b[k*8+i])
                CFMA(c[i*8+2],a[2],b[k*8+i])
                CFMA(c[i*8+3],a[3],b[k*8+i])
                CFMA(c[i*8+4],a[4],b[k*8+i])
                CFMA(c[i*8+5],a[5],b[k*8+i])
                CFMA(c[i*8+6],a[6],b[k*8+i])
                CFMA(c[i*8+7],a[7],b[k*8+i])
            } __syncthreads();
        }
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
        *sst=p[k];
        if(k==0)
        {
            d_b+=m;
            p[1]=*(d_a+=l);
            b[8+0]=d_b[ib0];
            b[8+1]=d_b[ib1];
            b[8+2]=d_b[ib2];
            b[8+3]=d_b[ib3];
            b[8+4]=d_b[ib4];
            b[8+5]=d_b[ib5];
            b[8+6]=d_b[ib6];
            b[8+7]=d_b[ib7];
        } __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){ a[i]=sld[i*16]; }
    #pragma unroll
        for( int i=0; i<8; ++i ){
            CFMA(c[i*8+0],a[0],b[k*8+i])
            CFMA(c[i*8+1],a[1],b[k*8+i])
            CFMA(c[i*8+2],a[2],b[k*8+i])
            CFMA(c[i*8+3],a[3],b[k*8+i])
            CFMA(c[i*8+4],a[4],b[k*8+i])
            CFMA(c[i*8+5],a[5],b[k*8+i])
            CFMA(c[i*8+6],a[6],b[k*8+i])
            CFMA(c[i*8+7],a[7],b[k*8+i])
        } 
        if(k==0){ __syncthreads(); }
    }
#pragma unroll
    for( int i=0; i<64; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    bat-=oy;
    onc-=oz;
#pragma unroll
    for( int i=0; i<8; ++i )
    {
        float2* temp=d_c+i*8*l;
        if(i*8<onc)
        {
            if(bat>0){ *temp=c[8*i+0]; } temp+=slice_size;
            if(bat>1){ *temp=c[8*i+1]; } temp+=slice_size;
            if(bat>2){ *temp=c[8*i+2]; } temp+=slice_size;
            if(bat>3){ *temp=c[8*i+3]; } temp+=slice_size;
            if(bat>4){ *temp=c[8*i+4]; } temp+=slice_size;
            if(bat>5){ *temp=c[8*i+5]; } temp+=slice_size;
            if(bat>6){ *temp=c[8*i+6]; } temp+=slice_size;
            if(bat>7){ *temp=c[8*i+7]; }
        }
    }
}