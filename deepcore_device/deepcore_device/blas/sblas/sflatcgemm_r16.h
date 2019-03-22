__global__ void dk_sflatcgemm_16x32( float2* d_c, 
    const float2* __restrict__ d_a, 
    const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, 
    int inc, int onc, int n, int m )
{
    __shared__ float2 smem[768];
    float2 p[12], a[8], b[8], c[64]={{0.f,0.f}};
    unsigned int bx=blockIdx.x;
    unsigned int n_tiles=(bat+15)>>4;
    unsigned int x=blockIdx.y;
    unsigned int y=bx%n_tiles;
    unsigned int z=bx/n_tiles;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int u=slot&1;
    unsigned int v=slot>>1;
    unsigned int ox=__imad(x,16,lane);
    unsigned int oy=__imad(y,16,u);
    unsigned int oz=__imad(z,32,v);
    unsigned int l=bat*slice_size;
    unsigned int ai=__imad(y,16,slot);
    unsigned int bi=__imad(z,32,slot);
    unsigned int ia0=((ai+0*8)<bat?(ai+0*8):(bat-1))*slice_size;
    unsigned int ia1=((ai+1*8)<bat?(ai+1*8):(bat-1))*slice_size;
    unsigned int ib0=((bi+0*8)<onc?(bi+0*8):(onc-1))*n;
    unsigned int ib1=((bi+1*8)<onc?(bi+1*8):(onc-1))*n;
    unsigned int ib2=((bi+2*8)<onc?(bi+2*8):(onc-1))*n;
    unsigned int ib3=((bi+3*8)<onc?(bi+3*8):(onc-1))*n;
    d_c+=oz*l+oy*slice_size+ox;
    d_a+=ox;
    d_b+=ox;
    p[0]=d_a[ia0];
    p[1]=d_a[ia1];
    p[2]=d_b[ib0]; 
    p[3]=d_b[ib1]; 
    p[4]=d_b[ib2]; 
    p[5]=d_b[ib3];
    float2* sst=&smem[tid];
    float2* asld=&smem[u*16+lane];
    float2* bsld=&smem[v*16+lane];
    for( int s=inc-2; s>0; s-=2 )
    {       
    #pragma unroll
        for( int k=0; k<2; ++k )
        {
        #pragma unroll
            for( int i=0; i<6; ++i ){ sst[i*128]=p[k*6+i]; }
            d_a+=l; d_b+=m;
            p[((k+1)%2)*6+0]=d_a[ia0];
            p[((k+1)%2)*6+1]=d_a[ia1];
            p[((k+1)%2)*6+2]=d_b[ib0]; 
            p[((k+1)%2)*6+3]=d_b[ib1]; 
            p[((k+1)%2)*6+4]=d_b[ib2]; 
            p[((k+1)%2)*6+5]=d_b[ib3];
            __syncthreads();
        #pragma unroll
            for( int i=0; i<8; ++i ){
                a[i]=asld[i*32];
                b[i]=bsld[i*64+256];
            }
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
        }
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<6; ++i ){ sst[i*128]=p[k*6+i]; }
        if(k==0)
        {
            d_a+=l; d_b+=m;
            p[6+0]=d_a[ia0];
            p[6+1]=d_a[ia1];
            p[6+2]=d_b[ib0]; 
            p[6+3]=d_b[ib1]; 
            p[6+4]=d_b[ib2]; 
            p[6+5]=d_b[ib3];
        } __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){
            a[i]=asld[i*32];
            b[i]=bsld[i*64+256];
        }
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
        float2* temp=d_c+i*4*l;
        if(i*4<onc)
        {
            if(0*2<bat){ *temp=c[8*i+0]; } temp+=2*slice_size;
            if(1*2<bat){ *temp=c[8*i+1]; } temp+=2*slice_size;
            if(2*2<bat){ *temp=c[8*i+2]; } temp+=2*slice_size;
            if(3*2<bat){ *temp=c[8*i+3]; } temp+=2*slice_size;
            if(4*2<bat){ *temp=c[8*i+4]; } temp+=2*slice_size;
            if(5*2<bat){ *temp=c[8*i+5]; } temp+=2*slice_size;
            if(6*2<bat){ *temp=c[8*i+6]; } temp+=2*slice_size;
            if(7*2<bat){ *temp=c[8*i+7]; }
        }
    }
}
__global__ void dk_sflatcgemm_16x64( float2* d_c, 
    const float2* __restrict__ d_a, 
    const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, 
    int inc, int onc, int n, int m )
{
    __shared__ float2 smem[1280];
    float2 p[10], a[8], b[8], c[64]={{0.f,0.f}};
    unsigned int bx=blockIdx.x;
    unsigned int n_tiles=(bat+15)>>4;
    unsigned int x=blockIdx.y;
    unsigned int y=bx%n_tiles;
    unsigned int z=bx/n_tiles;
    unsigned int tid=threadIdx.x;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int u=slot&1;
    unsigned int v=slot>>1;
    unsigned int ox=__imad(x,16,lane);
    unsigned int oy=__imad(y,16,u);
    unsigned int oz=__imad(z,64,v);
    unsigned int l=bat*slice_size;
    unsigned int ai=__imad(y,16,slot);
    unsigned int bi=__imad(z,64,slot);
    unsigned int pa=ai<bat?ai:(bat-1);
    unsigned int ib0=((bi+0*16)<onc?(bi+0*16):(onc-1))*n;
    unsigned int ib1=((bi+1*16)<onc?(bi+1*16):(onc-1))*n;
    unsigned int ib2=((bi+2*16)<onc?(bi+2*16):(onc-1))*n;
    unsigned int ib3=((bi+3*16)<onc?(bi+3*16):(onc-1))*n;
    d_c+=oz*l+oy*slice_size+ox;
    d_a+=pa*slice_size+ox;
    d_b+=ox;
    p[0]=d_a[  0];
    p[1]=d_b[ib0]; 
    p[2]=d_b[ib1]; 
    p[3]=d_b[ib2]; 
    p[4]=d_b[ib3];
    float2* sst=&smem[tid];
    float2* asld=&smem[__imad(u,16,lane)];
    float2* bsld=&smem[__imad(v,16,lane)];
    for( int s=inc-2; s>0; s-=2 )
    {       
    #pragma unroll
        for( int k=0; k<2; ++k )
        {
        #pragma unroll
            for( int i=0; i<5; ++i ){ sst[i*256]=p[k*5+i]; }
            d_a+=l; d_b+=m;
            p[((k+1)%2)*5+0]=d_a[  0];
            p[((k+1)%2)*5+1]=d_b[ib0]; 
            p[((k+1)%2)*5+2]=d_b[ib1]; 
            p[((k+1)%2)*5+3]=d_b[ib2]; 
            p[((k+1)%2)*5+4]=d_b[ib3];
            __syncthreads();
        #pragma unroll
            for( int i=0; i<8; ++i ){
                a[i]=asld[i* 32];
                b[i]=bsld[i*128+256];
            }
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
        } 
    }
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<5; ++i ){ sst[i*256]=p[k*5+i]; }
        if(k==0)
        {
            d_a+=l; d_b+=m;
            p[5]=d_a[  0];
            p[6]=d_b[ib0]; 
            p[7]=d_b[ib1]; 
            p[8]=d_b[ib2]; 
            p[9]=d_b[ib3];
        } __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){
            a[i]=asld[i* 32];
            b[i]=bsld[i*128+256];
        }
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
        if(onc>i*8)
        {
            if(bat>0*2){ *temp=c[8*i+0]; } temp+=2*slice_size;
            if(bat>1*2){ *temp=c[8*i+1]; } temp+=2*slice_size;
            if(bat>2*2){ *temp=c[8*i+2]; } temp+=2*slice_size;
            if(bat>3*2){ *temp=c[8*i+3]; } temp+=2*slice_size;
            if(bat>4*2){ *temp=c[8*i+4]; } temp+=2*slice_size;
            if(bat>5*2){ *temp=c[8*i+5]; } temp+=2*slice_size;
            if(bat>6*2){ *temp=c[8*i+6]; } temp+=2*slice_size;
            if(bat>7*2){ *temp=c[8*i+7]; }
        }
    }
}