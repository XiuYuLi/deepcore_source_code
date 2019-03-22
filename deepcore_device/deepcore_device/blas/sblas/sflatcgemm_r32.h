__global__ void dk_sflatcgemm_32x32( 
          float2*              d_c, 
    const float2* __restrict__ d_a, 
    const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, 
    int inc, int onc, int n, int m )
{
    __shared__ float2 smem[2048];
    float2 p[8], q[8], a[8], b[8], c[64]={{0.f,0.f}};
    unsigned int bx=blockIdx.x; 
    unsigned int n_tiles=(bat+31)>>5;
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
    unsigned int l=bat*slice_size;
    unsigned int ai=__imad(y,32,slot);
    unsigned int bi=__imad(z,32,slot);
    unsigned int la=((ai   )<bat?(ai   ):(bat-1))*slice_size;
    unsigned int ha=((ai+16)<bat?(ai+16):(bat-1))*slice_size;
    unsigned int lb=((bi   )<onc?(bi   ):(onc-1))*n;
    unsigned int hb=((bi+16)<onc?(bi+16):(onc-1))*n;
    d_c+=oz*l+oy*slice_size+ox;
    d_a+=ox;
    d_b+=ox;
    p[0]=d_a[la];
    p[1]=d_a[ha];
    p[2]=d_b[lb]; 
    p[3]=d_b[hb];
    d_a+=l; d_b+=m;
    p[4]=d_a[la];
    p[5]=d_a[ha]; 
    p[6]=d_b[lb]; 
    p[7]=d_b[hb];
    float2* sst=&smem[tid];
    float2* asld=&smem[u*16+lane];
    float2* bsld=&smem[v*16+lane];
    for( int s=inc-2; s>0; s-=2 )
    {       
    #pragma unroll
        for( int i=0; i<8; ++i ){ sst[i*256]=p[i]; }
        d_a+=l; d_b+=m;
        q[0]=d_a[la];
        q[1]=d_a[ha]; 
        q[2]=d_b[lb]; 
        q[3]=d_b[hb];
        d_a+=l; d_b+=m;
        q[4]=d_a[la];
        q[5]=d_a[ha]; 
        q[6]=d_b[lb]; 
        q[7]=d_b[hb];
        __syncthreads();
    #pragma unroll
        for( int k=0; k<2; ++k )
        {
        #pragma unroll
            for( int i=0; i<8; ++i ){
                a[i]=asld[k*1024    +i*64];
                b[i]=bsld[k*1024+512+i*64];
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
        } __syncthreads();
    #pragma unroll
        for( int i=0; i<8; ++i ){ p[i]=q[i]; }
    }
#pragma unroll
    for( int i=0; i<8; ++i ){ sst[i*256]=p[i]; }
    __syncthreads();
#pragma unroll
    for( int k=0; k<2; ++k )
    {
    #pragma unroll
        for( int i=0; i<8; ++i ){
            a[i]=asld[k*1024    +i*64];
            b[i]=bsld[k*1024+512+i*64];
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
        if(onc>i*4)
        {
            if(bat>0*4){ *temp=c[8*i+0]; } temp+=4*slice_size;
            if(bat>1*4){ *temp=c[8*i+1]; } temp+=4*slice_size;
            if(bat>2*4){ *temp=c[8*i+2]; } temp+=4*slice_size;
            if(bat>3*4){ *temp=c[8*i+3]; } temp+=4*slice_size;
            if(bat>4*4){ *temp=c[8*i+4]; } temp+=4*slice_size;
            if(bat>5*4){ *temp=c[8*i+5]; } temp+=4*slice_size;
            if(bat>6*4){ *temp=c[8*i+6]; } temp+=4*slice_size;
            if(bat>7*4){ *temp=c[8*i+7]; }
        }
    }
}