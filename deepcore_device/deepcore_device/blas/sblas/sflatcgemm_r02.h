__global__ void dk_sflatcgemm_2x32( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[32];
    float2 a[2], b[16], p, q, c[32]={{0.f,0.f}};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int n_tiles=slice_size>>4;
    unsigned int x=bid%n_tiles;
    unsigned int y=bid/n_tiles;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int l=slice_size<<1;
    unsigned int ox=(x<<4)|lane;
    unsigned int oy=(y<<5)|(slot<<4);
    d_c+=((y<<6)|(slot<<5))*slice_size+ox;
    d_a+=slot*slice_size+ox;
    d_b+=ox;
    p=d_a[0];
    float2* sst=&smem[tid];
    float2* sld=&smem[lane];
    for( int s=inc-1; s>=0; --s )
    {       
        *sst=p;
        if(s>0){ q=*(d_a+=l); }
        __syncthreads();
    #pragma unroll
        for( int i=0; i<16; ++i ){ b[i]=*(d_b+((oy+i)<onc?(oy+i):(onc-1))*n); }
        d_b+=m;
        a[0]=sld[0*16];
        a[1]=sld[1*16];
    #pragma unroll
        for( int i=0; i<16; ++i ){
            CFMA(c[i*2+0],a[0],b[i]);
            CFMA(c[i*2+1],a[1],b[i]);
        } __syncthreads();
        p=q;
    }
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    onc-=oy;
#pragma unroll
    for( int i=0; i<16; ++i ){
        if(i<onc){
            *d_c=c[2*i+0]; d_c+=slice_size;
            *d_c=c[2*i+1]; d_c+=slice_size;
        }
    }
}
__global__ void dk_sflatcgemm_2x128( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[128];
    float2 a[2], b[16], c[32]={{0.f,0.f}};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int n_tiles=slice_size>>4;
    unsigned int x=bid%n_tiles;
    unsigned int y=bid/n_tiles;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int l=slice_size<<3;
    unsigned int ox=(x<<4)|lane;
    unsigned int oy=(y<<7)|(slot<<4);
    d_c+=((y<<8)|(slot<<5))*slice_size+ox;
    d_a+=slot*slice_size+ox;
    d_b+=ox;
    float2 p=d_a[0];
    float2* sst=&smem[tid];
    float2* sld=&smem[lane];
    for( int s=inc-4; s>=0; s-=4 )
    {       
        *sst=p;
        __syncthreads();
        if(s>0){ p=*(d_a+=l); }
    #pragma unroll
        for( int k=0; k<4; ++k )
        {
        #pragma unroll
            for( int i=0; i<16; ++i ){ b[i]=*(d_b+((oy+i)<onc?(oy+i):(onc-1))*n); }
            d_b+=m;
            a[0]=sld[k*32+0*16];
            a[1]=sld[k*32+1*16];
        #pragma unroll
            for( int i=0; i<16; ++i ){
                CFMA(c[i*2+0],a[0],b[i]);
                CFMA(c[i*2+1],a[1],b[i]);
            }
        } __syncthreads();
    }
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    onc-=oy;
#pragma unroll
    for( int i=0; i<16; ++i ){
        if(i<onc){
            *d_c=c[2*i+0]; d_c+=slice_size;
            *d_c=c[2*i+1]; d_c+=slice_size;
        }
    }
}
__global__ void dk_sflatcgemm_2x256( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[256];
    float2 a[2], b[16], p, q, c[32]={{0.f,0.f}};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int n_tiles=slice_size>>4;
    unsigned int x=bid%n_tiles;
    unsigned int y=bid/n_tiles;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int l=slice_size<<4;
    unsigned int ox=(x<<4)|lane;
    unsigned int oy=(y<<8)|(slot<<4);
    d_c+=((y<<9)|(slot<<5))*slice_size+ox;
    d_a+=slot*slice_size+ox;
    d_b+=ox;
    p=d_a[0];
    float2* sst=&smem[tid];
    float2* sld=&smem[lane];
    for( int s=inc-8; s>=0; s-=8 )
    {       
        *sst=p;
        if(s>0){ q=*(d_a+=l); }
        __syncthreads();
    #pragma unroll
        for( int k=0; k<8; ++k )
        {
        #pragma unroll
            for( int i=0; i<16; ++i ){ b[i]=*(d_b+((oy+i)<onc?(oy+i):(onc-1))*n); }
            d_b+=m;
            a[0]=sld[k*32+0*16];
            a[1]=sld[k*32+1*16];
        #pragma unroll
            for( int i=0; i<16; ++i ){
                CFMA(c[i*2+0],a[0],b[i]);
                CFMA(c[i*2+1],a[1],b[i]);
            }
        } __syncthreads();
        p=q;
    }
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    onc-=oy;
#pragma unroll
    for( int i=0; i<16; ++i ){
        if(i<onc){
            *d_c=c[2*i+0]; d_c+=slice_size;
            *d_c=c[2*i+1]; d_c+=slice_size;
        }
    }
}