__global__ void dk_sflatcgemm_1x32( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[32];
    float2 b[16], c[16]={{0.f,0.f}};
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
    d_a+=slot*slice_size+ox;
    d_b+=ox;    
    d_c+=oy*slice_size+ox;
    float2 p=*d_a;
    float2* sst=&smem[tid];
    float2* sld=&smem[lane];
    for( int s=inc-2; s>=0; s-=2 )
    {       
        *sst=p;
        __syncthreads();
        if(s>0){ p=*(d_a+=l); }
    #pragma unroll
        for( int k=0; k<2; ++k ){
        #pragma unroll
            for( int i=0; i<16; ++i ){ b[i]=*(d_b+((oy+i)<onc?(oy+i):(onc-1))*n); }
            d_b+=m;
            float2 a=sld[k*16];
        #pragma unroll
            for( int i=0; i<16; ++i ){ CFMA(c[i],a,b[i]) }
        } __syncthreads();
    }
#pragma unroll
    for( int i=0; i<16; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    onc-=oy;
#pragma unroll
    for( int i=0; i<16; ++i ){
        if(i<onc){ *d_c=c[i]; } d_c+=slice_size;
    }
}
__global__ void dk_sflatcgemm_1x64( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[64];
    float2 b[16], c[16]={{0.f,0.f}};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int n_tiles=slice_size>>4;
    unsigned int x=bid%n_tiles;
    unsigned int y=bid/n_tiles;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int l=slice_size<<2;
    unsigned int ox=(x<<4)|lane;
    unsigned int oy=(y<<6)|(slot<<4);
    d_a+=slot*slice_size+ox;
    d_b+=ox;    
    d_c+=oy*slice_size+ox;
    float2 p=*d_a;
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
            float2 a=sld[k*16];
        #pragma unroll
            for( int i=0; i<16; ++i ){ CFMA(c[i],a,b[i]) }
        } __syncthreads();
    }
#pragma unroll
    for( int i=0; i<16; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    onc-=oy;
#pragma unroll
    for( int i=0; i<16; ++i ){
        if(i<onc){ *d_c=c[i]; } d_c+=slice_size;
    }
}
__global__ void dk_sflatcgemm_1x128( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[128];
    float2 b[16], c[16]={{0.f,0.f}};
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
    d_a+=slot*slice_size+ox;
    d_b+=ox;    
    d_c+=oy*slice_size+ox;
    float2 p=d_a[0];
    float2* sst=&smem[tid];
    float2* sld=&smem[lane];
    for( int s=inc-8; s>=0; s-=8 )
    {       
        *sst=p;
        __syncthreads();
        if(s>0){ p=*(d_a+=l); }
        for( int k=0; k<8; ++k ){
        #pragma unroll
            for( int i=0; i<16; ++i ){ b[i]=*(d_b+((oy+i)<onc?(oy+i):(onc-1))*n); }
            d_b+=m;
            float2 a=sld[k*16];
        #pragma unroll
            for( int i=0; i<16; ++i ){ CFMA(c[i],a,b[i]) }
        } __syncthreads();
    }
#pragma unroll
    for( int i=0; i<16; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    onc-=oy;
#pragma unroll
    for( int i=0; i<16; ++i ){
        if(i<onc){ *d_c=c[i]; } d_c+=slice_size;
    }
}