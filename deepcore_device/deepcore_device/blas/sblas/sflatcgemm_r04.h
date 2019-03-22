__global__ void dk_sflatcgemm_4x32( float2* d_c, 
    const float2* __restrict__ d_a, const float2* __restrict__ d_b, 
    float scale, int slice_size, int bat, int inc, int onc, int n, int m )
{
    __shared__ float2 smem[64];
    float2 p[2], a[4], b[16], c[32]={{0.f,0.f}};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int n_tiles=slice_size>>4;
    unsigned int x=bid%n_tiles;
    unsigned int y=bid/n_tiles;
    unsigned int lane=tid&15;
    unsigned int slot=tid>>4;
    unsigned int l=slice_size<<2;
    unsigned int ox=(x<<4)|lane;
    unsigned int oy=(y<<5)|(slot<<3);
    d_c+=((y<<7)|(slot<<5))*slice_size+ox;
    d_a+=slot*slice_size+ox;
    d_b+=(oy<onc?oy:(onc-8))*n+ox;
    p[0]=d_a[0];
    const float2* temp=d_b;
    b[0]=temp[0]; temp+=n;
    b[1]=temp[0]; temp+=n;
    b[2]=temp[0]; temp+=n;
    b[3]=temp[0]; temp+=n;
    b[4]=temp[0]; temp+=n;
    b[5]=temp[0]; temp+=n;
    b[6]=temp[0]; temp+=n;
    b[7]=temp[0];
    float2* sst=&smem[tid];
    float2* sld=&smem[lane];
    for( int s=inc-2; s>0; s-=2 )
    {       
    #pragma unroll
        for( int k=0; k<2; ++k )
        {
            *sst=p[k];
            p[(k+1)%2]=*(d_a+=l);
            temp=(d_b+=m);
            b[((k+1)%2)*8+0]=temp[0]; temp+=n;
            b[((k+1)%2)*8+1]=temp[0]; temp+=n;
            b[((k+1)%2)*8+2]=temp[0]; temp+=n;
            b[((k+1)%2)*8+3]=temp[0]; temp+=n;
            b[((k+1)%2)*8+4]=temp[0]; temp+=n;
            b[((k+1)%2)*8+5]=temp[0]; temp+=n;
            b[((k+1)%2)*8+6]=temp[0]; temp+=n;
            b[((k+1)%2)*8+7]=temp[0];
            __syncthreads();
            a[0]=sld[0*16];
            a[1]=sld[1*16];
            a[2]=sld[2*16];
            a[3]=sld[3*16];
        #pragma unroll
            for( int i=0; i<8; ++i ){
                CFMA(c[i*4+0],a[0],b[k*8+i]);
                CFMA(c[i*4+1],a[1],b[k*8+i]);
                CFMA(c[i*4+2],a[2],b[k*8+i]);
                CFMA(c[i*4+3],a[3],b[k*8+i]);
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
            temp=d_b+=m;
            b[8+0]=temp[0]; temp+=n;
            b[8+1]=temp[0]; temp+=n;
            b[8+2]=temp[0]; temp+=n;
            b[8+3]=temp[0]; temp+=n;
            b[8+4]=temp[0]; temp+=n;
            b[8+5]=temp[0]; temp+=n;
            b[8+6]=temp[0]; temp+=n;
            b[8+7]=temp[0];
        } __syncthreads();
        a[0]=sld[0*16];
        a[1]=sld[1*16];
        a[2]=sld[2*16];
        a[3]=sld[3*16];
    #pragma unroll
        for( int i=0; i<8; ++i ){
            CFMA(c[i*4+0],a[0],b[k*8+i]);
            CFMA(c[i*4+1],a[1],b[k*8+i]);
            CFMA(c[i*4+2],a[2],b[k*8+i]);
            CFMA(c[i*4+3],a[3],b[k*8+i]);
        } 
        if(k==0){ __syncthreads(); }
    }
#pragma unroll
    for( int i=0; i<32; ++i ){
        c[i].x*=scale;
        c[i].y*=scale;
    }
    onc-=oy;
#pragma unroll
    for( int i=0; i<8; ++i )
    {
        if(i<onc)
        {
            *d_c=c[4*i+0]; d_c+=slice_size;
            *d_c=c[4*i+1]; d_c+=slice_size;
            *d_c=c[4*i+2]; d_c+=slice_size;
            *d_c=c[4*i+3]; d_c+=slice_size;
        }
    }
}