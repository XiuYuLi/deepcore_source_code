
__global__ void dk_xfft128x128_r2c( float2* d_c, const __half2* __restrict__ d_r, const float2* __restrict__ d_RF, int nx, int ny, unsigned int ldr )
{
    __shared__ float smem[128*72];              
    float2 c[16], RF[15], temp;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&63;
    unsigned int y=tid>>6;
    unsigned int u=tid&7;                       
    unsigned int v=tid>>3;          
    float* spx=&smem[ 72*y+x];                  
    float* spy=&smem[576*y+x];                  
    float* spu=&smem[ 72*v+u];              
    float* spv=&smem[576*y+65*u+((v&7)<<3)];
    d_c+=(by*gridDim.x+bx)*65*128;
    d_r+=by*(ldr>>1)+(bx<<13)+tid;          
    RF[0]=d_RF[y];
#pragma unroll
    for( int i=0; i<16; ++i ){ c[i]=__half22float2(d_r[i*512]); }                   
    CALRF16(RF)                         
    FFT16(c,)                               
    MRF16(c,RF)                             
    RF[0]=d_RF[u<<1];                       
    CALRF8(RF)                              
    PERMUTE_S16_L8x2(spx,spy,c,576,4608,72,0xf)
    FFT8(&c[0],)                            
    FFT8(&c[8],)                            
    PERMUTE8x2(spx,spu,c,576,1152,4608,8,0xf)   
    FFT8(&c[0],)                            
    FFT8(&c[8],)                            
    MRF8(&c[0],RF)
    MRF8(&c[8],RF)                          
    PERMUTE8x2(spy,spv,c,4608,65,4608,1,0xf)    
    FFT8(&c[0],)                            
    FFT8(&c[8],)                            
    PERMUTE_S8x2_L16(spu,spx,c,4608,8,576,0xf)
    s_postproc_128x128_a( d_c, c, smem, d_RF, tid, x, y );
}
__global__ void dk_xfft128x128_r2c_pad( float2* d_c, const __half* __restrict__ d_r, 
    const float2* __restrict__ d_RF, int nx, int ny, unsigned int ldr, int pad_x, int pad_y )
{
    __shared__ float smem[128*72];
    float2 c[16]={{0.f,0.f}}, RF[15], temp;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&63;
    unsigned int y=tid>>6;
    unsigned int u=tid&7;
    unsigned int v=tid>>3;
    int p=(x<<1)-pad_x;
    int q=y-pad_y;
    float* spx=&smem[ 72*y+x];
    float* spy=&smem[576*y+x];
    float* spu=&smem[ 72*v+u];
    float* spv=&smem[576*y+65*u+((v&7)<<3)];    
    d_c+=(by*gridDim.x+bx)*65*128;
    d_r+=by*ldr+bx*ny*nx+p;
#pragma unroll
    for( int k=0; k<16; ++k, q+=8 ){
        if((q>=0)&(q<ny)){
            if(((p+0)>=0)&((p+0)<nx)){ c[k].x=__half2float(d_r[q*nx+0]); }
            if(((p+1)>=0)&((p+1)<nx)){ c[k].y=__half2float(d_r[q*nx+1]); }
        }
    }   
    RF[0]=d_RF[y];
    CALRF16(RF)
    FFT16(c,)
    MRF16(c,RF)
    RF[0]=d_RF[u<<1];
    PERMUTE_S16_L8x2(spx,spy,c,576,4608,72,0xf)
    CALRF8(RF)
    FFT8(&c[0],)
    FFT8(&c[8],)
    PERMUTE8x2(spx,spu,c,576,1152,4608,8,0xf)
    FFT8(&c[0],)
    FFT8(&c[8],)
    MRF8(&c[0],RF)
    MRF8(&c[8],RF)
    PERMUTE8x2(spy,spv,c,4608,65,4608,1,0xf)
    FFT8(&c[0],)
    FFT8(&c[8],)
    PERMUTE_S8x2_L16(spu,spx,c,4608,8,576,0xf)
    s_postproc_128x128_a( d_c, c, smem, d_RF, tid, x, y );
}
__global__ void 
#if SM>=50
__launch_bounds__(512,2)
#endif
dk_xfft128x128_r2c_ext( float2* d_c, const __half* __restrict__ d_r, const float2* __restrict__ d_RF, int nx, int ny, unsigned int ldr )
{
    __shared__ float smem[128*72];
    float2 c[16]={{0.f,0.f}}, RF[15], temp;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&63;
    unsigned int y=tid>>6;
    unsigned int u=tid&7;
    unsigned int v=tid>>3;
    float* spx=&smem[ 72*y+x];
    float* spy=&smem[576*y+x];
    float* spu=&smem[ 72*v+u];
    float* spv=&smem[576*y+65*u+((v&7)<<3)];    
    d_c+=(by*gridDim.x+bx)*65*128;
    d_r+=by*ldr+(bx*ny+y)*nx+(x<<1);
#pragma unroll
    for( int k=0; k<16; ++k ){ 
        if(((2*x+0)<nx)&((k*8+y)<ny)){ c[k].x=__half2float(d_r[k*8*nx+0]); }
        if(((2*x+1)<nx)&((k*8+y)<ny)){ c[k].y=__half2float(d_r[k*8*nx+1]); }
    }   
    RF[0]=d_RF[y];
    CALRF16(RF)
    FFT16(c,)
    MRF16(c,RF)
    RF[0]=d_RF[u<<1];
    PERMUTE_S16_L8x2(spx,spy,c,576,4608,72,0xf)
    CALRF8(RF)
    FFT8(&c[0],)
    FFT8(&c[8],)
    PERMUTE8x2(spx,spu,c,576,1152,4608,8,0xf)
    FFT8(&c[0],)
    FFT8(&c[8],)
    MRF8(&c[0],RF)
    MRF8(&c[8],RF)
    PERMUTE8x2(spy,spv,c,4608,65,4608,1,0xf)
    FFT8(&c[0],)
    FFT8(&c[8],)
    PERMUTE_S8x2_L16(spu,spx,c,4608,8,576,0xf)
    s_postproc_128x128_a( d_c, c, smem, d_RF, tid, x, y );
}
__global__ void 
#if SM>=50
__launch_bounds__(512,2)
#endif
dk_xfft128x128_r2c_flip( float2* d_c, const __half* __restrict__ d_r, const float2* __restrict__ d_RF, int nx, int ny, unsigned int ldr )
{
    __shared__ float smem[128*72];
    float2 c[16]={{0.f,0.f}}, RF[15], temp;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&63;
    unsigned int y=tid>>6;
    unsigned int u=tid&7;
    unsigned int v=tid>>3;
    int dx=nx-(x<<1)-1;
    float* spx=&smem[ 72*y+x];
    float* spy=&smem[576*y+x];
    float* spu=&smem[ 72*v+u];
    float* spv=&smem[576*y+65*u+((v&7)<<3)];    
    d_c+=(by*gridDim.x+bx)*65*128;
    d_r+=by*ldr+bx*ny*nx;
    int q=ny-y-1;
#pragma unroll
    for( int k=0; k<16; ++k ){ 
        int p=k*8+y;
        if(p<ny){
            if((dx+0)<nx){ c[k].x=__half2float(d_r[q*nx+dx  ]); }
            if((dx+1)<nx){ c[k].y=__half2float(d_r[q*nx+dx-1]); }
            q=ny-(p+8);
        }
    }   
    RF[0]=d_RF[y];
    CALRF16(RF)
    FFT16(c,)
    MRF16(c,RF)
    RF[0]=d_RF[u<<1];
    PERMUTE_S16_L8x2(spx,spy,c,576,4608,72,0xf)
    CALRF8(RF)
    FFT8(&c[0],)
    FFT8(&c[8],)
    PERMUTE8x2(spx,spu,c,576,1152,4608,8,0xf)
    FFT8(&c[0],)
    FFT8(&c[8],)
    MRF8(&c[0],RF)
    MRF8(&c[8],RF)
    PERMUTE8x2(spy,spv,c,4608,65,4608,1,0xf)
    FFT8(&c[0],)
    FFT8(&c[8],)
    PERMUTE_S8x2_L16(spu,spx,c,4608,8,576,0xf)
    s_postproc_128x128_a( d_c, c, smem, d_RF, tid, x, y );
}
