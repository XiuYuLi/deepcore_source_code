__device__ __forceinline__ static void s_postproc_128x128_a( float2* d_dst, float2* c, float* smem, const float2* d_RF, unsigned int i, unsigned int tx, unsigned int ty )
{
    float2 o[16];
    STORE16(&smem[i],c,512,.x)
    __syncthreads();    
    unsigned int x=i&31;
    unsigned int y=i>>5;
    unsigned int u=x?( 64-x):x;
    unsigned int v=y?(128-y):y;
    unsigned int p=y;
    unsigned int q=v;
#pragma unroll
    for( int k=0; k<16; k+=2 ){
        if(!((x==0)&(y>64))){
            o[k+0].x=smem[y*64+x];
            o[k+1].x=smem[v*64+u];
        }
        v=128-(y+=16);
    } __syncthreads();
    STORE16(&smem[i],c,512,.y)
    __syncthreads();
    float2 RF=d_RF[x];
#pragma unroll
    for( int k=0; k<16; k+=2 ){
        if(!((x==0)&(p>64))){
            o[k+0].y=smem[p*64+x];
            o[k+1].y=smem[q*64+u];          
            s_postproc(o[k],o[k+1],RF);
            d_dst[p*65   +x]=o[k+0];
            d_dst[q*65+64-x]=o[k+1];
        }
        q=128-(p+=16);
    }

    if(tx==32){
    #pragma unroll
        for( int k=0; k<16; ++k ){ ((float2*)smem)[4096+k*8+ty]=c[k]; }
    } __syncthreads();
    
    if(i<=64){
        v=i?(128-i):i;
        o[0]=((float2*)smem)[4096+i];
        o[1]=((float2*)smem)[4096+v];
        s_postproc(o[0],o[1],make_float2(0.f,-1.f));
        d_dst[i*65+32]=o[0];
        d_dst[v*65+32]=o[1];
    }
}
__device__ __forceinline__ static void s_postproc_128x128_b( float2* d_dst, float2* c, float* smem, const float2* d_RF, unsigned int i, unsigned int tx, unsigned int ty )
{
    STORE16(&smem[i],c,512,.x)
    __syncthreads();    
    if(tx==32){
    #pragma unroll
        for( int k=0; k<16; ++k ){ ((float2*)smem)[4096+k*8+ty]=c[k]; }
    }
    unsigned int x=i&31;
    unsigned int y=i>>5;
    unsigned int u=x?( 64-x):x;
    unsigned int v=y?(128-y):y;
    unsigned int p=y;
    unsigned int q=v;
#pragma unroll
    for( int k=0; k<16; k+=2 )
    {
        if(!((x==0)&(y>64))){
            c[k+0].x=smem[y*64+x];
            c[k+1].x=smem[v*64+u];
        }
        v=128-(y+=16);
    } __syncthreads();
    STORE16(&smem[i],c,512,.y)
    __syncthreads();
    float2 RF=d_RF[x];
#pragma unroll
    for( int k=0; k<16; k+=2 )
    {
        if(!((x==0)&(p>64))){
            c[k+0].y=smem[p*64+x];
            c[k+1].y=smem[q*64+u];          
            s_postproc(c[k],c[k+1],RF);
            d_dst[p*65   +x]=c[k+0];
            d_dst[q*65+64-x]=c[k+1];
        }
        q=128-(p+=16);
    }
    if(i<=64){
        v=i?(128-i):i;
        c[0]=((float2*)smem)[4096+i];
        c[1]=((float2*)smem)[4096+v];
        s_postproc(c[0],c[1],make_float2(0.f,-1.f));
        d_dst[i*65+32]=c[0];
        d_dst[v*65+32]=c[1];
    }
}

__global__ void dk_sfft128x128_r2c( float2* d_c, const float2* __restrict__ d_r, 
    const float2* __restrict__ d_RF, int nx, int ny, unsigned int ldr )
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
    LOAD16(c,d_r,512,)                      
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
__global__ void dk_sfft128x128_r2c_pad( float2* d_c, const float* __restrict__ d_r, 
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
            if(((p+0)>=0)&((p+0)<nx)){ c[k].x=d_r[q*nx+0]; }
            if(((p+1)>=0)&((p+1)<nx)){ c[k].y=d_r[q*nx+1]; }
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
dk_sfft128x128_r2c_ext( float2* d_c, const float* __restrict__ d_r, 
 const float2* __restrict__ d_RF, int nx, int ny, unsigned int ldr )
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
    d_r+=by*ldr+(bx*ny+y)*nx+(x<<1);
    CLEAR16C(c)
#pragma unroll
    for( int k=0; k<16; ++k ){ 
        if(((2*x+0)<nx)&((k*8+y)<ny)){ c[k].x=d_r[k*8*nx+0]; }
        if(((2*x+1)<nx)&((k*8+y)<ny)){ c[k].y=d_r[k*8*nx+1]; }
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
#if SM>=50
    s_postproc_128x128_b( d_c, c, smem, d_RF, tid, x, y );
#else
    s_postproc_128x128_a( d_c, c, smem, d_RF, tid, x, y );
#endif
}
__global__ void 
#if SM>=50
__launch_bounds__(512,2)
#endif
dk_sfft128x128_r2c_flip( float2* d_c, const float* __restrict__ d_r, 
    const float2* __restrict__ d_RF, int nx, int ny, unsigned int ldr )
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
            if((dx+0)<nx){ c[k].x=d_r[q*nx+dx  ]; }
            if((dx+1)<nx){ c[k].y=d_r[q*nx+dx-1]; }
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
#if SM>=50
    s_postproc_128x128_b( d_c, c, smem, d_RF, tid, x, y );
#else
    s_postproc_128x128_a( d_c, c, smem, d_RF, tid, x, y );
#endif
}
