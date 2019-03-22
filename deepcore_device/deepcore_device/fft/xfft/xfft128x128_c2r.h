__device__ __forceinline__ void xfft128x128_c2r_store( __half* dst, const __half* null, float2* c, int x, int y, int nx, int ny, float alpha )
{
    int dx=x<<1;
    int x0=dx?(128-dx):dx;
    int x1=127-dx;
    int p=y?(128-y):y;
#pragma unroll
    for( int k=0; k<16; ++k ){
        if(p<ny){
            if(x0<nx){ dst[p*nx+x0]=__float2half(alpha*c[k].x); }
            if(x1<nx){ dst[p*nx+x1]=__float2half(alpha*c[k].y); }
        }
        p=128-(y+=8);
    }
}
__device__ __forceinline__ void xfft128x128_c2r_store_relu( __half* dst, const __half* null, const float2* c, int x, int y, int nx, int ny, float alpha )
{
    int dx=x<<1;
    int x0=dx?(128-dx):dx;
    int x1=127-dx;
    int p=y?(128-y):y;
#pragma unroll
    for( int k=0; k<16; ++k ){
        if(p<ny){
            if(x0<nx){ dst[p*nx+x0]=__float2half(s_relu(alpha*c[k].x)); }
            if(x1<nx){ dst[p*nx+x1]=__float2half(s_relu(alpha*c[k].y)); }
        }
        p=128-(y+=8);
    }
}
__device__ __forceinline__ void xfft128x128_c2r_store_bias( __half* dst, const __half* bias, const float2* c, int x, int y, int nx, int ny, float alpha )
{
    int dx=x<<1;
    int x0=dx?(128-dx):dx;
    int x1=127-dx;
    int p=y?(128-y):y;
    float b=__half2float(*bias);
#pragma unroll
    for( int k=0; k<16; ++k ){
        if(p<ny){
            if(x0<nx){ dst[p*nx+x0]=__float2half(alpha*c[k].x+b); }
            if(x1<nx){ dst[p*nx+x1]=__float2half(alpha*c[k].y+b); }
        }
        p=128-(y+=8);
    }
}
__device__ __forceinline__ void xfft128x128_c2r_store_bias_relu( __half* dst, const __half* bias, const float2* c, int x, int y, int nx, int ny, float alpha )
{
    int dx=x<<1;
    int x0=dx?(128-dx):dx;
    int x1=127-dx;
    int p=y?(128-y):y;
    float b=__half2float(*bias);
#pragma unroll
    for( int k=0; k<16; ++k ){
        if(p<ny){
            if(x0<nx){ dst[p*nx+x0]=__float2half(s_relu(alpha*c[k].x+b)); }
            if(x1<nx){ dst[p*nx+x1]=__float2half(s_relu(alpha*c[k].y+b)); }
        }
        p=128-(y+=8);
    }
}
__device__ __forceinline__ void xfft128x128_c2r_store_drelu( __half* dst, const __half* a, const float2* c, int x, int y, int nx, int ny, float alpha )
{
    int dx=x<<1;
    int x0=dx?(128-dx):dx;
    int x1=127-dx;
    int p=y?(128-y):y;
#pragma unroll
    for( int k=0; k<16; ++k ){
        if(p<ny){
            if(x0<nx){ dst[p*nx+x0]=__float2half(alpha*c[k].x*s_drelu(__half2float(a[p*nx+x0]))); }
            if(x1<nx){ dst[p*nx+x1]=__float2half(alpha*c[k].y*s_drelu(__half2float(a[p*nx+x1]))); }
        }
        p=128-(y+=8);
    }
}
__device__ __forceinline__ void xfft128x128_c2r_store_xdrv( __half* dst, const __half* da, const float2* c, int x, int y, int nx, int ny, float alpha )
{
    int dx=x<<1;
    int x0=dx?(128-dx):dx;
    int x1=127-dx;
    int p=y?(128-y):y;
#pragma unroll
    for( int k=0; k<16; ++k ){
        if(p<ny){
            if(x0<nx){ dst[p*nx+x0]=__float2half(alpha*c[k].x*__half2float(da[p*nx+x0])); }
            if(x1<nx){ dst[p*nx+x1]=__float2half(alpha*c[k].y*__half2float(da[p*nx+x1])); }
        }
        p=128-(y+=8);
    }
}

#define xfft128x128_c2r(dir,suffix)        \
__global__ void dk_xfft128x128_c2r##suffix(\
          __half*              d_r  ,\
    const float2* __restrict__ d_c  ,\
    const float2* __restrict__ d_RF ,\
    const __half* __restrict__ d_x  ,\
    float                      alpha,\
    unsigned int               nx   ,\
    unsigned int               ny   ,\
    unsigned int               ldr ){\
    __shared__ float smem[128*72];\
    float2 c[16], RF[15], temp;   \
    unsigned int bx=blockIdx.x;   \
    unsigned int by=blockIdx.y;   \
    unsigned int tid=threadIdx.x; \
    unsigned int x=tid&63;        \
    unsigned int y=tid>>6;        \
    d_c+=(by*gridDim.x+bx)*65*128;\
    s_preproc_128x128( c, smem, d_c, d_RF, tid, x, y );\
    d_r+=by*ldr+bx*ny*nx;\
    if(dir==0){ d_x+=by; } else\
    if(dir==1){ d_x+=by*ldr+bx*ny*nx; }\
    unsigned int u=tid&7;     \
    unsigned int v=tid>>3;    \
    float* spx=&smem[ 72*y+x];\
    float* spy=&smem[576*y+x];\
    float* spu=&smem[72*v+u]; \
    float* spv=&smem[576*y+65*u+((v&7)<<3)];\
    RF[0]=d_RF[y];   \
    iCALRF16(RF)     \
    FFT16(c,i)       \
    MRF16(c,RF)      \
    RF[0]=d_RF[u<<1];\
    iCALRF8(RF)      \
    PERMUTE_S16_L8x2(spx,spy,c,576,4608,72,0xf)\
    FFT8(&c[0],i)\
    FFT8(&c[8],i)\
    PERMUTE8x2(spx,spu,c,576,1152,4608,8,0xf)\
    FFT8(&c[0],i) \
    FFT8(&c[8],i) \
    MRF8(&c[0],RF)\
    MRF8(&c[8],RF)\
    PERMUTE8x2(spy,spv,c,4608,65,4608,1,0xf)\
    FFT8(&c[0],i)\
    FFT8(&c[8],i)\
    PERMUTE_S8x2_L16(spu,spx,c,4608,8,576,0x7)\
    xfft128x128_c2r_store##suffix( d_r, d_x, c, x, y, nx, ny, alpha );\
}

xfft128x128_c2r(-1,)
xfft128x128_c2r( 0,_relu)
xfft128x128_c2r( 0,_bias)
xfft128x128_c2r( 0,_bias_relu)
xfft128x128_c2r( 1,_drelu)
xfft128x128_c2r( 1,_xdrv)

__global__ void dk_xfft128x128_c2r_grad( __half* d_r, const float2* __restrict__ d_c, 
    const float2* __restrict__ d_RF, float scale, unsigned int nx, unsigned int ny )
{
    __shared__ float smem[128*72];
    float2 c[16], RF[15], temp; 
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&63;
    unsigned int y=tid>>6;  
    d_c+=bid*65*128;
    s_preproc_128x128( c, smem, d_c, d_RF, tid, x, y );
    d_r+=bid*ny*nx;
    unsigned int u=tid&7;
    unsigned int v=tid>>3;
    float* spx=&smem[ 72*y+x];
    float* spy=&smem[576*y+x];
    float* spu=&smem[72*v+u];
    float* spv=&smem[576*y+65*u+((v&7)<<3)];
    RF[0]=d_RF[y];
    iCALRF16(RF)
    FFT16(c,i)
    MRF16(c,RF)
    RF[0]=d_RF[u<<1];
    iCALRF8(RF)
    PERMUTE_S16_L8x2(spx,spy,c,576,4608,72,0xf)
    FFT8(&c[0],i)
    FFT8(&c[8],i)
    PERMUTE8x2(spx,spu,c,576,1152,4608,8,0xf)
    FFT8(&c[0],i)
    FFT8(&c[8],i)
    MRF8(&c[0],RF)
    MRF8(&c[8],RF)
    PERMUTE8x2(spy,spv,c,4608,65,4608,1,0xf)
    FFT8(&c[0],i)
    FFT8(&c[8],i)
    PERMUTE_S8x2_L16(spu,spx,c,4608,8,576,0x7)
    int dx=x<<1;
    int x0=dx?(128-dx):dx;
    int x1=127-dx;
    int p=y?(128-y):y;
#pragma unroll
    for( int k=0; k<16; ++k ){
        if(p<ny){
            if(x0<nx){ d_r[p*nx+x0]=__float2half(scale*c[k].x); }
            if(x1<nx){ d_r[p*nx+x1]=__float2half(scale*c[k].y); }
        }
        p=128-(y+=8);
    }
}