__global__ void dk_sfft8x8_c2r_grad_perm( float* d_r, 
    const float2* __restrict__ d_c, const float* __restrict__ d_RF, 
    float scale, unsigned int ldr, unsigned int ldc, unsigned int nx, unsigned int ny )
{   
    __shared__ float smem[16*44];
    __shared__ float2 s_RF[4];
    float2 c[4], d[5];
    const int brev[]={0,2,1,3}; 
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int onc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&7;
    unsigned int y=tid>>3;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    unsigned int icell=(bx<<4)+y;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }  
    unsigned int flip_x=(8-x)&7;
    d_r+=by*ldr+icell*ny*nx+flip_x;
    d_c+=(q*onc+by)*ldc+(bx<<4)+p;
    s_load5( d, &smem[p*41+q], &smem[y*41+x], d_c, 8*onc*ldc );
    s_hifft8( c, d, &smem[y*44], s_RF, brev, x );
    s_vifft8( d, c, s_RF, brev, x );
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<4; ++i ){
            if((2*i+0)<ny){ d_r[0]=scale*d[i].x; d_r+=nx; }
            if((2*i+1)<ny){ d_r[0]=scale*d[i].y; d_r+=nx; }
        }
    }
}
__global__ void dk_sfft8x8_c2r_grad_perm_s3( float* d_r, 
    const float2* __restrict__ d_c, const float* __restrict__ d_RF, 
    float scale, unsigned int ldr, unsigned int ldc )
{   
    __shared__ float smem[16*44];
    __shared__ float2 s_RF[4];
    float2 c[4], d[5];
    const int brev[]={0,2,1,3}; 
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int onc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&7;
    unsigned int y=tid>>3;
    unsigned int p=tid&15;
    unsigned int q=tid>>4;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }  
    d_r+=by*ldr+bx*144+tid;
    d_c+=(q*onc+by)*ldc+(bx<<4)+p;
    s_load5( d, &smem[p*41+q], &smem[y*41+x], d_c, 8*onc*ldc );
    s_hifft8( c, d, &smem[y*44], s_RF, brev, x );
    s_vifft8( d, c, s_RF, brev, x );
    unsigned int flip_x=(8-x)&7;
    float* spr=&smem[y*9+flip_x];
    if(flip_x<3){
        spr[0]=d[0].x;
        spr[3]=d[0].y;
        spr[6]=d[1].x;
    } __syncthreads();
    d_r[0]=scale*smem[tid];
    if(tid<16){ d_r[128]=scale*smem[128+tid]; }
}