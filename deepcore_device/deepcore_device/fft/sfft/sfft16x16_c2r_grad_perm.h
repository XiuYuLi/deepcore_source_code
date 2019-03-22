__global__ void dk_sfft16x16_c2r_grad_perm( float* d_r, 
    const float2* __restrict__ d_c, const float* __restrict__ d_RF, 
    float scale, unsigned int ldr, unsigned int ldc, unsigned int nx, unsigned int ny )
{   
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[8], d[9];
    const int brev[]={0,4,2,6,1,5,3,7};
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int qnc=gridDim.y; 
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int icell=(bx<<4)+y;
    unsigned int flip_x=(16-x)&15;  
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }  
    d_r+=by*ldr+icell*ny*nx+flip_x;
    d_c+=(y*qnc+by)*ldc+(bx<<4)+x;
    s_load9( d, &smem[x*145+y], &smem[y*145+x], d_c, 16*qnc*ldc );
    s_hifft16( c, d, &smem[y*144], s_RF, brev, x );
    s_vifft16( d, c, s_RF, brev, x );
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<8; ++i ){
            if((2*i+0)<ny){ d_r[0]=scale*d[i].x; d_r+=nx; }
            if((2*i+1)<ny){ d_r[0]=scale*d[i].y; d_r+=nx; }
        }
    }
}
__global__ void dk_sfft16x16_c2r_grad_perm_s3( float* d_r, 
    const float2* __restrict__ d_c, const float* __restrict__ d_RF, 
    float scale, unsigned int ldr, unsigned int ldc )
{   
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[8], d[9];
    const int brev[]={0,4,2,6,1,5,3,7};
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int onc=gridDim.y; 
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }  
    d_r+=by*ldr+bx*144+tid;
    d_c+=(y*onc+by)*ldc+(bx<<4)+x;
    s_load9( d, &smem[x*145+y], &smem[y*145+x], d_c, 16*onc*ldc );
    s_hifft16( c, d, &smem[y*144], s_RF, brev, x );
    s_vifft16( d, c, s_RF, brev, x );
    unsigned int flip_x=(16-x)&15;
    float* spr=&smem[y*9+flip_x];
    if(flip_x<3){
        spr[0]=d[0].x;
        spr[3]=d[0].y;
        spr[6]=d[1].x;
    } __syncthreads();
    if(tid<144){ d_r[0]=scale*smem[tid]; }
}
__global__ void dk_sfft16x16_c2r_grad_perm_s5( float* d_r, 
    const float2* __restrict__ d_c, const float* __restrict__ d_RF, 
    float scale, unsigned int ldr, unsigned int ldc )
{   
    __shared__ float smem[16*145];
    __shared__ float2 s_RF[8];
    float2 c[8], d[9];
    const int brev[]={0,4,2,6,1,5,3,7};
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int onc=gridDim.y; 
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }  
    d_r+=by*ldr+bx*400+tid;
    d_c+=(y*onc+by)*ldc+(bx<<4)+x;
    s_load9( d, &smem[x*145+y], &smem[y*145+x], d_c, 16*onc*ldc );
    s_hifft16( c, d, &smem[y*144], s_RF, brev, x );
    s_vifft16( d, c, s_RF, brev, x );
    unsigned int flip_x=(16-x)&15;
    float* spr=&smem[y*25+flip_x];
    if(flip_x<5){
        spr[0*5]=d[0].x;
        spr[1*5]=d[0].y;
        spr[2*5]=d[1].x;
        spr[3*5]=d[1].y;
        spr[4*5]=d[2].x;
    } __syncthreads();
    spr=&smem[tid];
    d_r[0]=spr[0];
    if(tid<144){ d_r[256]=scale*spr[256]; }
}