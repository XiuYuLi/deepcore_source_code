__global__ void __launch_bounds__(256,2) dk_sfft32x32_c2r_grad_perm( 
    float* d_r, const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, 
    float scale, unsigned int ldr, unsigned int ldc, 
    unsigned int nx, unsigned int ny )
{   
    __shared__ float smem[8*560];
    __shared__ float2 s_RF[16];
    float2 c[16], d[17];
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int qnc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int p=tid&7;
    unsigned int q=tid>>3;
    unsigned int icell=(bx<<3)+y;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }  
    unsigned int flip_x=(32-x)&31;
    d_r+=by*ldr+icell*ny*nx+flip_x;
    d_c+=(q*qnc+by)*ldc+(bx<<3)+p;
    s_load( d, &smem[p*547+q], &smem[y*547+x], d_c, qnc*ldc*32 );
    s_hifft( c, d, &smem[y*560], s_RF, brev, x );
    s_vifft( d, c, s_RF, brev, x );
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((2*i+0)<ny){ d_r[0]=scale*d[i].x; } d_r+=nx;
            if((2*i+1)<ny){ d_r[0]=scale*d[i].y; } d_r+=nx;
        }
    }
}
__global__ void __launch_bounds__(256,2) dk_sfft32x32_c2r_grad_perm_s3( 
    float* d_r, const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, 
    float scale, unsigned int ldr, unsigned int ldc )
{   
    __shared__ float smem[8*560];
    __shared__ float2 s_RF[16];
    float2 c[16], d[17];
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int qnc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int p=tid&7;
    unsigned int q=tid>>3;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    d_r+=by*ldr+bx*72+tid;
    d_c+=(q*qnc+by)*ldc+(bx<<3)+p;
    s_load( d, &smem[p*547+q], &smem[y*547+x], d_c, qnc*ldc*32 );
    s_hifft( c, d, &smem[y*560], s_RF, brev, x );
    s_vifft( d, c, s_RF, brev, x );
    unsigned int flip_x=(32-x)&31;
    float* spr=&smem[y*9+flip_x];
    if(flip_x<3){
        spr[0]=d[0].x;
        spr[3]=d[0].y;
        spr[6]=d[1].x;
    } __syncthreads();
    if(tid<72){ d_r[0]=scale*smem[tid]; }
}
__global__ void __launch_bounds__(256,2) 
dk_sfft32x32_c2r_grad_perm_s5( float* d_r, 
    const float2* __restrict__ d_c, const float* __restrict__ d_RF, 
    float scale, unsigned int ldr, unsigned int ldc )
{   
    __shared__ float smem[8*560];
    __shared__ float2 s_RF[16];
    float2 c[16], d[17];
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int qnc=gridDim.y;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int p=tid&7;
    unsigned int q=tid>>3;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    d_r+=by*ldr+bx*200+tid;
    d_c+=(q*qnc+by)*ldc+(bx<<3)+p;
    s_load( d, &smem[p*547+q], &smem[y*547+x], d_c, qnc*ldc*32 );
    s_hifft( c, d, &smem[y*560], s_RF, brev, x );
    s_vifft( d, c, s_RF, brev, x ); 
    unsigned int flip_x=(32-x)&31;
    float* spr=&smem[y*25+flip_x];
    if(flip_x<5){
        spr[0*5]=d[0].x;
        spr[1*5]=d[0].y;
        spr[2*5]=d[1].x;
        spr[3*5]=d[1].y;
        spr[4*5]=d[2].x;
    } __syncthreads();
    if(tid<200){ d_r[0]=scale*smem[tid]; }
}