__global__ void dk_sfft32x32_c2r_grad( 
    float* d_r, 
    const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, 
    float scale, unsigned int nx, unsigned int ny )
{   
    __shared__ float smem[8*560];
    __shared__ float2 s_RF[16];
    float2 c[16], d[17];
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int icell=(bid<<3)+y;
    unsigned int flip_x=(32-x)&31;
    d_r+=icell*ny*nx+flip_x;
    d_c+=icell*544+x;   
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    __syncthreads();
#pragma unroll
    for( int i=0; i<17; ++i ){ d[i]=d_c[i*32]; }
    s_hifft( c, d, &smem[y*560], s_RF, brev, x );
    s_vifft( d, c, s_RF, brev, x );
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<16; ++i ){
            if((2*i+0)<ny){ d_r[0]=scale*d[i].x; d_r+=nx; } 
            if((2*i+1)<ny){ d_r[0]=scale*d[i].y; d_r+=nx; } 
        }
    }
}
__global__ void dk_sfft32x32_c2r_grad_s3( 
    float* d_r, 
    const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, float scale )
{   
    __shared__ float smem[8*560];
    __shared__ float2 s_RF[16];
    float2 c[16], d[17];
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int icell=(bid<<3)+y;
    d_r+=bid*72+tid;
    d_c+=icell*544+x;   
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    __syncthreads();
#pragma unroll
    for( int i=0; i<17; ++i ){ d[i]=d_c[i*32]; }
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
__global__ void dk_sfft32x32_c2r_grad_s5( 
    float* d_r, 
    const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, float scale )
{   
    __shared__ float smem[8*560];
    __shared__ float2 s_RF[16];
    float2 c[16], d[17];
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;
    unsigned int icell=(bid<<3)+y;
    d_r+=bid*200+tid;
    d_c+=icell*544+x;   
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    __syncthreads();
#pragma unroll
    for( int i=0; i<17; ++i ){ d[i]=d_c[i*32]; }
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