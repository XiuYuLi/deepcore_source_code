__global__ void dk_sfft16x16_c2r_grad( 
    float* d_r, 
    const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, 
    float scale, unsigned int nx, unsigned int ny )
{   
    __shared__ float smem[8*144];
    __shared__ float2 s_RF[8];
    float2 c[8], d[9];
    const int brev[]={0,4,2,6,1,5,3,7};
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;  
    unsigned int icell=(blockIdx.x<<3)+y;
    unsigned int flip_x=(16-x)&15;
    d_r+=icell*ny*nx+flip_x;
    d_c+=icell*144+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
#pragma unroll
    for( int i=0; i<9; ++i ){ d[i]=d_c[i*16]; }
    __syncthreads();
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
__global__ void dk_sfft16x16_c2r_grad_s3( 
    float* d_r, 
    const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, float scale )
{   
    __shared__ float smem[8*144];
    __shared__ float2 s_RF[8];
    float2 c[8], d[9];
    const int brev[]={0,4,2,6,1,5,3,7};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;  
    unsigned int icell=(bid<<3)+y;
    d_r+=bid*72+tid;
    d_c+=icell*144+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
#pragma unroll
    for( int i=0; i<9; ++i ){ d[i]=d_c[i*16]; }
    __syncthreads();
    s_hifft16( c, d, &smem[y*144], s_RF, brev, x );
    s_vifft16( d, c, s_RF, brev, x );
    unsigned int flip_x=(16-x)&15;
    float* spr=&smem[y*9+flip_x];
    if(flip_x<3){
        spr[0]=d[0].x;
        spr[3]=d[0].y;
        spr[6]=d[1].x;
    } __syncthreads();
    if(tid<72){ d_r[0]=scale*smem[tid]; }
}
__global__ void dk_sfft16x16_c2r_grad_s5( 
    float* d_r, 
    const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, float scale )
{   
    __shared__ float smem[8*144];
    __shared__ float2 s_RF[8];
    float2 c[8], d[9];
    const int brev[]={0,4,2,6,1,5,3,7};
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;  
    unsigned int icell=(bid<<3)+y;
    d_r+=bid*200+tid;
    d_c+=icell*144+x;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
#pragma unroll
    for( int i=0; i<9; ++i ){ d[i]=d_c[i*16]; }
    __syncthreads();
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
    if(tid<72){ d_r[128]=scale*spr[128]; }
}