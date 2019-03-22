__global__ void dk_sfft64x64_c2r_grad( 
    float* d_r, 
    const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, 
    float scale, unsigned int nx, unsigned int ny )
{   
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*32];
    __shared__ float2 s_RF[32];
    float2 c[32], d[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    int flip_x=tid?(64-tid):tid;
    d_c+=bid*33*64+tid;
    d_r+=bid*ny*nx+flip_x;
    ((float*)s_RF)[tid]=d_RF[tid];
#pragma unroll
    for( int i=0; i<33; ++i ){ d[i]=d_c[i*64]; }
    s_hifft64( c, d, smem, s_RF, brev, tid );
    s_vifft64( d, c, s_RF, brev, tid );
    if(flip_x<nx){
    #pragma unroll
        for( int i=0; i<32; ++i ){
            if((2*i+0)<ny){ d_r[0]=scale*d[i].x; d_r+=nx; } 
            if((2*i+1)<ny){ d_r[0]=scale*d[i].y; d_r+=nx; } 
        }
    }
}
__global__ void dk_sfft64x64_c2r_grad_s3( float* d_r, const float2* __restrict__ d_c, const float* __restrict__ d_RF, float scale )
{   
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*32];
    __shared__ float2 s_RF[32];
    float2 c[32], d[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    int flip_x=tid?(64-tid):tid;
    d_c+=bid*33*64+tid;
    d_r+=bid*9+flip_x;
    ((float*)s_RF)[tid]=d_RF[tid];
#pragma unroll
    for( int i=0; i<33; ++i ){ d[i]=d_c[i*64]; }
    s_hifft64( c, d, smem, s_RF, brev, tid );
    s_vifft64( d, c, s_RF, brev, tid );
    if(flip_x<3){
        d_r[0]=scale*d[0].x; 
        d_r[3]=scale*d[0].y; 
        d_r[6]=scale*d[1].x; 
    }
}
__global__ void dk_sfft64x64_c2r_grad_s5( 
    float* d_r, 
    const float2* __restrict__ d_c, 
    const float* __restrict__ d_RF, float scale )
{   
    const int brev[]={0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31};
    __shared__ float smem[66*32];
    __shared__ float2 s_RF[32];
    float2 c[32], d[33];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    int flip_x=tid?(64-tid):tid;
    d_c+=bid*33*64+tid;
    d_r+=bid*25+flip_x;
    ((float*)s_RF)[tid]=d_RF[tid];
#pragma unroll
    for( int i=0; i<33; ++i ){ d[i]=d_c[i*64]; }
    s_hifft64( c, d, smem, s_RF, brev, tid );
    s_vifft64( d, c, s_RF, brev, tid );
    if(flip_x<5){
        d_r[0*5]=scale*d[0].x; 
        d_r[1*5]=scale*d[0].y;
        d_r[2*5]=scale*d[1].x; 
        d_r[3*5]=scale*d[1].y; 
        d_r[4*5]=scale*d[2].x;
    }
}