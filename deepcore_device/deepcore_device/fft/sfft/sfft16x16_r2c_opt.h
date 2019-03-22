__global__ void dk_sfft16x16_r2c_s3( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[8*144+73];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bid<<3)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=icell*144+x;
    d_r+=bid*72+tid;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(tid<72){ smem[8*144+tid]=d_r[0]; }
    if(tid==0){ smem[8*144+72]=0.f; }
    __syncthreads();
    float* spr=&smem[8*144+(x<3?(y*9+x):72)];
    unsigned int ofs=x<3?3:0;
    c[0].x=*spr; spr+=ofs;
    c[0].y=*spr; spr+=ofs;
    c[1].x=*spr;
    c[1].y=0.f;
    s_vfft16_s3( c, spx, spy, s_RF, brev );
    s_hfft16_s3( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<9; ++i ){ d_c[i*16]=c[i]; }
}
__global__ void dk_sfft16x16_r2c_s5( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[8*144];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=icell*144+x;
    d_r+=icell*25+x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<5){
        c[0].x=d_r[0*5];
        c[0].y=d_r[1*5];
        c[1].x=d_r[2*5];
        c[1].y=d_r[3*5];
        c[2].x=d_r[4*5];
    } __syncthreads();
    s_vfft16_s5( c, spx, spy, s_RF, brev );
    s_hfft16_s5( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<9; ++i ){ d_c[i*16]=c[i]; }
}
__global__ void dk_sfft16x16_r2c_s7( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[8*144];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=icell*144+x;
    d_r+=icell*49+x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    c[3].x=0.f;
    c[3].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<7){
        c[0].x=d_r[0*7];
        c[0].y=d_r[1*7];
        c[1].x=d_r[2*7];
        c[1].y=d_r[3*7];
        c[2].x=d_r[4*7];
        c[2].y=d_r[5*7];
        c[3].x=d_r[6*7];
    } __syncthreads();
    s_vfft16_s7( c, spx, spy, s_RF, brev );
    s_hfft16_s7( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<9; ++i ){ d_c[i*16]=c[i]; }
}
__global__ void dk_sfft16x16_r2c_flip_s3( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[8*144];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=icell*144+x;
    d_r+=icell*9+2-x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<3){
        c[0].x=d_r[6];
        c[0].y=d_r[3];
        c[1].x=d_r[0];
    } __syncthreads();
    s_vfft16_s3( c, spx, spy, s_RF, brev );
    s_hfft16_s3( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<9; ++i ){ d_c[i*16]=c[i]; }
}
__global__ void dk_sfft16x16_r2c_flip_s5( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[8*144];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=icell*144+x;
    d_r+=icell*25+4-x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<5){
        c[0].x=d_r[20];
        c[0].y=d_r[15];
        c[1].x=d_r[10];
        c[1].y=d_r[ 5];
        c[2].x=d_r[ 0];
    } __syncthreads();
    s_vfft16_s5( c, spx, spy, s_RF, brev );
    s_hfft16_s5( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<9; ++i ){ d_c[i*16]=c[i]; }
}
__global__ void dk_sfft16x16_r2c_flip_s7( float2* d_c, const float* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,4,2,6,1,5,3,7};
    __shared__ float smem[8*144];
    __shared__ float2 s_RF[8];
    float2 c[9];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&15;
    unsigned int y=tid>>4;
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*144+x];
    float* spy=&smem[y*144+v*18+u];
    d_c+=icell*144+x;
    d_r+=icell*49+6-x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    c[3].x=0.f;
    c[3].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<7){
        c[0].x=d_r[42];
        c[0].y=d_r[35];
        c[1].x=d_r[28];
        c[1].y=d_r[21];
        c[2].x=d_r[14];
        c[2].y=d_r[ 7];
        c[3].x=d_r[ 0];
    } __syncthreads();
    s_vfft16_s7( c, spx, spy, s_RF, brev );
    s_hfft16_s7( c, &smem[y*144+v*18+u*8], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<9; ++i ){ d_c[i*16]=c[i]; }
}
