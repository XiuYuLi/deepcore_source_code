__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_s3( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544+73];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bid<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=bid*72+tid;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(tid<72){ smem[8*544+tid]=__half2float(d_r[0]); }
    if(tid==0){ smem[8*544+72]=0.f; }
    __syncthreads();
    float* spr=&smem[8*544+(x<3?(y*9+x):72)];
    unsigned int ofs=x<3?3:0;
    c[0].x=*spr; spr+=ofs;
    c[0].y=*spr; spr+=ofs;
    c[1].x=*spr;
    c[1].y=0.f;
    s_vfft_s3( c, spx, spy, s_RF, brev );
    s_hfft_s3( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=c[i]; }
}
__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_s5( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544+201];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bid<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    const __half2* temp=(const __half2*)d_r;
    d_c+=icell*544+x;
    temp+=bid*100+tid;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(tid<100){ ((float2*)smem)[4*544+tid]=__half22float2(*temp); }
    if(tid==0){ smem[8*544+200]=0.f; }
    __syncthreads();
    float* spr=&smem[8*544+(x<5?(y*25+x):200)];
    unsigned int ofs=x<5?5:0;
    c[0].x=*spr; spr+=ofs;
    c[0].y=*spr; spr+=ofs;
    c[1].x=*spr; spr+=ofs;
    c[1].y=*spr; spr+=ofs;
    c[2].x=*spr;
    c[2].y=0.f;
    s_vfft_s5( c, spx, spy, s_RF, brev );
    s_hfft_s5( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=c[i]; }
}
__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_s7( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544+393];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int bid=blockIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(bid<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    const __half2* temp=(const __half2*)d_r;
    d_c+=icell*544+x;
    temp+=bid*196+tid;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(tid<196){ ((float2*)smem)[4*544+tid]=__half22float2(*temp); }
    if(tid==0){ smem[8*544+392]=0.f; }
    __syncthreads();
    float* spr=&smem[8*544+(x<7?(y*49+x):392)];
    unsigned int ofs=x<7?7:0;
    c[0].x=*spr; spr+=ofs;
    c[0].y=*spr; spr+=ofs;
    c[1].x=*spr; spr+=ofs;
    c[1].y=*spr; spr+=ofs;
    c[2].x=*spr; spr+=ofs;
    c[2].y=*spr; spr+=ofs;
    c[3].x=*spr;
    c[3].y=0.f;
    s_vfft_s7( c, spx, spy, s_RF, brev );
    s_hfft_s7( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=c[i]; }
}
__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_flip_s3( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=icell*9+2-x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<3){
        c[0].x=__half2float(d_r[6]);
        c[0].y=__half2float(d_r[3]);
        c[1].x=__half2float(d_r[0]);
    } __syncthreads();
    s_vfft_s3( c, spx, spy, s_RF, brev );
    s_hfft_s3( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=c[i]; }
}
__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_flip_s5( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
    d_r+=icell*25+4-x;
    c[0].x=0.f;
    c[0].y=0.f;
    c[1].x=0.f;
    c[1].y=0.f;
    c[2].x=0.f;
    c[2].y=0.f;
    if(y==0){ ((float*)s_RF)[x]=d_RF[x]; }
    if(x<5){
        c[0].x=__half2float(d_r[20]);
        c[0].y=__half2float(d_r[15]);
        c[1].x=__half2float(d_r[10]);
        c[1].y=__half2float(d_r[ 5]);
        c[2].x=__half2float(d_r[ 0]);
    } __syncthreads();
    s_vfft_s5( c, spx, spy, s_RF, brev );
    s_hfft_s5( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=c[i]; }
}
__global__ void __launch_bounds__(256,2) dk_xfft32x32_r2c_flip_s7( float2* d_c, const __half* __restrict__ d_r, const float* __restrict__ d_RF )
{
    const int brev[]={0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    __shared__ float smem[8*544];
    __shared__ float2 s_RF[16];
    float2 c[17];
    unsigned int tid=threadIdx.x;
    unsigned int x=tid&31;
    unsigned int y=tid>>5;  
    unsigned int u=x&1;
    unsigned int v=x>>1;
    unsigned int icell=(blockIdx.x<<3)+y;
    float* spx=&smem[y*544+x];
    float* spy=&smem[y*544+v*34+u];
    d_c+=icell*544+x;
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
        c[0].x=__half2float(d_r[42]);
        c[0].y=__half2float(d_r[35]);
        c[1].x=__half2float(d_r[28]);
        c[1].y=__half2float(d_r[21]);
        c[2].x=__half2float(d_r[14]);
        c[2].y=__half2float(d_r[ 7]);
        c[3].x=__half2float(d_r[ 0]);
    } __syncthreads();
    s_vfft_s7( c, spx, spy, s_RF, brev );
    s_hfft_s7( c, &smem[y*544+v*34+u*16], spx, s_RF, brev, x, u );
#pragma unroll
    for( int i=0; i<17; ++i ){ d_c[i*32]=c[i]; }
}