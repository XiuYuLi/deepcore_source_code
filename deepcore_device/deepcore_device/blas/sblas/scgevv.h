__global__ void dk_scgevv( 
          char*              d_c  ,
    const char* __restrict__ d_a  ,
    const char* __restrict__ d_b  ,
    float                    alpha,
    int                      pnc  ,
    int                      qnc  ,
    int                      lda  ,
    int                      ldb  ,
    int                      ldc  ,
    int                      nbx )
{
    __shared__ char smem[1024];
    float2 c[32];
    float4 a[2], b[4];
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int tid=threadIdx.x;
    unsigned int dx=bx%nbx;
    unsigned int dy=bx/nbx;
    unsigned int u=tid&7;
    unsigned int v=tid>>3;
    unsigned int x=(dx<<5)+tid;
    unsigned int y=dy<<5;
    unsigned int nx=(pnc+1)>>1;
    unsigned int ny=(qnc+1)>>1;
    unsigned int ai=(dx<<4)+(tid&15);
    unsigned int bi=(dy<<4)+(tid&15);
    unsigned int sai=ai<nx?ai:(nx-1);
    unsigned int sbi=bi<ny?bi:(ny-1);
    d_a+=by*lda+(sai<<4);
    d_b+=by*ldb+(sbi<<4);
    d_c+=(by*qnc+y)*ldc+(x<<3);
    ny-=y;
    const float4* d_x=(const float4*)((tid>>4)==0?d_a:d_b);
    float4 p=*((const float4*)d_x);
    float4* asld=(float4*)&smem[u<<4];
    float4* bsld=(float4*)&smem[256+(v<<4)];
    ((float4*)smem)[tid]=p;
    __syncthreads();
    a[0]=asld[ 0];
    a[1]=asld[ 8];
    b[0]=bsld[ 0];
    b[1]=bsld[ 4];
    b[2]=bsld[ 8];
    b[3]=bsld[12];
    c[0*4+0].x=a[0].x*b[0].x+a[0].y*b[0].y;
    c[0*4+0].y=a[0].x*b[0].y-a[0].y*b[0].x;
    c[0*4+1].x=a[0].z*b[0].x+a[0].w*b[0].y;
    c[0*4+1].y=a[0].z*b[0].y-a[0].w*b[0].x;
    c[0*4+2].x=a[1].x*b[0].x+a[1].y*b[0].y;
    c[0*4+2].y=a[1].x*b[0].y-a[1].y*b[0].x;
    c[0*4+3].x=a[1].z*b[0].x+a[1].w*b[0].y;
    c[0*4+3].y=a[1].z*b[0].y-a[1].w*b[0].x;
    c[1*4+0].x=a[0].x*b[0].z+a[0].y*b[0].w;
    c[1*4+0].y=a[0].x*b[0].w-a[0].y*b[0].z;
    c[1*4+1].x=a[0].z*b[0].z+a[0].w*b[0].w;
    c[1*4+1].y=a[0].z*b[0].w-a[0].w*b[0].z;
    c[1*4+2].x=a[1].x*b[0].z+a[1].y*b[0].w;
    c[1*4+2].y=a[1].x*b[0].w-a[1].y*b[0].z;
    c[1*4+3].x=a[1].z*b[0].z+a[1].w*b[0].w;
    c[1*4+3].y=a[1].z*b[0].w-a[1].w*b[0].z;
    c[2*4+0].x=a[0].x*b[1].x+a[0].y*b[1].y;
    c[2*4+0].y=a[0].x*b[1].y-a[0].y*b[1].x;
    c[2*4+1].x=a[0].z*b[1].x+a[0].w*b[1].y;
    c[2*4+1].y=a[0].z*b[1].y-a[0].w*b[1].x;
    c[2*4+2].x=a[1].x*b[1].x+a[1].y*b[1].y;
    c[2*4+2].y=a[1].x*b[1].y-a[1].y*b[1].x;
    c[2*4+3].x=a[1].z*b[1].x+a[1].w*b[1].y;
    c[2*4+3].y=a[1].z*b[1].y-a[1].w*b[1].x;
    c[3*4+0].x=a[0].x*b[1].z+a[0].y*b[1].w;
    c[3*4+0].y=a[0].x*b[1].w-a[0].y*b[1].z;
    c[3*4+1].x=a[0].z*b[1].z+a[0].w*b[1].w;
    c[3*4+1].y=a[0].z*b[1].w-a[0].w*b[1].z;
    c[3*4+2].x=a[1].x*b[1].z+a[1].y*b[1].w;
    c[3*4+2].y=a[1].x*b[1].w-a[1].y*b[1].z;
    c[3*4+3].x=a[1].z*b[1].z+a[1].w*b[1].w;
    c[3*4+3].y=a[1].z*b[1].w-a[1].w*b[1].z;
    c[4*4+0].x=a[0].x*b[2].x+a[0].y*b[2].y;
    c[4*4+0].y=a[0].x*b[2].y-a[0].y*b[2].x;
    c[4*4+1].x=a[0].z*b[2].x+a[0].w*b[2].y;
    c[4*4+1].y=a[0].z*b[2].y-a[0].w*b[2].x;
    c[4*4+2].x=a[1].x*b[2].x+a[1].y*b[2].y;
    c[4*4+2].y=a[1].x*b[2].y-a[1].y*b[2].x;
    c[4*4+3].x=a[1].z*b[2].x+a[1].w*b[2].y;
    c[4*4+3].y=a[1].z*b[2].y-a[1].w*b[2].x;
    c[5*4+0].x=a[0].x*b[2].z+a[0].y*b[2].w;
    c[5*4+0].y=a[0].x*b[2].w-a[0].y*b[2].z;
    c[5*4+1].x=a[0].z*b[2].z+a[0].w*b[2].w;
    c[5*4+1].y=a[0].z*b[2].w-a[0].w*b[2].z;
    c[5*4+2].x=a[1].x*b[2].z+a[1].y*b[2].w;
    c[5*4+2].y=a[1].x*b[2].w-a[1].y*b[2].z;
    c[5*4+3].x=a[1].z*b[2].z+a[1].w*b[2].w;
    c[5*4+3].y=a[1].z*b[2].w-a[1].w*b[2].z;
    c[6*4+0].x=a[0].x*b[3].x+a[0].y*b[3].y;
    c[6*4+0].y=a[0].x*b[3].y-a[0].y*b[3].x;
    c[6*4+1].x=a[0].z*b[3].x+a[0].w*b[3].y;
    c[6*4+1].y=a[0].z*b[3].y-a[0].w*b[3].x;
    c[6*4+2].x=a[1].x*b[3].x+a[1].y*b[3].y;
    c[6*4+2].y=a[1].x*b[3].y-a[1].y*b[3].x;
    c[6*4+3].x=a[1].z*b[3].x+a[1].w*b[3].y;
    c[6*4+3].y=a[1].z*b[3].y-a[1].w*b[3].x;
    c[7*4+0].x=a[0].x*b[3].z+a[0].y*b[3].w;
    c[7*4+0].y=a[0].x*b[3].w-a[0].y*b[3].z;
    c[7*4+1].x=a[0].z*b[3].z+a[0].w*b[3].w;
    c[7*4+1].y=a[0].z*b[3].w-a[0].w*b[3].z;
    c[7*4+2].x=a[1].x*b[3].z+a[1].y*b[3].w;
    c[7*4+2].y=a[1].x*b[3].w-a[1].y*b[3].z;
    c[7*4+3].x=a[1].z*b[3].z+a[1].w*b[3].w;
    c[7*4+3].y=a[1].z*b[3].w-a[1].w*b[3].z;
    float2* sst=(float2*)&smem[tid<<4];
    float2* sld=(float2*)&smem[tid<<3];
#pragma unroll
    for( int i=0; i<32; ++i ){ c[i].x*=alpha; c[i].y*=alpha; }
    int ybd=qnc-y;
#pragma unroll
    for( int i=0; i<8; ++i )
    {
        __syncthreads();
        sst[ 0]=c[i*4+0];
        sst[ 1]=c[i*4+1];
        sst[16]=c[i*4+2];
        sst[17]=c[i*4+3];
        __syncthreads();
        if(x<pnc){
            if(((i/2)*8+0+(i%2))<ybd){ *((float2*)&d_c[((i/2)*8+0+(i%2))*ldc])=sld[0*32]; }
            if(((i/2)*8+2+(i%2))<ybd){ *((float2*)&d_c[((i/2)*8+2+(i%2))*ldc])=sld[1*32]; }
            if(((i/2)*8+4+(i%2))<ybd){ *((float2*)&d_c[((i/2)*8+4+(i%2))*ldc])=sld[2*32]; }
            if(((i/2)*8+6+(i%2))<ybd){ *((float2*)&d_c[((i/2)*8+6+(i%2))*ldc])=sld[3*32]; }
        }
    }
}