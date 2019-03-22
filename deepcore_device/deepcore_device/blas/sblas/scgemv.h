__global__ void dk_scgemv( char* d_c, 
                     const char* __restrict__ d_a, 
                     const char* __restrict__ d_b, 
                     float alpha, int nx, int ny, 
                     int lda, int ldb, int ldc )
{
    __shared__ float2 smem[128+512];
    float2 c[16]={{0.f,0.f}}, a, b[4];
    unsigned int x=threadIdx.x;
    unsigned int bx=blockIdx.x;
    unsigned int by=blockIdx.y;
    unsigned int u=x&31;
    unsigned int v=x>>5;    
    unsigned int bidx=(bx<<7)|u;
    unsigned int cidx=(bx<<7)|x;
    unsigned int qldb=ldb<<2;
    d_a+=by*lda+x*8;
    d_b+=by*ldb*nx+v*ldb+bidx*8;
    d_c+=by*ldc+cidx*8;
    float2* spx=&smem[x];
    for( int s=0, k=0; k<nx; k+=16 )
    {
        if(s==0){ *spx=((const float2*)d_a)[k]; __syncthreads(); }
    #pragma unroll
        for( int i=0; i<16; i+=4 )
        {
            if((bidx+0*32)<ny){ b[0]=*((const float2*)&d_b[0*256]); }
            if((bidx+1*32)<ny){ b[1]=*((const float2*)&d_b[1*256]); }
            if((bidx+2*32)<ny){ b[2]=*((const float2*)&d_b[2*256]); }
            if((bidx+3*32)<ny){ b[3]=*((const float2*)&d_b[3*256]); } 
            d_b+=qldb;
            a=smem[s*16+v+i];
            c[i+0].x+= a.x*b[0].x;
            c[i+0].y+= a.x*b[0].y;
            c[i+1].x+= a.x*b[1].x;
            c[i+1].y+= a.x*b[1].y;
            c[i+2].x+= a.x*b[2].x;
            c[i+2].y+= a.x*b[2].y;
            c[i+3].x+= a.x*b[3].x;
            c[i+3].y+= a.x*b[3].y;
            c[i+0].x+= a.y*b[0].y;
            c[i+0].y+=-a.y*b[0].x;
            c[i+1].x+= a.y*b[1].y;
            c[i+1].y+=-a.y*b[1].x;
            c[i+2].x+= a.y*b[2].y;
            c[i+2].y+=-a.y*b[2].x;
            c[i+3].x+= a.y*b[3].y;
            c[i+3].y+=-a.y*b[3].x;
        }
        if(((++s)&=7)==0){ __syncthreads(); }
    }
#pragma unroll
    for( int i=0; i<8; ++i ){
        c[i].x+=c[8+i].x;
        c[i].y+=c[8+i].y;
    }
#pragma unroll
    for( int i=0; i<4; ++i ){
        c[i].x+=c[4+i].x;
        c[i].y+=c[4+i].y;
    }
    float2* spy=&smem[128+(v<<7)+u];
    spy[0*32]=c[0];
    spy[1*32]=c[1];
    spy[2*32]=c[2];
    spy[3*32]=c[3];
    __syncthreads();
    c[0]=spx[128];
#pragma unroll
    for( int i=1; i<4; ++i ){
        c[i]=spx[128+i*128];
        c[0].x+=c[i].x;
        c[0].y+=c[i].y;
    }
    if(cidx<ny){ 
        *((float2*)d_c)=make_float2(alpha*c[0].x,alpha*c[0].y); 
    }
}