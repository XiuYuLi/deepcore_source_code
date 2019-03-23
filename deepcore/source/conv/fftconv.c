#include"../../include/conv/fftconv.h"

__local_func size_t idc_fftconv_createOp( idc_fftconvOp_t* Op, const cuda_context_t* p_ctx, uint32_t mask, int ng, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int qnx, int qny, int qnc, int ldq, int bat, int pad_x, int pad_y )
{
    int dir, prc, inc, onc, fft_size, axis, snx, lds, sny, onx, ony, ldo, dx, dy, n, grid_x, grid_y, npc, radix, is_split;
    cuda_kernel_t* p_kernel;
    
    dir=mask&0x1;
    snx=dir?qnx:pnx;
    sny=dir?qny:pny;
    lds=dir?ldq:ldp;
    onx=dir?pnx:qnx;
    ony=dir?pny:qny;
    ldo=dir?ldp:ldq;
    axis=idc_fftconv_choose_optimal_size( snx+(pad_x<<1), sny+(pad_y<<1), onx, ony, fnx, fny );
    fft_size=1<<(3+axis);
    dx=fft_size-fnx+1;
    dy=fft_size-fny+1;
    grid_x=(onx+dx-1)/dx;
    grid_y=(ony+dy-1)/dy;
    npc=bat*grid_x*grid_y;
    if((fft_size<=32)&&((npc==1)||(npc>8)))
        return idc_cellconv_createOp( Op, p_ctx, mask, ng, pnx, pny, pnc, ldp, fnx, fny, qnx, qny, qnc, ldq, bat, pad_x, pad_y );
    prc=(mask>>1)&0x3;
    n=prc==0?2:1;
    pnc/=ng;
    qnc/=ng;    
    inc=dir?qnc:pnc;
    onc=dir?pnc:qnc;
    Op->ng=ng;
    Op->ags=(inc*lds)<<n;
    Op->bgs=(pnc*qnc*fnx*fny)<<n;    
    Op->cgs=(onc*ldo)<<n;
    radix=fft_size>8?8:16;
    is_split=(grid_x|grid_y)>1;
    
    {
        int is_pad=(pad_x!=0)|(pad_y!=0);
        int is_ext=((snx!=fft_size)|(sny!=fft_size))&(1^is_pad);
        p_kernel=&Op->kfft[0];
        idc_create_fft_kernel_r2c( p_kernel, p_ctx, axis, prc, is_split, is_ext, is_pad, 0 );
        n=inc*npc;
        cuda_kernel_sgl( p_kernel, axis<3?((n+radix-1)/radix):bat, axis<3?1:inc, 1 );
        cuda_kernel_sep_i32( p_kernel, 3, snx );
        cuda_kernel_sep_i32( p_kernel, 4, sny );
        cuda_kernel_sep_i32( p_kernel, 5, lds );
        if(axis<3){
            cuda_kernel_sep_i32( p_kernel, 6, npc );
            if(is_split|(axis<3)){ cuda_kernel_sep_i32( p_kernel, 7, n ); }
            if(is_split){
                
                cuda_kernel_sep_i32( p_kernel, 8, grid_x );
                cuda_kernel_sep_i32( p_kernel, 9, grid_y );
                cuda_kernel_sep_i32( p_kernel,10, dx     );
                cuda_kernel_sep_i32( p_kernel,11, dy     );
            }
        }           
        if(is_pad){
            cuda_kernel_sep_i32( p_kernel, is_split?12:(axis<3?8:6), pad_x );
            cuda_kernel_sep_i32( p_kernel, is_split?13:(axis<3?9:7), pad_y );
        }
    }
    
    {
        int is_opt=((fnx==3)&(fny==3))|((fnx==5)&(fny==5))|((fnx==7)&(fny==7));
        p_kernel=&Op->kfft[1];
        n=pnc*qnc;
        if((fft_size>8)&(fft_size<=64)&is_opt){
            idc_create_fft_kernel_r2c_opt( p_kernel, p_ctx, axis, prc, dir, (fnx>3)+(fnx>5) );
        } else {
            idc_create_fft_kernel_r2c( p_kernel, p_ctx, axis, prc, 0, 1^dir, 0, dir );
        }
        if((fft_size<64)|((fft_size==64)&is_opt)){
            cuda_kernel_sgl( p_kernel, fft_size<64?((n+radix-1)/radix):n, 1, 1 );
        } else {
            cuda_kernel_sgl( p_kernel, pnc, qnc, 1 );
        }
        if((!is_opt)|(!is_opt&(fft_size==64))|(fft_size>64)|(fft_size==8)){
            cuda_kernel_sep_i32( p_kernel, 3, fnx );
            cuda_kernel_sep_i32( p_kernel, 4, fny );
            cuda_kernel_sep_i32( p_kernel, 5, pnc*fnx*fny );
            if(axis<3){             
                cuda_kernel_sep_i32( p_kernel, 6, pnc );
                cuda_kernel_sep_i32( p_kernel, 7, n   );
            }
        }
    }
    
    {
        int is_fuse=(mask>> 3)&0x1;
        int is_relu=(mask>>24)&0x1;
        p_kernel=&Op->kfft[2];
        idc_create_fft_kernel_c2r( p_kernel, p_ctx, axis, prc, dir, is_split, is_fuse, is_relu );
        n=onc*npc;
        cuda_kernel_sgl( p_kernel, axis<3?((n+radix-1)/radix):bat, axis<3?1:onc, 1 );
        cuda_kernel_sep_i32( p_kernel, 5, onx );
        cuda_kernel_sep_i32( p_kernel, 6, ony );
        cuda_kernel_sep_i32( p_kernel, 7, ldo );
        if(axis<3){
            cuda_kernel_sep_i32( p_kernel, 8, npc );
            cuda_kernel_sep_i32( p_kernel, 9, n   );
            if(is_split){
                cuda_kernel_sep_i32( p_kernel, 10, grid_x );
                cuda_kernel_sep_i32( p_kernel, 11, grid_y );
                cuda_kernel_sep_i32( p_kernel, 12, dx     );
                cuda_kernel_sep_i32( p_kernel, 13, dy     );
            }
        }
    }
    n=((fft_size>>1)+1)*fft_size+(fft_size>8?0:8);
    idc_flatcgemm_create_kernel( &Op->kcgemm, p_ctx, n, npc, pnc, qnc, dir );
    cuda_kernel_sep_f32( &Op->kcgemm, 3, (float)(1.0/(fft_size*fft_size)) );
    n<<=3;
    Op->divpt[0]=n*npc*inc;
    Op->divpt[1]=n*pnc*qnc;
    return (Op->divpt[0]+Op->divpt[1]+n*npc*onc);
}
__local_func size_t idc_fftconv_createOp_grad( idc_fftconvOp_t* Op, const cuda_context_t* p_ctx, int prc, int ng, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int qnx, int qny, int qnc, int ldq, int bat )
{
    int n, dx, dy, grid_x, grid_y, npc, b, fft_size, axis, radix, is_split;
    cuda_kernel_t* p_kernel;
        
    axis=idc_fftconv_choose_optimal_size( pnx, pny, qnx, qny, fnx, fny );
    fft_size=1<<(3+axis);
    dx=fft_size-fnx+1;
    dy=fft_size-fny+1;
    grid_x=(qnx+dx-1)/dx;
    grid_y=(qny+dy-1)/dy;
    npc=bat*grid_x*grid_y;
    if((fft_size<=32)&&((npc&7)==0)) return idc_cellconv_createOp_grad( Op, p_ctx, prc, ng, fnx, fny, pnx, pny, pnc, ldp, qnx, qny, qnc, ldq, bat );
    b=axis<3;
    radix=fft_size>8?8:16;
    is_split=(grid_x|grid_y)>1;
    n=prc==0?2:1;
    pnc/=ng;
    qnc/=ng;
    Op->ng=ng;
    Op->ags=(pnc*ldp)<<n;
    Op->bgs=(qnc*ldq)<<n;
    Op->cgs=(pnc*qnc*fnx*fny)<<n;
    
    {
        int is_ext=(pnx!=fft_size)|(pny!=fft_size);
        p_kernel=&Op->kfft[0];
        idc_create_fft_kernel_r2c( p_kernel, p_ctx, axis, prc, is_split, is_ext, 0, 0 );
        n=pnc*npc;
        cuda_kernel_sgl( p_kernel, b?((n+radix-1)/radix):bat, b?1:pnc, 1 );
        cuda_kernel_sep_i32( p_kernel, 3, pnx );
        cuda_kernel_sep_i32( p_kernel, 4, pny );
        cuda_kernel_sep_i32( p_kernel, 5, ldp );
        if(b){
            cuda_kernel_sep_i32( p_kernel, 6, npc );
            if(is_split){
                cuda_kernel_sep_i32( p_kernel, 7, n      );
                cuda_kernel_sep_i32( p_kernel, 8, grid_x );
                cuda_kernel_sep_i32( p_kernel, 9, grid_y );
                cuda_kernel_sep_i32( p_kernel,10, dx     );
                cuda_kernel_sep_i32( p_kernel,11, dy     );
            }
        }
    }
    
    {
        p_kernel=&Op->kfft[1];
        idc_create_fft_kernel_r2c( p_kernel, p_ctx, axis, prc, is_split, 1, 0, 0 );     
        n=qnc*npc;
        cuda_kernel_sgl( p_kernel, b?((n+radix-1)/radix):bat, b?1:qnc, 1 );     
        cuda_kernel_sep_i32( p_kernel, 3, qnx );
        cuda_kernel_sep_i32( p_kernel, 4, qny );
        cuda_kernel_sep_i32( p_kernel, 5, ldq );
        if(b){
            cuda_kernel_sep_i32( p_kernel, 6, npc );
            if(is_split){
                cuda_kernel_sep_i32( p_kernel, 7, n      );
                cuda_kernel_sep_i32( p_kernel, 8, grid_x );
                cuda_kernel_sep_i32( p_kernel, 9, grid_y );
                cuda_kernel_sep_i32( p_kernel,10, dx     );
                cuda_kernel_sep_i32( p_kernel,11, dy     );
            }
        }
    }
    
    {
        p_kernel=&Op->kfft[2];      
        n=pnc*qnc;
        cuda_kernel_sgl( p_kernel, fft_size<64?((n+radix-1)/radix):n, 1, 1 );
        if(((fft_size>8)|(fft_size<128))&(((fnx==3)&(fny==3))|((fnx==5)&(fny==5)))){
            idc_create_fft_kernel_c2r_grad_opt( p_kernel, p_ctx, axis, prc, fnx>3 );
        } else {
            idc_create_fft_kernel_c2r_grad( p_kernel, p_ctx, axis, prc );       
            cuda_kernel_sep_i32( p_kernel, 4, fnx );
            cuda_kernel_sep_i32( p_kernel, 5, fny );
        }
    }
    
    n=((fft_size>>1)+1)*fft_size+(fft_size>8?0:8);
    idc_flatcgevv_create_kernel( &Op->kcgemm, p_ctx, n, npc, pnc, qnc );
    cuda_kernel_sep_f32( &Op->kcgemm, 3, (float)(1.0/(bat*fft_size*fft_size)) );
    n<<=3;
    Op->divpt[0]=n*npc*pnc;
    Op->divpt[1]=n*npc*qnc;
    return (Op->divpt[0]+Op->divpt[1]+n*pnc*qnc);
}
__local_func void idc_fftconv( idc_fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_x, float alpha, CUstream s )
{
    uint32_t g;
    for( g=0; g<Op->ng; ++g ){
        CUdeviceptr d_a=d_aux;
        CUdeviceptr d_b=d_a+Op->divpt[0];
        CUdeviceptr d_c=d_b+Op->divpt[1];
        idc_fft2d_r2c( &Op->kfft[0], d_a, d_source+g*Op->ags, s );
        idc_fft2d_r2c( &Op->kfft[1], d_b, d_filter+g*Op->bgs, s );
        idc_cgemm( &Op->kcgemm, d_c, d_a, d_b, s );
        idc_fft2d_c2r( &Op->kfft[2], d_target+g*Op->cgs, d_c, d_x, alpha, s );
    }
}
__local_func void idc_fftconv_grad( idc_fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_z, CUdeviceptr d_x, CUdeviceptr d_y, float scale, CUstream s )
{
    uint32_t g;
    for( g=0; g<Op->ng; ++g ){
        CUdeviceptr d_a=d_aux;
        CUdeviceptr d_b=d_a+Op->divpt[0];
        CUdeviceptr d_c=d_b+Op->divpt[1];
        idc_fft2d_r2c( &Op->kfft[0], d_a, d_x+g*Op->ags, s );
        idc_fft2d_r2c( &Op->kfft[1], d_b, d_y+g*Op->bgs, s );
        idc_cgemm( &Op->kcgemm, d_c, d_a, d_b, s );
        idc_fft2d_c2r_grad( &Op->kfft[2], d_z+g*Op->cgs, d_c, scale, s );
    }
}