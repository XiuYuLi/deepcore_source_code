#include"../../include/conv/fftconv.h"

__local_func size_t idc_cellconv_createOp( idc_fftconvOp_t* Op, const cuda_context_t* p_ctx, uint32_t mask, int ng, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int qnx, int qny, int qnc, int ldq, int bat, int pad_x, int pad_y )
{
    int dir, prc, inc, onc, snx, sny, lds, onx, ony, ldo, dx, dy, grid_x, grid_y, cell_size, n, axis, npc, lda, ldb, is_split, perm_id;
    cuda_kernel_t* p_kernel;
    
    dir=mask&0x1;
    snx=dir?qnx:pnx;
    sny=dir?qny:pny;
    lds=dir?ldq:ldp;
    onx=dir?pnx:qnx;
    ony=dir?pny:qny;
    ldo=dir?ldp:ldq;
    axis=idc_cellconv_choose_optimal_size( onx, ony, fnx, fny );
    cell_size=1<<(3+axis);
    dx=cell_size-fnx+1;
    dy=cell_size-fny+1;
    grid_x=(onx+dx-1)/dx;
    grid_y=(ony+dy-1)/dy;
    npc=bat*grid_x*grid_y;
    if((npc>1)&&(npc<=8))
        return idc_fftconv_createOp( Op, p_ctx, mask, ng, pnx, pny, pnc, ldp, fnx, fny, qnx, qny, qnc, ldq, bat, pad_x, pad_y );
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
    lda=(npc>1)?npc:inc;
    lda=IDC_AFFIS(lda,16);
    ldb=IDC_AFFIS(onc,16);
    is_split=(grid_x|grid_y)>1;
    perm_id=is_split?2:(bat>1);

    {
        int is_pad=(pad_x|pad_y)!=0;
        int is_ext=((snx!=cell_size)|(sny!=cell_size))&(1^is_pad);
        n=npc>1?npc:inc;
        p_kernel=&Op->kfft[0];
        idc_create_cellfft_kernel_r2c( p_kernel, p_ctx, axis, prc, perm_id, is_ext, is_pad, 0 );
        cuda_kernel_sgl( p_kernel, (n+15)>>4, npc>1?inc:1, 1 );
        cuda_kernel_sep_i32( p_kernel, 3, snx );
        cuda_kernel_sep_i32( p_kernel, 4, sny );            
        cuda_kernel_sep_i32( p_kernel, 5, lda );
        cuda_kernel_sep_i32( p_kernel, 6, lds );
        cuda_kernel_sep_i32( p_kernel, 7, n   );
        if(is_split==0){
            if((perm_id>0)&&(is_pad==0)){ cuda_kernel_sep_i32( p_kernel, 8, dir ); }
        } else {
            cuda_kernel_sep_i32( p_kernel, 8, grid_x );
            cuda_kernel_sep_i32( p_kernel, 9, grid_y );
            cuda_kernel_sep_i32( p_kernel,10, dx     );
            cuda_kernel_sep_i32( p_kernel,11, dy     );
            cuda_kernel_sep_i32( p_kernel,12, 0      );
        }   
        if(is_pad){
            cuda_kernel_sep_i32( p_kernel, is_split?13:8, pad_x );
            cuda_kernel_sep_i32( p_kernel, is_split?14:9, pad_y );
        }
    }
    
    {
        int ldr=pnc*fnx*fny;
        int is_opt=(cell_size>8)&(((fnx==3)&(fny==3))|((fnx==5)&(fny==5))|((fnx==7)&(fny==7)));
        p_kernel=&Op->kfft[1];
        if(is_opt){
            idc_create_cellfft_kernel_r2c_opt( p_kernel, p_ctx, axis, prc, dir, (fnx>3)+(fnx>5) );
            cuda_kernel_sep_i32( p_kernel, 3, ldb );
            cuda_kernel_sep_i32( p_kernel, 4, ldr );
        } else {
            idc_create_cellfft_kernel_r2c( p_kernel, p_ctx, axis, prc, 1, 1^dir, 0, dir );
            cuda_kernel_sep_i32( p_kernel, 3, fnx   );
            cuda_kernel_sep_i32( p_kernel, 4, fny   );
            cuda_kernel_sep_i32( p_kernel, 5, ldb   );
            cuda_kernel_sep_i32( p_kernel, 6, ldr   );
            cuda_kernel_sep_i32( p_kernel, 7, onc   );
            cuda_kernel_sep_i32( p_kernel, 8, 1^dir );
        }
        cuda_kernel_sgl( p_kernel, (onc+15)>>4, inc, 1 );
    }
    
    {
        int is_fuse=(mask>> 3)&0x1;
        int is_relu=(mask>>24)&0x1;
        int radix=cell_size>16?8:16;
        int ldc=npc>1?lda:ldb;
        n=npc>1?npc:onc;
        p_kernel=&Op->kfft[2];
        idc_create_cellfft_kernel_c2r( p_kernel, p_ctx, axis, prc, dir, perm_id, is_fuse, is_relu );
        cuda_kernel_sgl( p_kernel, (n+radix-1)/radix, npc>1?onc:1, 1 ); 
        cuda_kernel_sep_i32( p_kernel, 5, onx );
        cuda_kernel_sep_i32( p_kernel, 6, ony );
        cuda_kernel_sep_i32( p_kernel, 7, ldo );
        cuda_kernel_sep_i32( p_kernel, 8, ldc );
        cuda_kernel_sep_i32( p_kernel, 9, n   );
        if(is_split){
            cuda_kernel_sep_i32( p_kernel, 10, grid_x );
            cuda_kernel_sep_i32( p_kernel, 11, grid_y );
            cuda_kernel_sep_i32( p_kernel, 12, dx     );
            cuda_kernel_sep_i32( p_kernel, 13, dy     );
        }
    }
    
    n=((cell_size>>1)+1)*cell_size;
    p_kernel=&Op->kcgemm;
    lda<<=3; ldb<<=3;
    if(npc>1){
        idc_cgemm_create_kernel( p_kernel, p_ctx, n, npc, inc, onc, lda, ldb );
    } else {
        idc_cgemv_create_kernel( p_kernel, p_ctx, n, inc, onc, lda, ldb, ldb );
    }
    cuda_kernel_sep_f32( p_kernel, 3, (float)(1.0/(cell_size*cell_size)) );
    Op->divpt[0]=n*(npc>1?inc:1)*lda;
    Op->divpt[1]=n*inc*ldb;
    return (Op->divpt[0]+Op->divpt[1]+n*(npc>1?(onc*lda):ldb));
}
__local_func size_t idc_cellconv_createOp_grad( idc_fftconvOp_t* Op, const cuda_context_t* p_ctx, int prc, int ng, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int qnx, int qny, int qnc, int ldq, int bat )
{
    int cell_size, n, dx, dy, grid_x, grid_y, npc, lda, ldb, axis, is_split, perm_id;
    cuda_kernel_t* p_kernel;
    
    axis=idc_cellconv_choose_optimal_size( qnx, qny, fnx, fny );
    cell_size=1<<(3+axis);
    dx=cell_size-fnx+1;
    dy=cell_size-fny+1;
    grid_x=(qnx+dx-1)/dx;
    grid_y=(qny+dy-1)/dy;
    npc=bat*grid_x*grid_y;
    if((npc&7)!=0)
        return idc_fftconv_createOp_grad( Op, p_ctx, prc, ng, pnx, pny, pnc, ldp, fnx, fny, qnx, qny, qnc, ldq, bat );
    n=prc==0?2:1;
    pnc/=ng;
    qnc/=ng;
    Op->ng=ng;
    Op->ags=(pnc*ldp)<<n;
    Op->bgs=(qnc*ldq)<<n;
    Op->cgs=(pnc*qnc*fnx*fny)<<n;
    lda=IDC_AFFIS(pnc,16);
    ldb=IDC_AFFIS(qnc,16);
    is_split=(grid_x|grid_y)>1;
    perm_id=is_split?2:1;
    
    {
        int is_ext=(pnx!=cell_size)|(pny!=cell_size);
        p_kernel=&Op->kfft[0];
        idc_create_cellfft_kernel_r2c( p_kernel, p_ctx, axis, prc, perm_id, is_ext, 0, 0 );
        n=is_split?pnc:npc;
        cuda_kernel_sgl( p_kernel, (n+15)>>4, is_split?npc:pnc, 1 );
        cuda_kernel_sep_i32( p_kernel, 3, pnx );
        cuda_kernel_sep_i32( p_kernel, 4, pny );
        cuda_kernel_sep_i32( p_kernel, 5, lda );
        cuda_kernel_sep_i32( p_kernel, 6, ldp );
        cuda_kernel_sep_i32( p_kernel, 7, n   );
        cuda_kernel_sep_i32( p_kernel, 8, 1   );
        if(is_split!=0){
            cuda_kernel_sep_i32( p_kernel, 8, grid_x );
            cuda_kernel_sep_i32( p_kernel, 9, grid_y );
            cuda_kernel_sep_i32( p_kernel,10, dx     );
            cuda_kernel_sep_i32( p_kernel,11, dy     );
            cuda_kernel_sep_i32( p_kernel,12, 1      );
        }
    }
    
    {
        p_kernel=&Op->kfft[1];
        n=is_split?qnc:npc;
        idc_create_cellfft_kernel_r2c( p_kernel, p_ctx, axis, prc, perm_id, 1, 0, 0 );
        cuda_kernel_sgl( p_kernel, (n+15)>>4, is_split?npc:qnc, 1 );        
        cuda_kernel_sep_i32( p_kernel, 3, qnx );
        cuda_kernel_sep_i32( p_kernel, 4, qny );
        cuda_kernel_sep_i32( p_kernel, 5, ldb );
        cuda_kernel_sep_i32( p_kernel, 6, ldq );
        cuda_kernel_sep_i32( p_kernel, 7, n   );
        cuda_kernel_sep_i32( p_kernel, 8, 0   );
        if(is_split){
            cuda_kernel_sep_i32( p_kernel, 8, grid_x );
            cuda_kernel_sep_i32( p_kernel, 9, grid_y );
            cuda_kernel_sep_i32( p_kernel,10, dx     );
            cuda_kernel_sep_i32( p_kernel,11, dy     );
            cuda_kernel_sep_i32( p_kernel,12, 1 );
        }
    }
    
    {
        int radix=cell_size>16?8:16;
        int is_opt=(cell_size>8)&(((fnx==3)&(fny==3))|((fnx==5)&(fny==5)));
        p_kernel=&Op->kfft[2];
        if(is_opt){
            idc_create_cellfft_kernel_c2r_grad_opt( p_kernel, p_ctx, axis, prc, fnx>3 );
        } else {
            idc_create_cellfft_kernel_c2r_grad( p_kernel, p_ctx, axis, prc );
            cuda_kernel_sep_i32( p_kernel, 6, fnx );
            cuda_kernel_sep_i32( p_kernel, 7, fny );
        }
        cuda_kernel_sgl( p_kernel, (pnc+radix-1)/radix, qnc, 1 );
        cuda_kernel_sep_i32( p_kernel, 4, pnc*fnx*fny );
        cuda_kernel_sep_i32( p_kernel, 5, lda         );    
    }
    
    p_kernel=&Op->kcgemm;
    n=((cell_size>>1)+1)*cell_size;
    lda<<=3; ldb<<=3;
    if(npc>1){
        idc_cgemm_create_kernel( p_kernel, p_ctx, n, pnc, npc, qnc, lda, ldb );
    } else {
        idc_cgevv_create_kernel( p_kernel, p_ctx, n, pnc, qnc, lda, ldb );
    }
    cuda_kernel_sep_f32( p_kernel, 3, (float)(1.0/(bat*cell_size*cell_size)) );
    Op->divpt[0]=n*npc*lda;
    Op->divpt[1]=n*npc*ldb;
    return (Op->divpt[0]+Op->divpt[1]+n*lda*qnc);
}