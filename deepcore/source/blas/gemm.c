#include"../../include/blas/gemm.h"
#include"../../include/blas/blasEx.h"

static void sgemm_create_kernel( idc_gemmOp_t* Op, const cuda_context_t* p_ctx, uint32_t mask, int ng, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
    uint32_t axis, dir, fuse, relu, o;
    static const char* s_head[][2] = {{"gemmcc_","gemmcr_"},{"gemmccx_","gemmcrx_"}};
    static const char* s_tile_shape[]={"128x32","128x64","128x128"};
    static const uint8_t block_size[]={127,127,255};
    char kname[64];
    cuda_kernel_t* p_kernel=&Op->kernel;
    axis=(cnc>32)+(cnc>64);
    dir=mask&0x1;
    fuse=(mask>> 3)&0x1;
    relu=(mask>>24)&0x1;
    o=idc_strcat( kname, ((mask>>1)&0x3)==0?"dk_s":"dk_x" );
    o+=idc_strcat( &kname[o], s_head[(axis==2)&&(ng>1)][dir] );
    o+=idc_strcat( &kname[o], s_tile_shape[axis] );
    if(dir==0){
        if(fuse){ o+=idc_strcat( &kname[o], "_bias" ); }
        if(relu){    idc_strcat( &kname[o], "_relu" ); }
    } else {
        if(relu){ idc_strcat( &kname[o], "_drelu" ); } else
        if(fuse){ idc_strcat( &kname[o], "_xdrv"  ); }
    }
    cuda_context_create_kernel( p_kernel, p_ctx->module_blas, kname );
    cuda_kernel_sao( p_kernel, AM_4P_7S );
    cuda_kernel_sbl( p_kernel, block_size[axis]+1, 1 );
    if((axis+=5)==7){
        cuda_kernel_sgl( p_kernel, (anr+127)>>7, (cnc+(1<<axis)-1)>>axis, ng );
    } else {
        cuda_kernel_sgl( p_kernel, (anr+127)>>7, ng, 1 ); 
    }
    cuda_kernel_sep_i32( p_kernel, 5, anr );
    cuda_kernel_sep_i32( p_kernel, 6, bnr );
    cuda_kernel_sep_i32( p_kernel, 7, cnc );
    cuda_kernel_sep_i32( p_kernel, 8, lda );
    cuda_kernel_sep_i32( p_kernel, 9, ldb );
    cuda_kernel_sep_i32( p_kernel,10, ldc );
}
static void sgemmrc_create_kernel( idc_gemmOp_t* Op, const cuda_context_t* p_ctx, uint32_t mask, int ng, int anr, int bnr, int cnc, int ldp, int ldb, int ldq )
{
    uint32_t axis, o;
    static const char* s_tile_shape[]={"32x32","64x64","128x128"};
    static const uint8_t block_size[]={31,127,255};
    char kname[64];
    cuda_kernel_t* p_kernel=&Op->kernel;
    axis=5+(cnc>32)+(cnc>64);
    o=idc_strcat( kname, ((mask>>1)&0x3)==0?"dk_sgemmrc_":"dk_xgemmrc_" );
    idc_strcat( &kname[o], s_tile_shape[axis-5] );
    cuda_context_create_kernel( p_kernel, p_ctx->module_blas, kname );
    cuda_kernel_sao( p_kernel, AM_3P_7S );
    cuda_kernel_sbl( p_kernel, block_size[axis-5]+1, 1 );
    cuda_kernel_sgl( p_kernel, (anr+(1<<axis)-1)>>axis, (cnc+(1<<axis)-1)>>axis, ng );
    cuda_kernel_sep_i32( p_kernel, 4, anr );
    cuda_kernel_sep_i32( p_kernel, 5, bnr );
    cuda_kernel_sep_i32( p_kernel, 6, cnc );
    cuda_kernel_sep_i32( p_kernel, 7, ldp );
    cuda_kernel_sep_i32( p_kernel, 8, ldb );
    cuda_kernel_sep_i32( p_kernel, 9, ldq );
}

__local_func void idc_gemm_createOp( idc_gemmOp_t* Op, const cuda_context_t* p_ctx, uint32_t mask, int ng, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
    sgemm_create_kernel( Op, p_ctx, mask, ng, anr, bnr, cnc, lda, ldb, ldc );
}
__local_func void idc_gemm_createOp_grad( idc_gemmOp_t* Op, const cuda_context_t* p_ctx, uint32_t mask, int ng, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
    sgemmrc_create_kernel( Op, p_ctx, mask, ng, anr, bnr, cnc, lda, ldb, ldc );
}
__local_func void idc_gemm( idc_gemmOp_t* Op, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_x, float alpha, CUstream s )
{
    cuda_kernel_t* p=&Op->kernel;
    cuda_kernel_sep_ptr( p, 0, d_c   );
    cuda_kernel_sep_ptr( p, 1, d_a   );
    cuda_kernel_sep_ptr( p, 2, d_b   );
    cuda_kernel_sep_ptr( p, 3, d_x   );
    cuda_kernel_sep_f32( p, 4, alpha );
    cuda_kernel_launch( p, s );
}
__local_func void idc_gemm_grad( idc_gemmOp_t* Op, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, float scale, CUstream s )
{
    cuda_kernel_t* p=&Op->kernel;
    cuda_kernel_sep_ptr( p, 0, d_c   );
    cuda_kernel_sep_ptr( p, 1, d_a   );
    cuda_kernel_sep_ptr( p, 2, d_b   );
    cuda_kernel_sep_f32( p, 3, scale );
    cuda_kernel_launch( p, s );
}

