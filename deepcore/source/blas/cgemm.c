#include"../../include/blas/cgemm.h"
#include"../../include/blas/blasEx.h"

static void scgemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int bat, int anr, int bnr, int cnc, int lda, int ldb )
{
    uint32_t axis, axis_x, axis_y, nbx, nby, o;
    static const char* s_tile_size[]={ "16", "32", "64", "128", "256" };
    char kname[32];
    axis=idc_get_optimal_cgemm_id( anr, cnc, p_ctx->n_sm, bat );
    axis_x=axis&0xf;
    axis_y=axis>>4;
    o=idc_strcat( kname, "dk_scgemm_" );
    o+=idc_strcat( &kname[o], s_tile_size[axis_x] );
    o+=idc_strcat( &kname[o], "x" );
    idc_strcat( &kname[o], s_tile_size[axis_y+1] );
    axis_x+=4;
    axis_y+=5;
    nbx=(anr+(1<<axis_x)-1)>>axis_x;
    nby=(cnc+(1<<axis_y)-1)>>axis_y;
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P_7S );
    cuda_kernel_sbl( p_kernel, (1<<(axis_x+axis_y))>>4, 1 );
    cuda_kernel_sgl( p_kernel, nbx*nby, bat, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f );
    cuda_kernel_sep_i32( p_kernel, 4, nbx );
    cuda_kernel_sep_i32( p_kernel, 5, anr );
    cuda_kernel_sep_i32( p_kernel, 6, bnr );
    cuda_kernel_sep_i32( p_kernel, 7, cnc );
    cuda_kernel_sep_i32( p_kernel, 8, lda );
    cuda_kernel_sep_i32( p_kernel, 9, ldb );
}
__local_func void idc_cgemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int bat, int anr, int bnr, int cnc, int lda, int ldb )
{
    scgemm_create_kernel( p_kernel, p_ctx, bat, anr, bnr, cnc, lda, ldb );
}
__local_func void idc_cgemm( cuda_kernel_t* p_kernel, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUstream s )
{
    cuda_kernel_sep_ptr( p_kernel, 0, d_c );
    cuda_kernel_sep_ptr( p_kernel, 1, d_a );
    cuda_kernel_sep_ptr( p_kernel, 2, d_b );
    cuda_kernel_launch( p_kernel, s );
}