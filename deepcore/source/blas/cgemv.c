#include"../../include/blas/cgemm.h"

__local_func void idc_cgemv_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int bat, int nr, int nc, int lda, int ldb, int ldc )
{
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, "dk_scgemv" );
    cuda_kernel_sao( p_kernel, AM_3P_6S );
    cuda_kernel_sgl( p_kernel, (nr+127)>>7, bat, 1 );
    cuda_kernel_sbl( p_kernel, 1<<7, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f );
    cuda_kernel_sep_i32( p_kernel, 4, nr  );
    cuda_kernel_sep_i32( p_kernel, 5, nc  );
    cuda_kernel_sep_i32( p_kernel, 6, lda );
    cuda_kernel_sep_i32( p_kernel, 7, ldb );
    cuda_kernel_sep_i32( p_kernel, 8, ldc );
}



