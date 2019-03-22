#include"../../include/blas/cgemm.h"

__local_func void idc_cgevv_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int bat, int nx, int ny, int lda, int ldb )
{
    uint32_t dx=((nx+31)>>5);
    uint32_t dy=((ny+31)>>5);
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, "dk_scgevv" );
    cuda_kernel_sao( p_kernel, AM_3P_7S );
    cuda_kernel_sgl( p_kernel, dx*dy, bat, 1 );
    cuda_kernel_sbl( p_kernel, 32, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f );
    cuda_kernel_sep_i32( p_kernel, 4, nx  );
    cuda_kernel_sep_i32( p_kernel, 5, ny  );
    cuda_kernel_sep_i32( p_kernel, 6, lda );
    cuda_kernel_sep_i32( p_kernel, 7, ldb );
    cuda_kernel_sep_i32( p_kernel, 8, lda );
    cuda_kernel_sep_i32( p_kernel, 9, dx  );
}