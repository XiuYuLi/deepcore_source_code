#include"../../include/blas/cgemm.h"

static void sflatcgevv_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int slice_size, int bat, int pnc, int qnc )
{
    static const char* knames[]={ "dk_sflatcgevv_16x32", "dk_sflatcgevv_16x64", "dk_sflatcgevv_32x32" };
    int x, y, i, nx, ny;
    x=((pnc>16)&(pnc<=32))|(pnc>48);
    y=((qnc<=32)|((qnc>64)&(qnc<=96)))?0:(x==0);
    i=(x<<1)|y;
    nx=1<<(4+x);
    ny=1<<(5+y);
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, knames[i] );
    cuda_kernel_sao( p_kernel, AM_3P_5S );
    cuda_kernel_sbl( p_kernel, i>0?256:128, 1 );
    cuda_kernel_sgl( p_kernel, ((pnc+nx-1)/nx)*((qnc+ny-1)/ny), slice_size>>4, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f        );
    cuda_kernel_sep_i32( p_kernel, 4, slice_size );
    cuda_kernel_sep_i32( p_kernel, 5, bat        );
    cuda_kernel_sep_i32( p_kernel, 6, pnc        );
    cuda_kernel_sep_i32( p_kernel, 7, qnc        );
}
__local_func void idc_flatcgevv_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int slice_size, int bat, int pnc, int qnc )
{
    sflatcgevv_create_kernel( p_kernel, p_ctx, slice_size, bat, pnc, qnc );
}