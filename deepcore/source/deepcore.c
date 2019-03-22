#pragma warning(disable:4101)
#include"../include/deepcore.h"
#include"../include/idc_tensor_shape.h"
#include"../include/cuda/cuda_platform.h"
#include"../include/conv/fftconv.h"
#include"../include/blas/gemm.h"
#pragma comment( lib, "cuda.lib" )
#define as_devptr(p) (CUdeviceptr)((uintptr_t)(p))

static cuda_platform_t* g_pPlat=NULL;
static cuda_context_t * g_pCtx =NULL;
static char           * g_pTemp=NULL;

DEEPCOREAPIENTRY dc_status_t dc_init()
{
    int i, s, status;
    if((g_pPlat=(cuda_platform_t*)malloc(sizeof(cuda_platform_t)))==NULL)
        return dc_error_out_of_memory;
    if((status=cuda_platform_init(g_pPlat))!=dc_success){
        free(g_pPlat); g_pPlat=NULL;
        return (dc_status_t)status;
    }
    if((g_pCtx=(cuda_context_t*)calloc(g_pPlat->n_devices,sizeof(cuda_context_t)))==NULL){
        free(g_pPlat); g_pPlat=NULL;
        return dc_error_out_of_memory;
    }
    if((g_pTemp=(char*)malloc(1<<24))==0){
        free(g_pPlat); g_pPlat=NULL;
        return dc_error_out_of_memory;
    }
    for( i=0; (i<g_pPlat->n_devices)&(status==dc_success); ++i ){
        g_pCtx[i].arch=g_pPlat->arch[i];
        status=cuda_context_create(&g_pCtx[i],g_pTemp);
    }
    if(status!=dc_success){
        while(i>=0){ cuda_context_release( &g_pCtx[i] ); }
        free(g_pTemp);
        free(g_pCtx);
        free(g_pPlat);
        g_pPlat=NULL;
    }
    return (dc_status_t)status;
}
DEEPCOREAPIENTRY int dc_get_device_count(){ return g_pPlat->n_devices; }
DEEPCOREAPIENTRY dc_status_t dc_set_device( int dev )
{
    if((dev<0)|(dev>=g_pPlat->n_devices)) return dc_error_out_of_range;
    cuda_context_bind(&g_pCtx[dev]);
    return dc_success;
}
DEEPCOREAPIENTRY dc_tensorshape_t dc_create_tensor_shape( int prc, uint32_t size, uint32_t ncbt )
{
    dc_tensorshape_t nx=(dc_tensorshape_t)((size>> 0)&0xffff);
    dc_tensorshape_t ny=(dc_tensorshape_t)((size>>16)&0xffff);
    dc_tensorshape_t bt=(dc_tensorshape_t)((ncbt>> 0)&0xffff);
    dc_tensorshape_t nc=(dc_tensorshape_t)((ncbt>>16)&0xffff);
    return ((((dc_tensorshape_t)prc)<<62)|((nc-1)<<36)|((bt-1)<<20)|((ny-1)<<10)|(nx-1));	
}
DEEPCOREAPIENTRY dc_tensorshape_t dc_create_tensor_shape_filter( int prc, uint32_t size, uint32_t chan )
{
    dc_tensorshape_t nx =(dc_tensorshape_t)((size>> 0)&0xffff);
    dc_tensorshape_t ny =(dc_tensorshape_t)((size>>16)&0xffff);
    dc_tensorshape_t pnc=(dc_tensorshape_t)((chan>> 0)&0xffff);
    dc_tensorshape_t qnc=(dc_tensorshape_t)((chan>>16)&0xffff);
    return ((((dc_tensorshape_t)prc)<<62)|0x0100000000000000L|((qnc-1)<<26)|((pnc-1)<<10)|((ny-1)<<5)|(nx-1));	
}
DEEPCOREAPIENTRY dc_tensorshape_t dc_create_tensor_shape_linear( size_t nb )
{
    return (0x0200000000000000L|((dc_tensorshape_t)nb));
}
DEEPCOREAPIENTRY dc_status_t dc_create_tensor( void** p_devptr, dc_tensorshape_t shape )
{
    CUdeviceptr devptr;
    uint32_t tt, prc, nx, ny, bt, nc, size, pitch, ext, enb;
    tt=((uint32_t)(shape>>56))&0x3f;
    prc=((uint32_t)(shape>>62))&0x3;
    if(tt==0){
        nx=(((uint32_t)(shape>> 0))&0x03ff)+1;
        ny=(((uint32_t)(shape>>10))&0x03ff)+1;
        bt=(((uint32_t)(shape>>20))&0xffff)+1;
        nc=(((uint32_t)(shape>>36))&0xffff)+1;
        size=bt*ny*nx;
        if(size<=32){
            pitch=idc_minlds(size);
        } else
        if((size>32)&&(size<=48)){
            pitch=IDC_AFFIS(size,16);
        } else {
            pitch=IDC_AFFIS(size,32);
        }
        ext=((nc&7)!=0)&(size==pitch);
        size=pitch*nc+ext;
    } else
    if(tt==1){
        nx=(((uint32_t)(shape>> 0))&0x001f)+1;
        ny=(((uint32_t)(shape>> 5))&0x001f)+1;
        bt=(((uint32_t)(shape>>10))&0xffff)+1;
        nc=(((uint32_t)(shape>>26))&0xffff)+1;
        size=bt*ny*nx;
        pitch=IDC_AFFIS(size,8);
        size=nc*pitch;
    } else {
        size=(uint32_t)(shape);
    }
    enb=prc==0?4:2;
    if(cuMemAlloc( &devptr, size*enb )!=CUDA_SUCCESS)
        return dc_error_out_of_device_memory;
    *p_devptr=(void*)devptr;
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_release_tensor( void* p_devptr )
{
    if(p_devptr!=0){ cuMemFree(as_devptr(p_devptr)); }
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_tensor_zero( void* p, dc_tensorshape_t shape, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape;
    idc_get_tensor_shape( &ishape, shape );
    if(ishape.diy>1){
        cuMemsetD2D8Async( as_devptr(p), ishape.ldx, 0, ishape.dix, ishape.diy, s );
    } else {
        cuMemsetD8Async( as_devptr(p), 0, ishape.dix, s );
    }
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_tensor_subzero( void* p, dc_tensorshape_t shape, size_t xnb, size_t y, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape;
    idc_get_tensor_shape( &ishape, shape );
    if((xnb>ishape.dix)||(y>ishape.diy)) 
        return dc_error_invalid_value;
    if(y>1){
        cuMemsetD2D8Async( as_devptr(p), ishape.ldx, 0, xnb, y, s );
    } else {
        cuMemsetD8Async( as_devptr(p), 0, xnb, s );
    }
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_tensor_copy( void* p_dst, dc_tensorshape_t shape_dst, const void* p_src, dc_tensorshape_t shape_src, size_t xnb, size_t diy, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape_dst, ishape_src;
    idc_get_tensor_shape( &ishape_dst, shape_dst );
    idc_get_tensor_shape( &ishape_src, shape_src );
    if((ishape_src.dix<xnb)|(ishape_dst.dix<xnb)|(ishape_src.diy<diy)|(ishape_dst.diy<diy))
        return dc_error_out_of_range;
    mem2d.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.dstDevice    =as_devptr(p_dst);
    mem2d.dstPitch     =ishape_dst.ldx;	
    mem2d.dstXInBytes  =0;
    mem2d.dstY         =0;
    mem2d.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.srcDevice    =as_devptr(p_src);
    mem2d.srcPitch     =ishape_src.ldx;
    mem2d.srcXInBytes  =0;
    mem2d.srcY         =0;
    mem2d.WidthInBytes =xnb;
    mem2d.Height       =diy;
    cuMemcpy2DAsync( &mem2d, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_tensor_subcopy( void* p_dst, dc_tensorshape_t shape_dst, const void* p_src, dc_tensorshape_t shape_src, size_t xnb, size_t diy, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape_dst, ishape_src;
    idc_get_tensor_shape( &ishape_dst, shape_dst );
    idc_get_tensor_shape( &ishape_src, shape_src );
    if((ishape_src.dix<xnb)|(ishape_dst.dix<xnb)|(ishape_src.diy<diy)|(ishape_dst.diy<diy))
        return dc_error_invalid_value;
    mem2d.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.dstDevice    =as_devptr(p_dst);
    mem2d.dstPitch     =ishape_dst.ldx;	
    mem2d.dstXInBytes  =0;
    mem2d.dstY         =0;
    mem2d.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.srcDevice    =as_devptr(p_src);
    mem2d.srcPitch     =ishape_src.ldx;
    mem2d.srcXInBytes  =0;
    mem2d.srcY         =0;
    mem2d.WidthInBytes =xnb;
    mem2d.Height       =diy;
    cuMemcpy2DAsync( &mem2d, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_tensor_store( void* p_dst, dc_tensorshape_t shape, const void* p_src, size_t src_pitch, size_t xnb, size_t diy, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape;
    idc_get_tensor_shape( &ishape, shape );
    if((ishape.dix<xnb)|(ishape.diy<diy))
        return dc_error_out_of_range;
    mem2d.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.dstDevice    =as_devptr(p_dst);
    mem2d.dstPitch     =ishape.ldx;	
    mem2d.dstXInBytes  =0;
    mem2d.dstY         =0;
    mem2d.srcMemoryType=CU_MEMORYTYPE_HOST;
    mem2d.srcHost      =p_src;
    mem2d.srcPitch     =src_pitch;	
    mem2d.srcXInBytes  =0;
    mem2d.srcY         =0;
    mem2d.WidthInBytes =xnb;
    mem2d.Height       =diy;
    cuMemcpy2DAsync( &mem2d, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_tensor_load( void* p_dst, size_t dst_pitch, const void* p_src, dc_tensorshape_t shape, size_t xnb, size_t diy, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape;
    idc_get_tensor_shape( &ishape, shape );
    if((ishape.dix<xnb)|(ishape.diy<diy))
        return dc_error_out_of_range;
    mem2d.dstMemoryType=CU_MEMORYTYPE_HOST;
    mem2d.dstHost      =p_dst;
    mem2d.dstPitch     =dst_pitch;
    mem2d.dstXInBytes  =0;
    mem2d.dstY         =0;
    mem2d.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.srcDevice    =as_devptr(p_src);
    mem2d.srcPitch     =ishape.ldx;	
    mem2d.srcXInBytes  =0;
    mem2d.srcY         =0;
    mem2d.WidthInBytes =xnb;
    mem2d.Height       =diy;
    cuMemcpy2DAsync( &mem2d, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_create_fftconvOp( dc_fftconvOp* Op, size_t* p_auxsize, uint32_t mask, int ng, dc_tensorshape_t pshape, dc_tensorshape_t fshape, dc_tensorshape_t qshape, uint32_t pad )
{
    idc_op_param_t param;
    int dir, idev, anx, any, bnx, bny, pu, pv;
    idc_get_op_param( &param, pshape, fshape, qshape );
    dir=mask&0x1;
    if((dir!=0)&&(((mask&(dcMaskActivationRelu|dcMaskMulDrv))^(dcMaskActivationRelu|dcMaskMulDrv)))==0)
        return dc_error_mutually_exclusive;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    pu=(pad>>0)&0xff;
    pv=(pad>>8)&0xff;
    if(dir!=0){
        pu=param.fnx-pu-1;
        pv=param.fny-pv-1;
    }
    anx=(dir!=0?param.qnx:param.pnx)+(pu<<1);
    any=(dir!=0?param.qny:param.pny)+(pv<<1);
    bnx=dir!=0?param.pnx:param.qnx;
    bny=dir!=0?param.pny:param.qny;
    if(((anx-param.fnx+1)!=bnx)|((any-param.fny+1)!=bny))
        return dc_error_mismatch;
    if((*Op=(dc_fftconvOp)malloc(sizeof(idc_fftconvOp_t)))==NULL)
        return dc_error_out_of_memory;
    *p_auxsize=idc_fftconv_createOp( (idc_fftconvOp_t*)(*Op), &g_pCtx[idev], mask, ng, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.qnx, param.qny, param.qnc, param.ldq, param.bat, pu, pv );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_create_fftconvOp_grad( dc_fftconvOp* Op, size_t* p_auxsize, uint32_t mask, int ng, dc_tensorshape_t pshape, dc_tensorshape_t fshape, dc_tensorshape_t qshape )
{
    idc_op_param_t param;
    int idev;
    idc_get_op_param( &param, pshape, fshape, qshape );
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    if(((param.pnx-param.qnx+1)!=param.fnx)|((param.pny-param.qny+1)!=param.fny)) 
        return dc_error_mismatch;
    if((*Op=(dc_fftconvOp)malloc(sizeof(idc_fftconvOp_t)))==NULL)
        return dc_error_out_of_memory;
    *p_auxsize=idc_fftconv_createOp_grad( (idc_fftconvOp_t*)(*Op), &g_pCtx[idev], param.prc, ng, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.qnx, param.qny, param.qnc, param.ldq, param.bat );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_create_cellconvOp( dc_cellconvOp* Op, size_t* p_auxsize, uint32_t mask, int ng, dc_tensorshape_t pshape, dc_tensorshape_t fshape, dc_tensorshape_t qshape, uint32_t pad )
{
    idc_op_param_t param;
    int dir, idev, anx, any, bnx, bny, pu, pv;    
    idc_get_op_param( &param, pshape, fshape, qshape );
    dir=mask&0x1;
    if((dir!=0)&&(((mask&(dcMaskActivationRelu|dcMaskMulDrv))^(dcMaskActivationRelu|dcMaskMulDrv)))==0)
        return dc_error_mutually_exclusive;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    pu=(pad>>0)&0xff;
    pv=(pad>>8)&0xff;
    if(dir!=0){
        pu=param.fnx-pu-1;
        pv=param.fny-pv-1;
    }
    anx=(dir!=0?param.qnx:param.pnx)+(pu<<1);
    any=(dir!=0?param.qny:param.pny)+(pv<<1);
    bnx=dir!=0?param.pnx:param.qnx;
    bny=dir!=0?param.pny:param.qny;
    if(((anx-param.fnx+1)!=bnx)|((any-param.fny+1)!=bny))
        return dc_error_mismatch;
    if((*Op=(dc_cellconvOp)malloc(sizeof(idc_fftconvOp_t)))==NULL)
        return dc_error_out_of_memory;
    *p_auxsize=idc_cellconv_createOp( (idc_fftconvOp_t*)(*Op), &g_pCtx[idev], (param.prc<<1)|mask, ng, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.qnx, param.qny, param.qnc, param.ldq, param.bat, pu, pv );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_create_cellconvOp_grad( dc_cellconvOp* Op, size_t* p_auxsize, uint32_t mask, int ng, dc_tensorshape_t pshape, dc_tensorshape_t fshape, dc_tensorshape_t qshape )
{
    idc_op_param_t param;
    int idev;    
    idc_get_op_param( &param, pshape, fshape, qshape );    
    if(((param.pnx-param.qnx+1)!=param.fnx)|((param.pny-param.qny+1)!=param.fny))
        return dc_error_mismatch;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    if((*Op=(dc_cellconvOp)malloc(sizeof(idc_fftconvOp_t)))==0)
        return dc_error_out_of_memory;
    *p_auxsize=idc_cellconv_createOp_grad( (idc_fftconvOp_t*)(*Op), &g_pCtx[idev], param.prc, ng, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.qnx, param.qny, param.qnc, param.ldq, param.bat );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_create_gemmOp( dc_gemmOp* Op, uint32_t mask, int ng, dc_tensorshape_t ashape, dc_tensorshape_t bshape, dc_tensorshape_t cshape )
{
    idc_op_param_t param;
    int idev, anr, lda, ldb, s;    
    idc_get_op_param( &param, ashape, bshape, cshape );
    if((param.pnx!=param.qnx)|(param.pny!=param.qny))
        return dc_error_mismatch;
    if(((mask&0x1)!=0)&&(((mask&(dcMaskActivationRelu|dcMaskMulDrv))^(dcMaskActivationRelu|dcMaskMulDrv)))==0)
        return dc_error_mutually_exclusive;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;      
    if((*Op=(dc_gemmOp)malloc(sizeof(idc_gemmOp_t)))==NULL)
        return dc_error_out_of_memory;
    s=param.prc==0?2:1;
    anr=param.bat*param.pnx*param.pny;
    ldb=param.ldf<<s;
    lda=param.ldp<<s;
    idc_gemm_createOp( (idc_gemmOp_t*)(*Op), &g_pCtx[idev], (param.prc<<1)|mask, ng, anr, param.pnc/ng, param.qnc/ng, lda, ldb, lda );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_create_gemmOp_grad( dc_gemmOp* Op, uint32_t mask, int ng, dc_tensorshape_t ashape, dc_tensorshape_t bshape, dc_tensorshape_t cshape )
{
    idc_op_param_t param;
    int idev, n, s;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    idc_get_op_param( &param, ashape, bshape, cshape );
    if((param.pnx!=param.qnx)|(param.pny!=param.qny))
        return dc_error_mismatch;     
    if((*Op=(dc_gemmOp)malloc(sizeof(idc_gemmOp_t)))==NULL)
        return dc_error_out_of_memory;
    n=param.bat*param.pny*param.pnx;
    s=param.prc==0?2:1;
    ((idc_gemmOp_t*)(*Op))->ldx=param.ldp<<s;
    ((idc_gemmOp_t*)(*Op))->dix=n<<s;
    ((idc_gemmOp_t*)(*Op))->diy=(s<<16)|(param.pnc-1);
    idc_gemm_createOp_grad( (idc_gemmOp_t*)(*Op), &g_pCtx[idev], (param.prc<<1)|mask, ng, param.pnc/ng, n, param.qnc/ng, param.ldp<<s, param.ldq<<s, param.ldf<<s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_fftconv( dc_fftconvOp Op, void* d_aux, void* d_dst, const void* d_src, const void* d_filter, const void* d_x, float alpha, CUstream s )
{
    idc_fftconv( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_x), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_fftconv_grad( dc_fftconvOp Op, void* d_aux, void* d_grad, const void* d_p, const void* d_q, float scale, CUstream s )
{
    idc_fftconv_grad( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_grad), as_devptr(d_p), as_devptr(d_q), scale, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_cellconv( dc_cellconvOp Op, void* d_aux, void* d_dst, const void* d_src, const void* d_filter, const void* d_x, float alpha, CUstream s )
{
    idc_fftconv( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_x), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_cellconv_grad( dc_cellconvOp Op, void* d_aux, void* d_grad, const void* d_p, const void* d_q, float scale, CUstream s )
{
    idc_fftconv_grad( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_grad), as_devptr(d_p), as_devptr(d_q), scale, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_gemm( dc_gemmOp Op, void* d_c, const void* d_a, const void* d_b, const void* d_x, float alpha, CUstream s )
{
    idc_gemm( (idc_gemmOp_t*)Op, as_devptr(d_c), as_devptr(d_a), as_devptr(d_b), as_devptr(d_x), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_gemm_grad( dc_gemmOp Op, void* d_c, const void* d_a, const void* d_b, float scale, CUstream s )
{
    idc_gemmOp_t* iOp=(idc_gemmOp_t*)Op;
    uint32_t ldx=iOp->ldx;
    uint32_t dix=iOp->dix;
    uint32_t diy=iOp->diy;
    uint32_t align=8<<(diy>>16);
    if((dix&(align-1))!=0){
        cuMemsetD2D8Async( as_devptr(d_a)+dix, ldx, 0, IDC_AFFIS(dix,align)-dix, (diy&0xffff)+1, s );
    }
    idc_gemm_grad( iOp, as_devptr(d_c), as_devptr(d_a), as_devptr(d_b), scale, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_destroy_fftconvOp( dc_fftconvOp Op )
{   
    if(((void*)Op)!=NULL){ free((void*)Op); }
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_destroy_cellconvOp( dc_cellconvOp Op )
{
    if(((void*)Op)!=NULL){ free((void*)Op); }
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_destroy_gemmOp( dc_gemmOp Op )
{
    if(((void*)Op)!=NULL){ free((void*)Op); }
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t dc_exit()
{
    if(g_pPlat!=NULL){
        int i;
        for( i=0; i<g_pPlat->n_devices; ++i ){ 
            cuda_context_release( &g_pCtx[i] ); 
        }
        free(g_pPlat); g_pPlat=NULL;
        free(g_pCtx );
        free(g_pTemp);
    }
    return dc_success;
}
