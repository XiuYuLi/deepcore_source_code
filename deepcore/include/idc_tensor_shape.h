#ifndef __idc_tensor_shape_h__
#define __idc_tensor_shape_h__

#include"idc_bop.h"
#include"idc_macro.h"

typedef struct idc_tensor_shape{
    uint32_t dix;
    uint32_t ldx;
    uint32_t diy;
} idc_tensor_shape_t;

typedef struct idc_op_param{
    uint32_t prc;
    uint32_t pnx;
    uint32_t pny;
    uint32_t pnc;
    uint32_t ldp;
    uint32_t qnx;
    uint32_t qny;   
    uint32_t qnc;
    uint32_t ldq;
    uint32_t bat;
    uint32_t fnx;
    uint32_t fny;
    uint32_t ldf;
} idc_op_param_t;

INLINE void idc_get_tensor_shape( idc_tensor_shape_t* p_shape, uint64_t shape )
{
    uint32_t tt, prc, s, nx, ny, bt, dix, ldx, diy;
    tt=((uint32_t)(shape>>56))&0x3f;
    prc=(uint32_t)(shape>>62);
    s=prc==0?2:1;
    if(tt==0){
        nx=(((uint32_t)(shape>> 0))&0x03ff)+1;
        ny=(((uint32_t)(shape>>10))&0x03ff)+1;
        bt=(((uint32_t)(shape>>20))&0xffff)+1;
        dix=bt*ny*nx;
        if(dix<=32){
            ldx=idc_minlds(dix);
        } else
        if((dix>32)&&(dix<=48)){
            ldx=IDC_AFFIS(dix,16);
        } else {
            ldx=IDC_AFFIS(dix,32);
        }
        ldx<<=s;
        dix<<=s;
        diy=(((uint32_t)(shape>>36))&0xffff)+1;
    } else
    if(tt==1){
        nx=(((uint32_t)(shape>> 0))&0x001f)+1;
        ny=(((uint32_t)(shape>> 5))&0x001f)+1;
        bt=(((uint32_t)(shape>>10))&0xffff)+1;
        dix=bt*ny*nx;
        ldx=IDC_AFFIS(dix,8);
        ldx<<=s;
        dix<<=s;
        diy=(((uint32_t)(shape>>26))&0xffff)+1;
    } else {
        dix=(uint32_t)(shape);
        ldx=dix;
        diy=1;
    }
    p_shape->ldx=ldx;    
    p_shape->dix=dix;
    p_shape->diy=diy;
}
INLINE void idc_get_op_param( idc_op_param_t* p_param, uint64_t pshape, uint64_t fshape, uint64_t qshape )
{
    uint32_t dix, ldx;
    p_param->prc=  (uint32_t)(pshape>>62);
    p_param->pnx=(((uint32_t)(pshape>> 0))&0x03ff)+1;
    p_param->pny=(((uint32_t)(pshape>>10))&0x03ff)+1;
    p_param->pnc=(((uint32_t)(pshape>>36))&0xffff)+1;
    p_param->fnx=(((uint32_t)(fshape>> 0))&0x001f)+1;
    p_param->fny=(((uint32_t)(fshape>> 5))&0x001f)+1;
    p_param->qnx=(((uint32_t)(qshape>> 0))&0x03ff)+1;
    p_param->qny=(((uint32_t)(qshape>>10))&0x03ff)+1;
    p_param->bat=(((uint32_t)(qshape>>20))&0xffff)+1;
    p_param->qnc=(((uint32_t)(qshape>>36))&0xffff)+1;
    dix=p_param->bat*p_param->pny*p_param->pnx;
    if(dix<=32){
        ldx=idc_minlds(dix);
    } else
    if((dix>32)&&(dix<=48)){
        ldx=IDC_AFFIS(dix,16);
    } else {
        ldx=IDC_AFFIS(dix,32);
    }
    p_param->ldp=ldx;
    dix=p_param->bat*p_param->qny*p_param->qnx;
    if(dix<=32){
        ldx=idc_minlds(dix);
    } else
    if((dix>32)&&(dix<=48)){
        ldx=IDC_AFFIS(dix,16);
    } else {
        ldx=IDC_AFFIS(dix,32);
    }
    p_param->ldq=ldx;
    dix=p_param->pnc*p_param->fny*p_param->fnx;
    p_param->ldf=IDC_AFFIS(dix,8);
}

#endif