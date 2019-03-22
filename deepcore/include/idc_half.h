#ifndef __idc_half_h__
#define __idc_half_h__

#include"idc_macro.h"
#include<stdint.h>

INLINE unsigned short idc_float2half( float f )
{
    uint32_t c=*((uint32_t*)&f);
    return (((c>>16)&0x8000)|((((c&0x7f800000)-0x38000000)>>13)&0x7c00)|((c>>13)&0x03ff));
}

#endif