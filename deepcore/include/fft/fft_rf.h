#ifndef __idc_fft_calcRF_h__
#define __idc_fft_calcRF_h__

#include<math.h>
#include<vector_types.h>
#include"../idc_macro.h"

#define PI 3.1415926535897931e+0

INLINE void idc_fft_calcRF( float2* p, int n, double rt )
{
    int i=0;
    do{ 
        p[i].x=(float)cos(i*rt*-PI);
        p[i].y=(float)sin(i*rt*-PI);
    }while((++i)<n);
}

#endif