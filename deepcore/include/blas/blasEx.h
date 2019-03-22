#ifndef __blasEx_h__
#define __blasEx_h__

#include"../idc_macro.h"

__local_func unsigned int idc_get_optimal_sgemm_id( int, int, int );
__local_func unsigned int idc_get_optimal_cgemm_id( int, int, int, int );

#endif