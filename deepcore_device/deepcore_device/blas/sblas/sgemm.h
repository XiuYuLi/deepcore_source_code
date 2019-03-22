#ifndef __sgemm_h__
#define __sgemm_h__

#define BOP4x4(c,a,b){   \
    (c)[ 0]+=(a).x*(b).x;\
    (c)[ 1]+=(a).y*(b).x;\
    (c)[ 2]+=(a).z*(b).x;\
    (c)[ 3]+=(a).w*(b).x;\
    (c)[ 4]+=(a).x*(b).y;\
    (c)[ 5]+=(a).y*(b).y;\
    (c)[ 6]+=(a).z*(b).y;\
    (c)[ 7]+=(a).w*(b).y;\
    (c)[ 8]+=(a).x*(b).z;\
    (c)[ 9]+=(a).y*(b).z;\
    (c)[10]+=(a).z*(b).z;\
    (c)[11]+=(a).w*(b).z;\
    (c)[12]+=(a).x*(b).w;\
    (c)[13]+=(a).y*(b).w;\
    (c)[14]+=(a).z*(b).w;\
    (c)[15]+=(a).w*(b).w;\
}

#define BOP8x8(c,a,b){            \
    BOP4x4(&c[0*16],(a)[0],(b)[0])\
    BOP4x4(&c[1*16],(a)[1],(b)[0])\
    BOP4x4(&c[2*16],(a)[0],(b)[1])\
    BOP4x4(&c[3*16],(a)[1],(b)[1])\
}

#define BOP8x4(c,a,b){            \
    BOP4x4(&c[0*16],(a)[0],(b)[0])\
    BOP4x4(&c[1*16],(a)[1],(b)[0])\
}

#define BOP4x8(c,a,b){            \
    BOP4x4(&c[0*16],(a)[0],(b)[0])\
    BOP4x4(&c[1*16],(a)[0],(b)[1])\
}

#include"sgemm_base.h"
#include"sgemmcc_128x128.h"
#include"sgemmcc_128x064.h"
#include"sgemmcc_128x032.h"
#include"sgemmcr_128x128.h"
#include"sgemmcr_128x064.h"
#include"sgemmcr_128x032.h"
#include"sgemmrc_032x032.h"
#include"sgemmrc_064x064.h"
#include"sgemmrc_128x128.h"
#include"../xblas/xgemm.h"

#endif