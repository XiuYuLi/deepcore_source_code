#ifndef __sgemm_base_h__
#define __sgemm_base_h__

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

#define SZERO16(c){\
    (c)[ 0]=0.f;   \
    (c)[ 1]=0.f;   \
    (c)[ 2]=0.f;   \
    (c)[ 3]=0.f;   \
    (c)[ 4]=0.f;   \
    (c)[ 5]=0.f;   \
    (c)[ 6]=0.f;   \
    (c)[ 7]=0.f;   \
    (c)[ 8]=0.f;   \
    (c)[ 9]=0.f;   \
    (c)[10]=0.f;   \
    (c)[11]=0.f;   \
    (c)[12]=0.f;   \
    (c)[13]=0.f;   \
    (c)[14]=0.f;   \
    (c)[15]=0.f;   \
}

#define SZERO32(c){\
    SZERO16(&c[ 0])\
    SZERO16(&c[16])\
}

#define SZERO64(c){\
    SZERO16(&c[ 0])\
    SZERO16(&c[16])\
    SZERO16(&c[32])\
    SZERO16(&c[48])\
}

#include"../activation/activation.h"
#include"sgemm_epilog.h"
#include"../xblas/xgemm_epilog.h"

#endif