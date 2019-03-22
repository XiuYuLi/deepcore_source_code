
#define CFMA4x4(c,a,b){         \
    c[ 0].x+= (a)[0].x*(b)[0].x;\
    c[ 0].y+= (a)[0].x*(b)[0].y;\
    c[ 0].x+= (a)[0].y*(b)[0].y;\
    c[ 0].y+=-(a)[0].y*(b)[0].x;\
    c[ 1].x+= (a)[0].z*(b)[0].x;\
    c[ 1].y+= (a)[0].z*(b)[0].y;\
    c[ 1].x+= (a)[0].w*(b)[0].y;\
    c[ 1].y+=-(a)[0].w*(b)[0].x;\
    c[ 2].x+= (a)[1].x*(b)[0].x;\
    c[ 2].y+= (a)[1].x*(b)[0].y;\
    c[ 2].x+= (a)[1].y*(b)[0].y;\
    c[ 2].y+=-(a)[1].y*(b)[0].x;\
    c[ 3].x+= (a)[1].z*(b)[0].x;\
    c[ 3].y+= (a)[1].z*(b)[0].y;\
    c[ 3].x+= (a)[1].w*(b)[0].y;\
    c[ 3].y+=-(a)[1].w*(b)[0].x;\
    c[ 4].x+= (a)[0].x*(b)[0].z;\
    c[ 4].y+= (a)[0].x*(b)[0].w;\
    c[ 4].x+= (a)[0].y*(b)[0].w;\
    c[ 4].y+=-(a)[0].y*(b)[0].z;\
    c[ 5].x+= (a)[0].z*(b)[0].z;\
    c[ 5].y+= (a)[0].z*(b)[0].w;\
    c[ 5].x+= (a)[0].w*(b)[0].w;\
    c[ 5].y+=-(a)[0].w*(b)[0].z;\
    c[ 6].x+= (a)[1].x*(b)[0].z;\
    c[ 6].y+= (a)[1].x*(b)[0].w;\
    c[ 6].x+= (a)[1].y*(b)[0].w;\
    c[ 6].y+=-(a)[1].y*(b)[0].z;\
    c[ 7].x+= (a)[1].z*(b)[0].z;\
    c[ 7].y+= (a)[1].z*(b)[0].w;\
    c[ 7].x+= (a)[1].w*(b)[0].w;\
    c[ 7].y+=-(a)[1].w*(b)[0].z;\
    c[ 8].x+= (a)[0].x*(b)[1].x;\
    c[ 8].y+= (a)[0].x*(b)[1].y;\
    c[ 8].x+= (a)[0].y*(b)[1].y;\
    c[ 8].y+=-(a)[0].y*(b)[1].x;\
    c[ 9].x+= (a)[0].z*(b)[1].x;\
    c[ 9].y+= (a)[0].z*(b)[1].y;\
    c[ 9].x+= (a)[0].w*(b)[1].y;\
    c[ 9].y+=-(a)[0].w*(b)[1].x;\
    c[10].x+= (a)[1].x*(b)[1].x;\
    c[10].y+= (a)[1].x*(b)[1].y;\
    c[10].x+= (a)[1].y*(b)[1].y;\
    c[10].y+=-(a)[1].y*(b)[1].x;\
    c[11].x+= (a)[1].z*(b)[1].x;\
    c[11].y+= (a)[1].z*(b)[1].y;\
    c[11].x+= (a)[1].w*(b)[1].y;\
    c[11].y+=-(a)[1].w*(b)[1].x;\
    c[12].x+= (a)[0].x*(b)[1].z;\
    c[12].y+= (a)[0].x*(b)[1].w;\
    c[12].x+= (a)[0].y*(b)[1].w;\
    c[12].y+=-(a)[0].y*(b)[1].z;\
    c[13].x+= (a)[0].z*(b)[1].z;\
    c[13].y+= (a)[0].z*(b)[1].w;\
    c[13].x+= (a)[0].w*(b)[1].w;\
    c[13].y+=-(a)[0].w*(b)[1].z;\
    c[14].x+= (a)[1].x*(b)[1].z;\
    c[14].y+= (a)[1].x*(b)[1].w;\
    c[14].x+= (a)[1].y*(b)[1].w;\
    c[14].y+=-(a)[1].y*(b)[1].z;\
    c[15].x+= (a)[1].z*(b)[1].z;\
    c[15].y+= (a)[1].z*(b)[1].w;\
    c[15].x+= (a)[1].w*(b)[1].w;\
    c[15].y+=-(a)[1].w*(b)[1].z;\
}

#define CFMA(c,a,b){    \
    (c).x+= (a).x*(b).x;\
    (c).y+= (a).x*(b).y;\
    (c).x+= (a).y*(b).y;\
    (c).y+=-(a).y*(b).x;\
}

#define CFMA8(c,a,b){  \
    CFMA((c)[0],a[0],b)\
    CFMA((c)[1],a[1],b)\
    CFMA((c)[2],a[2],b)\
    CFMA((c)[3],a[3],b)\
    CFMA((c)[4],a[4],b)\
    CFMA((c)[5],a[5],b)\
    CFMA((c)[6],a[6],b)\
    CFMA((c)[7],a[7],b)\
}

#define CFMA8x8(c,a,b){  \
    CFMA8(&c[0*8],a,b[0])\
    CFMA8(&c[1*8],a,b[1])\
    CFMA8(&c[2*8],a,b[2])\
    CFMA8(&c[3*8],a,b[3])\
    CFMA8(&c[4*8],a,b[4])\
    CFMA8(&c[5*8],a,b[5])\
    CFMA8(&c[6*8],a,b[6])\
    CFMA8(&c[7*8],a,b[7])\
}

#include"scgemm_epilog.h"
#include"scgemm_016x032.h"
#include"scgemm_016x064.h"
#include"scgemm_016x128.h"
#include"scgemm_016x256.h"
#include"scgemm_032x032.h"
#include"scgemm_032x064.h"
#include"scgemm_032x128.h"
#include"scgemm_064x032.h"
#include"scgemm_064x064.h"
#include"scgemm_128x032.h"
#include"scgemv.h"
#include"scgevv.h"
#include"sflatcgemm_r01.h"
#include"sflatcgemm_r02.h"
#include"sflatcgemm_r04.h"
#include"sflatcgemm_r08.h"
#include"sflatcgemm_r16.h"
#include"sflatcgemm_r32.h"
#include"sflatcgevv_16x32.h"
#include"sflatcgevv_16x64.h"
#include"sflatcgevv_32x32.h"