#ifndef __sfft_h__
#define __sfft_h__

#define BFLYU(a,b,c,s){         \
    temp.x=((c)/(s))*b.x+(-b.y);\
    temp.y=((c)/(s))*b.y+( b.x);\
    b.x=a.x+(-(s))*temp.x;      \
    b.y=a.y+(-(s))*temp.y;      \
    a.x+=(s)*temp.x;            \
    a.y+=(s)*temp.y;            \
}

#define iBFLYU(a,b,c,s){        \
    temp.x=((c)/(s))*b.x+( b.y);\
    temp.y=((c)/(s))*b.y+(-b.x);\
    b.x=a.x+(-(s))*temp.x;      \
    b.y=a.y+(-(s))*temp.y;      \
    a.x+=(s)*temp.x;            \
    a.y+=(s)*temp.y;            \
}

#define BFLYU10(a,b){\
    temp.x=b.x;      \
    temp.y=b.y;      \
    b.x=a.x-b.x;     \
    b.y=a.y-b.y;     \
    a.x+=temp.x;     \
    a.y+=temp.y;     \
}

#define BFLYU01(a,b){\
    temp.x=b.y;      \
    temp.y=b.x;      \
    b.x=a.x-temp.x;  \
    b.y=a.y+temp.y;  \
    a.x+=temp.x;     \
    a.y-=temp.y;     \
}

#define iBFLYU01(a,b){\
    temp.x=b.y;       \
    temp.y=b.x;       \
    b.x=a.x+temp.x;   \
    b.y=a.y-temp.y;   \
    a.x-=temp.x;      \
    a.y+=temp.y;      \
}

#define FFT4(c,dir){           \
    BFLYU10((c)[0],(c)[2])     \
    BFLYU10((c)[1],(c)[3])     \
    BFLYU10((c)[0],(c)[1])     \
    dir##BFLYU01((c)[2],(c)[3])\
}

#define FFT8(c,dir){                                      \
    BFLYU10((c)[0],(c)[4])                                \
    BFLYU10((c)[1],(c)[5])                                \
    BFLYU10((c)[2],(c)[6])                                \
    BFLYU10((c)[3],(c)[7])                                \
    BFLYU10((c)[0],(c)[2])                                \
    BFLYU10((c)[1],(c)[3])                                \
    dir##BFLYU01((c)[4],(c)[6])                           \
    dir##BFLYU01((c)[5],(c)[7])                           \
    BFLYU10((c)[0],(c)[1])                                \
    dir##BFLYU01((c)[2],(c)[3])                           \
    dir##BFLYU((c)[4],(c)[5], 0.707106781f,-0.707106781f);\
    dir##BFLYU((c)[6],(c)[7],-0.707106781f,-0.707106781f);\
}

#define FFT8_M1(c,dir){\
    (c)[4]=(c)[0];     \
    (c)[2]=(c)[0];     \
    (c)[6]=(c)[4];     \
    (c)[1]=(c)[0];     \
    (c)[3]=(c)[2];     \
    (c)[5]=(c)[4];     \
    (c)[7]=(c)[6];     \
}

#define FFT8_M2(c,dir){                                   \
    (c)[4]=(c)[0];                                        \
    (c)[5]=(c)[1];                                        \
    (c)[2]=(c)[0];                                        \
    (c)[3]=(c)[1];                                        \
    (c)[6]=(c)[4];                                        \
    (c)[7]=(c)[5];                                        \
    BFLYU10((c)[0],(c)[1])                                \
    dir##BFLYU01((c)[2],(c)[3])                           \
    dir##BFLYU((c)[4],(c)[5], 0.707106781f,-0.707106781f);\
    dir##BFLYU((c)[6],(c)[7],-0.707106781f,-0.707106781f);\
}

#define FFT8_M3(c,dir){                                   \
    (c)[4]=(c)[0];                                        \
    (c)[5]=(c)[1];                                        \
    (c)[6]=(c)[2];                                        \
    (c)[7]=(c)[5];                                        \
    (c)[3]=(c)[1];                                        \
    BFLYU10((c)[0],(c)[2])                                \
    dir##BFLYU01((c)[4],(c)[6])                           \
    BFLYU10((c)[0],(c)[1])                                \
    dir##BFLYU01((c)[2],(c)[3])                           \
    dir##BFLYU((c)[4],(c)[5], 0.707106781f,-0.707106781f);\
    dir##BFLYU((c)[6],(c)[7],-0.707106781f,-0.707106781f);\
}

#define FFT8_M4(c,dir){                                   \
    (c)[4]=(c)[0];                                        \
    (c)[5]=(c)[1];                                        \
    (c)[6]=(c)[2];                                        \
    (c)[7]=(c)[3];                                        \
    BFLYU10((c)[0],(c)[2])                                \
    BFLYU10((c)[1],(c)[3])                                \
    dir##BFLYU01((c)[4],(c)[6])                           \
    dir##BFLYU01((c)[5],(c)[7])                           \
    BFLYU10((c)[0],(c)[1])                                \
    dir##BFLYU01((c)[2],(c)[3])                           \
    dir##BFLYU((c)[4],(c)[5], 0.707106781f,-0.707106781f);\
    dir##BFLYU((c)[6],(c)[7],-0.707106781f,-0.707106781f);\
}

#define FFT16(c,dir){                                      \
    BFLYU10((c)[0],(c)[ 8])                                \
    BFLYU10((c)[1],(c)[ 9])                                \
    BFLYU10((c)[2],(c)[10])                                \
    BFLYU10((c)[3],(c)[11])                                \
    BFLYU10((c)[4],(c)[12])                                \
    BFLYU10((c)[5],(c)[13])                                \
    BFLYU10((c)[6],(c)[14])                                \
    BFLYU10((c)[7],(c)[15])                                \
                                                           \
    BFLYU10((c)[0],(c)[4])                                 \
    BFLYU10((c)[1],(c)[5])                                 \
    BFLYU10((c)[2],(c)[6])                                 \
    BFLYU10((c)[3],(c)[7])                                 \
    dir##BFLYU01((c)[ 8],(c)[12])                          \
    dir##BFLYU01((c)[ 9],(c)[13])                          \
    dir##BFLYU01((c)[10],(c)[14])                          \
    dir##BFLYU01((c)[11],(c)[15])                          \
                                                           \
    BFLYU10((c)[0],(c)[2])                                 \
    BFLYU10((c)[1],(c)[3])                                 \
    dir##BFLYU01((c)[4],(c)[6])                            \
    dir##BFLYU01((c)[5],(c)[7])                            \
    dir##BFLYU((c)[ 8],(c)[10], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 9],(c)[11], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[12],(c)[14],-0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[13],(c)[15],-0.707106781f,-0.707106781f)\
                                                           \
    BFLYU10((c)[0],(c)[1])                                 \
    dir##BFLYU01((c)[2],(c)[3])                            \
    dir##BFLYU((c)[ 4],(c)[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 6],(c)[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 8],(c)[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU((c)[10],(c)[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[12],(c)[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[14],(c)[15],-0.923879533f,-0.382683432f)\
}

#define FFT16_M1(c,dir){\
    (c)[ 8]=(c)[ 0];    \
    (c)[ 4]=(c)[ 0];    \
    (c)[12]=(c)[ 8];    \
    (c)[ 2]=(c)[ 0];    \
    (c)[ 6]=(c)[ 4];    \
    (c)[10]=(c)[ 8];    \
    (c)[14]=(c)[12];    \
    (c)[ 1]=(c)[ 0];    \
    (c)[ 3]=(c)[ 2];    \
    (c)[ 5]=(c)[ 4];    \
    (c)[ 7]=(c)[ 6];    \
    (c)[ 9]=(c)[ 8];    \
    (c)[11]=(c)[10];    \
    (c)[13]=(c)[12];    \
    (c)[15]=(c)[14];    \
}

#define FFT16_M2(c,dir){                                   \
    (c)[ 8]=(c)[ 0];                                       \
    (c)[ 9]=(c)[ 1];                                       \
    (c)[ 4]=(c)[ 0];                                       \
    (c)[ 5]=(c)[ 1];                                       \
    (c)[12]=(c)[ 8];                                       \
    (c)[13]=(c)[ 9];                                       \
    (c)[ 2]=(c)[ 0];                                       \
    (c)[ 3]=(c)[ 1];                                       \
    (c)[ 6]=(c)[ 4];                                       \
    (c)[ 7]=(c)[ 5];                                       \
    (c)[10]=(c)[ 8];                                       \
    (c)[11]=(c)[ 9];                                       \
    (c)[14]=(c)[12];                                       \
    (c)[15]=(c)[13];                                       \
    BFLYU10((c)[0],(c)[1])                                 \
    dir##BFLYU01((c)[2],(c)[3])                            \
    dir##BFLYU((c)[ 4],(c)[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 6],(c)[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 8],(c)[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU((c)[10],(c)[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[12],(c)[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[14],(c)[15],-0.923879533f,-0.382683432f)\
}

#define FFT16_M3(c,dir){                                   \
    (c)[ 8]=(c)[ 0];                                       \
    (c)[ 9]=(c)[ 1];                                       \
    (c)[10]=(c)[ 2];                                       \
    (c)[ 4]=(c)[ 0];                                       \
    (c)[ 5]=(c)[ 1];                                       \
    (c)[ 6]=(c)[ 2];                                       \
    (c)[12]=(c)[ 8];                                       \
    (c)[13]=(c)[ 9];                                       \
    (c)[14]=(c)[10];                                       \
    (c)[ 3]=(c)[ 1];                                       \
    (c)[ 7]=(c)[ 5];                                       \
    (c)[11]=(c)[ 9];                                       \
    (c)[15]=(c)[13];                                       \
    BFLYU10((c)[0],(c)[2])                                 \
    dir##BFLYU01((c)[4],(c)[6])                            \
    dir##BFLYU((c)[ 8],(c)[10], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[12],(c)[14],-0.707106781f,-0.707106781f)\
    BFLYU10((c)[0],(c)[1])                                 \
    dir##BFLYU01((c)[2],(c)[3])                            \
    dir##BFLYU((c)[ 4],(c)[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 6],(c)[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 8],(c)[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU((c)[10],(c)[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[12],(c)[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[14],(c)[15],-0.923879533f,-0.382683432f)\
}

#define FFT16_M4(c,dir){                                   \
    (c)[ 8]=(c)[ 0];                                       \
    (c)[ 9]=(c)[ 1];                                       \
    (c)[10]=(c)[ 2];                                       \
    (c)[11]=(c)[ 3];                                       \
    (c)[ 4]=(c)[ 0];                                       \
    (c)[ 5]=(c)[ 1];                                       \
    (c)[ 6]=(c)[ 2];                                       \
    (c)[ 7]=(c)[ 3];                                       \
    (c)[12]=(c)[ 8];                                       \
    (c)[13]=(c)[ 9];                                       \
    (c)[14]=(c)[10];                                       \
    (c)[15]=(c)[11];                                       \
    BFLYU10((c)[0],(c)[2])                                 \
    BFLYU10((c)[1],(c)[3])                                 \
    dir##BFLYU01((c)[4],(c)[6])                            \
    dir##BFLYU01((c)[5],(c)[7])                            \
    dir##BFLYU((c)[ 8],(c)[10], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 9],(c)[11], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[12],(c)[14],-0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[13],(c)[15],-0.707106781f,-0.707106781f)\
    BFLYU10((c)[0],(c)[1])                                 \
    dir##BFLYU01((c)[2],(c)[3])                            \
    dir##BFLYU((c)[ 4],(c)[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 6],(c)[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 8],(c)[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU((c)[10],(c)[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[12],(c)[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[14],(c)[15],-0.923879533f,-0.382683432f)\
}

#define FFT32(c,dir){                                  \
    BFLYU10(c[ 0],c[16])                               \
    BFLYU10(c[ 1],c[17])                               \
    BFLYU10(c[ 2],c[18])                               \
    BFLYU10(c[ 3],c[19])                               \
    BFLYU10(c[ 4],c[20])                               \
    BFLYU10(c[ 5],c[21])                               \
    BFLYU10(c[ 6],c[22])                               \
    BFLYU10(c[ 7],c[23])                               \
    BFLYU10(c[ 8],c[24])                               \
    BFLYU10(c[ 9],c[25])                               \
    BFLYU10(c[10],c[26])                               \
    BFLYU10(c[11],c[27])                               \
    BFLYU10(c[12],c[28])                               \
    BFLYU10(c[13],c[29])                               \
    BFLYU10(c[14],c[30])                               \
    BFLYU10(c[15],c[31])                               \
                                                       \
    BFLYU10(c[0],c[ 8])                                \
    BFLYU10(c[1],c[ 9])                                \
    BFLYU10(c[2],c[10])                                \
    BFLYU10(c[3],c[11])                                \
    BFLYU10(c[4],c[12])                                \
    BFLYU10(c[5],c[13])                                \
    BFLYU10(c[6],c[14])                                \
    BFLYU10(c[7],c[15])                                \
    dir##BFLYU01(c[16],c[24])                          \
    dir##BFLYU01(c[20],c[28])                          \
    dir##BFLYU01(c[18],c[26])                          \
    dir##BFLYU01(c[22],c[30])                          \
    dir##BFLYU01(c[17],c[25])                          \
    dir##BFLYU01(c[21],c[29])                          \
    dir##BFLYU01(c[19],c[27])                          \
    dir##BFLYU01(c[23],c[31])                          \
                                                       \
    BFLYU10(c[0],c[4])                                 \
    BFLYU10(c[1],c[5])                                 \
    BFLYU10(c[2],c[6])                                 \
    BFLYU10(c[3],c[7])                                 \
    dir##BFLYU01(c[ 8],c[12])                          \
    dir##BFLYU01(c[ 9],c[13])                          \
    dir##BFLYU01(c[10],c[14])                          \
    dir##BFLYU01(c[11],c[15])                          \
    dir##BFLYU(c[16],c[20], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[17],c[21], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[18],c[22], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[19],c[23], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[24],c[28],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[25],c[29],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[26],c[30],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[27],c[31],-0.707106781f,-0.707106781f)\
                                                       \
    BFLYU10(c[0],c[2])                                 \
    BFLYU10(c[1],c[3])                                 \
    dir##BFLYU01(c[4],c[6])                            \
    dir##BFLYU01(c[5],c[7])                            \
    dir##BFLYU(c[ 8],c[10], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 9],c[11], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[12],c[14],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[13],c[15],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[16],c[18], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[17],c[19], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[20],c[22],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[21],c[23],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[24],c[26], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[25],c[27], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[28],c[30],-0.923879533f,-0.382683432f)\
    dir##BFLYU(c[29],c[31],-0.923879533f,-0.382683432f)\
                                                       \
    BFLYU10(c[0],c[1])                                 \
    dir##BFLYU01(c[2],c[3])                            \
    dir##BFLYU(c[ 4],c[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 6],c[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 8],c[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[10],c[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[12],c[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[14],c[15],-0.923879533f,-0.382683432f)\
    dir##BFLYU(c[16],c[17], 0.980785280f,-0.195090322f)\
    dir##BFLYU(c[18],c[19],-0.195090322f,-0.980785280f)\
    dir##BFLYU(c[20],c[21], 0.555570233f,-0.831469612f)\
    dir##BFLYU(c[22],c[23],-0.831469612f,-0.555570233f)\
    dir##BFLYU(c[24],c[25], 0.831469612f,-0.555570233f)\
    dir##BFLYU(c[26],c[27],-0.555570233f,-0.831469612f)\
    dir##BFLYU(c[28],c[29], 0.195090322f,-0.980785280f)\
    dir##BFLYU(c[30],c[31],-0.980785280f,-0.195090322f)\
}

#define FFT32_M2(c,dir){                               \
    c[16]=c[ 0];                                       \
    c[17]=c[ 1];                                       \
    c[ 8]=c[ 0];                                       \
    c[ 9]=c[ 1];                                       \
    c[24]=c[16];                                       \
    c[25]=c[17];                                       \
    c[ 4]=c[ 0];                                       \
    c[ 5]=c[ 1];                                       \
    c[12]=c[ 8];                                       \
    c[13]=c[ 9];                                       \
    c[20]=c[16];                                       \
    c[21]=c[17];                                       \
    c[28]=c[24];                                       \
    c[29]=c[25];                                       \
    c[ 2]=c[ 0];                                       \
    c[ 3]=c[ 1];                                       \
    c[ 6]=c[ 4];                                       \
    c[ 7]=c[ 5];                                       \
    c[10]=c[ 8];                                       \
    c[11]=c[ 9];                                       \
    c[14]=c[12];                                       \
    c[15]=c[13];                                       \
    c[18]=c[16];                                       \
    c[19]=c[17];                                       \
    c[22]=c[20];                                       \
    c[23]=c[21];                                       \
    c[26]=c[24];                                       \
    c[27]=c[25];                                       \
    c[30]=c[28];                                       \
    c[31]=c[29];                                       \
    BFLYU10(c[0],c[1])                                 \
    dir##BFLYU01(c[2],c[3])                            \
    dir##BFLYU(c[ 4],c[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 6],c[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 8],c[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[10],c[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[12],c[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[14],c[15],-0.923879533f,-0.382683432f)\
    dir##BFLYU(c[16],c[17], 0.980785280f,-0.195090322f)\
    dir##BFLYU(c[18],c[19],-0.195090322f,-0.980785280f)\
    dir##BFLYU(c[20],c[21], 0.555570233f,-0.831469612f)\
    dir##BFLYU(c[22],c[23],-0.831469612f,-0.555570233f)\
    dir##BFLYU(c[24],c[25], 0.831469612f,-0.555570233f)\
    dir##BFLYU(c[26],c[27],-0.555570233f,-0.831469612f)\
    dir##BFLYU(c[28],c[29], 0.195090322f,-0.980785280f)\
    dir##BFLYU(c[30],c[31],-0.980785280f,-0.195090322f)\
}

#define FFT32_M3(c,dir){                                   \
    c[16]=c[ 0];                                       \
    c[17]=c[ 1];                                       \
    c[18]=c[ 2];                                       \
    c[ 8]=c[ 0];                                       \
    c[ 9]=c[ 1];                                       \
    c[10]=c[ 2];                                       \
    c[24]=c[16];                                       \
    c[26]=c[18];                                       \
    c[25]=c[17];                                       \
    c[ 4]=c[ 0];                                       \
    c[ 5]=c[ 1];                                       \
    c[ 6]=c[ 2];                                       \
    c[12]=c[ 8];                                       \
    c[13]=c[ 9];                                       \
    c[14]=c[10];                                       \
    c[20]=c[16];                                       \
    c[21]=c[17];                                       \
    c[22]=c[18];                                       \
    c[28]=c[24];                                       \
    c[29]=c[25];                                       \
    c[30]=c[26];                                       \
    c[ 3]=c[ 1];                                       \
    c[ 7]=c[ 5];                                       \
    c[11]=c[ 9];                                       \
    c[15]=c[13];                                       \
    c[19]=c[17];                                       \
    c[23]=c[21];                                       \
    c[27]=c[25];                                       \
    c[31]=c[29];                                       \
    BFLYU10(c[0],c[2])                                 \
    dir##BFLYU01(c[4],c[6])                            \
    dir##BFLYU(c[ 8],c[10], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[12],c[14],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[16],c[18], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[20],c[22],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[24],c[26], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[28],c[30],-0.923879533f,-0.382683432f)\
    BFLYU10(c[0],c[1])                                 \
    dir##BFLYU01(c[2],c[3])                            \
    dir##BFLYU(c[ 4],c[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 6],c[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 8],c[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[10],c[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[12],c[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[14],c[15],-0.923879533f,-0.382683432f)\
    dir##BFLYU(c[16],c[17], 0.980785280f,-0.195090322f)\
    dir##BFLYU(c[18],c[19],-0.195090322f,-0.980785280f)\
    dir##BFLYU(c[20],c[21], 0.555570233f,-0.831469612f)\
    dir##BFLYU(c[22],c[23],-0.831469612f,-0.555570233f)\
    dir##BFLYU(c[24],c[25], 0.831469612f,-0.555570233f)\
    dir##BFLYU(c[26],c[27],-0.555570233f,-0.831469612f)\
    dir##BFLYU(c[28],c[29], 0.195090322f,-0.980785280f)\
    dir##BFLYU(c[30],c[31],-0.980785280f,-0.195090322f)\
}

#define FFT32_M4(c,dir){                               \
    c[16]=c[ 0];                                       \
    c[17]=c[ 1];                                       \
    c[18]=c[ 2];                                       \
    c[19]=c[ 3];                                       \
    c[ 8]=c[ 0];                                       \
    c[ 9]=c[ 1];                                       \
    c[10]=c[ 2];                                       \
    c[11]=c[ 3];                                       \
    c[24]=c[16];                                       \
    c[26]=c[18];                                       \
    c[25]=c[17];                                       \
    c[27]=c[19];                                       \
    c[ 4]=c[ 0];                                       \
    c[ 5]=c[ 1];                                       \
    c[ 6]=c[ 2];                                       \
    c[ 7]=c[ 3];                                       \
    c[12]=c[ 8];                                       \
    c[13]=c[ 9];                                       \
    c[14]=c[10];                                       \
    c[15]=c[11];                                       \
    c[20]=c[16];                                       \
    c[21]=c[17];                                       \
    c[22]=c[18];                                       \
    c[23]=c[19];                                       \
    c[28]=c[24];                                       \
    c[29]=c[25];                                       \
    c[30]=c[26];                                       \
    c[31]=c[27];                                       \
    BFLYU10(c[0],c[2])                                 \
    BFLYU10(c[1],c[3])                                 \
    dir##BFLYU01(c[4],c[6])                            \
    dir##BFLYU01(c[5],c[7])                            \
    dir##BFLYU(c[ 8],c[10], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 9],c[11], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[12],c[14],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[13],c[15],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[16],c[18], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[17],c[19], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[20],c[22],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[21],c[23],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[24],c[26], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[25],c[27], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[28],c[30],-0.923879533f,-0.382683432f)\
    dir##BFLYU(c[29],c[31],-0.923879533f,-0.382683432f)\
    BFLYU10(c[0],c[1])                                 \
    dir##BFLYU01(c[2],c[3])                            \
    dir##BFLYU(c[ 4],c[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 6],c[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 8],c[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[10],c[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[12],c[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[14],c[15],-0.923879533f,-0.382683432f)\
    dir##BFLYU(c[16],c[17], 0.980785280f,-0.195090322f)\
    dir##BFLYU(c[18],c[19],-0.195090322f,-0.980785280f)\
    dir##BFLYU(c[20],c[21], 0.555570233f,-0.831469612f)\
    dir##BFLYU(c[22],c[23],-0.831469612f,-0.555570233f)\
    dir##BFLYU(c[24],c[25], 0.831469612f,-0.555570233f)\
    dir##BFLYU(c[26],c[27],-0.555570233f,-0.831469612f)\
    dir##BFLYU(c[28],c[29], 0.195090322f,-0.980785280f)\
    dir##BFLYU(c[30],c[31],-0.980785280f,-0.195090322f)\
}

#define CALRF4(RF){                         \
    RF[1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
}

#define iCALRF4(RF){                        \
    RF[0].y=-RF[0].y;                       \
    RF[1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
}

#define CALRF8(RF){                         \
    RF[1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
    RF[3].x=RF[1].x*RF[1].x-RF[1].y*RF[1].y;\
    RF[3].y=RF[1].x*RF[1].y+RF[1].y*RF[1].x;\
    RF[4].x=RF[1].x*RF[2].x-RF[1].y*RF[2].y;\
    RF[4].y=RF[1].x*RF[2].y+RF[1].y*RF[2].x;\
    RF[5].x=RF[2].x*RF[2].x-RF[2].y*RF[2].y;\
    RF[5].y=RF[2].x*RF[2].y+RF[2].y*RF[2].x;\
    RF[6].x=RF[2].x*RF[3].x-RF[2].y*RF[3].y;\
    RF[6].y=RF[2].x*RF[3].y+RF[2].y*RF[3].x;\
}

#define iCALRF8(RF){                        \
    RF[0].y=-RF[0].y;                       \
    RF[1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
    RF[3].x=RF[1].x*RF[1].x-RF[1].y*RF[1].y;\
    RF[3].y=RF[1].x*RF[1].y+RF[1].y*RF[1].x;\
    RF[4].x=RF[1].x*RF[2].x-RF[1].y*RF[2].y;\
    RF[4].y=RF[1].x*RF[2].y+RF[1].y*RF[2].x;\
    RF[5].x=RF[2].x*RF[2].x-RF[2].y*RF[2].y;\
    RF[5].y=RF[2].x*RF[2].y+RF[2].y*RF[2].x;\
    RF[6].x=RF[2].x*RF[3].x-RF[2].y*RF[3].y;\
    RF[6].y=RF[2].x*RF[3].y+RF[2].y*RF[3].x;\
}

#define CALRF16(RF){                         \
    RF[ 1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[ 1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[ 2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[ 2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
    RF[ 3].x=RF[1].x*RF[1].x-RF[1].y*RF[1].y;\
    RF[ 3].y=RF[1].x*RF[1].y+RF[1].y*RF[1].x;\
    RF[ 4].x=RF[1].x*RF[2].x-RF[1].y*RF[2].y;\
    RF[ 4].y=RF[1].x*RF[2].y+RF[1].y*RF[2].x;\
    RF[ 5].x=RF[2].x*RF[2].x-RF[2].y*RF[2].y;\
    RF[ 5].y=RF[2].x*RF[2].y+RF[2].y*RF[2].x;\
    RF[ 6].x=RF[2].x*RF[3].x-RF[2].y*RF[3].y;\
    RF[ 6].y=RF[2].x*RF[3].y+RF[2].y*RF[3].x;\
    RF[ 7].x=RF[3].x*RF[3].x-RF[3].y*RF[3].y;\
    RF[ 7].y=RF[3].x*RF[3].y+RF[3].y*RF[3].x;\
    RF[ 8].x=RF[3].x*RF[4].x-RF[3].y*RF[4].y;\
    RF[ 8].y=RF[3].x*RF[4].y+RF[3].y*RF[4].x;\
    RF[ 9].x=RF[4].x*RF[4].x-RF[4].y*RF[4].y;\
    RF[ 9].y=RF[4].x*RF[4].y+RF[4].y*RF[4].x;\
    RF[10].x=RF[4].x*RF[5].x-RF[4].y*RF[5].y;\
    RF[10].y=RF[4].x*RF[5].y+RF[4].y*RF[5].x;\
    RF[11].x=RF[5].x*RF[5].x-RF[5].y*RF[5].y;\
    RF[11].y=RF[5].x*RF[5].y+RF[5].y*RF[5].x;\
    RF[12].x=RF[5].x*RF[6].x-RF[5].y*RF[6].y;\
    RF[12].y=RF[5].x*RF[6].y+RF[5].y*RF[6].x;\
    RF[13].x=RF[6].x*RF[6].x-RF[6].y*RF[6].y;\
    RF[13].y=RF[6].x*RF[6].y+RF[6].y*RF[6].x;\
    RF[14].x=RF[6].x*RF[7].x-RF[6].y*RF[7].y;\
    RF[14].y=RF[6].x*RF[7].y+RF[6].y*RF[7].x;\
}

#define iCALRF16(RF){                        \
    RF[0].y=-RF[0].y;                        \
    RF[ 1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[ 1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[ 2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[ 2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
    RF[ 3].x=RF[1].x*RF[1].x-RF[1].y*RF[1].y;\
    RF[ 3].y=RF[1].x*RF[1].y+RF[1].y*RF[1].x;\
    RF[ 4].x=RF[1].x*RF[2].x-RF[1].y*RF[2].y;\
    RF[ 4].y=RF[1].x*RF[2].y+RF[1].y*RF[2].x;\
    RF[ 5].x=RF[2].x*RF[2].x-RF[2].y*RF[2].y;\
    RF[ 5].y=RF[2].x*RF[2].y+RF[2].y*RF[2].x;\
    RF[ 6].x=RF[2].x*RF[3].x-RF[2].y*RF[3].y;\
    RF[ 6].y=RF[2].x*RF[3].y+RF[2].y*RF[3].x;\
    RF[ 7].x=RF[3].x*RF[3].x-RF[3].y*RF[3].y;\
    RF[ 7].y=RF[3].x*RF[3].y+RF[3].y*RF[3].x;\
    RF[ 8].x=RF[3].x*RF[4].x-RF[3].y*RF[4].y;\
    RF[ 8].y=RF[3].x*RF[4].y+RF[3].y*RF[4].x;\
    RF[ 9].x=RF[4].x*RF[4].x-RF[4].y*RF[4].y;\
    RF[ 9].y=RF[4].x*RF[4].y+RF[4].y*RF[4].x;\
    RF[10].x=RF[4].x*RF[5].x-RF[4].y*RF[5].y;\
    RF[10].y=RF[4].x*RF[5].y+RF[4].y*RF[5].x;\
    RF[11].x=RF[5].x*RF[5].x-RF[5].y*RF[5].y;\
    RF[11].y=RF[5].x*RF[5].y+RF[5].y*RF[5].x;\
    RF[12].x=RF[5].x*RF[6].x-RF[5].y*RF[6].y;\
    RF[12].y=RF[5].x*RF[6].y+RF[5].y*RF[6].x;\
    RF[13].x=RF[6].x*RF[6].x-RF[6].y*RF[6].y;\
    RF[13].y=RF[6].x*RF[6].y+RF[6].y*RF[6].x;\
    RF[14].x=RF[6].x*RF[7].x-RF[6].y*RF[7].y;\
    RF[14].y=RF[6].x*RF[7].y+RF[6].y*RF[7].x;\
}

#define MRF4(c,RF){\
    (c)[1]=s_cmul(RF[1],(c)[1]);\
    (c)[2]=s_cmul(RF[0],(c)[2]);\
    (c)[3]=s_cmul(RF[2],(c)[3]);\
}

#define MRF8(c,RF){\
    (c)[1]=s_cmul((RF)[3],(c)[1]);\
    (c)[2]=s_cmul((RF)[1],(c)[2]);\
    (c)[3]=s_cmul((RF)[5],(c)[3]);\
    (c)[4]=s_cmul((RF)[0],(c)[4]);\
    (c)[5]=s_cmul((RF)[4],(c)[5]);\
    (c)[6]=s_cmul((RF)[2],(c)[6]);\
    (c)[7]=s_cmul((RF)[6],(c)[7]);\
}

#define MRF16(c,RF){                 \
    (c)[ 1]=s_cmul((RF)[ 7],(c)[ 1]);\
    (c)[ 2]=s_cmul((RF)[ 3],(c)[ 2]);\
    (c)[ 3]=s_cmul((RF)[11],(c)[ 3]);\
    (c)[ 4]=s_cmul((RF)[ 1],(c)[ 4]);\
    (c)[ 5]=s_cmul((RF)[ 9],(c)[ 5]);\
    (c)[ 6]=s_cmul((RF)[ 5],(c)[ 6]);\
    (c)[ 7]=s_cmul((RF)[13],(c)[ 7]);\
    (c)[ 8]=s_cmul((RF)[ 0],(c)[ 8]);\
    (c)[ 9]=s_cmul((RF)[ 8],(c)[ 9]);\
    (c)[10]=s_cmul((RF)[ 4],(c)[10]);\
    (c)[11]=s_cmul((RF)[12],(c)[11]);\
    (c)[12]=s_cmul((RF)[ 2],(c)[12]);\
    (c)[13]=s_cmul((RF)[10],(c)[13]);\
    (c)[14]=s_cmul((RF)[ 6],(c)[14]);\
    (c)[15]=s_cmul((RF)[14],(c)[15]);\
}

#define MRF4x4(c,RF){         \
    c[ 1]=s_cmul(RF[1],c[ 1]);\
    c[ 5]=s_cmul(RF[1],c[ 5]);\
    c[ 9]=s_cmul(RF[1],c[ 9]);\
    c[13]=s_cmul(RF[1],c[13]);\
    c[ 2]=s_cmul(RF[0],c[ 2]);\
    c[ 6]=s_cmul(RF[0],c[ 6]);\
    c[10]=s_cmul(RF[0],c[10]);\
    c[14]=s_cmul(RF[0],c[14]);\
    c[ 3]=s_cmul(RF[2],c[ 3]);\
    c[ 7]=s_cmul(RF[2],c[ 7]);\
    c[11]=s_cmul(RF[2],c[11]);\
    c[15]=s_cmul(RF[2],c[15]);\
}

#define CLEAR4C(c){\
    (c)[0].x=0.f;  \
    (c)[0].y=0.f;  \
    (c)[1].x=0.f;  \
    (c)[1].y=0.f;  \
    (c)[2].x=0.f;  \
    (c)[2].y=0.f;  \
    (c)[3].x=0.f;  \
    (c)[3].y=0.f;  \
}

#define CLEAR8C(c){\
    CLEAR4C(&c[0]) \
    CLEAR4C(&c[4]) \
}

#define CLEAR16C(c){\
    CLEAR4C(&c[ 0]) \
    CLEAR4C(&c[ 4]) \
    CLEAR4C(&c[ 8]) \
    CLEAR4C(&c[12]) \
}

__device__ __forceinline__ float2 s_cmul( const float2& a, const float2& b )
{
    return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+b.x*a.y);
}
__device__ __forceinline__ float2 s_icmul( const float2& a, const float2& b )
{
    return make_float2(a.x*b.x+a.y*b.y,a.x*b.y-b.x*a.y);
}
__device__ __forceinline__ void s_postproc( float2& a, float2& b, const float2& RF )
{
    float hax=0.5f*a.x;
    float hay=0.5f*a.y;
    float p0=( 0.5f)*b.x+hax;
    float p1=(-0.5f)*b.y+hay;
    float q0=( 0.5f)*b.y+hay;
    float q1=(-0.5f)*b.x+hax;
    a.x=__fmaf_rn( q0, RF.x, __fmaf_rn( q1, RF.y, p0));
    a.y=__fmaf_rn( q0, RF.y, __fmaf_rn(-q1, RF.x, p1));
    b.x=__fmaf_rn(-q0, RF.x, __fmaf_rn(-q1, RF.y, p0));
    b.y=__fmaf_rn( q0, RF.y, __fmaf_rn(-q1, RF.x,-p1));
}
__device__ __forceinline__ void s_preproc( float2& a, float2& b, const float2& RF )
{
    float p0=a.x+b.x;
    float p1=a.y-b.y;
    float q0=a.y+b.y;
    float q1=a.x-b.x;
    a.x=__fmaf_rn(-q0, RF.x, __fmaf_rn( q1, RF.y, p0));
    a.y=__fmaf_rn( q1, RF.x, __fmaf_rn( q0, RF.y, p1));
    b.x=__fmaf_rn( q0, RF.x, __fmaf_rn(-q1, RF.y, p0));
    b.y=__fmaf_rn( q1, RF.x, __fmaf_rn( q0, RF.y,-p1));
}

#include"sfft8x8.h"
#include"sfft16x16.h"
#include"sfft32x32.h"
#include"sfft64x64.h"
#include"sfft128x128_r2c.h"
#include"sfft128x128_c2r.h"
#include"../xfft/xfft.h"

#endif