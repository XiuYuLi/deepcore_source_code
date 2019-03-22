#ifndef __half_h__
#define __half_h__

#include<cuda_fp16.h>

#define HZERO       __float2half2_rn(0.f)
#define HLO(x)      __low2half(x)
#define HHI(x)      __high2half(x)
#define HLOLO(x)    __low2half2(x)
#define HHIHI(x)    __high2half2(x)
#define HPERM(x)    __lowhigh2highlow(x)
#define HIX(x)      __halves2half2(__hneg(HLO(x)),HHI(x))
#define HIY(x)      __halves2half2(HLO(x),__hneg(HHI(x)))

typedef struct __align__( 8) __half4__{ __half2 x; __half2 y;   } __half4;
typedef struct __align__(16) __half8__{ __half4 lo; __half4 hi; } __half8;

__device__ __forceinline__ __half4 make_half4( __half2 x, __half2 y )
{
    __half4 pk;
    pk.x=x; pk.y=y;
    return pk;
}
__device__ __forceinline__ __half4 make_half4( __half x, __half y, __half z, __half w )
{
    __half4 pk;
    pk.x=__halves2half2(x,y);
    pk.y=__halves2half2(z,w);
    return pk;
}
__device__ __forceinline__ __half8 make_half8( const __half4& a, const __half4& b )
{
    __half8 c;
    c.lo=a;
    c.hi=b;
    return c;
}
__device__ __forceinline__ __half8 make_half8( __half2 x, __half2 y, __half2 z, __half2 w )
{
    __half8 pk;
    pk.lo.x=x; pk.lo.y=y; pk.hi.x=z; pk.hi.y=w;
    return pk;
}
__device__ __forceinline__ float4 __half42float4( __half4 val )
{
    union{ struct{ float2 lo; float2 hi; } x; float4 y; } castor;
    castor.x.lo=__half22float2(val.x);
    castor.x.hi=__half22float2(val.y);
    return castor.y;
}
__device__ __forceinline__ __half2 __byte_perm( __half2 a, __half2 b, unsigned int s )
{
    __half2 c;
    c.x=__byte_perm(a.x,b.x,s);
    return c;
}
__device__ __forceinline__ __half2 __uint_as_half2( unsigned int i )
{
    __half2 h; h.x=i; return h;
}
__device__ __forceinline__ unsigned int __half2_as_uint( __half2 h )
{
    return h.x;
}
__device__ __forceinline__ float operator+( float a, __half b )
{
    return (a+__half2float(b));
}
__device__ __forceinline__ float operator-( float a, __half b )
{
    return (a-__half2float(b));
}
__device__ __forceinline__ float operator*( float a, __half b )
{
    return (a*__half2float(b));
}
__device__ __forceinline__ float operator/( float a, __half b )
{
    return (a/__half2float(b));;
}
__device__ __forceinline__ float operator+( __half a, float b )
{
    return (__half2float(a)+b);
}
__device__ __forceinline__ float operator-( __half a, float b )
{
    return (__half2float(a)-b);
}
__device__ __forceinline__ float operator*( __half a, float b )
{
    return (__half2float(a)*b);
}
__device__ __forceinline__ float operator/( __half a, float b )
{
    return (__half2float(a)/b);
}
__device__ __forceinline__ void operator+=( float& a, __half b )
{
    a+=__half2float(b);
}
__device__ __forceinline__ void operator-=( float& a, __half b )
{
    a-=__half2float(b);
}
__device__ __forceinline__ void operator*=( float& a, __half b )
{
    a*=__half2float(b);
}

#endif