#ifndef __vop_h__
#define __vop_h__

__device__ __forceinline__ float2 operator+( const float2& a, const float2& b )
{
    return make_float2(a.x+b.x,a.y+b.y);
}
__device__ __forceinline__ float2 operator-( const float2& a, const float2& b )
{
    return make_float2(a.x-b.x,a.y-b.y);
}
__device__ __forceinline__ float2 operator*( const float2& a, const float2& b )
{
    return make_float2(a.x*b.x,a.y*b.y);
}
__device__ __forceinline__ float2 operator*( const float2& a, float b )
{
    return make_float2(a.x*b,a.y*b);
}
__device__ __forceinline__ float2 operator*(  float a, const float2& b )
{
    return make_float2(a*b.x,a*b.y);
}
__device__ __forceinline__ void operator+=( float2& a, const float2& b )
{
    a.x+=b.x; a.y+=b.y;
}
__device__ __forceinline__ void operator-=( float2& a, const float2& b )
{
    a.x-=b.x; a.y-=b.y;
}
__device__ __forceinline__ void operator*=( float2& a, const float2& b )
{
    a.x*=b.x; a.y*=b.y;
}
__device__ __forceinline__ void operator*=( float2& a, float b )
{
    a.x*=b; a.y*=b;
}

#endif