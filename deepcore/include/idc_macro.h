#ifndef __idc_macro_h__
#define __idc_macro_h__

#define IDC_MAX_DEVICES_PER_NODE  32
#define IDC_AFFIS(n,a)            (((n)+(a)-1)&(~((a)-1)))

#ifdef _MSC_VER
#define INLINE __forceinline
#else
#define INLINE inline
#endif

#ifdef __GNUC__
  #if __GNUC__>=4
    #define __local_func __attribute__((visibility("hidden")))
  #else
	#define __local_func
  #endif
#else
  #define __local_func
#endif

#endif
