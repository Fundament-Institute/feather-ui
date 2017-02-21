/* Black Sphere Studios Utility Library
   Copyright ©2017 Black Sphere Studios

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef __BSS_UTIL_H__
#define __BSS_UTIL_H__

#include "bss_util_c.h"
#include <assert.h>
#include <cmath>
#include <memory>
#include <cstring> // for memcmp
#include <emmintrin.h> // for SSE intrinsics
#include <float.h>
#include <string>
#include <stdio.h>
#include <array>
#include <limits>
#include <ostream>
#ifdef BSS_PLATFORM_POSIX
#include <stdlib.h> // For abs(int) on POSIX systems
#include <fpu_control.h> // for FPU control on POSIX systems
#endif

namespace bss_util { 
  static const VersionType BSSUTIL_VERSION = { { { BSS_VERSION_MAJOR,BSS_VERSION_MINOR,BSS_VERSION_REVISION } } };
  
  BSS_COMPILER_DLLEXPORT extern void SetWorkDirToCur(); //Sets the working directory to the actual goddamn location of the EXE instead of the freaking start menu, or possibly the desktop. The possibilities are endless! Fuck you, windows.
  BSS_COMPILER_DLLEXPORT extern void ForceWin64Crash(); // I can't believe this function exists (forces 64-bit windows to not silently ignore fatal errors)
  BSS_COMPILER_DLLEXPORT extern unsigned long long bssFileSize(const char* path);
  BSS_COMPILER_DLLEXPORT extern unsigned long long bssFileSize(const wchar_t* path);
  BSS_COMPILER_DLLEXPORT extern long GetTimeZoneMinutes(); //Returns the current time zone difference from UTC in minutes

  //Useful numbers
  const double PI = 3.141592653589793238462643383279;
  const double PI_HALF = PI*0.5;
  const double PI_DOUBLE = PI*2.0;  
  const double E_CONST = 2.718281828459045235360287471352;
  const double SQRT_TWO = 1.414213562373095048801688724209;
  const float PIf = 3.141592653589793238462643383279f; // Convenience definitions
  const float PI_HALFf = PIf*0.5f;
  const float PI_DOUBLEf = PIf*2.0f;  
  const float E_CONSTf = 2.718281828459045235360287471352f;
  const float SQRT_TWOf = 1.414213562373095048801688724209f;
  const float FLT_EPS = 1.192092896e-07F;
  const double DBL_EPS = 2.2204460492503131e-016;
  const int32_t FLT_INTEPS = *(int32_t*)(&FLT_EPS);
  const int64_t DBL_INTEPS = *(int64_t*)(&DBL_EPS);

  // Get max size of an arbitrary number of bits, either signed or unsigned (assuming one's or two's complement implementation)
  template<uint8_t BITS>
  struct BitLimit
  {
    static const uint16_t BYTES = ((T_CHARGETMSB(BITS)>>3) << (0+((BITS%8)>0))) + (BITS<8);
    typedef typename std::conditional<sizeof(char) == BYTES, char,  //rounds the type up if necessary.
      typename std::conditional<sizeof(short) == BYTES, short,
      typename std::conditional<sizeof(int) == BYTES, int,
      typename std::conditional<sizeof(int64_t) == BYTES, int64_t,
#ifdef BSS_HASINT128
      typename std::conditional<sizeof(__int128) == BYTES, __int128, void>::type>::type>::type>::type>::type SIGNED;
#else
      void>::type>::type>::type>::type SIGNED;
#endif
    typedef typename std::make_unsigned<SIGNED>::type UNSIGNED;

    static const UNSIGNED UNSIGNED_MIN = 0;
    static const UNSIGNED UNSIGNED_MAX = (((UNSIGNED)2) << (BITS - 1)) - ((UNSIGNED)1); //these are all done carefully to ensure no overflow is ever utilized unless appropriate and it respects an arbitrary bit limit. We use 2<<(BITS-1) here to avoid shifting more bits than there are bits in the type.
    static const SIGNED SIGNED_MIN_RAW = (SIGNED)(((UNSIGNED)1) << (BITS - 1)); // When we have normal bit lengths (8,16, etc) this will correctly result in a negative value in two's complement.
    static const UNSIGNED SIGNED_MIN_HELPER = (((UNSIGNED)~0) << (BITS - 1)); // However if we have unusual bit lengths (3,19, etc) the raw bit representation will be technically correct in the context of that sized integer, but since we have to round to a real integer size to represent the number, the literal interpretation will be wrong. This yields the proper minimum value.
    static const SIGNED SIGNED_MIN = (SIGNED)SIGNED_MIN_HELPER;
    static const SIGNED SIGNED_MAX = ((~SIGNED_MIN_RAW)&UNSIGNED_MAX);
  };
  template<typename T>
  struct TBitLimit : public BitLimit<sizeof(T)<<3> {};

  // Typesafe malloc implementation, for when you don't want to use New because you don't want constructors to be called.
  template<typename T>
  BSS_FORCEINLINE T* bssmalloc(size_t sz) { return reinterpret_cast<T*>(malloc(sz * sizeof(T))); }
  template<typename T>
  BSS_FORCEINLINE T* bsscalloc(size_t sz) { return reinterpret_cast<T*>(calloc(sz, sizeof(T))); }

  // template inferred version of T_GETBIT and T_GETBITRANGE
  template<class T>
  inline T GetBitMask(int bit) noexcept { return T_GETBIT(T,bit); }
  template<class T>
  inline T GetBitMask(int low, int high) noexcept { return T_GETBITRANGE(T,low,high); }
  template<class T>
  inline T bssSetBit(T word, T bit, bool value) noexcept { return T_SETBIT(word,bit,value); }

  template<class T>
  BSS_FORCEINLINE bool bssGetBit(T* p, size_t index) { return (p[index / (sizeof(T) << 3)] & (1 << (index % (sizeof(T) << 3)))) != 0; }

  // Replaces one character with another in a string
  template<typename T>
  inline T* strreplace(T* string, const T find, const T replace) noexcept
	{
    static_assert(std::is_integral<T>::value,"T must be integral");
		if(!string) return 0;
    size_t curpos = (size_t)-1; //this will wrap around to 0 when we increment

		while(string[++curpos] != '\0') //replace until the null-terminator
			if(string[curpos] == find)
				string[curpos] = replace;

		return string;
	}

#ifndef BSS_COMPILER_MSC
  template<int SZ>
  BSS_FORCEINLINE char* strcpyx0(char (&dst)[SZ], const char* src) { return strncpy(dst,src,SZ-1); }
#endif

  // Counts number of occurences of character c in string, up to the null terminator
  template<typename T>
  inline size_t strccount(const T* string, T c) noexcept
  {
    static_assert(std::is_integral<T>::value,"T must be integral");
    size_t ret=0;
    while(*string) { if(*string==c) ++ret; ++string; }
    return ret;
  }
  
  // Counts number of occurences of character c in string, up to length characters
  template<typename T>
  inline size_t strccount(const T* string, T c, size_t length) noexcept
  {
    static_assert(std::is_integral<T>::value,"T must be integral");
    size_t ret=0;
    for(size_t i = 0; (i < length) && ((*string)!=0); ++i)
      if(string[i]==c) ++ret;
    return ret;
  }

  // template swap function, h should be optimized out by the compiler
  template<typename T>
  BSS_FORCEINLINE void rswap(T& p, T& q) noexcept
  {
    T h(std::move(p));
    p=std::move(q);
    q=std::move(h);
  }
  
  // This yields mathematically correct integer division (i/div) towards negative infinity, but only if div is a positive integer. This
  // implementation obviously must have integral types for T and D, but can be explicitely specialized to handle vectors or other structs.
  template<typename T, typename D>
  inline T intdiv(T i, D div) noexcept
  { 
    static_assert(std::is_integral<T>::value,"T must be integral");
    static_assert(std::is_integral<D>::value,"D must be integral");
    assert(div>0); 
    return (i/div) - ((i<0)&((i%div)!=0)); // If i is negative and has a nonzero remainder, subtract one to correct the truncation.
  }

  // Performs a mathematically correct modulo, unlike the modulo operator, which doesn't actually perform modulo, it performs a remainder operation. THANKS GUYS!
  template<typename T> //T must be integral
  BSS_FORCEINLINE T bssmod(T x, T m) noexcept
  {
		static_assert(std::is_signed<T>::value && std::is_integral<T>::value,"T must be a signed integral type or this function is pointless");
    x%=m;
    return (x+((-(T)(x<0))&m));
    //return (x+((x<0)*m)); // This is a tad slower
  }

  // Performs a mathematically correct floating point modulo, unlike fmod, which performs a remainder operation, not a modulo operation.
  template<typename T> //T must be floating point
  BSS_FORCEINLINE T bssfmod(T x, T m) noexcept
  {
    static_assert(std::is_floating_point<T>::value, "T must be a floating point type");
    return x - floor(x/m)*m;
  }
  // Uses rswap to reverse the order of an array
  template<typename T>
  inline void bssreverse(T* src, size_t length) noexcept
  {
    assert(length>0);
    for(size_t i = 0, j=(--length); i < j; j=length-(++i)) // Equivelent to: for(size_t i = 0,j=length-1; i < j; j=length-1-(++i))
      rswap(src[i],src[j]);
  }
  template<typename T, size_t size>
  BSS_FORCEINLINE void bssreverse(T (&p)[size]) noexcept { bssreverse(p,size); }

  /* Trims space from left end of string by returning a different pointer. It is possible to use const char or const wchar_t as a type
     here because the string itself is not modified. */
  template<typename T>
  inline T* strltrim(T* str) noexcept
  {
    static_assert(std::is_integral<T>::value,"T must be integral");
    for(;*str>0 && *str<33;++str);
    return str;
  }
  
  // Trims space from right end of string by inserting a null terminator in the appropriate location
  template<typename T>
  inline T* strrtrim(T* str) noexcept
  {
    static_assert(std::is_integral<T>::value,"T must be integral");
    T* inter=str+strlen(str);

    for(;inter>str && *inter<33;--inter);
    *(++inter)='\0';
    return str;
  }

  // Trims space from left and right ends of a string
  template<typename T>
  BSS_FORCEINLINE T* strtrim(T* str) noexcept
  {
    return strrtrim(strltrim(str));
  }

  template<typename T>
  inline int ToArgV(T** argv, T* cmdline)
  {
    int count=0;
    while(*cmdline)
    {
      if(!isspace(*cmdline))
      {
        if(*cmdline == '"') {
          if(argv!=0) argv[count++]=cmdline+1;
          else ++count;
          while(*++cmdline !=0 && cmdline[1] !=0 && (*cmdline != '"' || !isspace(cmdline[1])));
        } else {
          if(argv!=0) argv[count++]=cmdline;
          else ++count;
          while(*++cmdline !=0 && !isspace(*cmdline));
        }
        if(!*cmdline) break;
        if(argv!=0) *cmdline=0;
      }
      ++cmdline;
    }
    return count;
  }

  // Processes an argv command line using the given function and divider. Use ToArgV if you only have a string (for windows command lines)
  template<typename F> //std::function<void(const char* const*,int num)>
  inline void ProcessCmdArgs(int argc, const char* const* argv, F && fn, char divider='-')
  {
    if(!argc||!argv) return;
    const char* const* cur=argv;
    int len=1;
    for(int i = 1; i<argc; ++i)
    {
      if(argv[i][0]==divider)
      {
        fn(cur, len);
        cur=argv+i;
        len=1;
      }
      else
        ++len;
    }
    fn(cur, len);
  }

#ifdef BSS_VARIADIC_TEMPLATES
  template<typename... A>
  struct __safeFormat
  {
    template<int N, typename Arg, typename... Args>
    BSS_FORCEINLINE static void helper(std::ostream& o, int n, Arg arg, Args... args)
    {
      if(sizeof...(Args) == (N - n)) // This works because Args is actually 1 less than the total argument count already.
        o << arg;
      else
        helper<N, Args...>(o, n, args...);
    }
    template<int N>
    BSS_FORCEINLINE static void helper(std::ostream& o, int n) { o << "{INVALID PARAMETER " << n << "}"; }

    template<void(*FN)(std::ostream&, int, A...)>
    BSS_FORCEINLINE static void F(std::ostream& o, const char* format, A... args)
    {
      const char* pos = format;
      while(format[0])
      {
        if(format[0] == '{' && format[1] >= '0' && format[1] <= '9' && (format[2] == '}' || (format[2] >= '0' && format[2] <= '9' && format[3] == '}')))
        {
          if(format - pos > 0)
            o.write(pos, format - pos);
          int n = 0;
          if(format[2] == '}')
          {
            n = format[1] - '0';
            format += 3;
          }
          else
          {
            n = format[2] - '0';
            n = (format[1] - '0') * 10;
            format += 4;
          }
          pos = format;
          FN(o, n, args...);
        }
        else
          ++format;
      }
      if(format - pos > 0)
        o.write(pos, format - pos);
    }
  };

  template<typename... Args>
  inline void SafeFormat(std::ostream& o, const char* format, Args... args)
  {
    __safeFormat<Args...>::template F<&__safeFormat<Args...>::template helper<(sizeof...(Args) - 1), Args...>>(o, format, args...);
  }
#endif

  // Converts 32-bit unicode int to a series of utf8 encoded characters, appending them to the string
  inline void OutputUnicode(std::string& s, int c) noexcept
  {
    if(c < 0x0080) s += c;
    else if(c < 0x0800) { s += (0xC0 | c >> (6 * 1)); s += (0x80 | (c & 0x3F)); }
    else if(c < 0x10000) { s += (0xE0 | c >> (6 * 2)); s += (0x80 | (c & 0x0FC0) >> (6 * 1)); s += (0x80 | (c & 0x3F)); }
    else if(c < 0x10FFFF) { s += (0xF0 | c >> (6 * 3)); s += (0x80 | (c & 0x03F000) >> (6 * 2)); s += (0x80 | (c & 0x0FC0) >> (6 * 1)); s += (0x80 | (c & 0x3F)); }
    // Otherwise this is an illegal codepoint so just ignore it
  }

  // Flips the endianness of a memory location
  BSS_FORCEINLINE void flipendian(char* target, char n) noexcept
  {
    char t;
    char end = (n>>1);
    --n;
    for(char i=0; i<end; ++i)
    {
      t=target[n-i];
      target[n-i]=target[i];
      target[i]=t;
    }
  }

  template<int I>
  BSS_FORCEINLINE void flipendian(char* target) noexcept { flipendian((char*)target, I); }
  template<> BSS_FORCEINLINE void flipendian<0>(char* target) noexcept { }
  template<> BSS_FORCEINLINE void flipendian<1>(char* target) noexcept { }
  template<> BSS_FORCEINLINE void flipendian<2>(char* target) noexcept { char t = target[0]; target[0] = target[1]; target[1] = t; }

  template<typename T>
  BSS_FORCEINLINE void flipendian(T* target) noexcept { flipendian<sizeof(T)>((char*)target); }

    // This is a bit-shift method of calculating the next number in the fibonacci sequence by approximating the golden ratio with 0.6171875 (1/2 + 1/8 - 1/128)
  template<typename T>
  BSS_FORCEINLINE T fbnext(T x) noexcept
  {
    static_assert(std::is_integral<T>::value,"T must be integral");
    return T_FBNEXT(x);
    //return x + 1 + (x>>1) + (x>>3) - (x>>7) + (x>>10) - (x>>13) - (x>>17) - (x>>21) + (x>>24); // 0.61803394 (but kind of pointless)
  }

  // Gets the sign of any integer (0 is assumed to be positive)
  template<typename T>
  BSS_FORCEINLINE T tsign(T n) noexcept
  {
    static_assert(std::is_integral<T>::value, "T must be an integer.");
    return 1 | (n >> ((sizeof(T) << 3) - 1));
  }

  // Gets the sign of any number, where a value of 0 returns 0
  template<typename T>
  BSS_FORCEINLINE char tsignzero(T n) noexcept
  {
    return (n > 0) - (n < 0);
  }
  
  // Gets the shortest distance between two angles in radians
  template<typename T>
  BSS_FORCEINLINE T angledist(T a, T b) noexcept
  {
    static_assert(std::is_floating_point<T>::value,"T must be float, double, or long double");
    return ((T)PI) - fabs(fmod(fabs(a - b), ((T)PI_DOUBLE)) - ((T)PI));
  }
  
  // Gets the SIGNED shortest distance between two angles starting with (a - b) in radians
  template<typename T>
  BSS_FORCEINLINE T angledistsgn(T a, T b) noexcept
  {
    static_assert(std::is_floating_point<T>::value,"T must be float, double, or long double");
    return fmod(bssfmod(b - a, (T)PI_DOUBLE) + ((T)PI), (T)PI_DOUBLE) - ((T)PI);
  }

  // Smart compilers will use SSE2 instructions to eliminate the massive overhead of int -> float conversions. This uses SSE2 instructions
  // to round a float to an integer using whatever the rounding mode currently is (usually it will be round-to-nearest)
  BSS_FORCEINLINE int32_t fFastRound(float f) noexcept
  {
#ifdef BSS_SSE_ENABLED
    return _mm_cvt_ss2si(_mm_load_ss(&f));
#else
    return (int32_t)roundf(f);
#endif
  }

  BSS_FORCEINLINE int32_t fFastRound(double f) noexcept
  {
#ifdef BSS_SSE_ENABLED
    return _mm_cvtsd_si32(_mm_load_sd(&f));
#else
    return (int32_t)round(f);
#endif
  }

  BSS_FORCEINLINE int32_t fFastTruncate(float f) noexcept
  {
#ifdef BSS_SSE_ENABLED
    return _mm_cvtt_ss2si(_mm_load_ss(&f));
#else
    return (int32_t)f;
#endif
  }

  BSS_FORCEINLINE int32_t fFastTruncate(double f) noexcept
  {
#ifdef BSS_SSE_ENABLED
    return _mm_cvttsd_si32(_mm_load_sd(&f));
#else
    return (int32_t)f;
#endif
  }

#if defined(BSS_CPU_x86) || defined(BSS_CPU_x86_64) || defined(BSS_CPU_IA_64)
#ifndef MSC_MANAGED
  BSS_FORCEINLINE void fSetRounding(bool nearest) noexcept
  {
    uint32_t a;
#ifdef BSS_PLATFORM_WIN32
    _controlfp_s(&a, nearest?_RC_NEAR:_RC_CHOP, _MCW_RC);
#else
    _FPU_GETCW(a);
    a = (a&(~(_FPU_RC_NEAREST|_FPU_RC_DOWN|_FPU_RC_UP|_FPU_RC_ZERO)))|(nearest?_FPU_RC_NEAREST:_FPU_RC_ZERO);
    _FPU_SETCW(a);
#endif
  }
  BSS_FORCEINLINE void fSetDenormal(bool on) noexcept
  {
    uint32_t a;
#ifdef BSS_PLATFORM_WIN32
    _controlfp_s(&a, on?_DN_SAVE:_DN_FLUSH, _MCW_DN);
#else
    _FPU_GETCW(a); // Linux doesn't know what the denormal flags are, so we just hardcode the values in from windows' flags.
    a = (a&(~0x03000000))|(on?0:0x01000000);
    _FPU_SETCW(a);
#endif
  }
#endif

  // Returns true if FPU is in single precision mode and false otherwise (false for both double and extended precision)
  BSS_FORCEINLINE bool FPUsingle() noexcept
  { 
    uint32_t i;
#ifdef BSS_PLATFORM_WIN32
    i = _mm_getcsr();
#else
    _FPU_GETCW(i);
#endif
    return ((i&(0x0300))==0); //0x0300 is the mask for the precision bits, 0 indicates single precision
  }
#endif

  // Extremely fast rounding function that truncates properly, but only works in double precision mode; see http://stereopsis.com/FPU.html
  /*BSS_FORCEINLINE int32_t fFastDoubleRound(double val)
  {
    const double _double2fixmagic = 4503599627370496.0*1.5; //2^52 for 52 bits of mantissa
    assert(!FPUsingle());
	  val		= val + _double2fixmagic;
	  return ((int32_t*)&val)[0]; 
  }
  BSS_FORCEINLINE int32_t fFastDoubleRound(float val) { return fFastDoubleRound((double)val); }

  // Single precision version of the above function. While precision problems are mostly masked in the above function by limiting it to
  // int32_t, in this function they are far more profound due to there only being 24 bits of mantissa to work with. Use with caution. 
  BSS_FORCEINLINE int32_t fFastSingleRound(double val)
  {
    const double _single2fixmagic = 16777216.0*1.5; //2^24 for 24 bits of mantissa
    assert(FPUsingle());
	  val		= val + _single2fixmagic;
	  return (int32_t)(((((uint64_t*)&val)[0])&0xFFFFF0000000)>>28);
  }
  BSS_FORCEINLINE int32_t fFastSingleRound(float val) { return fFastSingleRound((double)val); } */

	// This is a super fast floating point comparison function with a significantly higher tolerance and no
	// regard towards the size of the floats.
	inline bool fwidecompare(float fleft, float fright) noexcept
	{
		int32_t left = *reinterpret_cast<int32_t*>(&fleft); //This maps our float to an int so we can do bitshifting operations on it
		int32_t right = *reinterpret_cast<int32_t*>(&fright); //see above
		uint8_t dif = abs((0x7F800000&left)-(0x7F800000&right))>>23; // This grabs the 8 exponent bits and subtracts them.
		if(dif>1) // An exponent difference of 2 or greater means the numbers are different.
			return false;
		return !dif?((0x007FFF80&left)==(0x007FFF80&right)):!(abs((0x007FFF80&left)-(0x007FFF80&right))-0x007FFF80); //If there is no difference in exponent we tear off the last 7 bits and compare the value, otherwise we tear off the last 7 bits, subtract, and then subtract the highest possible significand to compensate for the extra exponent.
	}

  // Highly optimized traditional tolerance based approach to comparing floating point numbers, found here: http://www.randydillon.org/Papers/2007/everfast.htm
  inline bool fcompare(float af, float bf, int32_t maxDiff=1) noexcept
  { 
    //assert(af!=0.0f && bf!=0.0f); // Use fsmall for this
    int32_t ai = *reinterpret_cast<int32_t*>(&af);
    int32_t bi = *reinterpret_cast<int32_t*>(&bf);
    int32_t test = (-(int32_t)(((uint32_t)(ai^bi))>>31));
    assert((0 == test) || (0xFFFFFFFF == (uint32_t)test));
    int32_t diff = ((ai + test) ^ (test & 0x7fffffff)) - bi;
    int32_t v1 = maxDiff + diff;
    int32_t v2 = maxDiff - diff;
    return (v1|v2) >= 0;
  }

  inline bool fcompare(double af, double bf, int64_t maxDiff=1) noexcept
  { 
    //assert(af!=0.0 && bf!=0.0); // Use fsmall for this
    int64_t ai = *reinterpret_cast<int64_t*>(&af);
    int64_t bi = *reinterpret_cast<int64_t*>(&bf);
    int64_t test = (-(int64_t)(((uint64_t)(ai^bi))>>63));
    assert((0 == test) || (0xFFFFFFFFFFFFFFFF == (uint64_t)test));
    int64_t diff = ((ai + test) ^ (test & 0x7fffffffffffffff)) - bi;
    int64_t v1 = maxDiff + diff;
    int64_t v2 = maxDiff - diff;
    return (v1|v2) >= 0;
  }

  // This determines if a float is sufficiently close to 0
  BSS_FORCEINLINE bool fsmall(float f, float eps = FLT_EPS) noexcept
  {
    uint32_t i = ((*reinterpret_cast<uint32_t*>(&f)) & 0x7FFFFFFF); //0x7FFFFFFF strips off the sign bit (which is always the highest bit)
    uint32_t e = *reinterpret_cast<uint32_t*>(&eps);
    return i <= e;
  }

  // This determines if a double is sufficiently close to 0
  BSS_FORCEINLINE bool fsmall(double f, double eps = DBL_EPS) noexcept
  {
    uint64_t i = ((*reinterpret_cast<uint64_t*>(&f)) & 0x7FFFFFFFFFFFFFFF); //0x7FFFFFFFFFFFFFFF strips off the sign bit (which is always the highest bit)
    uint64_t e = *reinterpret_cast<uint64_t*>(&eps);
    return i <= e;
  }

  inline bool fcomparesmall(float af, float bf, int32_t maxDiff=1, float eps = FLT_EPS) noexcept
  {
    if(af==0.0) return fsmall(bf, eps);
    if(bf==0.0) return fsmall(af, eps);
    return fcompare(af, bf, maxDiff);
  }

  inline bool fcomparesmall(double af, double bf, int64_t maxDiff=1, double eps = DBL_EPS) noexcept
  {
    if(af==0.0) return fsmall(bf, eps);
    if(bf==0.0) return fsmall(af, eps);
    return fcompare(af, bf, maxDiff);
  }

  // This is a super fast length approximation for 2D coordinates; See http://www.azillionmonkeys.com/qed/sqroot.html for details (Algorithm by Paul Hsieh)
  inline float flength(float x, float y) noexcept
  {
    x = fabs(x);
    y = fabs(y);
    //(1 + 1/(4-2*sqrt(2)))/2 = 0.92677669529663688
    float hold=0.7071067811865475f*(x+y), mval=(x > y)?x:y;
    return 0.92677669529663688f * ((hold > mval)?hold:mval);
  }
  
  // The classic fast square root approximation, which is often mistakenly attributed to John Carmack. The algorithm is in fact over 15 years old and no one knows where it came from.
  inline float fFastSqrt(float number) noexcept
  {
    const float f = 1.5F;
    int32_t i;
    float x, y;

    x = number * 0.5F;
    y = number;
    i = *reinterpret_cast<int32_t*>(&y);
    i = 0x5f3759df - (i >> 1);
    y = *(float *)&i;
    y = y * (f - (x * y * y));
    y = y * (f - (x * y * y)); //extra iteration for added accuracy
    return number * y;
  }

  // Adaptation of the class fast square root approximation for double precision, based on http://www.azillionmonkeys.com/qed/sqroot.html
  inline double dFastSqrt(double number) noexcept
  {
    const double f = 1.5;
    uint32_t* i;
    double x, y;

    x = number*0.5;
    y = number;
    i = reinterpret_cast<uint32_t*>(&y) + 1;
    *i = (0xbfcdd90a - *i) >> 1; // estimate of 1/sqrt(number)

    y = y * (f - (x * y * y));
    y = y * (f - (x * y * y));
    y = y * (f - (x * y * y)); //Newton raphson converges quadratically, so one additional iteration doubles the precision
    //y = y * ( f - ( x * y * y ) ); 
    return number * y;
  }

  // bit-twiddling based method of calculating an integral square root from Wilco Dijkstra - http://www.finesse.demon.co.uk/steven/sqrt.html
  template<typename T, size_t bits> // WARNING: bits should be HALF the actual number of bits in (T)!
  inline T IntFastSqrt(T n) noexcept
  {
    static_assert(std::is_integral<T>::value,"T must be integral");
    T root = 0, t;

    for(size_t i = bits; i>0;)
    {
      --i;
      t = ((root + (((T)1) << (i))) << (i)); 
      if (n >= t)   
      {   n -= t;   
          root |= 2 << (i); 
      }
    }
    return root >> 1;
  }
  template<typename T>
  BSS_FORCEINLINE T IntFastSqrt(T n) noexcept
  {
    return IntFastSqrt<T,sizeof(T)<<2>(n); //done to ensure loop gets unwound (the bit conversion here is <<2 because the function wants HALF the bits in T, so <<3 >>1 -> <<2)
  }
  
  template<typename T> //assumes integer type if not one of the floating point types
  BSS_FORCEINLINE T FastSqrt(T n) noexcept { return IntFastSqrt(n); } //Picks correct method for calculating any square root quickly
  template<> BSS_FORCEINLINE float FastSqrt(float n) noexcept { return fFastSqrt(n); }
  template<> BSS_FORCEINLINE double FastSqrt(double n) noexcept { return dFastSqrt(n); }
  template<> BSS_FORCEINLINE long double FastSqrt(long double n) noexcept { return dFastSqrt((double)n); }

  // Distance calculation (squared)
  template<typename T>
  BSS_FORCEINLINE T distsqr(T X, T Y, T x, T y) noexcept
  {
    T tx=X-x,ty=Y-y; return (tx*tx)+(ty*ty); //It doesn't matter if you use temporary values for floats, but it does if you use ints (for unknown reasons)
  }

  // Distance calculation
  template<typename T>
  BSS_FORCEINLINE T dist(T X, T Y, T x, T y) noexcept
  {
    return FastSqrt<T>(distsqr<T>(X,Y,x,y));
  }

  // Average aggregation without requiring a total variable that can overflow. Nextnum should be the current avg count incremented by 1.
  template<typename T, typename CT_> // T must be float or double, CT_ must be integral
  BSS_FORCEINLINE T bssavg(T curavg, T nvalue, CT_ nextnum) noexcept
  { // USAGE: avg = bssavg<double, int>(avg, value, ++total);
    static_assert(std::is_integral<CT_>::value,"CT_ must be integral");
    static_assert(std::is_floating_point<T>::value,"T must be float, double, or long double");
    return curavg + ((nvalue-curavg)/(T)nextnum);
  }

  // Sum of squares of differences aggregation using an algorithm by Knuth. Nextnum should be the current avg count incremented by 1.
  template<typename T, typename CT_> // T must be float or double, CT_ must be integral
  BSS_FORCEINLINE void bssvariance(T& curvariance, T& avg, T nvalue, CT_ nextnum) noexcept
  { // USAGE: bssvariance<double, int>(variance, avg, value, ++total); Then use sqrt(variance/(n-1)) to get the actual standard deviation
    static_assert(std::is_integral<CT_>::value, "CT_ must be integral");
    static_assert(std::is_floating_point<T>::value, "T must be float, double, or long double");
    T delta = nvalue - avg;
    avg += delta/(T)nvalue;
    curvariance += delta*(nvalue - avg);
  }

  // Searches an arbitrary series of bytes for another arbitrary series of bytes
  inline const void* bytesearch(const void* search, size_t length, const void* find, size_t flength) noexcept
  {
    if(!search || !length || !find || !flength || length < flength) return 0;

    uint8_t* s=(uint8_t*)search;
    length-=flength;
    for(size_t i = 0; i <= length; ++i) // i <= length works because of the above length-=flength
    {
      search=s+i;
      if(!memcmp(search,find,flength))
        return search;
    }
    return 0;
  }

  BSS_FORCEINLINE void* bytesearch(void* search, size_t length, void* find, size_t flength) noexcept
  {
    return const_cast<void*>(bytesearch((const void*)search,length,(const void*)find,flength));
  }

  // Counts the number of bits in v (up to 128-bit types) using the parallel method detailed here: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
  template<typename T>
  inline uint8_t bitcount(T v) noexcept
  {
    static_assert(std::is_integral<T>::value,"T must be integral");
    v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
    v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
    v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
    return (uint8_t)((T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * (sizeof(char)<<3));
  }

  //Unlike FastSqrt, these are useless unless you are on a CPU without SSE instructions, or have a terrible std implementation.
  //// Fast sin function with 0.078% error when extra precision is left in. See http://www.devmaster.net/forums/showthread.php?t=5784
  //inline float FastSin(float x)
  //{
  //  x-= (int)(x*(1/(float)PI_DOUBLE))*(float)PI_DOUBLE;
  //  const float B = 4/(float)PI;
  //  const float C = -4/(float)PI_DOUBLE;

  //  float y = B * x + C * x * fabs(x);

  //  //const float Q = 0.775f;
  //  const float P = 0.225f;

  //  return (P * (y * fabs(y) - y) + y);   // Q * y + P * y * fabs(y) (extra precision)
  //}

  //inline float FastCos(float x)
  //{
  //  return FastSin(x+(float)PI_HALF);
  //}
  template<typename T, bool nullterminate = false>
  BSS_FORCEINLINE std::pair<std::unique_ptr<T[]>, size_t> bssloadfile(const char* file)
  {
    FILE* f = 0;
    WFOPEN(f, file, "rb");
    fseek(f, 0, SEEK_END);
    size_t ln = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::unique_ptr<T[]> a(new T[ln + !!nullterminate]);
    fread(a.get(), sizeof(T), ln, f);
    fclose(f);
    if(nullterminate)
      a[ln] = 0;
    return std::pair<std::unique_ptr<T[]>, size_t>(std::move(a), ln + !!nullterminate);
  }
  // Round a number up to the next power of 2 (32 -> 32, 33 -> 64, etc.)
  inline uint64_t nextpow2(uint64_t v) noexcept
  {
	  v -= 1;
	  v |= (v >> 1);
	  v |= (v >> 2);
	  v |= (v >> 4);
	  v |= (v >> 8);
	  v |= (v >> 16);
	  v |= (v >> 32);
	
	  return v + 1;
  }
  inline uint32_t nextpow2(uint32_t v) noexcept
  {
	  v -= 1;
	  v |= (v >> 1);
	  v |= (v >> 2);
	  v |= (v >> 4);
	  v |= (v >> 8);
	  v |= (v >> 16);
	
	  return v + 1;
  }
  inline uint16_t nextpow2(uint16_t v) noexcept
  {
	  v -= 1;
	  v |= (v >> 1);
	  v |= (v >> 2);
	  v |= (v >> 4);
	  v |= (v >> 8);
	
	  return v + 1;
  }
  inline uint8_t nextpow2(uint8_t v) noexcept
  {
	  v -= 1;
	  v |= (v >> 1);
	  v |= (v >> 2);
	  v |= (v >> 4);
	
	  return v + 1;
  }

#ifdef BSS_PLATFORM_WIN32
  inline uint32_t bsslog2(uint32_t v) noexcept
  {
    if(!v) return 0;
    unsigned long r; 
    _BitScanReverse(&r, v); 
    return r; 
  }
#elif defined(BSS_COMPILER_GCC)
  inline uint32_t bsslog2(uint32_t v) noexcept { return !v?0:((sizeof(uint32_t)<<3)-1-__builtin_clz(v)); }
#else
  // Bit-twiddling hack for base 2 log by Sean Eron Anderson
  inline uint32_t bsslog2(uint8_t v) noexcept
  {
    const uint32_t b[] = {0x2, 0xC, 0xF0};
    const uint32_t S[] = {1, 2, 4};

    uint32_t r = 0; // result of bsslog2(v) will go here
    if (v & b[2]) { v >>= S[2]; r |= S[2]; } 
    if (v & b[1]) { v >>= S[1]; r |= S[1]; } 
    if (v & b[0]) { v >>= S[0]; r |= S[0]; } 

    return r;
  }
  inline uint32_t bsslog2(uint16_t v) noexcept
  {
    const uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00};
    const uint32_t S[] = {1, 2, 4, 8};

    uint32_t r = 0; // result of bsslog2(v) will go here
    if (v & b[3]) { v >>= S[3]; r |= S[3]; } 
    if (v & b[2]) { v >>= S[2]; r |= S[2]; } 
    if (v & b[1]) { v >>= S[1]; r |= S[1]; } 
    if (v & b[0]) { v >>= S[0]; r |= S[0]; } 

    return r;
  }

  inline uint32_t bsslog2(uint32_t v) noexcept
  {
    const uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    const uint32_t S[] = {1, 2, 4, 8, 16};

    uint32_t r = 0; // result of bsslog2(v) will go here
    if (v & b[4]) { v >>= S[4]; r |= S[4]; } 
    if (v & b[3]) { v >>= S[3]; r |= S[3]; } 
    if (v & b[2]) { v >>= S[2]; r |= S[2]; } 
    if (v & b[1]) { v >>= S[1]; r |= S[1]; } 
    if (v & b[0]) { v >>= S[0]; r |= S[0]; } 

    return r;
  }
#endif
  inline uint32_t bsslog2(uint64_t v) noexcept
  {
#if defined(BSS_COMPILER_MSC) && defined(BSS_64BIT)
    if(!v) return 0;
    unsigned long r; 
    _BitScanReverse64(&r, v); 
#elif defined(BSS_COMPILER_GCC) && defined(BSS_64BIT)
    uint32_t r = !v?0:((sizeof(uint64_t)<<3)-1-__builtin_clz(v));
#else
    const uint64_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000, 0xFFFFFFFF00000000};
    const uint32_t S[] = {1, 2, 4, 8, 16, 32};

    uint32_t r = 0; // result of bsslog2(v) will go here
    if (v & b[5]) { v >>= S[5]; r |= S[5]; } 
    if (v & b[4]) { v >>= S[4]; r |= S[4]; } 
    if (v & b[3]) { v >>= S[3]; r |= S[3]; } 
    if (v & b[2]) { v >>= S[2]; r |= S[2]; } 
    if (v & b[1]) { v >>= S[1]; r |= S[1]; } 
    if (v & b[0]) { v >>= S[0]; r |= S[0]; } 
#endif

    return r;
  }
  inline uint32_t bsslog2_p2(uint32_t v) noexcept //Works only if v is a power of 2
  {
    assert(v && !(v & (v - 1))); //debug version checks to ensure its a power of two
#ifdef BSS_COMPILER_MSC
    unsigned long r;
    _BitScanReverse(&r,v);
#elif defined(BSS_COMPILER_GCC)
    uint32_t r = (sizeof(uint32_t)<<3)-1-__builtin_clz(v);
#else
    const uint32_t b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000};
    uint32_t r = (v & b[0]) != 0;
    r |= ((v & b[4]) != 0) << 4;
    r |= ((v & b[3]) != 0) << 3;
    r |= ((v & b[2]) != 0) << 2;
    r |= ((v & b[1]) != 0) << 1;
#endif
    return r;
  }

  template<class T>
  inline typename std::make_unsigned<T>::type bssabs(T x) noexcept
  {
    static_assert(std::is_signed<T>::value, "T must be signed for this to work properly.");
    T const mask = x >> ((sizeof(T) << 3) - 1); // Uses a bit twiddling hack to take absolute value without branching: https://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
    return (x + mask) ^ mask;
  }

  template<class T>
  inline typename std::make_signed<T>::type bssnegate(T x, char negate) noexcept
  {
    static_assert(std::is_unsigned<T>::value, "T must be unsigned for this to work properly.");
    return (x ^ -negate) + negate;
  }

  template<bool ENABLE, typename T>
  struct __bssabsnegate_h
  {
    BSS_FORCEINLINE static typename std::make_unsigned<T>::type _bssabs(T x) { return bssabs<T>(x); }
    BSS_FORCEINLINE static void _bssnegate(T& x, T& y, char negate) {
      if(negate)
      {
        y = ~y + !x; // only add one if the x addition would overflow, which can only happen if x is the maximum value
        x = ~x + 1;
      }
    }
  };
  template<typename T>
  struct __bssabsnegate_h<false, T>
  {
    BSS_FORCEINLINE static T _bssabs(T x) { return x; }
    BSS_FORCEINLINE static void _bssnegate(T& x, T& y, char negate) { }
  };

  // Double width multiplication followed by a right shift and truncation.
  template<class T>
  inline T __bssmultiplyextract__h(T xs, T ys, T shift) noexcept
  {
    typedef typename std::make_unsigned<T>::type U;
    U x = __bssabsnegate_h<std::is_signed<T>::value, T>::_bssabs(xs);
    U y = __bssabsnegate_h<std::is_signed<T>::value, T>::_bssabs(ys);
    static const U halfbits = (sizeof(U) << 2);
    static const U halfmask = ((U)~0) >> halfbits;
    U a = x >> halfbits, b = x & halfmask;
    U c = y >> halfbits, d = y & halfmask;

    U ac = a * c;
    U bc = b * c;
    U ad = a * d;
    U bd = b * d;

    U mid34 = (bd >> halfbits) + (bc & halfmask) + (ad & halfmask);

    U high = ac + (bc >> halfbits) + (ad >> halfbits) + (mid34 >> halfbits); // high
    U low = (mid34 << halfbits) | (bd & halfmask); // low
    __bssabsnegate_h<std::is_signed<T>::value, U>::_bssnegate(low, high, (xs < 0) ^ (ys < 0));

    if(shift >= (sizeof(U) << 3))
      return ((T)high) >> (shift - (sizeof(U) << 3));
    low = (low >> shift);
    high = (high << ((sizeof(T) << 3) - shift)) & (-(shift>0)); // shifting left by 64 bits is undefined, so we use a bit trick to set high to zero if shift is 0 without branching.
    return (T)(low | high);
  }
  template<class T>
  BSS_FORCEINLINE T bssmultiplyextract(T x, T y, T shift) noexcept
  {
    typedef typename std::conditional<std::is_signed<T>::value, typename BitLimit<sizeof(T) << 4>::SIGNED, typename BitLimit<sizeof(T) << 4>::UNSIGNED>::type U;
    return (T)(((U)x * (U)y) >> shift);
  }
#ifndef BSS_HASINT128
  template<>
  BSS_FORCEINLINE int64_t bssmultiplyextract<int64_t>(int64_t x, int64_t y, int64_t shift) noexcept { return __bssmultiplyextract__h<int64_t>(x, y, shift); }
  template<>
  BSS_FORCEINLINE uint64_t bssmultiplyextract<uint64_t>(uint64_t x, uint64_t y, uint64_t shift) noexcept { return __bssmultiplyextract__h<uint64_t>(x, y, shift); }
#endif

  template<class I>
  inline I days_from_civil(I y, unsigned m, unsigned d) noexcept
  {
    static_assert(std::numeric_limits<unsigned>::digits >= 18, "This algorithm has not been ported to a 16 bit unsigned integer");
    static_assert(std::numeric_limits<I>::digits >= 20, "This algorithm has not been ported to a 16 bit signed integer");
    y -= m <= 2;
    const I era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);      // [0, 399]
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;  // [0, 365]
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;         // [0, 146096]
    return era * 146097 + static_cast<I>(doe) - 719468;
  }

  // Generic function application to an array
  template<class T, size_t I, T (*F)(T,T)>
  BSS_FORCEINLINE std::array<T, I> arraymap(const std::array<T, I>& l, const std::array<T, I>& r) noexcept
  {
    std::array<T, I> x;
    for(size_t i = 0; i < I; ++i)
      x[i] = F(l[i], r[i]);
    return x;
  }

  // Basic lerp function with no bounds checking
  template<class T, class D = double>
  BSS_FORCEINLINE T lerp(T a, T b, D t) noexcept
  {
    return T((D(1.0) - t)*a) + T(t*b);
	  //return a+((T)((b-a)*t)); // This is susceptible to floating point errors when t = 1
  }

#ifdef BSS_VARIADIC_TEMPLATES

  // Generates a packed sequence of numbers
  template<int ...> struct bssSeq {};
  template<int N, int ...S> struct bssSeq_gens : bssSeq_gens<N-1, N-1, S...> {};
  template<int ...S> struct bssSeq_gens<0, S...>{ typedef bssSeq<S...> type; };
#endif

  BSS_COMPILER_DLLEXPORT extern void bssdll_delete_delfunc(void* p);

  //unique_ptr deleter class that forces the deletion to occur in this DLL
  template<class _Ty>
	struct bssdll_delete
	{	// default deleter for unique_ptr
	  typedef bssdll_delete<_Ty> _Myt;

	  void operator()(_Ty *_Ptr) const {
		  if (0 < sizeof (_Ty))	{ // won't compile for incomplete type
        _Ptr->~_Ty(); // call destructor because delete won't
			  bssdll_delete_delfunc(_Ptr);
		  }
    }
	};
  
  template<class _Ty>
	struct bssdll_delete<_Ty[]>
	{	// default deleter for unique_ptr to array of unknown size
	  typedef bssdll_delete<_Ty> _Myt;

	  void operator()(_Ty *_Ptr) const {
		  if (0 < sizeof (_Ty))	{ // won't compile for incomplete type
        _Ptr->~_Ty(); // call destructor because delete won't
			  bssdll_delete_delfunc(_Ptr);
      }
		}
	};
} 

#endif

  // Simple 128bit integer for x86 instruction set. Replaced with __int128 if possible
//#ifdef BSS32BIT
//  struct int128
//  {
//    inline static int128& operator+=(const int128& right) { _add(right.ints[0],right.ints[1],right.ints[2],right.ints[3]); return *this; }
//
//    int32_t ints[4];
//
//  private:
//    inline static void _add(int32_t a,int32_t b,int32_t c,int32_t d) //Done because I suck with assembly :D
//    {
//      __asm {
//#ifdef BSS_NO_FASTCALL //if we are using fastcall we don't need these instructions
//        mov ECX, a
//        mov EDX, b
//#endif
//        mov EAX, c
//        mov EBX, d
//        add ints[0], ECX
//        adc ints[1], EDX
//        adc ints[2], EAX
//        adc ints[3], EBX
//      }
//    }
//  };
//  struct uint128
//  {
//    inline static uint128& operator+=(const uint128& right) { _add(right.ints[0],right.ints[1],right.ints[2],right.ints[3]); return *this; }
//
//    uint32_t ints[4];
//
//  private:
//    inline static void _add(uint32_t a,uint32_t b,uint32_t c,uint32_t d) //Done because I suck with assembly :D
//    {
//      __asm {
//#ifdef BSS_NO_FASTCALL //if we are using fastcall we don't need these instructions
//        mov ECX, a
//        mov EDX, b
//#endif
//        mov EAX, c
//        mov EBX, d
//        add ints[0], ECX
//        adc ints[1], EDX
//        adc ints[2], EAX
//        adc ints[3], EBX
//      }
//    }
//  };
//#else
//  typedef __int128 int128;
//  typedef unsigned __int128 uint128;
//#endif
