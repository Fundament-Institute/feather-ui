// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __BSS_UTIL_C_H__
#define __BSS_UTIL_C_H__

#include "bss_defines.h"
#include <wchar.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif
  
struct VersionType
{
  union {
    struct {
      unsigned char Major;
      unsigned char Minor;
      unsigned short Revision;
    };
    unsigned int Version;
  };
};

union bssCPUInfo {
  struct {
    unsigned short cores; // Number of logical cores (not physical cores). If you need more than 65535 cores, you're apparently using code written in 2013 in the year 2080, so stop being a douchebag and upgrade already.
    unsigned char SSE; // 1: Supports MMX, 2: Supports SSE+MMX, 3: up to SSE2, 4: SSE3, 5: SSSE3, 6: SSE4.1, 7: SSE4.2, 8: AVX, 9: AVX2
    unsigned int flags; // 1 - supports CMPXCHG8b, 2 - supports CMPXCHG16b, 4 - supports AMD SSE4a, 8 - supports CMOV, 16 - supports CFLUSH, 32 - supports POPCNT
  };
  unsigned long long _raw;
};

struct tm;

BSS_COMPILER_DLLEXPORT extern union bssCPUInfo bssGetCPUInfo();
BSS_COMPILER_DLLEXPORT extern size_t BSS_FASTCALL UTF8toUTF16(const char*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen);
BSS_COMPILER_DLLEXPORT extern size_t BSS_FASTCALL UTF16toUTF8(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen);
BSS_COMPILER_DLLEXPORT extern size_t BSS_FASTCALL UTF8toUTF32(const char*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
BSS_COMPILER_DLLEXPORT extern size_t BSS_FASTCALL UTF32toUTF8(const int*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen);
BSS_COMPILER_DLLEXPORT extern size_t BSS_FASTCALL UTF32toUTF16(const int*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen);
BSS_COMPILER_DLLEXPORT extern size_t BSS_FASTCALL UTF16toUTF32(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
BSS_COMPILER_DLLEXPORT extern int itoa_r(int value, char* buffer, int size, unsigned int radix); // For various stupid reasons we must reimplement multiple threadsafe versions of various functions because MinGW doesn't have them.
BSS_COMPILER_DLLEXPORT extern const char* GetProgramPath();
BSS_COMPILER_DLLEXPORT extern size_t GetWorkingSet();
BSS_COMPILER_DLLEXPORT extern size_t GetPeakWorkingSet();
//BSS_COMPILER_DLLEXPORT extern struct tm* gmtime64_r(const long long*BSS_RESTRICT clock, struct tm*BSS_RESTRICT result);
#ifdef BSS_PLATFORM_MINGW
BSS_COMPILER_DLLEXPORT extern char* strtok_r(char* s, const char* delim, char** lasts);
BSS_COMPILER_DLLEXPORT extern wchar_t* wcstok_r(wchar_t* s, const wchar_t* delim, wchar_t** lasts);
#endif

#ifdef  __cplusplus
}

// templatized implementation of itoa_r so it can work with stack allocated arrays
template<int SZ>
BSS_FORCEINLINE int BSS_FASTCALL _itoa_r(int value, char (&buffer)[SZ], unsigned int radix) { return itoa_r(value,buffer,SZ,radix); }
#endif

#endif
