// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__COMPILER_H
#define GL__COMPILER_H

// CPU Architecture (possible pre-defined macros found on http://predef.sourceforge.net/prearch.html)
#if defined(_M_X64) || defined(__amd64__) || defined(__amd64) || defined(_AMD64_) || defined(__x86_64__) || \
  defined(__x86_64) || defined(_LP64)
  #define BSS_CPU_x86_64 // x86-64 architecture
  #define BSS_64BIT
#elif defined(__ia64__) || defined(_IA64) || defined(__IA64__) || defined(__ia64) || defined(_M_IA64)
  #define BSS_CPU_IA_64 // Itanium (IA-64) architecture
  #define BSS_64BIT
#elif defined(_M_IX86) || defined(__i386) || defined(__i386__) || defined(__X86__) || defined(_X86_) || \
  defined(__I86__) || defined(__THW_INTEL__) || defined(__INTEL__)
  #define BSS_CPU_x86 // x86 architecture
  #define BSS_32BIT
#elif defined(__arm__) || defined(__thumb__) || defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || defined(_ARM)
  #define BSS_CPU_ARM // ARM architecture
  //#ifndef(???) //ARMv8 will support 64-bit so we'll have to detect that somehow, and it's the first to make NEON
  //standardized.
  #define BSS_32BIT
//#else
//#define BSS_64BIT
//#endif
#elif defined(__mips__) || defined(mips) || defined(_MIPS_ISA) || defined(__mips) || defined(__MIPS__)
  #define BSS_CPU_MIPS
  #define BSS_64BIT
#elif defined(__powerpc) || defined(__powerpc__) || defined(__POWERPC__) || defined(__ppc__) || defined(_M_PPC) || \
  defined(_ARCH_PPC)
  #define BSS_CPU_POWERPC
  #define BSS_32BIT
#else
  #define BSS_CPU_UNKNOWN // Unknown CPU architecture (should force architecture independent C implementations for all
                          // utilities)
#endif

// Compiler detection and macro generation
#if defined(__clang__) // Clang (must be before GCC, because clang also pretends it's GCC)
  #define FG_COMPILER_CLANG
  #define FG_COMPILER_DLLEXPORT __attribute__((dllexport))
  #define FG_COMPILER_DLLIMPORT __attribute__((dllimport))
  #define FG_COMPILER_FASTCALL  __attribute__((fastcall))
  #define FG_COMPILER_NAKED     __attribute__((naked))
  #define FG_FORCEINLINE        __attribute__((always_inline)) inline
  #define FG_RESTRICT           __restrict__
  #define FG_ALIGN(n)           __attribute__((aligned(n)))
  #define FG_ALIGNED(sn, n)     sn FG_ALIGN(n)
#elif defined __GNUC__ // GCC
  #define FG_COMPILER_GCC
  #define FG_COMPILER_DLLEXPORT __attribute__((dllexport))
  #define FG_COMPILER_DLLIMPORT __attribute__((dllimport))
  #define FG_COMPILER_FASTCALL  __attribute__((fastcall))
  #define FG_COMPILER_NAKED     __attribute__((naked))
  #define FG_FORCEINLINE        __attribute__((always_inline)) inline
  #define FG_RESTRICT           __restrict__
  #define FG_ALIGN(n)           __attribute__((aligned(n)))
  #define FG_ALIGNED(sn, n)     sn FG_ALIGN(n)
#elif defined _MSC_VER // VC++
  #define FG_COMPILER_MSC
  #define FG_COMPILER_DLLEXPORT __declspec(dllexport)
  #define FG_COMPILER_DLLIMPORT __declspec(dllimport)
  #define FG_COMPILER_FASTCALL  __fastcall
  #define FG_FORCEINLINE        __forceinline
  #define FG_RESTRICT           __restrict
  #define FG_ALIGN(n)           __declspec(align(n))
  #define FG_ALIGNED(sn, n)     FG_ALIGN(n) sn
  #define FG_ASSUME(x)    __assume(x)
  #define _HAS_EXCEPTIONS 0
#endif

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(__TOS_WFG__) || defined(__WINDOWS__)
  #define FG_PLATFORM_WIN32
#else
  #define FG_PLATFORM_POSIX
#endif

#ifdef FG_PLATFORM_WIN32
  #define ALLOCA(x)                 _alloca(x)
  #define MEMCPY(d, size, s, len)   memcpy_s(d, size, s, len)
  #define ALIGNEDALLOC(size, align) _aligned_malloc(size, align)
  #define ALIGNEDFREE(p)            _aligned_free(p)
#else
  #define ALLOCA(x)                 alloca(x)
  #define MEMCPY(d, size, s, len)   memcpy(d, s, len)
  #define ALIGNEDALLOC(size, align) aligned_alloc(align, size)
  #define ALIGNEDFREE(p)            free(p)
#endif

#ifdef FG_COMPILER_GCC
  #ifndef NDEBUG
    #define FG_DEBUG
  #endif
#else
  #if defined(DEBUG) || defined(_DEBUG)
    #define FG_DEBUG
  #endif
#endif

#ifdef FG_DEBUG
  #define fgassert(x)   \
    if(!(x))            \
    {                   \
      int* p = nullptr; \
      *p     = 1;       \
    }
#else
  #define fgassert(x)
#endif

#define LOGEMPTY
#define LOGFAILURE(x, f, ...)                                             \
  {                                                                       \
    HRESULT hr = (x);                                                     \
    if(FAILED(hr))                                                        \
    {                                                                     \
      (*instance->_log)(instance->_root, FG_Level_Error, f, __VA_ARGS__); \
    }                                                                     \
  }
#define LOGFAILURERET(x, r, f, ...)                                       \
  {                                                                       \
    HRESULT hr = (x);                                                     \
    if(FAILED(hr))                                                        \
    {                                                                     \
      (*instance->_log)(instance->_root, FG_Level_Error, f, __VA_ARGS__); \
      return r;                                                           \
    }                                                                     \
  }
#define LOGFAILURERETNULL(x, f, ...) LOGFAILURERET(x, LOGEMPTY, f, __VA_ARGS__)

#ifdef FG_32BIT
  #define kh_ptr_hash_func kh_int_hash_func
#else
  #define kh_ptr_hash_func(key) kh_int64_hash_func((uint64_t)key)
#endif

#endif
