// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__COMPILER_H
#define GL__COMPILER_H

// Compiler detection and macro generation
#if defined(__clang__) // Clang (must be before GCC, because clang also pretends it's GCC)
#define FG_COMPILER_CLANG
#define FG_COMPILER_DLLEXPORT __attribute__((dllexport))
#define FG_COMPILER_DLLIMPORT __attribute__((dllimport))
#define FG_COMPILER_FASTCALL __attribute__((fastcall))
#define FG_COMPILER_NAKED __attribute__((naked))
#define FG_FORCEINLINE __attribute__((always_inline)) inline
#define FG_RESTRICT __restrict__
#define FG_ALIGN(n) __attribute__((aligned(n)))
#define FG_ALIGNED(sn, n) sn FG_ALIGN(n)
#elif defined __GNUC__ // GCC
#define FG_COMPILER_GCC
#define FG_COMPILER_DLLEXPORT __attribute__((dllexport))
#define FG_COMPILER_DLLIMPORT __attribute__((dllimport))
#define FG_COMPILER_FASTCALL __attribute__((fastcall))
#define FG_COMPILER_NAKED __attribute__((naked))
#define FG_FORCEINLINE __attribute__((always_inline)) inline
#define FG_RESTRICT __restrict__
#define FG_ALIGN(n) __attribute__((aligned(n)))
#define FG_ALIGNED(sn, n) sn FG_ALIGN(n)
#elif defined _MSC_VER // VC++
#define FG_COMPILER_MSC
#define FG_COMPILER_DLLEXPORT __declspec(dllexport)
#define FG_COMPILER_DLLIMPORT __declspec(dllimport)
#define FG_COMPILER_FASTCALL __fastcall
#define FG_FORCEINLINE __forceinline
#define FG_RESTRICT __restrict
#define FG_ALIGN(n) __declspec(align(n))
#define FG_ALIGNED(sn, n) FG_ALIGN(n) sn
#define FG_SSE_ENABLED
#define FG_ASSUME(x) __assume(x)
#define _HAS_EXCEPTIONS 0
#endif

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(__TOS_WFG__) || defined(__WINDOWS__)
#define FG_PLATFORM_WIN32
#else
#define FG_PLATFORM_POSIX
#endif

#ifdef FG_PLATFORM_WIN32
#define ALLOCA(x) _alloca(x)
#define MEMCPY(d, size, s, len) memcpy_s(d, size, s, len)
#define ALIGNEDALLOC(size, align) _aligned_malloc(size, align)
#define ALIGNEDFREE(p) _aligned_free(p)
#else
#define ALLOCA(x) alloca(x)
#define MEMCPY(d, size, s, len) memcpy(d, s, len)
#define ALIGNEDALLOC(size, align) aligned_alloc(align, size)
#define ALIGNEDFREE(p) free(p)
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
#define fgassert(x) if(!(x)) { int* p = nullptr; *p = 1; }
#else
#define fgassert(x) 
#endif

#define LOGEMPTY
#define LOGFAILURE(x, f, ...) { HRESULT hr = (x); if(FAILED(hr)) { (*instance->_log)(instance->_root, FG_Level_ERROR, f, __VA_ARGS__); } }
#define LOGFAILURERET(x, r, f, ...) { HRESULT hr = (x); if(FAILED(hr)) { (*instance->_log)(instance->_root, FG_Level_ERROR, f, __VA_ARGS__); return r; } }
#define LOGFAILURERETNULL(x, f, ...) LOGFAILURERET(x,LOGEMPTY,f,__VA_ARGS__)

#ifdef FG_32BIT
#define kh_ptr_hash_func kh_int_hash_func
#else
#define kh_ptr_hash_func(key) kh_int64_hash_func((uint64_t)key)
#endif

#endif
