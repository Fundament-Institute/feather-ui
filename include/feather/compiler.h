// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__COMPILER_H
#define FG__COMPILER_H

// CPU Architecture (possible pre-defined macros found on http://predef.sourceforge.net/prearch.html)
#if defined(_M_X64) || defined(__amd64__) || defined(__amd64) || defined(_AMD64_) || defined(__x86_64__) || defined(__x86_64) || defined(_LP64)
#define FG_CPU_x86_64  //x86-64 architecture
#define FG_64BIT
#elif defined(__ia64__) || defined(_IA64) || defined(__IA64__) || defined(__ia64) || defined(_M_IA64)
#define FG_CPU_IA_64 //Itanium (IA-64) architecture
#define FG_64BIT
#elif defined(_M_IX86) || defined(__i386) || defined(__i386__) || defined(__X86__) || defined(_X86_) || defined(__I86__) || defined(__THW_INTEL__) || defined(__INTEL__)
#define FG_CPU_x86  //x86 architecture
#define FG_32BIT
#elif defined(__arm__) || defined(__thumb__) || defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || defined(_ARM) || defined(__aarch64__) 
#ifdef __aarch64__  
#define FG_CPU_ARM64 //ARM 64-bit architecture
#define FG_64BIT
#else
#define FG_CPU_ARM //ARM 32-bit architecture
#define FG_32BIT
#endif

#ifdef __ARM_ARCH_2_
#define FG_CPU_ARM_V 2
#elif defined(__ARM_ARCH_3__) || defined(__ARM_ARCH_3M__)
#define FG_CPU_ARM_V 3
#elif defined(__ARM_ARCH_4T__) || defined(__TARGET_ARM_4T)
#define FG_CPU_ARM_V 4
#elif defined(__ARM_ARCH_5__) || defined(__ARM_ARCH_5E__) || defined(__ARM_ARCH_5T__) || defined(__ARM_ARCH_5TE__) || defined(__ARM_ARCH_5TEJ__) || (_M_ARM == 5)
#define FG_CPU_ARM_V 5
#elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__) || defined(__ARM_ARCH_6T2__) || (_M_ARM == 6)
#define FG_CPU_ARM_V 6
#elif defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__) || (_M_ARM == 7)
#define FG_CPU_ARM_V 7
#elif defined(__ARM_ARCH_8__) || (_M_ARM == 8)
#define FG_CPU_ARM_V 8
#endif

#if (FG_CPU_ARM_V > 4) || defined(__ARM_ARCH_4T__)
#define ARCH_HAS_BX
#endif
#if FG_CPU_ARM_V > 4
#define ARCH_HAS_BLX
#endif

#elif defined(__mips__) || defined(mips) || defined(_MIPS_ISA) || defined(__mips) || defined(__MIPS__)
#define FG_CPU_MIPS
#define FG_64BIT
#elif defined(__powerpc) || defined(__powerpc__) || defined(__POWERPC__) || defined(__ppc__) || defined(_M_PPC) || defined(_ARCH_PPC)
#if defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__) || defined(__64BIT__) || defined(_LP64) || defined(__LP64__)
#define FG_CPU_POWERPC64
#define FG_64BIT
#else
#define FG_CPU_POWERPC
#define FG_32BIT
#endif
#else
#define FG_CPU_UNKNOWN //Unknown CPU architecture (should force architecture independent C implementations)
#endif

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

// Platform detection
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(__TOS_WFG__) || defined(__WINDOWS__)
#define FG_PLATFORM_WIN32
#elif defined(_POSIX_VERSION) || defined(_XOPEN_VERSION) || defined(unix) || defined(__unix__) || defined(__unix)
#define FG_PLATFORM_POSIX
#endif

#ifdef _WIN32_WCE
#define FG_PLATFORM_WIN32_CE // Implies WIN32
#elif defined(__APPLE__) || defined(__MACH__) || defined(macintosh) || defined(Macintosh)
#define FG_PLATFORM_APPLE // Should also define POSIX, use only for Apple OS specific features
#elif defined(__CYGWFG__)
#define FG_PLATFORM_CYGWIN // Should also define POSIX, use only to deal with Cygwin weirdness
#elif defined(__ANDROID__) || defined(__ANDROID_API__) 
#define FG_PLATFORM_ANDROID // Should also define POSIX, use for Android specific features.
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__) || defined(BSD) // Also defines POSIX
#define FG_PLATFORM_BSD // Should also define POSIX
#elif defined(sun) || defined(__sun) 
# define FG_PLATFORM_SOLARIS
# if !defined(__SVR4) && !defined(__svr4__)
#   define FG_PLATFORM_SUNOS
# endif
#endif

#if defined(__linux__) || defined(__linux)
#define FG_PLATFORM_LINUX // Should also define POSIX, use only for linux specific features
#endif

#if !(defined(FG_PLATFORM_WIN32) || defined(FG_PLATFORM_POSIX) || defined(FG_PLATFORM_WIN32_CE) || defined(FG_PLATFORM_APPLE))
#error "Unknown Platform"
#endif

// Endianness detection
#if defined(FG_PLATFORM_WIN32) || defined(FG_PLATFORM_WIN32_CE) || defined(FG_CPU_x86_64) || defined(FG_CPU_x86) || defined(FG_CPU_IA_64) // Windows, x86, x86_64 and itanium all only run in little-endian (except on HP-UX but we don't support that)
# define FG_ENDIAN_LITTLE
#elif defined(FG_CPU_ARM)
# ifdef FG_PLATFORM_LINUX
#   define FG_ENDIAN_LITTLE
# endif
#elif defined(FG_CPU_POWERPC)
# ifdef FG_PLATFORM_SOLARIS
#   define FG_ENDIAN_LITTLE
# elif defined(FG_PLATFORM_APPLE) || defined(FG_PLATFORM_BSD) || defined(FG_PLATFORM_LINUX)
#   define FG_ENDIAN_BIG
# endif
#elif defined(FG_CPU_MIPS) // MIPS is a bitch to detect endianness on
# ifdef FG_PLATFORM_LINUX
#   define FG_ENDIAN_BIG
# endif
#endif

#if !defined(FG_ENDIAN_LITTLE) && !defined(FG_ENDIAN_BIG)
#error "Unknown Endianness"
#endif

// Debug detection
#ifdef FG_COMPILER_GCC
#ifndef NDEBUG
#define FG_DEBUG
#endif
#else
#if defined(DEBUG) || defined(_DEBUG)
#define FG_DEBUG
#endif
#endif

#endif
