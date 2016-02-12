// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __BSS_COMPILER_H__
#define __BSS_COMPILER_H__

// CPU Architecture (possible pre-defined macros found on http://predef.sourceforge.net/prearch.html)
#if defined(_M_X64) || defined(__amd64__) || defined(__amd64) || defined(_AMD64_) || defined(__x86_64__) || defined(__x86_64) || defined(_LP64)
#define BSS_CPU_x86_64  //x86-64 architecture
#define BSS_64BIT
#elif defined(__ia64__) || defined(_IA64) || defined(__IA64__) || defined(__ia64) || defined(_M_IA64)
#define BSS_CPU_IA_64 //Itanium (IA-64) architecture
#define BSS_64BIT
#elif defined(_M_IX86) || defined(__i386) || defined(__i386__) || defined(__X86__) || defined(_X86_) || defined(__I86__) || defined(__THW_INTEL__) || defined(__INTEL__)
#define BSS_CPU_x86  //x86 architecture
#define BSS_32BIT
#elif defined(__arm__) || defined(__thumb__) || defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || defined(_ARM)
#define BSS_CPU_ARM //ARM architecture
//#ifndef(???) //ARMv8 will support 64-bit so we'll have to detect that somehow, and it's the first to make NEON standardized.
#define BSS_32BIT
//#else
//#define BSS_64BIT
//#endif
#elif defined(__mips__) || defined(mips) || defined(_MIPS_ISA) || defined(__mips) || defined(__MIPS__)
#define BSS_CPU_MIPS
#define BSS_64BIT
#elif defined(__powerpc) || defined(__powerpc__) || defined(__POWERPC__) || defined(__ppc__) || defined(_M_PPC) || defined(_ARCH_PPC)
#define BSS_CPU_POWERPC
#define BSS_32BIT
#else
#define BSS_CPU_UNKNOWN //Unknown CPU architecture (should force architecture independent C implementations for all utilities)
#endif

#define BSS_COMPILER_DELETEFUNC = delete;
#define BSS_COMPILER_DELETEOPFUNC = delete;

// Compiler detection and macro generation
#if defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC) // Intel C++ compiler
#define BSS_COMPILER_INTEL
#define BSS_COMPILER_DLLEXPORT __declspec(dllexport)
#define BSS_COMPILER_DLLIMPORT __declspec(dllimport)
#define BSS_TEMPLATE_DLLEXPORT
#define MEMBARRIER_READWRITE __memory_barrier()
#define MEMBARRIER_READ MEMBARRIER_READWRITE
#define MEMBARRIER_WRITE MEMBARRIER_READWRITE
#define BSS_COMPILER_FASTCALL
#define BSS_COMPILER_STDCALL
#define BSS_COMPILER_NAKED
#define BSS_FORCEINLINE inline
#define BSS_RESTRICT __restrict__
#define BSS_ALIGNED(sn, n) sn
#define MSC_FASTCALL BSS_FASTCALL
#define GCC_FASTCALL 
#define BSS_DELETEFUNC BSS_COMPILER_DELETEFUNC
#define BSS_DELETEFUNCOP BSS_COMPILER_DELETEOPFUNC
#define BSS_UNREACHABLE() 
#define BSS_ASSUME(x) 
#define BSS_EXPLICITSTATIC static

#elif defined(__clang__) // Clang (must be before GCC, because clang also pretends it's GCC)
#define BSS_COMPILER_CLANG
#define BSS_COMPILER_DLLEXPORT __attribute__((dllexport))
#define BSS_COMPILER_DLLIMPORT __attribute__((dllimport))
#define BSS_TEMPLATE_DLLEXPORT
#define MEMBARRIER_READWRITE __asm__ __volatile__ ("" ::: "memory");
#define MEMBARRIER_READ MEMBARRIER_READWRITE
#define MEMBARRIER_WRITE MEMBARRIER_READWRITE
#define BSS_COMPILER_FASTCALL __attribute__((fastcall))
#define BSS_COMPILER_STDCALL __attribute__((stdcall))
#define BSS_COMPILER_NAKED __attribute__((naked)) // Will only work on ARM, AVR, MCORE, RX and SPU. 
#define BSS_FORCEINLINE __attribute__((always_inline))
#define BSS_RESTRICT __restrict__
#define BSS_ALIGN(n) __attribute__((aligned(n)))
#define BSS_ALIGNED(sn, n) sn BSS_ALIGN(n)
#define BSS_VARIADIC_TEMPLATES

#define MSC_FASTCALL 
#define GCC_FASTCALL BSS_FASTCALL
#define BSS_DELETEFUNC BSS_COMPILER_DELETEFUNC
#define BSS_DELETEFUNCOP BSS_COMPILER_DELETEOPFUNC
#define BSS_EXPLICITSTATIC // GCC says that putting "static" on explicit templatizations of functions is illegal. VC++ breaks if you don't.

#elif defined __GNUC__ // GCC
#define BSS_COMPILER_GCC
#define BSS_COMPILER_DLLEXPORT __attribute__((dllexport))
#define BSS_COMPILER_DLLIMPORT __attribute__((dllimport))
#define BSS_TEMPLATE_DLLEXPORT
#define MEMBARRIER_READWRITE __asm__ __volatile__ ("" ::: "memory");
#define MEMBARRIER_READ MEMBARRIER_READWRITE
#define MEMBARRIER_WRITE MEMBARRIER_READWRITE
#define BSS_COMPILER_FASTCALL __attribute__((fastcall))
#define BSS_COMPILER_STDCALL __attribute__((stdcall))
#define BSS_COMPILER_NAKED __attribute__((naked)) // Will only work on ARM, AVR, MCORE, RX and SPU. 
#define BSS_FORCEINLINE __attribute__((always_inline))
#define BSS_RESTRICT __restrict__
#define BSS_ALIGN(n) __attribute__((aligned(n)))
#define BSS_ALIGNED(sn, n) sn BSS_ALIGN(n)
#define BSS_VARIADIC_TEMPLATES

#define MSC_FASTCALL 
#define GCC_FASTCALL BSS_FASTCALL
#define BSS_DELETEFUNC BSS_COMPILER_DELETEFUNC
#define BSS_DELETEFUNCOP BSS_COMPILER_DELETEOPFUNC
#define BSS_EXPLICITSTATIC // GCC says that putting "static" on explicit templatizations of functions is illegal. VC++ breaks if you don't.

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if __has_builtin(__builtin_unreachable) || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)
#define PORTABLE_UNREACHABLE() __builtin_unreachable()
#define PORTABLE_ASSUME(x) do { if (!(x)) { __builtin_unreachable(); } } while (0)
#endif

#elif defined _MSC_VER // VC++
#define BSS_COMPILER_MSC
#define BSS_COMPILER_DLLEXPORT __declspec(dllexport)
#define BSS_COMPILER_DLLIMPORT __declspec(dllimport)
#define BSS_TEMPLATE_DLLEXPORT BSS_COMPILER_DLLEXPORT
#define MEMBARRIER_READWRITE _ReadWriteBarrier()
#define MEMBARRIER_READ _ReadBarrier()
#define MEMBARRIER_WRITE _WriteBarrier()
#define BSS_COMPILER_FASTCALL __fastcall
#define BSS_COMPILER_STDCALL __stdcall
#define BSS_COMPILER_NAKED __declspec(naked) 
#define BSS_FORCEINLINE __forceinline
#define BSS_RESTRICT __restrict
#define BSS_ALIGN(n) __declspec(align(n))
#define BSS_ALIGNED(sn, n) BSS_ALIGN(n) sn
#define BSS_VERIFY_HEAP _ASSERTE(_CrtCheckMemory())
#define FUNCPTRCC(m,CC) CC m
#define MSC_FASTCALL BSS_FASTCALL
#define GCC_FASTCALL 
#define BSS_EXPLICITSTATIC static // GCC says that putting "static" on explicit templatizations of functions is illegal. VC++ breaks if you don't.
#define BSS_UNREACHABLE() __assume(0)
#define BSS_ASSUME(x) __assume(x)
#if (_MANAGED == 1) || (_M_CEE == 1)
#define MSC_MANAGED
#endif

#if _MSC_VER >= 1800
#define BSS_VARIADIC_TEMPLATES
#define BSS_DELETEFUNC BSS_COMPILER_DELETEFUNC
#define BSS_DELETEFUNCOP BSS_COMPILER_DELETEOPFUNC
#elif _MSC_VER >= 1700
#define BSS_DELETEFUNC BSS_COMPILER_DELETEFUNC
#define BSS_DELETEFUNCOP BSS_COMPILER_DELETEOPFUNC
#elif _MSC_VER >= 1600
#define BSS_COMPILER_MSC2010
#define BSS_DELETEFUNC { assert(false); }
#define BSS_DELETEFUNCOP { assert(false); return *this; }
#endif
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
#define BSS_PLATFORM_MINGW // Should also define WIN32, use only for minGW specific bugs
#endif

#define BSS_ALIGNED_STRUCT(n) struct BSS_ALIGN(n)
#define BSS_ALIGNED_CLASS(n) class BSS_ALIGN(n)
#define BSS_ALIGNED_UNION(n) union BSS_ALIGN(n)

// Platform detection
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(__TOS_WIN__) || defined(__WINDOWS__)
#define BSS_PLATFORM_WIN32
#elif defined(_POSIX_VERSION) || defined(_XOPEN_VERSION) || defined(unix) || defined(__unix__) || defined(__unix)
#define BSS_PLATFORM_POSIX
#endif

#ifdef _WIN32_WCE
#define BSS_PLATFORM_WIN32_CE // Implies WIN32
#elif defined(__APPLE__) || defined(__MACH__) || defined(macintosh) || defined(Macintosh)
#define BSS_PLATFORM_APPLE // Should also define POSIX, use only for Apple OS specific features
#elif defined(__CYGWIN__)
#define BSS_PLATFORM_CYGWIN // Should also define POSIX, use only to deal with Cygwin weirdness
#elif defined(__ANDROID__) || defined(__ANDROID_API__) 
#define BSS_PLATFORM_ANDROID // Should also define POSIX, use for Android specific features.
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__) || defined(BSD) // Also defines POSIX
#define BSS_PLATFORM_BSD // Should also define POSIX
#elif defined(sun) || defined(__sun) 
# define BSS_PLATFORM_SOLARIS
# if !defined(__SVR4) && !defined(__svr4__)
#   define BSS_PLATFORM_SUNOS
# endif
#endif

#if defined(__linux__) || defined(__linux)
#define BSS_PLATFORM_LINUX // Should also define POSIX, use only for linux specific features
#endif

#if !(defined(BSS_PLATFORM_WIN32) || defined(BSS_PLATFORM_POSIX) || defined(BSS_PLATFORM_WIN32_CE) || defined(BSS_PLATFORM_APPLE))
#error "Unknown Platform"
#endif

// Endianness detection
#if defined(BSS_PLATFORM_WIN32) || defined(BSS_PLATFORM_WIN32_CE) || defined(BSS_CPU_x86_64) || defined(BSS_CPU_x86) || defined(BSS_CPU_IA_64) // Windows, x86, x86_64 and itanium all only run in little-endian (except on HP-UX but we don't support that)
# define BSS_ENDIAN_LITTLE
#elif defined(BSS_CPU_ARM)
# ifdef BSS_PLATFORM_LINUX
#   define BSS_ENDIAN_LITTLE
# endif
#elif defined(BSS_CPU_POWERPC)
# ifdef BSS_PLATFORM_SOLARIS
#   define BSS_ENDIAN_LITTLE
# elif defined(BSS_PLATFORM_APPLE) || defined(BSS_PLATFORM_BSD) || defined(BSS_PLATFORM_LINUX)
#   define BSS_ENDIAN_BIG
# endif
#elif defined(BSS_CPU_MIPS) // MIPS is a bitch to detect endianness on
# ifdef BSS_PLATFORM_LINUX
#   define BSS_ENDIAN_BIG
# endif
#endif

#if !defined(BSS_ENDIAN_LITTLE) && !defined(BSS_ENDIAN_BIG)
#error "Unknown Endianness"
#endif

// Debug detection
#ifdef BSS_COMPILER_GCC
#ifndef NDEBUG
#define BSS_DEBUG
#endif
#else
#if defined(DEBUG) || defined(_DEBUG)
#define BSS_DEBUG
#endif
#endif

#endif
