// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __LOCKLESS_H__BSS__
#define __LOCKLESS_H__BSS__

#include "bss_defines.h"
#ifdef BSS_COMPILER_MSC
#include <intrin.h>
#endif
#ifndef BSS_COMPILER_MSC2010
#include <atomic>
#endif

#ifdef BSS_CPU_x86
#define BSSASM_PREG ECX
#define BSSASM_PREGA EAX
#elif defined(BSS_CPU_x86_64)
#define BSSASM_PREG RCX
#define BSSASM_PREGA RAX
#endif

// These are all implemented using compiler intrinsics on MS compilers because the inline assembly does not behave well when inlined. GCC 
// intrinsics are not used due to inconsistent locking barriers, and the wide availability of standardized inline assembly for them.
namespace bss_util {
  template<typename T> struct bss_PTag { T* p; size_t tag; }; // Stores a pointer and a tag value
  
#if defined(BSS_CPU_x86_64) || defined(BSS_CPU_x86)
#pragma warning(push)
#pragma warning(disable : 4793)

  // Enforces a CPU barrier to prevent any reordering attempts
	BSS_FORCEINLINE void CPU_Barrier()
	{
#ifdef BSS_COMPILER_CLANG
    __sync_synchronize();
#elif defined(BSS_COMPILER_MSC)
#ifdef BSS_CPU_x86
		int32_t Barrier;
		__asm { /*lock*/ xchg Barrier, eax } //xchg locks itself if both operands are registers and blows up if you lock it anyway in a multiprocessor environment.
#elif defined(BSS_CPU_x86_64)
    __faststorefence();
#elif defined(BSS_CPU_IA_64)
    __mf();
#endif
#elif defined(BSS_COMPILER_GCC)
    int32_t Barrier = 0;
    __asm__ __volatile__("xchgl %%eax,%0 ":"=r" (Barrier));
#endif
	}
#pragma warning(pop)

  template<typename T, int size>
  struct ATOMIC_XADDPICK { };
  
#ifdef BSS_PLATFORM_WIN32
  template<typename T> struct ATOMIC_XADDPICK<T, 1> { BSS_FORCEINLINE static T atomic_xadd(volatile T* p, T v) { return (T)_InterlockedExchangeAdd8((volatile char*)p, v); } };
  template<typename T> struct ATOMIC_XADDPICK<T, 2> { BSS_FORCEINLINE static T atomic_xadd(volatile T* p, T v) { return (T)_InterlockedExchangeAdd16((volatile short*)p, v); } };
  template<typename T> struct ATOMIC_XADDPICK<T, 4> { BSS_FORCEINLINE static T atomic_xadd(volatile T* p, T v) { return (T)_InterlockedExchangeAdd((volatile long*)p, v); } };
  template<typename T> struct ATOMIC_XADDPICK<T, 8> { BSS_FORCEINLINE static T atomic_xadd(volatile T* p, T v) { return (T)_InterlockedExchangeAdd64((volatile int64_t*)p, v); } };
#elif defined(BSS_COMPILER_GCC) || defined(BSS_COMPILER_CLANG)
#define ATOMIC_XADDPICK_MACRO(INST,SIZE)  template<typename T> \
  struct ATOMIC_XADDPICK<T,SIZE> \
  { \
    static BSS_FORCEINLINE T atomic_xadd(volatile T* p, T val) \
    { \
        __asm__ __volatile__("lock " INST " %0,%1" :"+r" (val),"+m" (*p): : "memory"); \
        return val; \
    } \
  }

  ATOMIC_XADDPICK_MACRO("xaddb", 1);
  ATOMIC_XADDPICK_MACRO("xaddw", 2);
  ATOMIC_XADDPICK_MACRO("xaddl", 4);
#ifdef BSS_CPU_x86_64
  ATOMIC_XADDPICK_MACRO("xaddq", 8);
#endif
#endif
  
  template<typename T> // This performs an atomic addition, and returns the value of the variable BEFORE the addition. This is faster if p is 32-bit aligned.
  BSS_FORCEINLINE T atomic_xadd(volatile T* p, T val=1)
  { 
    return ATOMIC_XADDPICK<T,sizeof(T)>::atomic_xadd(p,val);
  }

  template<typename T, int size>
  struct ATOMIC_XCHGPICK { };
  
#ifdef BSS_PLATFORM_WIN32
  template<typename T> struct ATOMIC_XCHGPICK<T, 1> { static BSS_FORCEINLINE T atomic_xchg(volatile T* p, T val) { return (T)_InterlockedExchange8((volatile char*)p, (char)val); } };
  template<typename T> struct ATOMIC_XCHGPICK<T, 2> { static BSS_FORCEINLINE T atomic_xchg(volatile T* p, T val) { return (T)_InterlockedExchange16((volatile short*)p, (short)val); } };
  template<typename T> struct ATOMIC_XCHGPICK<T, 4> { static BSS_FORCEINLINE T atomic_xchg(volatile T* p, T val) { return (T)_InterlockedExchange((volatile long*)p, (long)val); } };
  template<typename T> struct ATOMIC_XCHGPICK<T, 8> { static BSS_FORCEINLINE T atomic_xchg(volatile T* p, T val) { return (T)_InterlockedExchange64((volatile int64_t*)p, (int64_t)val); } };
#elif defined(BSS_COMPILER_GCC) || defined(BSS_COMPILER_CLANG)
#define ATOMIC_XCHGPICK_MACRO(INST,SIZE,TYPE)  template<typename T> \
  struct ATOMIC_XCHGPICK<T,SIZE> \
  { \
    static BSS_FORCEINLINE T atomic_xchg(volatile T* ptr, T x) \
    { \
		__asm__ __volatile__(INST " %0,%1" :"=r" ((TYPE)x) :"m" (*(volatile TYPE*)ptr), "0" ((TYPE)x) :"memory"); \
		return x; \
    } \
  }

  ATOMIC_XCHGPICK_MACRO("xchgb",1,uint8_t);
  ATOMIC_XCHGPICK_MACRO("xchgw",2,uint16_t);
  ATOMIC_XCHGPICK_MACRO("xchgl",4,uint32_t);
#ifdef BSS_CPU_x86_64
  ATOMIC_XCHGPICK_MACRO("xchgq",8,unsigned long long);
#endif
#endif

  template<typename T> // This performs an atomic exchange, and returns the old value of *p. This is faster if p is 32-bit aligned.
  BSS_FORCEINLINE T atomic_xchg(volatile T* p, T val)
  { 
    return ATOMIC_XCHGPICK<T,sizeof(T)>::atomic_xchg(p,val);
  }

  template<typename T, int size>
  struct ASMCAS_REGPICK_WRITE { };
  
#ifdef BSS_PLATFORM_WIN32
  template<typename T> struct ASMCAS_REGPICK_WRITE<T, 1> {
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) { return _InterlockedCompareExchange8((volatile char*)dest, *(char*)&newval, *(char*)&oldval) == *(char*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_WRITE<T, 2> {
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) { return _InterlockedCompareExchange16((volatile short*)dest, *(short*)&newval, *(short*)&oldval) == *(short*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_WRITE<T, 4> {
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) { return _InterlockedCompareExchange((volatile long*)dest, *(long*)&newval, *(long*)&oldval) == *(long*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_WRITE<T, 8> {
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) { return _InterlockedCompareExchange64((volatile int64_t*)dest, *(int64_t*)&newval, *(int64_t*)&oldval) == *(int64_t*)&oldval; }
  };
#ifdef BSS_64BIT
  template<typename T> struct ASMCAS_REGPICK_WRITE<T, 16> {
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) {
      assert(!(((size_t)dest) % 16));
      return _InterlockedCompareExchange128((volatile int64_t*)dest, ((int64_t*)&newval)[1], ((int64_t*)&newval)[0], (int64_t*)&oldval);
    }
  };
#endif
#else
  template<typename T> struct ASMCAS_REGPICK_WRITE<T,1> { 
    BSS_FORCEINLINE static  uint8_t asmcas(volatile T *dest, T newval, T oldval) { return __sync_bool_compare_and_swap((volatile char*)dest,*(char*)&oldval, *(char*)&newval)!=0; } 
  };
  template<typename T> struct ASMCAS_REGPICK_WRITE<T,2> { 
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) { return __sync_bool_compare_and_swap((volatile short*)dest,*(int16_t*)&oldval, *(int16_t*)&newval)!=0; } 
  };
  template<typename T> struct ASMCAS_REGPICK_WRITE<T,4> { 
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) { return __sync_bool_compare_and_swap((volatile long*)dest,*(int32_t*)&oldval, *(int32_t*)&newval)!=0; } 
  };
  template<typename T> struct ASMCAS_REGPICK_WRITE<T,8> { 
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) { return __sync_bool_compare_and_swap((volatile long long*)dest,*(int64_t*)&oldval, *(int64_t*)&newval)!=0; } 
  };
#ifdef BSS_64BIT
  template<typename T> struct ASMCAS_REGPICK_WRITE<T,16> { 
    BSS_FORCEINLINE static uint8_t asmcas(volatile T *dest, T newval, T oldval) { return __sync_bool_compare_and_swap((volatile __int128*)dest,*(__int128*)&oldval, *(__int128*)&newval)!=0; } 
  };
#endif
#endif

  // Provides assembly level Compare and Exchange operation. Returns 1 if successful or 0 on failure. Implemented via
  // inline template specialization so that the proper assembly is generated for the type given. If a type is provided
  // that isn't exactly 1,2,4,or 8 bytes,the compiler will explode.
  template<typename T>
  inline uint8_t asmcas(volatile T *pval, T newval, T oldval)
  {
    return ASMCAS_REGPICK_WRITE<T,sizeof(T)>::asmcas(pval,newval,oldval);
  }

  template<typename T, int size>
  struct ASMCAS_REGPICK_READ { };

#ifdef BSS_PLATFORM_WIN32
  template<typename T> struct ASMCAS_REGPICK_READ<T, 1> {
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(char*)&retval = _InterlockedCompareExchange8((volatile char*)dest, *(char*)&newval, *(char*)&oldval); return *(char*)&retval == *(char*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_READ<T, 2> {
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(int16_t*)&retval = _InterlockedCompareExchange16((volatile short*)dest, *(int16_t*)&newval, *(int16_t*)&oldval); return *(int16_t*)&retval == *(int16_t*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_READ<T, 4> {
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(int32_t*)&retval = _InterlockedCompareExchange((volatile long*)dest, *(int32_t*)&newval, *(int32_t*)&oldval); return *(int32_t*)&retval == *(int32_t*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_READ<T, 8> {
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(int64_t*)&retval = _InterlockedCompareExchange64((volatile long long*)dest, *(int64_t*)&newval, *(int64_t*)&oldval); return *(int64_t*)&retval == *(int64_t*)&oldval; }
  };
#ifdef BSS_64BIT
  template<typename T> struct ASMCAS_REGPICK_READ<T, 16> {
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) {
      assert(!(((size_t)dest) % 16));
      char r = _InterlockedCompareExchange128((volatile int64_t*)dest, ((int64_t*)&newval)[1], ((int64_t*)&newval)[0], (int64_t*)&retval); return r != 0;
    }
  };
#endif
#else
  template<typename T> struct ASMCAS_REGPICK_READ<T,1> { 
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(char*)&retval = __sync_val_compare_and_swap((volatile char*)dest, *(char*)&oldval, *(char*)&newval); return *(char*)&retval==*(char*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_READ<T,2> { 
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(short*)&retval = __sync_val_compare_and_swap((volatile short*)dest, *(int16_t*)&oldval, *(int16_t*)&newval); return *(int16_t*)&retval==*(int16_t*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_READ<T,4> { 
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(long*)&retval = __sync_val_compare_and_swap((volatile long*)dest, *(int32_t*)&oldval, *(int32_t*)&newval); return *(int32_t*)&retval==*(int32_t*)&oldval; }
  };
  template<typename T> struct ASMCAS_REGPICK_READ<T,8> { 
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(long long*)&retval = __sync_val_compare_and_swap((volatile long long*)dest, *(int64_t*)&oldval, *(int64_t*)&newval); return *(int64_t*)&retval==*(int64_t*)&oldval; }
  };
#ifdef BSS_64BIT
  template<typename T> struct ASMCAS_REGPICK_READ<T,16> { 
    BSS_FORCEINLINE static bool asmcas(volatile T *dest, T newval, T oldval, T& retval) { *(__int128*)&retval = __sync_val_compare_and_swap((volatile __int128*)dest, *(__int128*)&oldval, *(__int128*)&newval); return *(__int128*)&retval==*(__int128*)&oldval; }
  };
#endif
#endif

  // Provides assembly level Compare and Exchange operation and sets retval to the previous value held by pval. Returns true if operation succeeded.
  template<typename T>
  inline bool asmcasr(volatile T *pval, T newval, T oldval, T& retval)
  {
    return ASMCAS_REGPICK_READ<T,sizeof(T)>::asmcas(pval,newval,oldval,retval);
  }

  template<typename T, int size>
  struct ASMBTS_PICK { };

#ifdef BSS_PLATFORM_WIN32
    template<typename T> struct ASMBTS_PICK<T, 4> {
    BSS_FORCEINLINE static bool asmbts(T* pval, T bit) { return _interlockedbittestandset((long*)pval, (long)bit) != 0; }
  };
  template<typename T> struct ASMBTS_PICK<T, 8> {
    BSS_FORCEINLINE static bool asmbts(T* pval, T bit) { return _interlockedbittestandset64((long long*)pval, (long long)bit) != 0; }
  };
#else
  template<typename T> struct ASMBTS_PICK<T, 4> {
    BSS_FORCEINLINE static bool asmbts(T* pval, T bit) {
      uint8_t retval;
      __asm__ __volatile__(
          "lock bts %[bit], %[x]\n\t"
          "setc     %b[rv]\n\t"
          : [x] "+m" (*pval), [rv] "=rm"(retval)
          : [bit] "ri" (bit));
      return retval != 0;
    }
  };
  template<typename T> struct ASMBTS_PICK<T, 8> {
    BSS_FORCEINLINE static bool asmbts(T* pval, T bit) {
      uint8_t retval;
      __asm__ __volatile__(
        "lock bts %[bit], %[x]\n\t"
        "setc     %b[rv]\n\t"
        : [x] "+m" (*pval), [rv] "=rm"(retval)
        : [bit] "ri" (bit));
      return retval != 0;
    }
  };
#endif

  template<typename T>
  inline bool asmbts(T* pval, T bit)
  {
    return ASMBTS_PICK<T, sizeof(T)>::asmbts(pval, bit);
  }

  template<typename T, int size>
  struct ASMBTR_PICK { };

#ifdef BSS_PLATFORM_WIN32
  template<typename T> struct ASMBTR_PICK<T, 4> {
    BSS_FORCEINLINE static bool asmbtr(T* pval, T bit) { return _interlockedbittestandreset((long*)pval, (long)bit) != 0; }
  };
  template<typename T> struct ASMBTR_PICK<T, 8> {
    BSS_FORCEINLINE static bool asmbtr(T* pval, T bit) { return _interlockedbittestandreset64((long long*)pval, (long long)bit) != 0; }
  };
#else
  template<typename T> struct ASMBTR_PICK<T, 4> {
    BSS_FORCEINLINE static bool asmbtr(T* pval, T bit) {
      uint8_t retval;
      __asm__ __volatile__(
        "lock btr %[bit], %[x]\n\t"
        "setc     %b[rv]\n\t"
        : [x] "+m" (*pval), [rv] "=rm"(retval)
        : [bit] "ri" (bit));
      return retval != 0;
    }
  };
  template<typename T> struct ASMBTR_PICK<T, 8> {
    BSS_FORCEINLINE static bool asmbtr(T* pval, T bit) {
      uint8_t retval;
      __asm__ __volatile__(
        "lock btr %[bit], %[x]\n\t"
        "setc     %b[rv]\n\t"
        : [x] "+m" (*pval), [rv] "=rm"(retval)
        : [bit] "ri" (bit));
      return retval != 0;
    }
  };
#endif
  template<typename T>
  inline bool asmbtr(T* pval, T bit)
  {
    return ASMBTR_PICK<T, sizeof(T)>::asmbtr(pval, bit);
  }
    
//  
//#ifdef BSS_CPU_x86_64
//  template<typename T>
//  struct ASMCAS_REGPICK_WRITE<T,16>
//  {
//    inline static uint8_t BSS_FORCEINLINE asmcas(volatile T *dest, T newval, T oldval)
//    {
//      uint8_t rval;
//      __asm { 
//        lea rsi,oldval; 
//        lea rdi,newval; 
//        mov rax,[rsi]; 
//        mov rdx,8[rsi]; 
//        mov rbx,[rdi]; 
//        mov rcx,8[rdi]; 
//        mov rsi,dest; 
//        //lock CMPXCHG16B [rsi] is equivalent to the following except that it's atomic:
//        //ZeroFlag = (rdx:rax == *rsi); 
//        //if (ZeroFlag) *rsi = rcx:rbx; 
//        //else rdx:rax = *rsi; 
//        lock CMPXCHG16B [rsi];
//        sete rval;
//      } 
//      return rval;
//    }
//  };
//#endif
//#define ASMCAS_REGPICK_READ_MACRO(SIZE,REG_D,REG_A) template<typename T> \
//  struct ASMCAS_REGPICK_READ<T,SIZE> \
//  { \
//    inline static uint8_t BSS_FORCEINLINE asmcas(volatile T *pval, T newval, T oldval) \
//    { \
//      ASMCAS_FSTMOV(REG_D) \
//      __asm mov REG_A, oldval \
//      __asm lock cmpxchg [BSSASM_PREG], REG_D \
//    }; \
//  }
//  
//  ASMCAS_REGPICK_READ_MACRO(1,DL,AL);
//  ASMCAS_REGPICK_READ_MACRO(2,DX,AX);
//  ASMCAS_REGPICK_READ_MACRO(4,EDX,EAX);
//#ifdef BSS_CPU_x86_64
//  ASMCAS_REGPICK_READ_MACRO(8,RDX,RAX);
//#endif
//
//#ifdef BSS_CPU_x86
//  template<typename T>
//  struct ASMCAS_REGPICK_READ<T,8>
//  {
//  public:
//    inline static BSS_FORCEINLINE T asmcas(volatile T *dest, T newval, T oldval)
//    {
//      __asm { 
//        lea esi,oldval; 
//        lea edi,newval; 
//        mov eax,[esi]; 
//        mov edx,4[esi]; 
//        mov ebx,[edi]; 
//        mov ecx,4[edi]; 
//        mov esi,dest; 
//        //lock CMPXCHG8B [esi] is equivalent to the following except that it's atomic: 
//        //ZeroFlag = (edx:eax == *esi); 
//        //if (ZeroFlag) *esi = ecx:ebx; 
//        //else edx:eax = *esi; 
//        lock CMPXCHG8B [esi];
//      } 
//    }
//  };
//#endif
//
//#ifdef BSS_CPU_x86_64
//  template<typename T>
//  struct ASMCAS_REGPICK_READ<T,16>
//  {
//  public:
//    inline static T asmcas(volatile T *dest, T newval, T oldval)
//    {
//      __asm { 
//        lea rsi,oldval; 
//        lea rdi,newval; 
//        mov rax,[rsi]; 
//        mov rdx,8[rsi]; 
//        mov rbx,[rdi]; 
//        mov rcx,8[rdi]; 
//        mov rsi,dest; 
//        //lock CMPXCHG16B [rsi] is equivalent to the following except that it's atomic: 
//        //ZeroFlag = (rdx:rax == *rsi); 
//        //if (ZeroFlag) *rsi = rcx:rbx; 
//        //else rdx:rax = *rsi; 
//        lock CMPXCHG16B [rsi];
//      } 
//    }
//  };
//#endif

  /* Assembly level 8-byte compare exchange operation. Compare with old value if you want to know if it succeeded */
  //inline int64_t asmcas8b(volatile int64_t *dest,int64_t newval,int64_t oldval) 
  //{ 
  //  //value returned in eax::edx 
  //  __asm { 
  //    lea esi,oldval; 
  //    lea edi,newval; 
  //    mov eax,[esi]; 
  //    mov edx,4[esi]; 
  //    mov ebx,[edi]; 
  //    mov ecx,4[edi]; 
  //    mov esi,dest; 
  //    //lock CMPXCHG8B [esi] is equivalent to the following except 
  //    //that it's atomic: 
  //    //ZeroFlag = (edx:eax == *esi); 
  //    //if (ZeroFlag) *esi = ecx:ebx; 
  //    //else edx:eax = *esi; 
  //    lock CMPXCHG8B [esi]; 
  //  } 
  //}
#endif
 //defined(BSS_CPU_x86_64) || defined(BSS_CPU_x86)
}

#ifdef BSS_COMPILER_MSC2010
namespace std
{
  enum memory_order {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
  };
  // Re-implements just enough of atomic_flag for it to work with our data structures
  struct BSS_COMPILER_DLLEXPORT atomic_flag
  {	
    atomic_flag() {}
    bool test_and_set(memory_order _Order = memory_order_seq_cst) volatile
    {
      return _interlockedbittestandset((volatile long*)&_My_flag, 0)!=0;
    }
    bool test_and_set(memory_order _Order = memory_order_seq_cst)
    {	// atomically set *this to true and return previous value
      return _interlockedbittestandset((volatile long*)&_My_flag, 0)!=0;
    }
    void clear(memory_order _Order = memory_order_seq_cst) volatile 
    {	// atomically clear *this
      _Atomic_store_4((volatile unsigned long *)&_My_flag, 0, _Order);
    }
    void clear(memory_order _Order = memory_order_seq_cst)
    {	// atomically clear *this
      _Atomic_store_4((volatile unsigned long *)&_My_flag, 0, _Order);
    }

  private:
    long _My_flag;

    atomic_flag(const atomic_flag&) BSS_DELETEFUNC
    atomic_flag& operator=(const atomic_flag&) BSS_DELETEFUNCOP

    inline void _Atomic_store_4(volatile unsigned long *_Tgt, unsigned long _Value, memory_order _Order) volatile
    {	/* store _Value atomically */
      switch(_Order)
      {
      case memory_order_relaxed:
        *_Tgt = _Value;
        break;
      case memory_order_release:
        _ReadWriteBarrier();
        *_Tgt = _Value;
        break;
      case memory_order_seq_cst:
        _InterlockedExchange((volatile long *)_Tgt, _Value);
        break;
      }
    }
  };
  template<class T>
  struct atomic
  {	
    atomic(const atomic&) BSS_DELETEFUNC
    atomic& operator=(const atomic&) BSS_DELETEFUNCOP

  public:
    inline atomic(){}
    inline atomic(T v) : _val(v) {}

    inline T operator=(T _Right) { _val = _Right; return _val; }
    inline T operator=(T _Right) volatile { _val = _Right; return _val; }
    inline T fetch_add(T arg, memory_order = memory_order_seq_cst) { return atomic_xadd<T>(&_val, arg); }
    inline T fetch_add(T arg, memory_order = memory_order_seq_cst) volatile { return atomic_xadd<T>(&_val, arg); }
    inline T load(memory_order = memory_order_seq_cst) { T r = _val; _ReadWriteBarrier(); return r; }
    inline T load(memory_order = memory_order_seq_cst) volatile { T r = _val; _ReadWriteBarrier(); return r; }
    inline void store(T arg, memory_order = memory_order_seq_cst) { atomic_xchg<T>(&_val, arg); }
    inline void store(T arg, memory_order = memory_order_seq_cst) volatile { atomic_xchg<T>(&_val, arg); }
    inline T exchange(T arg, memory_order = memory_order_seq_cst) { return atomic_xchg<T>(&_val, arg); }
    inline T exchange(T arg, memory_order = memory_order_seq_cst) volatile { return atomic_xchg<T>(&_val, arg); }

  protected:
    volatile T _val;
  };
}
#endif
#endif
