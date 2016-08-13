// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __BSS_ALLOC_H__
#define __BSS_ALLOC_H__

#include "bss_defines.h"
#include <memory>
#include <malloc.h> // Must be included because GCC is weird
#include <assert.h>

namespace bss_util {
  // An implementation of a standard allocation policy
  template<typename T>
  struct BSS_COMPILER_DLLEXPORT StandardAllocPolicy {
    typedef T* pointer;
    typedef T value_type;
    template<typename U>
    struct rebind { typedef StandardAllocPolicy<U> other; };

    inline pointer allocate(size_t cnt, const pointer p = nullptr) noexcept {
      //return reinterpret_cast<pointer>(::operator new(cnt * sizeof (T))); // note that while operator new does not call a constructor (it can't), it's much easier to override for leak tests.
      return reinterpret_cast<pointer>(realloc(p, cnt*sizeof(T)));
    }
    //inline void deallocate(pointer p, size_t = 0) { ::operator delete(p); }
    inline void deallocate(pointer p, size_t = 0) noexcept { free(p); }
    inline size_t max_size() const noexcept { return ((size_t)(-1)/sizeof(T)); }
  };

  // Implementation of a null allocation policy. Doesn't free anything, always returns 0 on all allocations.
  template<typename T>
  struct BSS_COMPILER_DLLEXPORT NullAllocPolicy {
    typedef T* pointer;
    typedef T value_type;
    template<typename U>
    struct rebind { typedef NullAllocPolicy<U> other; };

    inline pointer allocate(size_t cnt, const pointer p = nullptr) noexcept { return nullptr; }
    inline void deallocate(pointer p, size_t = 0) noexcept { }
    inline size_t max_size() const noexcept { return ((size_t)(-1)/sizeof(T)); }
  };

  // Static implementation of the standard allocation policy, used for cArrayBase
  template<typename T>
  struct BSS_COMPILER_DLLEXPORT StaticAllocPolicy {
    typedef T* pointer;
    typedef T value_type;
    template<typename U> struct rebind { typedef StaticAllocPolicy<U> other; };

    inline static pointer allocate(size_t cnt, const pointer p = nullptr) noexcept { return reinterpret_cast<pointer>(realloc(p, cnt*sizeof(T))); }
    inline static void deallocate(pointer p, size_t = 0) noexcept { free(p); }
  };

  // Static null allocator. Doesn't free anything, always returns 0 on all allocations.
  template<typename T>
  struct BSS_COMPILER_DLLEXPORT StaticNullPolicy {
    typedef T* pointer;
    typedef T value_type;
    template<typename U> struct rebind { typedef StaticNullPolicy<U> other; };

    inline static pointer allocate(size_t cnt, const pointer = nullptr) noexcept { return nullptr; }
    inline static void deallocate(pointer p, size_t = 0) noexcept { }
  };

  // Internal class used by cAllocTracker
  template<typename T, typename _Ax>
  class i_AllocTracker
  {
    typedef typename _Ax::pointer pointer;
  public:
    inline i_AllocTracker(const i_AllocTracker& copy) : _allocator(copy._alloc_extern?copy._allocator:new _Ax()), _alloc_extern(copy._alloc_extern) {}
    inline i_AllocTracker(i_AllocTracker&& mov) : _allocator(mov._allocator), _alloc_extern(mov._alloc_extern) { mov._alloc_extern=true; }
    inline i_AllocTracker(_Ax* ptr=0) :	_allocator((!ptr)?(new _Ax()):(ptr)), _alloc_extern(ptr!=0) {}
    inline ~i_AllocTracker() { if(!_alloc_extern) delete _allocator; }
    inline pointer _allocate(size_t cnt, const pointer p = 0) noexcept { return _allocator->allocate(cnt, p); }
    inline void _deallocate(pointer p, size_t s = 0) noexcept { _allocator->deallocate(p, s); }

    inline i_AllocTracker& operator =(const i_AllocTracker& copy) noexcept { if(&copy==this) return *this; if(!_alloc_extern) delete _allocator; _allocator = copy._alloc_extern?copy._allocator:new _Ax(); _alloc_extern=copy._alloc_extern; return *this; }
    inline i_AllocTracker& operator =(i_AllocTracker&& mov) noexcept { if(&mov==this) return *this; if(!_alloc_extern) delete _allocator; _allocator = mov._allocator; _alloc_extern=mov._alloc_extern; mov._alloc_extern=true; return *this; }

  protected:
    _Ax* _allocator;
    bool _alloc_extern;
  };

  // Explicit specialization of cAllocTracker that removes the memory usage for a standard allocation policy, as it isn't needed.
  template<typename T>
  class i_AllocTracker<T, StandardAllocPolicy<T>>
  {
    typedef typename StandardAllocPolicy<T>::pointer pointer;
  public:
    inline i_AllocTracker(StandardAllocPolicy<T>* ptr = 0) {}
    inline pointer _allocate(size_t cnt, const pointer = 0) noexcept { return (T*)malloc(cnt * sizeof(T)); }
    inline void _deallocate(pointer p, size_t = 0) noexcept { free(p); }
  };

  // This implements stateful allocators.
  template<typename _Ax>
  class cAllocTracker : public i_AllocTracker<typename _Ax::value_type, _Ax>
  {
    typedef i_AllocTracker<typename _Ax::value_type, _Ax> BASE;
  public:
    inline cAllocTracker(const cAllocTracker& copy) : BASE(copy) {}
    inline cAllocTracker(cAllocTracker&& mov) : BASE(std::move(mov)) {}
    inline cAllocTracker(_Ax* ptr=0) : BASE(ptr) {}
  };
}



//  
//  // This takes an allocator and makes a static implementation that can be used in standard containers, but its heavily DLL-dependent.
//	template<typename T, typename Alloc=void>
//  class BSS_COMPILER_DLLEXPORT StaticAllocPolicy : public AllocPolicySize<T> {
//	public:
//    typedef typename AllocPolicySize<T>::pointer pointer;
//    template<typename U>
//    struct rebind { typedef StaticAllocPolicy<U,Alloc> other; };
//
//    inline explicit StaticAllocPolicy() {}
//    inline ~StaticAllocPolicy() {}
//    inline explicit StaticAllocPolicy(StaticAllocPolicy const&) {}
//    template <typename U>
//    inline explicit StaticAllocPolicy(StaticAllocPolicy<U,Alloc> const&) {}
//
//    inline static pointer allocate(size_t cnt, const pointer = nullptr) { return _alloc.allocate(cnt); }
//    inline static void deallocate(pointer p, size_t = 0) { _alloc.deallocate(p); }
//
//    static Alloc _alloc;
//	};
//
// template<typename T>
//  class BSS_COMPILER_DLLEXPORT StaticAllocPolicy<T,void> : public AllocPolicySize<T> {
//  public:
//    typedef typename AllocPolicySize<T>::pointer pointer;
//    template<typename U>
//    struct rebind { typedef StaticAllocPolicy<U,void> other; };
//
//    inline explicit StaticAllocPolicy() {}
//    inline ~StaticAllocPolicy() {}
//    inline explicit StaticAllocPolicy(StaticAllocPolicy const&) {}
//    template <typename U>
//    inline explicit StaticAllocPolicy(StaticAllocPolicy<U,void> const&) {}
//
//    inline static pointer allocate(size_t cnt, const pointer = nullptr) { 
//        //return reinterpret_cast<pointer>(::operator new(cnt * sizeof (T))); // note that while operator new does not call a constructor (it can't), it's much easier to override for leak tests.
//        return reinterpret_cast<pointer>(malloc(cnt*sizeof(T)));
//    }
//    //inline static void deallocate(pointer p, size_t = 0) { ::operator delete(p); }
//    inline static void deallocate(pointer p, size_t = 0) { free(p); }
//    inline static pointer reallocate(pointer p, size_t cnt) { return reinterpret_cast<pointer>(realloc(p,cnt*sizeof(T))); }
//  };
//
//#define BSSBUILD_STATIC_POLICY(name,alloc) template<class T> \
//  class BSS_COMPILER_DLLEXPORT name : public StaticAllocPolicy<T,alloc<T>> \
//  { \
//  public: \
//    template<typename U> \
//    struct rebind { typedef name<U> other; }; \
//    inline explicit name() {} \
//    inline ~name() {} \
//    inline explicit name(name const&) {} \
//    template <typename U> \
//    inline explicit name(name<U> const&) {} \
//  }

#endif
