// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "compiler.h"
#include <memory>
#include <malloc.h> // Must be included because GCC is weird
#include <assert.h>
#include <string.h>

namespace D2D {
  // Align should be a power of two for platform independence.
  inline void* aligned_realloc(void* p, size_t size, size_t align)
  {
#ifdef FG_COMPILER_MSC
    return _aligned_realloc(p, size, align);
#else
    void* n = aligned_alloc(align, size);
    if(p)
    {
      size_t old = malloc_usable_size(p);
      memcpy(n, p, (old < size ? old : size));
      free(p);
    }
    return n;
#endif
  }

  FG_FORCEINLINE static constexpr size_t AlignSize(size_t sz, size_t align)
  {
    return ((sz / align) + ((sz % align) != 0)) * align;
  }

  // An implementation of a standard allocator, with optional alignment
  template<typename T, int ALIGN = 0> class FG_COMPILER_DLLEXPORT StandardAllocator
  {
    union SIZETRACK
    {
      size_t sz;
      char align[!ALIGN ? 1 : ALIGN];
    };

  public:
    typedef T value_type;
    typedef void policy_type;
    template<class U> using rebind = StandardAllocator<U, ALIGN>;
    StandardAllocator()            = default;
    template<class U> constexpr StandardAllocator(const StandardAllocator<U>&) noexcept {}

    inline T* allocate(size_t cnt, T* p = nullptr, size_t old = 0) noexcept
    {
#ifdef FG_DEBUG
      static_assert(sizeof(SIZETRACK) == (sizeof(size_t) > ALIGN ? sizeof(size_t) : ALIGN),
                    "SIZETRACK has unexpected padding");
      if(SIZETRACK* r = reinterpret_cast<SIZETRACK*>(p))
      {
        assert(!old || r[-1].sz == (old * sizeof(T)));
        p = reinterpret_cast<T*>(r - 1);
  #ifdef FG_COMPILER_MSC
        if constexpr(ALIGN > alignof(void*))
          assert(old > 0 && old <= _aligned_msize(p, ALIGN, 0));
        else
          assert(old > 0 && old <= _msize(p));
  #endif
      }
      SIZETRACK* r;
      if constexpr(ALIGN > alignof(void*)) // realloc/malloc always return pointer-size aligned memory anyway
        r = reinterpret_cast<SIZETRACK*>(aligned_realloc(p, (cnt * sizeof(T)) + sizeof(SIZETRACK), ALIGN));
      else
        r = reinterpret_cast<SIZETRACK*>(realloc(p, (cnt * sizeof(T)) + sizeof(SIZETRACK)));
      assert(r != 0);
      r->sz = cnt * sizeof(T);
      return reinterpret_cast<T*>(r + 1);
#else
      if constexpr(ALIGN > alignof(void*)) // realloc/malloc always return pointer-size aligned memory anyway
        return reinterpret_cast<T*>(aligned_realloc(p, cnt * sizeof(T), ALIGN));
      else
        return reinterpret_cast<T*>(realloc(p, cnt * sizeof(T)));
#endif
    }
    inline void deallocate(T* p, size_t sz = 0) noexcept
    {
#ifdef FG_DEBUG
      SIZETRACK* r = reinterpret_cast<SIZETRACK*>(p);
      assert(!sz || r[-1].sz == (sz * sizeof(T)));
      p = reinterpret_cast<T*>(r - 1);
#endif
      assert(p != 0);
      if constexpr(ALIGN > alignof(void*))
        ALIGNEDFREE(p);
      else
        free(p);
    }
  };

  template<typename T> struct FG_COMPILER_DLLEXPORT AlignedAllocator : StandardAllocator<T, alignof(T)>
  {
    AlignedAllocator() = default;
    template<class U> constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
  };
}