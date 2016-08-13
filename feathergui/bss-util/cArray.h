// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_ARRAY_H__BSS__
#define __C_ARRAY_H__BSS__

#include "bss_alloc.h"
#include <string.h>
#include <initializer_list>

namespace bss_util {
  template<class T, typename CType = size_t>
  struct BSS_COMPILER_DLLEXPORT cArraySlice
  {
    cArraySlice(const cArraySlice& copy) : start(copy.start), length(copy.length) {}
    cArraySlice(T* s, CType l) : start(s), length(l) { }
    cArraySlice() {}
    inline cArraySlice& operator=(const cArraySlice& right) { start = right.start; length = right.length; return *this; }
    inline cArraySlice& operator++() { assert(length > 0); ++start; --length; return *this; }
    inline cArraySlice operator++(int) { assert(length > 0); return cArraySlice(start + 1, length - 1); }
    inline const cArraySlice operator()(CType start) const { return operator()(start, length); }
    inline const cArraySlice operator()(CType start, CType end) const { return operator()(start, length); }
    inline cArraySlice operator()(CType start) { return operator()(start, length); }
    inline cArraySlice operator()(CType start, CType end)
    { 
      assert(abs(start) < length);
      assert(end <= length);
      start = bssmod(start, length);
      if(end <= 0) end = length - end;
      assert(end >= start);
      return cArraySlice(start + start, end - start);
    }
    inline bool operator!() const { return !start || !length; }
    inline operator bool() const { return Valid(); }

    BSS_FORCEINLINE bool Valid() const { return start != 0 && length != 0; }
    BSS_FORCEINLINE const T& Front() const { assert(length>0); return start[0]; }
    BSS_FORCEINLINE const T& Back() const { assert(length>0); return start[length - 1]; }
    BSS_FORCEINLINE T& Front() { assert(length>0); return start[0]; }
    BSS_FORCEINLINE T& Back() { assert(length>0); return start[length - 1]; }
    BSS_FORCEINLINE const T* begin() const noexcept { return start; }
    BSS_FORCEINLINE const T* end() const noexcept { return start + length; }
    BSS_FORCEINLINE T* begin() noexcept { return start; }
    BSS_FORCEINLINE T* end() noexcept { return start + length; }
    BSS_FORCEINLINE T& operator [](CType i) noexcept { assert(i < length); return start[i]; }
    BSS_FORCEINLINE const T& operator [](CType i) const noexcept { assert(i < length); return start[i]; }

    inline operator cArraySlice<const T, CType>() const { return cArraySlice<const T, CType>(start, length); }

    T* start;
    CType length;
  };

  // Helper function for inserting a range into a simple array
  template<class T, typename CType = size_t>
  static inline void InsertRangeSimple(T* a, CType length, CType index, const T* t, CType tsize) noexcept
  {
    assert(index >= 0 && length >= index);
    memmove(a + index + tsize, a + index, sizeof(T)*(length - index));
    memcpy(a + index, t, sizeof(T)*tsize);
  }

  template<class T, typename CType = size_t>
  static inline void RemoveRangeSimple(T* a, CType length, CType index, CType range) noexcept
  {
    assert(index >= 0 && length > index);
    memmove(a + index, a + index + range, sizeof(T)*(length - index - range));
  }

  enum ARRAY_TYPE : uint8_t { CARRAY_SIMPLE=0, CARRAY_CONSTRUCT=1, CARRAY_SAFE=2, CARRAY_MOVE=3 };

  // Handles the very basic operations of an array. Constructor management is done by classes that inherit this class.
  template<class T, typename CType = size_t, typename Alloc = StaticAllocPolicy<T>>
  class BSS_COMPILER_DLLEXPORT cArrayBase
  {
  public:
    typedef CType CT_; // There are cases when you need access to these types even if you don't inherit (see cRandomQueue in bss_algo.h)
    typedef T T_;
    static_assert(std::is_integral<CType>::value, "CType must be integral");

    inline cArrayBase(cArrayBase&& mov) : _array(mov._array), _capacity(mov._capacity)
    {
      mov._array = 0;
      mov._capacity = 0;
    }
    inline explicit cArrayBase(CT_ capacity = 0) : _array(_getalloc(capacity)), _capacity(capacity) {}
    inline ~cArrayBase()
    {
      _free(_array);
    }
    inline CT_ Capacity() const noexcept { return _capacity; }
    BSS_FORCEINLINE void SetCapacity(CT_ capacity) noexcept { _capacity = capacity; _array = _getalloc(_capacity, _array); }
    BSS_FORCEINLINE void SetCapacityDiscard(CT_ capacity) noexcept
    {
      _free(_array);
      _capacity = capacity;
      _array = _getalloc(_capacity, 0);
    }
    cArrayBase& operator=(cArrayBase&& mov) noexcept
    {
      _free(_array);
      _array = mov._array;
      _capacity = mov._capacity;
      mov._array = 0;
      mov._capacity = 0;
      return *this;
    }
    inline cArraySlice<T, CType> GetSlice() const noexcept { return cArraySlice<T, CType>(_array, _capacity); }

  protected:
    static inline T* _getalloc(CT_ n, T* prev = 0) noexcept
    { 
      if(!n)
      {
        _free(prev);
        return 0;
      }
      return (T*)Alloc::allocate((size_t)n, prev);
    }
    static inline void _free(T* p) noexcept { if(p) Alloc::deallocate(p); }
    template<class U, typename UType, ARRAY_TYPE ArrayType, typename V>
    friend struct cArrayInternal;

    T* _array;
    CType _capacity;
  };

#ifdef BSS_DEBUG
#define BSS_DEBUGFILL(p, old, n, Ty) memset(p + bssmin(n, old), 0xfd, (size_t)(((n > old)?(n - old):(old - n))*sizeof(Ty)))
#else
#define BSS_DEBUGFILL(p, old, n, Ty)  
#endif

  template<class T, typename CType = size_t, ARRAY_TYPE ArrayType = CARRAY_SIMPLE, typename Alloc = StaticAllocPolicy<T>>
  struct BSS_COMPILER_DLLEXPORT cArrayInternal
  {
    static void _copymove(T* BSS_RESTRICT dest, T* BSS_RESTRICT src, CType n) noexcept { if(dest == nullptr) return; assert(dest != src); _copy(dest, src, n); }
    static void _copy(T* BSS_RESTRICT dest, const T* BSS_RESTRICT src, CType n) noexcept { if(dest == nullptr) return; assert(dest != src); memcpy(dest, src, sizeof(T)*n); }
    template<typename U>
    static void _insert(T* a, CType length, CType index, U && t) noexcept
    {
      assert(index >= 0 && length >= index);
      memmove(a + index + 1, a + index, sizeof(T)*(length - index));
      new(a + index) T(std::forward<U>(t));
    }
    static void _remove(T* a, CType length, CType index) noexcept { assert(index >= 0 && length > index); memmove(a + index, a + index + 1, sizeof(T)*(length - index - 1)); }
    static void _setlength(T* a, CType old, CType n) noexcept { BSS_DEBUGFILL(a, old, n, T); }
    static void _setcapacity(cArrayBase<T, CType, Alloc>& a, CType length, CType capacity) noexcept
    {
      if(capacity <= a._capacity)
        a._capacity = capacity;
      else
        a.SetCapacity(capacity);
    }
  };

  template<class T, typename CType, typename Alloc>
  struct BSS_COMPILER_DLLEXPORT cArrayInternal<T, CType, CARRAY_CONSTRUCT, Alloc>
  {
    static void _copymove(T* BSS_RESTRICT dest, T* BSS_RESTRICT src, CType n) noexcept { if(dest == nullptr) return; assert(dest != src); memcpy(dest, src, sizeof(T)*n); }
    static void _copy(T* BSS_RESTRICT dest, const T* BSS_RESTRICT src, CType n) noexcept {
      if(dest == nullptr) return;
      assert(dest != src);
      for(CType i = 0; i < n; ++i)
        new(dest + i) T(src[i]);
    }
    template<typename U>
    static void _insert(T* a, CType length, CType index, U && t) noexcept { cArrayInternal<T, CType, CARRAY_SIMPLE, Alloc>::_insert(a, length, index, std::forward<U>(t)); }
    static void _remove(T* a, CType length, CType index) noexcept {
      a[index].~T();
      assert(index >= 0 && length > index);
      memmove(a + index, a + index + 1, sizeof(T)*(length - index - 1));
    }
    static void _setlength(T* a, CType old, CType n) noexcept
    {
      for(CType i = n; i < old; ++i)
        a[i].~T();
      BSS_DEBUGFILL(a, old, n, T);
      for(CType i = old; i < n; ++i)
        new(a + i) T();
    }
    static void _setcapacity(cArrayBase<T, CType, Alloc>& a, CType length, CType capacity) noexcept
    {
      a.SetCapacity(capacity);
    }
  };

  template<class T, typename CType, typename Alloc>
  struct BSS_COMPILER_DLLEXPORT cArrayInternal<T, CType, CARRAY_SAFE, Alloc>
  {
    static void _copymove(T* BSS_RESTRICT dest, T* BSS_RESTRICT src, CType n) noexcept
    {
      if(dest == nullptr) return;
      assert(dest != src);
      for(CType i = 0; i < n; ++i)
        new(dest + i) T(std::move(src[i]));
      _setlength(src, n, 0);
    }
    static void _copy(T* BSS_RESTRICT dest, const T* BSS_RESTRICT src, CType n) noexcept
    {
      if(dest == nullptr) return;
      assert(dest != src);
      for(CType i = 0; i < n; ++i)
        new(dest + i) T(src[i]);
    }
    template<typename U>
    static void _insert(T* a, CType length, CType index, U && t) noexcept
    {
      assert(index >= 0 && length >= index);
      if(index < length)
      {
        new(a + length) T(std::move(a[length - 1]));
        std::move_backward<T*, T*>(a + index, a + length - 1, a + length);
        a[index] = std::forward<U>(t);
      }
      else
        new(a + index) T(std::forward<U>(t));
    }
    static void _remove(T* a, CType length, CType index) noexcept {
      assert(index >= 0 && length > index);
      std::move<T*, T*>(a + index + 1, a + length, a + index);
      a[length - 1].~T();
    }
    static void _setlength(T* a, CType old, CType n) noexcept { cArrayInternal<T, CType, CARRAY_CONSTRUCT, Alloc>::_setlength(a, old, n); }
    static void _setcapacity(cArrayBase<T, CType, Alloc>& a, CType length, CType capacity) noexcept
    {
      T* n = cArrayBase<T, CType, Alloc>::_getalloc(capacity, 0);
      if(n != nullptr) _copymove(n, a._array, bssmin(length, capacity));
      cArrayBase<T, CType, Alloc>::_free(a._array);
      a._array = n;
      a._capacity = capacity;
    }
  };

  template<class T, typename CType, typename Alloc>
  struct BSS_COMPILER_DLLEXPORT cArrayInternal<T, CType, CARRAY_MOVE, Alloc>
  {
    typedef cArrayInternal<T, CType, CARRAY_SAFE, Alloc> SAFE;
    static void _copymove(T* BSS_RESTRICT dest, T* BSS_RESTRICT src, CType n) noexcept { if(dest == nullptr) return; assert(dest != src); SAFE::_copymove(dest, src, n); }
    static void _copy(T* BSS_RESTRICT dest, const T* BSS_RESTRICT src, CType n) noexcept { assert(false); }
    template<typename U>
    static void _insert(T* a, CType length, CType index, U && t) noexcept { SAFE::_insert(a, length, index, std::forward<U>(t)); }
    static void _remove(T* a, CType length, CType index) noexcept { SAFE::_remove(a, length, index); }
    static void _setlength(T* a, CType old, CType n) noexcept { SAFE::_setlength(a, old, n); }
    static void _setcapacity(cArrayBase<T, CType, Alloc>& a, CType length, CType capacity) noexcept { SAFE::_setcapacity(a, length, capacity); }
  };

  // Wrapper for underlying arrays that expose the array, making them independently usable without blowing up everything that inherits them
  template<class T, typename CType = size_t, ARRAY_TYPE ArrayType = CARRAY_SIMPLE, typename Alloc = StaticAllocPolicy<T>>
  class BSS_COMPILER_DLLEXPORT cArray : protected cArrayBase<T, CType, Alloc>, protected cArrayInternal<T, CType, ArrayType, Alloc>
  {
  protected:
    typedef cArrayInternal<T, CType, ArrayType, Alloc> BASE;
    typedef cArrayBase<T, CType, Alloc> AT_;
    typedef typename AT_::CT_ CT_;
    typedef typename AT_::T_ T_;
    using AT_::_array;
    using AT_::_capacity;

  public:
    inline cArray(const cArray& copy) : AT_(copy._capacity) { BASE::_copy(_array, copy._array, _capacity); } // We have to declare this because otherwise its interpreted as deleted
    inline cArray(cArray&& mov) : AT_(std::move(mov)) {}
    inline cArray(const std::initializer_list<T>& list) : AT_(list.size())
    {
      auto end = list.end();
      CType c = 0;
      for(auto i = list.begin(); i != end && c < _capacity; ++i)
        new(_array + (c++)) T(*i);
      _capacity = c;
    }
    inline explicit cArray(CT_ capacity = 0) : AT_(capacity) { BASE::_setlength(_array, 0, _capacity); }
    inline cArray(const cArraySlice<const T, CType>& slice) : AT_(slice.length) { BASE::_copy(_array, slice.begin, slice.length); }
    inline ~cArray() { BASE::_setlength(_array, _capacity, 0); }
    BSS_FORCEINLINE CT_ Add(const T_& t) { _insert(t, _capacity); return _capacity - 1; }
    BSS_FORCEINLINE CT_ Add(T_&& t) { _insert(std::move(t), _capacity); return _capacity - 1; }
    BSS_FORCEINLINE void Remove(CT_ index)
    {
      BASE::_remove(_array, _capacity--, index); // we don't bother reallocating the array because it got smaller, so we just ignore part of it.
    }
    BSS_FORCEINLINE void RemoveLast() { Remove(_capacity - 1); }
    BSS_FORCEINLINE void Insert(const T_& t, CT_ index = 0) { _insert(t, index); }
    BSS_FORCEINLINE void Insert(T_&& t, CT_ index = 0) { _insert(std::move(t), index); }
    BSS_FORCEINLINE void Set(const cArraySlice<const T, CType>& slice) { Set(slice.start, slice.length); }
    BSS_FORCEINLINE void Set(const T_* p, CT_ n)
    {
      BASE::_setlength(_array, _capacity, 0);
      AT_::SetCapacityDiscard(n);
      BASE::_copy(_array, p, _capacity);
    }
    BSS_FORCEINLINE bool Empty() const noexcept { return !_capacity; }
    BSS_FORCEINLINE void Clear() noexcept { SetCapacity(0); }
    inline cArraySlice<T, CType> GetSlice() const noexcept { return cArraySlice<T, CType>(_array, _capacity); }
    BSS_FORCEINLINE void SetCapacity(CT_ capacity) noexcept
    { 
      if(capacity == _capacity) return;
      CT_ old = _capacity;
      if(capacity < _capacity) BASE::_setlength(_array, _capacity, capacity);
      BASE::_setcapacity(*this, _capacity, capacity);
      if(capacity > old) BASE::_setlength(_array, old, capacity);
    }
    BSS_FORCEINLINE void SetCapacityDiscard(CT_ capacity) noexcept
    {
      if(capacity == _capacity) return;
      BASE::_setlength(_array, _capacity, 0);
      AT_::SetCapacityDiscard(capacity);
      BASE::_setlength(_array, 0, _capacity);
    }
    BSS_FORCEINLINE CT_ Capacity() const noexcept { return _capacity; }
    inline const T_& Front() const noexcept { assert(_capacity>0); return _array[0]; }
    inline T_& Front() noexcept { assert(_capacity>0); return _array[0]; }
    inline const T_& Back() const noexcept { assert(_capacity>0); return _array[_capacity - 1]; }
    inline T_& Back() noexcept { assert(_capacity>0); return _array[_capacity - 1]; }
    BSS_FORCEINLINE operator T_*() noexcept { return _array; }
    BSS_FORCEINLINE operator const T_*() const noexcept { return _array; }
#if defined(BSS_64BIT) && defined(BSS_DEBUG) 
    BSS_FORCEINLINE T_& operator [](uint64_t i) { assert(i < _capacity); return _array[i]; } // for some insane reason, this works on 64-bit, but not on 32-bit
    BSS_FORCEINLINE const T_& operator [](uint64_t i) const { assert(i < _capacity); return _array[i]; }
#endif
    inline const T_* begin() const noexcept { return _array; }
    inline const T_* end() const noexcept { return _array + _capacity; }
    inline T_* begin() noexcept { return _array; }
    inline T_* end() noexcept { return _array + _capacity; }

    BSS_FORCEINLINE cArray& operator=(const cArray& copy) noexcept
    {
      BASE::_setlength(_array, _capacity, 0);
      AT_::SetCapacityDiscard(copy._capacity);
      BASE::_copy(_array, copy._array, _capacity);
      return *this;
    }
    BSS_FORCEINLINE cArray& operator=(cArray&& mov) noexcept { BASE::_setlength(_array, _capacity, 0); AT_::operator=(std::move(mov)); return *this; }
    BSS_FORCEINLINE cArray& operator=(const cArraySlice<const T, CType>& copy) noexcept { Set(copy); return *this; }
    BSS_FORCEINLINE cArray& operator +=(const cArray& add) noexcept
    { 
      CType old = _capacity;
      BASE::_setcapacity(*this, _capacity, _capacity + add._capacity);
      BASE::_copy(_array + old, add._array, add._capacity);
      return *this;
    }
    BSS_FORCEINLINE cArray operator +(const cArray& add) const noexcept { cArray r(*this); return (r += add); }

  protected:
    template<typename U>
    inline void _insert(U && data, CType index)
    {
      CType length = _capacity;
      BASE::_setcapacity(*this, _capacity, _capacity + 1);
      BASE::_insert(_array, length, index, std::forward<U>(data));
    }
  };
}

#endif
