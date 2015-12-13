// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_ARRAY_H__BSS__
#define __C_ARRAY_H__BSS__

#include "bss_alloc.h"
#include <string.h>
#include <initializer_list>

namespace bss_util {
  enum ARRAY_TYPE : unsigned char { CARRAY_SIMPLE=0, CARRAY_CONSTRUCT=1, CARRAY_SAFE=2 };

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
    inline CT_ Capacity() const { return _capacity; }
    BSS_FORCEINLINE void SetCapacity(CT_ capacity) { _capacity = capacity; _array = _getalloc(_capacity, _array); }
    BSS_FORCEINLINE void SetCapacityDiscard(CT_ capacity)
    {
      _free(_array);
      _capacity = capacity;
      _array = _getalloc(_capacity, 0);
    }
    cArrayBase& operator=(cArrayBase&& mov)
    {
      _free(_array);
      _array = mov._array;
      _capacity = mov._capacity;
      mov._array = 0;
      mov._capacity = 0;
      return *this;
    }

  protected:
    static inline T* _getalloc(CT_ n, T* prev = 0)
    { 
      if(!n)
      {
        _free(prev);
        return 0;
      }
      return (T*)Alloc::allocate(n, prev);
    }
    static inline void _free(T* p) { if(p) Alloc::deallocate(p); }
    template<class U, typename UType, ARRAY_TYPE ArrayType, typename V>
    friend struct cArrayInternal;

    T* _array;
    CType _capacity;
  };

  template<class T, typename CType = size_t, ARRAY_TYPE ArrayType = CARRAY_SIMPLE, typename Alloc = StaticAllocPolicy<T>>
  struct BSS_COMPILER_DLLEXPORT cArrayInternal
  {
    static void _copymove(T* dest, T* src, CType n) { _copy(dest, src, n); }
    static void _copy(T* dest, const T* src, CType n) { memcpy(dest, src, sizeof(T)*n); }
    static void _move(T* a, CType dest, CType src, CType n) { memmove(a + dest, a + src, sizeof(T)*n); }
    static void _setlength(T* a, CType old, CType n) {}
    static void _setcapacity(cArrayBase<T, CType, Alloc>& a, CType capacity)
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
    static void _copymove(T* dest, T* src, CType n) { memcpy(dest, src, sizeof(T)*n); }
    static void _copy(T* dest, const T* src, CType n) {
      for(CType i = 0; i < n; ++i)
        new(dest + i) T(src[i]);
    }
    static void _move(T* a, CType dest, CType src, CType n) { memmove(a + dest, a + src, sizeof(T)*n); }
    static void _setlength(T* a, CType old, CType n)
    {
      for(CType i = old; i < n; ++i)
        new(a + i) T();
      for(CType i = n; i < old; ++i)
        a[i].~T();
    }
    static void _setcapacity(cArrayBase<T, CType, Alloc>& a, CType capacity)
    {
      a.SetCapacity(capacity);
    }
  };

  template<class T, typename CType, typename Alloc>
  struct BSS_COMPILER_DLLEXPORT cArrayInternal<T, CType, CARRAY_SAFE, Alloc>
  {
    static void _copymove(T* dest, T* src, CType n)
    {
      for(CType i = 0; i < n; ++i)
        new(dest + i) T(std::move(src[i]));
      _setlength(src, n, 0);
    }
    static void _copy(T* dest, const T* src, CType n)
    {
      for(CType i = 0; i < n; ++i)
        new(dest + i) T(src[i]);
    }
    static void _move(T* a, CType dest, CType src, CType n)
    {
      if(src < dest)
        std::move_backward<T*, T*>(a + src, a + src + n, a + dest + n);
      else
        std::move<T*, T*>(a + src, a + src + n, a + dest);
    }
    static void _setlength(T* a, CType old, CType n) { cArrayInternal<T, CType, CARRAY_CONSTRUCT, Alloc>::_setlength(a, old, n); }
    static void _setcapacity(cArrayBase<T, CType, Alloc>& a, CType capacity)
    {
      T* n = cArrayBase<T, CType, Alloc>::_getalloc(capacity, 0);
      _copymove(n, a._array, bssmin(a._capacity, capacity));
      cArrayBase<T, CType, Alloc>::_free(a._array);
      a._array = n;
      a._capacity = capacity;
    }
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
    inline cArray(const std::initializer_list<T> list) : AT_(list.size())
    {
      auto end = list.end();
      CType c = 0;
      for(auto i = list.begin(); i != end && c < _capacity; ++i)
        new(_array + (c++)) T(*i);
      _capacity = c;
    }
    inline explicit cArray(CT_ capacity = 0) : AT_(capacity) { BASE::_setlength(_array, 0, _capacity); }
    inline ~cArray() { BASE::_setlength(_array, _capacity, 0); }
    BSS_FORCEINLINE CT_ Add(const T_& t) { _insert(t, _capacity); return _capacity - 1; }
    BSS_FORCEINLINE CT_ Add(T_&& t) { _insert(std::move(t), _capacity); return _capacity - 1; }
    BSS_FORCEINLINE void Remove(CT_ index)
    {
      BASE::_move(_array, index, index + 1, _capacity - index - 1);
      SetCapacity(_capacity - 1);
    }
    BSS_FORCEINLINE void RemoveLast() { Remove(_capacity - 1); }
    BSS_FORCEINLINE void Insert(const T_& t, CT_ index = 0) { _insert(t, index); }
    BSS_FORCEINLINE void Insert(T_&& t, CT_ index = 0) { _insert(std::move(t), index); }
    BSS_FORCEINLINE bool Empty() const { return !_capacity; }
    BSS_FORCEINLINE void Clear() { SetCapacity(0); }
    BSS_FORCEINLINE void SetCapacity(CT_ capacity)
    { 
      if(capacity == _capacity) return;
      CT_ old = _capacity;
      if(capacity < _capacity) BASE::_setlength(_array, _capacity, capacity);
      BASE::_setcapacity(*this, capacity);
      if(capacity > old) BASE::_setlength(_array, old, capacity);
    }
    BSS_FORCEINLINE void SetCapacityDiscard(CT_ capacity)
    {
      if(capacity == _capacity) return;
      BASE::_setlength(_array, _capacity, 0);
      SetCapacityDiscard(capacity);
      BASE::_setlength(_array, 0, _capacity);
    }
    BSS_FORCEINLINE CT_ Capacity() const { return _capacity; }
    inline const T_& Front() const { assert(_capacity>0); return _array[0]; }
    inline T_& Front() { assert(_capacity>0); return _array[0]; }
    inline const T_& Back() const { assert(_capacity>0); return _array[_capacity - 1]; }
    inline T_& Back() { assert(_capacity>0); return _array[_capacity - 1]; }
    BSS_FORCEINLINE operator T_*() { return _array; }
    BSS_FORCEINLINE operator const T_*() const { return _array; }
    inline const T_* begin() const { return _array; }
    inline const T_* end() const { return _array + _capacity; }
    inline T_* begin() { return _array; }
    inline T_* end() { return _array + _capacity; }

    BSS_FORCEINLINE cArray& operator=(const cArray& copy)
    {
      BASE::_setlength(_array, _capacity, 0);
      AT_::SetCapacityDiscard(copy._capacity);
      BASE::_copy(_array, copy._array, _capacity);
      return *this;
    }
    BSS_FORCEINLINE cArray& operator=(cArray&& mov) { BASE::_setlength(_array, _capacity, 0); AT_::operator=(std::move(mov)); return *this; }
    BSS_FORCEINLINE cArray& operator +=(const cArray& add)
    { 
      CType old = _capacity;
      BASE::_setcapacity(*this, _capacity + add._capacity);
      BASE::_copy(_array + old, add._array, add._capacity);
      return *this;
    }
    BSS_FORCEINLINE cArray operator +(const cArray& add) const { cArray r(*this); return (r += add); }

  protected:
    template<typename U>
    inline void _insert(U && data, CType index)
    {
      CType length = _capacity;
      BASE::_setcapacity(*this, _capacity + 1);
      if(index < length)
      {
        new(_array + length) T(std::forward<U>(_array[length - 1]));
        BASE::_move(_array, index + 1, index, length - index - 1);
        _array[index] = std::forward<U>(data);
      }
      else
        new(_array + index) T(std::forward<U>(data));
    }
  };
}

#endif
