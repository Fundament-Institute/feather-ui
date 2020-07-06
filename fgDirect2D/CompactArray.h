// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef D2D__COMPACT_ARRAY_H
#define D2D__COMPACT_ARRAY_H

#include <initializer_list>
#include <string.h>
#include <assert.h>
#include "Alloc.h"
#include "../backend.h"

namespace D2D {
  // Helper function for inserting a range into a simple array.
  template<class T, typename CType = size_t>
  inline void InsertRangeSimple(T* dest, CType length, CType index, const T* src, CType srcsize) noexcept
  {
    assert(index >= 0 && length >= index && dest != 0);
    memmove(dest + index + srcsize, dest + index, sizeof(T) * (length - index));
    memcpy(dest + index, src, sizeof(T) * srcsize);
  }

  template<class T, typename CType = size_t>
  inline void RemoveRangeSimple(T* dest, CType length, CType index, CType range) noexcept
  {
    assert(index >= 0 && length > index && dest != 0);
    memmove(dest + index, dest + index + range, sizeof(T) * (length - index - range));
  }

  // Array using an inline stack for small arrays before allocating on the heap. Can only be used with simple data.
  template<class T, int I = 2, class CType = size_t, class Alloc = StandardAllocator<T>>
  class FG_COMPILER_DLLEXPORT CompactArray final : Alloc
  {
  protected:
    typedef T Ty;
    typedef CType CT;
    static_assert(std::is_unsigned<CT>::value, "CT must be unsigned");
    static const CT COMPACTFLAG = CT(1) << ((sizeof(CT) << 3) - 1);
    static const CT COMPACTMASK = ~COMPACTFLAG;

  public:
    inline CompactArray(const CompactArray& copy) : Alloc(copy), _length(COMPACTFLAG)
    {
      SetLength(copy.Length());
      MEMCPY(begin(), Length() * sizeof(T), copy.begin(), copy.Length() * sizeof(T));
    }
    inline CompactArray(CompactArray&& mov) : Alloc(std::move(mov))
    {
      MEMCPY(this, sizeof(CompactArray), &mov, sizeof(CompactArray));
      mov._length = COMPACTFLAG;
    }
    inline CompactArray() : _length(COMPACTFLAG) {}
    inline CompactArray(const std::initializer_list<T>& list) : _length(COMPACTFLAG)
    {
      SetLength(list.size());
      auto end = list.end();
      CT j     = 0;
      CT len   = Length();

      for(auto i = list.begin(); i != end && j < len; ++i)
        new(_array + (j++)) T(*i);
    }
    inline ~CompactArray()
    {
      if(!(_length & COMPACTFLAG) && _array != 0)
        Alloc::deallocate(_array, _capacity);
    }
    inline CT Add(T item)
    {
      SetCapacity(Length() + 1);
      new(end()) T(item);
      return (_length++) & COMPACTMASK;
    }
    inline void Remove(CT index) { RemoveRangeSimple<T, CT>(begin(), (_length--) & COMPACTMASK, index, 1); }
    FG_FORCEINLINE void RemoveLast() { Remove(Length() - 1); }
    FG_FORCEINLINE void Insert(T item, CT index = 0)
    {
      SetCapacity(Length() + 1);
      InsertRangeSimple<T, CT>(begin(), (_length++) & COMPACTMASK, index, &item, 1);
    }
    FG_FORCEINLINE void Set(const T* p, CT n)
    {
      SetLength(n);
      MEMCPY(begin(), Length() * sizeof(T), p, n * sizeof(T));
    }
    FG_FORCEINLINE bool Empty() const { return !Length(); }
    FG_FORCEINLINE void Clear() { _length = (_length & COMPACTFLAG); }
    inline void SetLength(CT length)
    {
      SetCapacity(length);
#ifdef FG_DEBUG
      if(length > Length())
        bssFillN<T>(end(), length - Length(), 0xfd);
#endif
      _length = (length | (_length & COMPACTFLAG));
    }
    inline void SetCapacity(CT capacity)
    {
      if(capacity > I)
      {
        capacity *= 2;

        if(_length & COMPACTFLAG)
        {
          T* a    = Alloc::allocate(capacity, 0);
          _length = Length();

          if(_length)
            MEMCPY(a, capacity * sizeof(T), _internal, _length * sizeof(T));

          _array    = a;
          _capacity = capacity;
        }
        else
        {
          _array    = Alloc::allocate(capacity, _array, _capacity);
          _capacity = capacity;
        }
      }
    }
    FG_FORCEINLINE CT Length() const { return _length & COMPACTMASK; }
    FG_FORCEINLINE CT Capacity() const { return (_length & COMPACTFLAG) ? I : _capacity; }
    FG_FORCEINLINE const T& Front() const
    {
      assert(Length() > 0);
      return begin()[0];
    }
    FG_FORCEINLINE const T& Back() const
    {
      assert(Length() > 0);
      return begin()[Length() - 1];
    }
    FG_FORCEINLINE T& Front()
    {
      assert(Length() > 0);
      return begin()[0];
    }
    FG_FORCEINLINE T& Back()
    {
      assert(Length() > 0);
      return begin()[Length() - 1];
    }
    FG_FORCEINLINE const T* begin() const noexcept { return (_length & COMPACTFLAG) ? _internal : _array; }
    FG_FORCEINLINE const T* end() const noexcept { return begin() + Length(); }
    FG_FORCEINLINE T* begin() noexcept { return (_length & COMPACTFLAG) ? _internal : _array; }
    FG_FORCEINLINE T* end() noexcept { return begin() + Length(); }
    FG_FORCEINLINE operator T*() { return begin(); }
    FG_FORCEINLINE operator const T*() const { return begin(); }
    inline CompactArray& operator=(const CompactArray& copy)
    {
      SetLength(copy.Length());
      MEMCPY(begin(), Length() * sizeof(T), copy.begin(), copy.Length() * sizeof(T));
      return *this;
    }
    inline CompactArray& operator=(CompactArray&& mov)
    {
      if(!(_length & COMPACTFLAG) && _array != 0)
        Alloc::deallocate(_array, _capacity);
      Alloc::operator=(std::move(mov));
      MEMCPY(this, sizeof(CompactArray), &mov, sizeof(CompactArray));
      mov._length = COMPACTFLAG;
      return *this;
    }

  protected:
    union
    {
      struct
      {
        T* _array;
        CT _capacity;
      };
      struct
      {
        T _internal[I];
      };
    };

    CT _length;
  };
}

#endif
