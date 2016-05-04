// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_ARRAY_SORT_H__BSS__
#define __C_ARRAY_SORT_H__BSS__

#include "bss_algo.h"
#include "bss_compare.h"
#include "cDynArray.h"

namespace bss_util {
  // A dynamic array that keeps its contents sorted using insertion sort and uses a binary search to retrieve items.
  template<typename T, char(*CFunc)(const T&, const T&)=CompT<T>, typename CType = size_t, ARRAY_TYPE ArrayType = CARRAY_SIMPLE, typename Alloc=StaticAllocPolicy<T>>
  class BSS_COMPILER_DLLEXPORT cArraySort
  {
  public:
    typedef CType CT_;
    typedef const T& constref;
    typedef T&& moveref;

    inline cArraySort(const cArraySort& copy) : _array(copy._array) {}
    inline cArraySort(cArraySort&& mov) : _array(std::move(mov._array)) {}
    inline cArraySort(const cArraySlice<const T, CType>& slice) : _array(slice) {}
    inline explicit cArraySort(CT_ size=0) : _array(size) {}
    inline ~cArraySort() { }
    BSS_FORCEINLINE CT_ BSS_FASTCALL Insert(constref data) { CT_ loc = _insert(data); _array.Insert(data, loc); return loc; }
    BSS_FORCEINLINE CT_ BSS_FASTCALL Insert(moveref data) { CT_ loc = _insert(std::move(data)); _array.Insert(std::move(data), loc); return loc; }
    inline void Clear() { _array.Clear(); }
    inline void BSS_FASTCALL Discard(uint32_t num) { _array.SetLength((num>_array.Length())?0:(_array.Length() - num)); }
    BSS_FORCEINLINE bool Empty() const { return _array.Empty(); }
    BSS_FORCEINLINE void BSS_FASTCALL Reserve(CT_ capacity) { _array.Reserve(capacity); }
    BSS_FORCEINLINE CT_ Length() const { return _array.Length(); }
    BSS_FORCEINLINE CT_ Capacity() const { return _array.Capacity(); }
    inline const T& Front() const { return _array.Front(); }
    inline T& Front() { return _array.Front(); }
    inline const T& Back() const { return _array.Back(); }
    inline T& Back() { return _array.Back(); }
    inline const T* begin() const noexcept { return _array.begin(); }
    inline const T* end() const noexcept { return _array.end(); }
    inline T* begin() noexcept { return _array.begin(); }
    inline T* end() noexcept { return _array.end(); }
    BSS_FORCEINLINE cArraySlice<T, CT_> GetSlice() const noexcept { return _array.GetSlice(); }

    CT_ BSS_FASTCALL ReplaceData(CT_ index, constref data)
    {
      T swap;
      _array[index]=data;
      while(index>0 && CFunc(_array[index-1], _array[index])>0)
      { //we do a swap here and hope that the compiler is smart enough to optimize it
        swap=_array[index];
        _array[index]=_array[index-1];
        _array[--index]=swap;
      }
      while(index<(_array.Length()-1) && CFunc(_array[index+1], _array[index])<0)
      { //we do a swap here and hope that the compiler is smart enough to optimize it
        swap=_array[index];
        _array[index]=_array[index+1];
        _array[++index]=swap;
      }
      return index;
    }

    inline bool BSS_FASTCALL Remove(CT_ index)
    {
      if(index >= _array.Length()) return false;
      _array.Remove(index);
      return true;
    }

    BSS_FORCEINLINE CT_ BSS_FASTCALL Find(constref data) const
    {
      return binsearch_exact<T, T, CT_, CFunc>(_array, data, 0, _array.Length());
    }

    // Can actually return -1 if there isn't anything in the array
    inline CT_ BSS_FASTCALL FindNear(constref data, bool before=true) const
    {
      CT_ retval=before?binsearch_before<T, CT_, CFunc>(_array, _array.Length(), data):binsearch_after<T, CT_, CFunc>(_array, _array.Length(), data);
      return (retval<_array.Length())?retval:(CT_)(-1); // This is only needed for before=false in case it returns a value outside the range.
    }

    BSS_FORCEINLINE constref operator [](CT_ index) const { return _array[index]; }
    BSS_FORCEINLINE T& operator [](CT_ index) { return _array[index]; }
    inline cArraySort& operator=(const cArraySort& right) { _array = right._array; return *this; }
    inline cArraySort& operator=(cArraySort&& mov) { _array = std::move(mov._array); return *this; }
    inline cArraySort& operator=(const cArraySlice<const T, CType>& copy) { _array = copy; return *this; }

  protected:
    template<typename U>
    CT_ BSS_FASTCALL _insert(U && data)
    {
      if(_array.Empty())
        return 0;
      else
      {
        CT_ loc;
        if((*CFunc)(data, _array[0]) < 0) loc = 0;
        else if((*CFunc)(data, _array.Back()) >= 0) loc = _array.Length();
        else loc = binsearch_after<T, CT_, CFunc>(_array, _array.Length(), std::forward<U>(data));
        return loc;
      }
    }

    cDynArray<T, CType, ArrayType, Alloc> _array;
  };
}

#endif
