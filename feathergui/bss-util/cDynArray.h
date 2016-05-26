// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_DYN_ARRAY_H__BSS__
#define __C_DYN_ARRAY_H__BSS__

#include "cArray.h"
#include "cBitField.h"
#include "bss_util.h"

namespace bss_util {
  // Dynamic array implemented using ArrayType (should only be used when constructors could potentially not be needed)
  template<class T, typename CType = size_t, ARRAY_TYPE ArrayType = CARRAY_SIMPLE, typename Alloc=StaticAllocPolicy<T>>
  class BSS_COMPILER_DLLEXPORT cDynArray : protected cArrayBase<T, CType, Alloc>, protected cArrayInternal<T, CType, ArrayType, Alloc>
  {
  protected:
    typedef cArrayInternal<T, CType, ArrayType, Alloc> BASE;
    typedef cArrayBase<T, CType, Alloc> AT_;
    typedef typename AT_::CT_ CT_;
    typedef typename AT_::T_ T_;
    using AT_::_array;
    using AT_::_capacity;

  public:
    inline cDynArray(const cDynArray& copy) : AT_(copy._capacity), _length(copy._length) { BASE::_copy(_array, copy._array, _length); }
    inline cDynArray(cDynArray&& mov) : AT_(std::move(mov)), _length(mov._length) { mov._length = 0; }
    inline cDynArray(const cArraySlice<const T, CType>& slice) : AT_(slice.length), _length(slice.length) { BASE::_copy(_array, slice.start, slice.length); }
    inline explicit cDynArray(CT_ capacity=0) : AT_(capacity), _length(0) {}
    inline cDynArray(const std::initializer_list<T>& list) : AT_(list.size()), _length(0)
    {
      auto end = list.end();
      for(auto i = list.begin(); i != end && _length < _capacity; ++i)
        new(_array + (_length++)) T(*i);
    }
    inline ~cDynArray() { BASE::_setlength(_array, _length, 0); }
    BSS_FORCEINLINE CT_ Add(const T_& t) { _checksize(); new(_array + _length) T(t); return _length++; }
    BSS_FORCEINLINE CT_ Add(T_&& t) { _checksize(); new(_array + _length) T(std::move(t)); return _length++; }
#ifdef BSS_VARIADIC_TEMPLATES
    template<typename... Args>
    BSS_FORCEINLINE CT_ AddConstruct(Args&&... args) { _checksize(); new(_array + _length) T(std::forward<Args>(args)...); return _length++; }
#endif
    inline void Remove(CT_ index)
    {
      assert(index < _length);
      BASE::_remove(_array, _length--, index);
    }
    BSS_FORCEINLINE void RemoveLast() { Remove(_length - 1); }
    BSS_FORCEINLINE void Insert(const T_& t, CT_ index=0) { _insert(t, index); }
    BSS_FORCEINLINE void Insert(T_&& t, CT_ index=0) { _insert(std::move(t), index); }
    BSS_FORCEINLINE bool Empty() const { return !_length; }
    BSS_FORCEINLINE void Clear() { SetLength(0); }
    inline void SetLength(CT_ length)
    { 
      if(length < _length) BASE::_setlength(_array, _length, length);
      if(length > _capacity) BASE::_setcapacity(*this, _length, length);
      if(length > _length) BASE::_setlength(_array, _length, length);
      _length = length;
    }
    inline void Reserve(CT_ capacity) { if(capacity>_capacity) BASE::_setcapacity(*this, _length, capacity); }
    BSS_FORCEINLINE CT_ Length() const { return _length; }
    BSS_FORCEINLINE CT_ Capacity() const { return _capacity; }
    BSS_FORCEINLINE const T_& Front() const { assert(_length>0); return _array[0]; }
    BSS_FORCEINLINE const T_& Back() const { assert(_length>0); return _array[_length-1]; }
    BSS_FORCEINLINE T_& Front() { assert(_length>0); return _array[0]; }
    BSS_FORCEINLINE T_& Back() { assert(_length>0); return _array[_length-1]; }
    BSS_FORCEINLINE const T_* begin() const noexcept { return _array; }
    BSS_FORCEINLINE const T_* end() const noexcept { return _array+_length; }
    BSS_FORCEINLINE T_* begin() noexcept { return _array; }
    BSS_FORCEINLINE T_* end() noexcept { return _array+_length; }
    BSS_FORCEINLINE cArraySlice<T_, CT_> GetSlice() const noexcept { return AT_::GetSlice(); }
#if defined(BSS_64BIT) && defined(BSS_DEBUG) 
    BSS_FORCEINLINE T_& operator [](uint64_t i) { assert(i < _length); return _array[i]; } // for some insane reason, this works on 64-bit, but not on 32-bit
    BSS_FORCEINLINE const T_& operator [](uint64_t i) const { assert(i < _length); return _array[i]; }
#endif
    BSS_FORCEINLINE operator T_*() { return _array; }
    BSS_FORCEINLINE operator const T_*() const { return _array; }
    inline cDynArray& operator=(const cArray<T, CType, ArrayType, Alloc>& copy)
    { 
      BASE::_setlength(_array, _length, 0);
      if(copy._capacity > _capacity) 
        AT_::SetCapacityDiscard(copy._capacity);
      BASE::_copy(_array, copy._array, copy._capacity);
      _length = copy._capacity;
      return *this;
    }
    inline cDynArray& operator=(cArray<T, CType, ArrayType, Alloc>&& mov) { BASE::_setlength(_array, _length, 0); AT_::operator=(std::move(mov)); _length = mov._capacity; return *this; }
    inline cDynArray& operator=(const cDynArray& copy)
    {
      BASE::_setlength(_array, _length, 0);
      if(copy._length > _capacity)
        AT_::SetCapacityDiscard(copy._capacity);
      BASE::_copy(_array, copy._array, copy._length);
      _length = copy._length;
      return *this;
    }
    inline cDynArray& operator=(cDynArray&& mov) { BASE::_setlength(_array, _length, 0); AT_::operator=(std::move(mov)); _length = mov._length; mov._length = 0; return *this; }
    inline cDynArray& operator=(const cArraySlice<const T, CType>& copy)
    {
      BASE::_setlength(_array, _length, 0);
      if(copy.length > _capacity)
        AT_::SetCapacityDiscard(copy.length);
      BASE::_copy(_array, copy.start, copy.length);
      _length = copy.length;
      return *this;
    }
      inline cDynArray& operator +=(const cDynArray& add)
    { 
      BASE::_setcapacity(*this, _length, _length + add._length);
      BASE::_copy(_array + _length, add._array, add._length);
      _length+=add._length;
      return *this;
    }
    inline cDynArray operator +(const cDynArray& add) const
    {
      cDynArray r(_length + add._length);
      BASE::_copy(r._array, _array, _length);
      BASE::_copy(r._array + _length, add._array, add._length);
      r._length = _length + add._length;
      return r;
    }

  protected:
    template<typename U>
    inline void _insert(U && data, CType index)
    {
      _checksize();
      BASE::_insert(_array, _length++, index, std::forward<U>(data));
    }
    BSS_FORCEINLINE void _checksize()
    {
      if(_length >= _capacity)
        BASE::_setcapacity(*this, _length, T_FBNEXT(_capacity));
      assert(_length<_capacity);
    }

    CT_ _length;
  };

  template<typename CType, ARRAY_TYPE ArrayType, typename Alloc>
  class BSS_COMPILER_DLLEXPORT cDynArray<bool, CType, ArrayType, Alloc> : protected cArrayBase<uint8_t, CType, typename Alloc::template rebind<uint8_t>::other>
  {
  protected:
    typedef uint8_t STORE;
    typedef cArrayBase<STORE, CType, typename Alloc::template rebind<STORE>::other> AT_;
    typedef typename AT_::CT_ CT_;
    typedef typename AT_::T_ T_;
    using AT_::_array;
    using AT_::_capacity;
    static const CT_ DIV_AMT = (sizeof(STORE) << 3);
    static const CT_ MOD_AMT = (sizeof(STORE) << 3) - 1;
    static_assert(std::is_unsigned<CType>::value, "CType must be an unsigned integral type.");

  public:
    inline cDynArray(const cDynArray& copy) : AT_(copy._capacity), _length(copy._length) { memcpy(_array, copy._array, copy._capacity*sizeof(STORE)); }
    inline cDynArray(cDynArray&& mov) : AT_(std::move(mov)), _length(mov._length) { mov._length = 0; }
    inline explicit cDynArray(CT_ capacity = 0) : AT_(_maxchunks(capacity)/DIV_AMT), _length(0) {}
    inline cDynArray(const std::initializer_list<bool>& list) : AT_(_maxchunks(list.size())/DIV_AMT), _length(0)
    {
      auto end = list.end();
      for(auto i = list.begin(); i != end && _length < (_capacity*DIV_AMT); ++i)
        Add(*i);
    }
    inline ~cDynArray() { }
    BSS_FORCEINLINE CT_ Add(bool t) { _checksize(); GetBit(_length++) = t; return _length - 1; }
    inline void Remove(CT_ index)
    {
      assert(index < _length);
      _shiftdel(index);
      --_length;
    }
    BSS_FORCEINLINE void RemoveLast() { --_length; }
    BSS_FORCEINLINE void Insert(bool t, CT_ index = 0)
    {
      _shiftins(index);
      ++_length;
      GetBit(index) = t;
    }
    BSS_FORCEINLINE bool Empty() const { return !_length; }
    BSS_FORCEINLINE void Clear() { SetLength(0); }
    inline void SetLength(CT_ length)
    {
      if(length > (_capacity*DIV_AMT)) AT_::SetCapacity(_maxchunks(length) / DIV_AMT);
      for(CT_ i = _length; i < length; ++i) GetBit(i) = false;
      _length = length;
    }
    inline void Reserve(CT_ capacity) { capacity = _maxchunks(capacity) / DIV_AMT; if(capacity > _capacity) SetCapacity(capacity); }
    inline void Flip() { for(CT_ i = 0; i < _capacity; ++i) _array[i] = ~_array[i]; }
    CT_ BSS_FASTCALL CountBits(CT_ bitindex, CT_ length) const
    {
      if(!length) return 0;
      length += bitindex;
      assert(length <= _length);
      CT_ start = bitindex / DIV_AMT;
      CT_ end = _maxchunks(length) / DIV_AMT;
      STORE smask = ~((((STORE)1) << (bitindex - (start*DIV_AMT))) - 1);
      STORE emask = (STORE)(~0) >> ((end*DIV_AMT) - length);

      if(start == (--end))
        return bitcount<STORE>((_array[start] & smask)&emask);

      CT_ c = bitcount<STORE>(_array[start] & smask);
      c += bitcount<STORE>(_array[end] & emask);

      for(CT_ i = start + 1; i < end; ++i)
        c += bitcount<STORE>(_array[i]);

      return c;
    }
    BSS_FORCEINLINE CT_ Length() const { return _length; }
    BSS_FORCEINLINE CT_ Capacity() const { return _capacity*DIV_AMT; }
    BSS_FORCEINLINE bool Front() const { assert(_length>0); return _array[0]; }
    BSS_FORCEINLINE bool Back() const { assert(_length>0); return _array[_length - 1]; }
    BSS_FORCEINLINE _cBIT_REF<STORE> Front() { assert(_length>0); return GetBit(0); }
    BSS_FORCEINLINE _cBIT_REF<STORE> Back() { assert(_length>0); return GetBit(_length - 1); }
    BSS_FORCEINLINE bool operator[](CT_ index) const { assert(index < _length); return GetBitConst(index); }
    BSS_FORCEINLINE _cBIT_REF<STORE> operator[](CT_ index) { assert(index < _length); return GetBit(index); }
    // This gets the raw storage byte in such a way that unused bits are always set to zero. Useful for debugging.
    inline STORE GetRawByte(CT_ index) const { assert(index < _capacity); return (index < (_capacity - 1)) ? _array[index] : (_array[index]&((STORE)(~0) >> ((_capacity*DIV_AMT) - _length))); }

    class BSS_COMPILER_DLLEXPORT _cBIT_ITER : public std::iterator<std::bidirectional_iterator_tag, _cBIT_REF<STORE>>
    {
    public:
      inline _cBIT_ITER(const _cBIT_REF<STORE>& src) : _bits(const_cast<STORE*>(&src.GetState().first)), _bit(src.GetState().second) { }
      inline const _cBIT_REF<STORE> operator*() const { return _cBIT_REF<STORE>(_bit, *_bits); }
      inline _cBIT_REF<STORE> operator*() { return _cBIT_REF<STORE>(_bit, *_bits); }
      inline _cBIT_ITER& operator++() { _incthis(); return *this; } //prefix
      inline _cBIT_ITER operator++(int) { _cBIT_ITER r(*this); ++*this; return r; } //postfix
      inline _cBIT_ITER& operator--() { _decthis(); return *this; } //prefix
      inline _cBIT_ITER operator--(int) { _cBIT_ITER r(*this); --*this; return r; } //postfix
      inline const _cBIT_ITER& operator++() const { _incthis(); return *this; } //prefix
      inline const _cBIT_ITER operator++(int) const { _cBIT_ITER r(*this); ++*this; return r; } //postfix
      inline const _cBIT_ITER& operator--() const { _decthis(); return *this; } //prefix
      inline const _cBIT_ITER operator--(int) const { _cBIT_ITER r(*this); --*this; return r; } //postfix
      inline bool operator==(const _cBIT_ITER& _Right) const { return (_bits == _Right._bits) && (_bit == _Right._bit); }
      inline bool operator!=(const _cBIT_ITER& _Right) const { return !operator==(_Right); }

    protected:
      void _incthis()
      {
        _bit = (_bit << 1);
        if(!_bit)
        { 
          ++_bits;
          _bit = 1;
        }
      }
      void _decthis()
      {
        _bit = (_bit >> 1);
        if(!_bit)
        {
          --_bits;
          _bit = (1 << MOD_AMT);
        }
      }

      STORE* _bits;
      STORE _bit;
    };

    BSS_FORCEINLINE const _cBIT_ITER begin() const { return GetBitConst(0); }
    BSS_FORCEINLINE const _cBIT_ITER end() const { return GetBitConst(_length); }
    BSS_FORCEINLINE _cBIT_ITER begin() { return GetBit(0); }
    BSS_FORCEINLINE _cBIT_ITER end() { return GetBit(_length); }

    inline cDynArray& operator=(const cDynArray& copy)
    {
      if(copy._length > (_capacity*DIV_AMT))
        AT_::SetCapacityDiscard(copy._capacity);
      memcpy(_array, copy._array, _capacity * sizeof(STORE));
      _length = copy._length;
      return *this;
    }
    inline cDynArray& operator=(cDynArray&& mov) { AT_::operator=(std::move(mov)); _length = mov._length; mov._length = 0; return *this; }
    /*inline cDynArray& operator +=(const cDynArray& add)
    {
      Reserve(*this, _length + add._length);
      memcpybits(_array + (_length/DIV_AMT), _length&MOD_AMT, add._array, 0, add._length);
      _length += add._length;
      return *this;
    }
    inline cDynArray operator +(const cDynArray& add) const
    {
      cDynArray r(_length + add._length);
      memcpy(r._array, _array, (_maxchunks(_length) / DIV_AMT)*sizeof(STORE));
      memcpybits(r._array + (_length / DIV_AMT), _length&MOD_AMT, add._array, 0, add._length);
      r._length = _length + add._length;
      return r;
    }*/

  protected:
    BSS_FORCEINLINE _cBIT_REF<STORE> GetBit(CT_ bitindex) { return _cBIT_REF<STORE>(((STORE)1) << (bitindex&MOD_AMT), *(_array + (bitindex / DIV_AMT))); }
    BSS_FORCEINLINE bool GetBitConst(CT_ bitindex) const { return (_array[(bitindex / DIV_AMT)] & (((STORE)1) << (bitindex&MOD_AMT))) != 0; }
    BSS_FORCEINLINE static CT_ _maxchunks(CT_ numbits) { return T_NEXTMULTIPLE(numbits, MOD_AMT); }
    inline void _shiftdel(CT_ index)
    {
      CT_ ind = index / DIV_AMT;
      CT_ off = index & MOD_AMT;
      CT_ len = _maxchunks(_length)/DIV_AMT;
      STORE mask = (STORE)(~0) << off;
      _array[ind] = ((_array[ind] >> 1)&mask) | (_array[ind] & (~mask));

      for(CT_ i = ind+1; i < len; ++i)
      {
        _array[i - 1] = (_array[i - 1] & ((STORE)(~0) >> 1)) | ((_array[i] & 1) << MOD_AMT);
        _array[i] >>= 1;
      }
    }
    inline void _shiftins(CT_ index)
    {
      _checksize();
      CT_ ind = index / DIV_AMT;
      CT_ off = index & MOD_AMT;
      CT_ len = _maxchunks(_length + 1) / DIV_AMT;
      STORE mask = (STORE)(~0) << off;
      STORE store = _array[ind];
      _array[ind] = ((_array[ind] & mask) << 1) | (_array[ind] & (~mask));
      STORE test = _array[ind];

      for(CT_ i = ind + 1; i < len; ++i)
      {
        _array[i] = (store >> MOD_AMT) | ((_array[i]<<1) & (STORE)(~1));
        store = _array[i];
      }
    }

    BSS_FORCEINLINE void _checksize()
    {
      if(_length >= (_capacity*DIV_AMT))
        AT_::SetCapacity(T_FBNEXT(_capacity));
      assert(_length<(_capacity*DIV_AMT));
    }

    CT_ _length;
  };

  // A dynamic array that can dynamically adjust the size of each element
  template<typename CT_, typename Alloc=StaticAllocPolicy<uint8_t>>
  class BSS_COMPILER_DLLEXPORT cArbitraryArray : protected cArrayBase<uint8_t, CT_, Alloc>
  {
  protected:
    typedef cArrayBase<uint8_t, CT_, Alloc> AT_;
    using AT_::_array;
    using AT_::_capacity;

  public:
    inline cArbitraryArray(const cArbitraryArray& copy) : AT_(copy._capacity), _length(copy._length), _element(copy._element) { memcpy(_array, copy._array, _length*_element); }
    inline cArbitraryArray(cArbitraryArray&& mov) : AT_(std::move(mov)), _length(mov._length), _element(mov._element) {}
    inline cArbitraryArray(CT_ size=0, CT_ element=1): AT_(size*element), _length(0), _element(element) {}
    template<typename T>
    inline CT_ Add(const T& t)
    {
      if((_length*_element)>=_capacity) AT_::SetCapacity(T_FBNEXT(_length)*_element);
      memcpy(_array+(_length*_element), &t, bssmin(sizeof(T), _element));
      return _length++;
    }
    inline void Remove(CT_ index)
    {
      index *= _element;
      assert(_capacity>0 && index<_capacity);
      memmove(_array+index, _array+index+_element, _capacity-index-_element);
      --_length;
    }
    inline void RemoveLast() { --_length; }
    inline void SetElement(const void* newarray, CT_ element, CT_ num) // num is a count of how many elements are in the array
    {
      if(((uint8_t*)newarray)==_array) return;
      _element=element;
      _length=num;
      if((_length*_element)>_capacity) AT_::SetCapacityDiscard(_length*_element);
      if(_array)
        memcpy(_array, newarray, element*num);
    }
    template<typename T> // num is a count of how many elements are in the array
    BSS_FORCEINLINE void SetElement(const T* newarray, CT_ num) { SetElement(newarray, sizeof(T), num); }
    template<typename T, uint32_t NUM>
    BSS_FORCEINLINE void SetElement(const T(&newarray)[NUM]) { SetElement(newarray, sizeof(T), NUM); }
    void SetElement(CT_ element)
    {
      if(element==_element) return;
      _capacity=element*_length;
      uint8_t* narray = !_length?0:(uint8_t*)Alloc::allocate(_capacity);
      memset(narray, 0, _capacity);
      CT_ m=bssmin(element, _element);
      for(CT_ i = 0; i < _length; ++i)
        memcpy(narray+(i*element), _array+(i*_element), m);
      AT_::_free(_array);
      _array=narray;
      _element=element;
      assert(element>0);
    }
    inline CT_ Element() const { return _element; }
    inline bool Empty() const { return !_length; }
    inline void Clear() { _length=0; }
    inline void SetLength(CT_ length) { _length=length; length*=_element; if(length>_capacity) AT_::SetCapacity(length); }
    inline CT_ Length() const { return _length; }
    // These are templatized accessors, but we don't do an assertion on if the size of T is the same as _element, because there are many cases where this isn't actually true.
    template<typename T> inline const T& Front() const { assert(_length>0); return _array[0]; }
    template<typename T> inline const T& Back() const { assert(_length>0); return _array[(_length-1)*_element]; }
    template<typename T> inline T& Front() { assert(_length>0); return _array[0]; }
    template<typename T> inline T& Back() { assert(_length>0); return _array[(_length-1)*_element]; }
    template<typename T> inline const T* begin() const { return _array; }
    template<typename T> inline const T* end() const { return _array+(_length*_element); }
    template<typename T> inline T* begin() { return _array; }
    template<typename T> inline T* end() { return _array+(_length*_element); }
    template<typename T> inline const T& Get(CT_ index) const { return *(T*)Get(index); }
    template<typename T> inline T& Get(CT_ index) { return *(T*)Get(index); }
    BSS_FORCEINLINE const void* Get(CT_ index) const { assert(index<_length); return (_array+(index*_element)); }
    BSS_FORCEINLINE void* Get(CT_ index) { assert(index<_length); return (_array+(index*_element)); }

    inline const void* operator[](CT_ index) const { return Get(index); }
    inline void* operator[](CT_ index) { return Get(index); }
    inline cArbitraryArray& operator=(const cArbitraryArray& copy) { SetCapacityDiscard(copy._capacity); memcpy(_array, copy._array, _capacity); _length = copy._length; _element = copy._element; return *this; }
    inline cArbitraryArray& operator=(cArbitraryArray&& mov) { AT_::operator=(std::move(mov)); _length = mov._length; _element = mov._element; return *this; }

  protected:
    CT_ _length; // Total length of the array in number of elements
    CT_ _element; // Size of each element
  };
}

#endif
