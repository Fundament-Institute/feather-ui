// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_BITFIELD_H__BSS__ //These are used in case this header file is used by two different projects dependent on each other, resulting in duplicates which cannot be differentiated by #pragma once
#define __C_BITFIELD_H__BSS__

#include "bss_defines.h"
#include <utility>

namespace bss_util
{
  template<typename T>
  struct BSS_COMPILER_DLLEXPORT _cBIT_REF
  {
    inline _cBIT_REF(T bit, T& bits) : _bit(bit), _bits(bits) {}
    inline _cBIT_REF& operator=(bool right) { _bits=T_SETBIT(_bits,_bit,(char)right); return *this; }
    BSS_FORCEINLINE _cBIT_REF& operator=(const _cBIT_REF& right) { return operator=((bool)right); }
    BSS_FORCEINLINE operator bool() const { return (_bits&_bit) != 0; }
    inline void flip() { _bits ^= _bit; }
    BSS_FORCEINLINE bool operator!() const { return !operator bool(); }
    BSS_FORCEINLINE bool operator==(bool right) const { return right == operator bool(); }
    BSS_FORCEINLINE bool operator!=(bool right) const { return right != operator bool(); }
    BSS_FORCEINLINE bool operator&&(bool right) const { return right && operator bool(); }
    BSS_FORCEINLINE bool operator||(bool right) const { return right || operator bool(); }
    BSS_FORCEINLINE char operator|(char right) const { return right | (char)operator bool(); }
    BSS_FORCEINLINE char operator&(char right) const { return right & (char)operator bool(); }
    BSS_FORCEINLINE char operator^(char right) const { return right ^ (char)operator bool(); }
    BSS_FORCEINLINE std::pair<const T&, T> GetState() const { return std::pair<const T&, T>(_bits, _bit); }

  protected:
    T& _bits;
    T _bit;
  };

  // Generic implementation of using an integral type's component bits to store flags.
  template<typename T=uint32_t>
  class BSS_COMPILER_DLLEXPORT cBitField
  {
  public:
    // Initializes the bitfield with the given flag values, if any
    inline explicit cBitField(T init=0) : _bits(init) {}
    // Sets the entire bitfield to the given value
    BSS_FORCEINLINE void BSS_FASTCALL Set(T bits) { _bits=bits; }
    BSS_FORCEINLINE void BSS_FASTCALL Set(T bits, T mask) { _bits^=(bits^(_bits&mask)); }
    // Gets the entire bitfield
    BSS_FORCEINLINE T Get() const { return _bits; }
    BSS_FORCEINLINE bool Get(T bit) const { return (_bits&bit)!=0; }

    inline cBitField& operator=(T right) { _bits=right; return *this; }
    BSS_FORCEINLINE bool operator[](T bit) const { return Get(bit); }
    BSS_FORCEINLINE _cBIT_REF<T> operator[](T bit) { return _cBIT_REF<T>(bit,_bits); }
    inline operator T() const { return _bits; }
    inline cBitField& operator+=(T bit) { _bits|=bit; return *this; }
    inline cBitField& operator-=(T bit) { _bits&=(~bit); return *this; }

  protected:
    T _bits;
  };
}

#endif
