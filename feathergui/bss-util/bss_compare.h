// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __BSS_COMPARE_H__
#define __BSS_COMPARE_H__

#include "bss_defines.h"

#define SGNCOMPARE(left,right) (((left)>(right))-((left)<(right)))
#define PRICOMPARE(left,right,p) (SGNCOMPARE(left,right)<<p)

namespace bss_util {
  template<typename T> // Returns -1,0,1 if l<r,l==r,l>r, respectively
  BSS_FORCEINLINE char CompT(const T& left, const T& right) noexcept { return SGNCOMPARE(left,right); }

  template<typename T> // Returns -1,0,1 if l<r,l==r,l>r, respectively
  BSS_FORCEINLINE char CompTInv(const T& left, const T& right) noexcept { return SGNCOMPARE(right,left); }

  template<typename T> // Returns -1 if l<r or 0 otherwise
  BSS_FORCEINLINE char CompT_LT(const T& left, const T& right) noexcept { return -(left<right); }

  template<typename T> // Returns 1 if l>r or 0 otherwise
  BSS_FORCEINLINE char CompT_GT(const T& left, const T& right) noexcept { return (left>right); }

  template<typename T> 
  BSS_FORCEINLINE char CompT_EQ(const T& left, const T& right) noexcept { return (left==right); }

  template<typename T> 
  BSS_FORCEINLINE char CompT_NEQ(const T& left, const T& right) noexcept { return (left!=right); }

  template<typename T, char (*CFunc)(const typename T::first_type&, const typename T::first_type&)>
  BSS_FORCEINLINE char CompTFirst(const T& left, const T& right) noexcept { return CFunc(left.first, right.first); }

  template<typename T, char (*CFunc)(const typename T::second_type&, const typename T::second_type&)>
  BSS_FORCEINLINE char CompTSecond(const T& left, const T& right) noexcept { return CFunc(left.second, right.second); }

  template<typename T>
  BSS_FORCEINLINE char CompStr(const T& left, const T& right)
  {
    int result = strcmp(left, right);
    return SGNCOMPARE(result,0);
  }
  template<typename T>
  BSS_FORCEINLINE char CompIStr(const T& left, const T& right)
  {
    int result = STRICMP(left, right);
    return SGNCOMPARE(result,0);
  }
  template<typename T>
  BSS_FORCEINLINE bool CompStrLT(const T& left, const T& right) { return strcmp(left, right)<0; }
  template<typename T>
  BSS_FORCEINLINE bool CompIStrLT(const T& left, const T& right) { return stricmp(left, right)<0; }
  template<typename T>
  BSS_FORCEINLINE char CompStrW(const T& left, const T& right)
  {
    int result = wcscmp(left, right);
    return SGNCOMPARE(result,0);
  }
  template<typename T>
  BSS_FORCEINLINE char CompIStrW(const T& left, const T& right)
  {
    int result = WCSICMP(left, right);
    return SGNCOMPARE(result,0);
  }
}

#endif
