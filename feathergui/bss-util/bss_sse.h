// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __BSS_SSE_H__
#define __BSS_SSE_H__

#include "bss_defines.h"
#include <assert.h>
#include <emmintrin.h>

// Define compiler-specific intrinsics for working with SSE.
#define BSS_SSE_LOAD_APS _mm_load_ps
#define BSS_SSE_LOAD_UPS _mm_loadu_ps
#define BSS_SSE_SET_PS _mm_set_ps
#define BSS_SSE_SET1_PS _mm_set_ps1
#define BSS_SSE_STORE_APS _mm_store_ps
#define BSS_SSE_STORE_UPS _mm_storeu_ps
#define BSS_SSE_ADD_PS _mm_add_ps
#define BSS_SSE_SUB_PS _mm_sub_ps
#define BSS_SSE_MUL_PS _mm_mul_ps
#define BSS_SSE_DIV_PS _mm_div_ps
#define BSS_SSE_MIN_PS bss_mm_min_ps
#define BSS_SSE_MAX_PS bss_mm_max_ps
#define BSS_SSE_CMPEQ_PS _mm_cmpeq_ps
#define BSS_SSE_CMPNEQ_PS _mm_cmpneq_ps

#define BSS_SSE_SS_F32 _mm_cvtss_f32

#define BSS_SSE_LOAD_APD _mm_load_pd
#define BSS_SSE_LOAD_UPD _mm_loadu_pd
#define BSS_SSE_SET_PD _mm_set_pd
#define BSS_SSE_SET1_PD _mm_set1_pd
#define BSS_SSE_STORE_APD _mm_store_pd
#define BSS_SSE_STORE_UPD _mm_storeu_pd
#define BSS_SSE_ADD_PD _mm_add_pd
#define BSS_SSE_SUB_PD _mm_sub_pd
#define BSS_SSE_MUL_PD _mm_mul_pd
#define BSS_SSE_DIV_PD _mm_div_pd
//#define BSS_SSE_MIN_PD _mm_min_pd // These are apparently SSE4.1
//#define BSS_SSE_MAX_PD _mm_max_pd
#define BSS_SSE_MIN_PD bss_mm_min_pd
#define BSS_SSE_MAX_PD bss_mm_max_pd
#define BSS_SSE_CMPLT_PS _mm_cmplt_ps
#define BSS_SSE_CMPLTE_PS _mm_cmple_ps
#define BSS_SSE_CMPGT_PS _mm_cmpgt_ps
#define BSS_SSE_CMPGTE_PS _mm_cmpge_ps
#define BSS_SSE_CMPEQ_PD _mm_cmpeq_pd
#define BSS_SSE_CMPNEQ_PD _mm_cmpneq_pd
#define BSS_SSE_CMPLT_PD _mm_cmplt_pd
#define BSS_SSE_CMPLTE_PD _mm_cmple_pd
#define BSS_SSE_CMPGT_PD _mm_cmpgt_pd
#define BSS_SSE_CMPGTE_PD _mm_cmpge_pd
#define BSS_SSE_SD_F64 _mm_cvtsd_f64

#define BSS_SSE_LOAD_ASI128 _mm_load_si128
#define BSS_SSE_LOAD_USI128 _mm_loadu_si128 
#define BSS_SSE_SET_EPI32 _mm_set_epi32
#define BSS_SSE_SET1_EPI32 _mm_set1_epi32
#define BSS_SSE_STORE_ASI128 _mm_store_si128 
#define BSS_SSE_STORE_USI128 _mm_storeu_si128 
#define BSS_SSE_ADD_EPI32 _mm_add_epi32
#define BSS_SSE_SUB_EPI32 _mm_sub_epi32
#define BSS_SSE_MUL_EPI32 bss_mm_mul_epi32
#define BSS_SSE_MIN_EPI32 bss_mm_min_epi32
#define BSS_SSE_MAX_EPI32 bss_mm_max_epi32
#define BSS_SSE_SHUFFLE_EPI32 _mm_shuffle_epi32
#define BSS_SSE_AND _mm_and_si128 
#define BSS_SSE_ANDNOT _mm_andnot_si128 
#define BSS_SSE_OR _mm_or_si128 
#define BSS_SSE_XOR _mm_xor_si128
#define BSS_SSE_SL_EPI32 _mm_sll_epi32
#define BSS_SSE_SLI_EPI32 _mm_slli_epi32
#define BSS_SSE_SR_EPI32 _mm_srl_epi32
#define BSS_SSE_SRI_EPI32 _mm_srli_epi32
#define BSS_SSE_SAR_EPI32 _mm_sra_epi32 //Arithmetic right shift
#define BSS_SSE_SARI_EPI32 _mm_srai_epi32
#define BSS_SSE_CMPEQ_EPI32 _mm_cmpeq_epi32
#define BSS_SSE_CMPLT_EPI32 _mm_cmplt_epi32
#define BSS_SSE_CMPGT_EPI32 _mm_cmpgt_epi32
#define BSS_SSE_SI128_SI32 _mm_cvtsi128_si32

#define BSS_SSE_SET_EPI8 _mm_set_epi8
#define BSS_SSE_SET1_EPI8 _mm_set1_epi8
#define BSS_SSE_ADD_EPI8 _mm_add_epi8
#define BSS_SSE_SUB_EPI8 _mm_sub_epi8
#define BSS_SSE_CMPEQ_EPI8 _mm_cmpeq_epi8
#define BSS_SSE_CMPLT_EPI8 _mm_cmplt_epi8
#define BSS_SSE_CMPGT_EPI8 _mm_cmpgt_epi8
#define BSS_SSE_SI128_SI8 (char)_mm_cvtsi128_si32

#define BSS_SSE_SET_EPI16 _mm_set_epi16
#define BSS_SSE_SET1_EPI16 _mm_set1_epi16
#define BSS_SSE_ADD_EPI16 _mm_add_epi16
#define BSS_SSE_SUB_EPI16 _mm_sub_epi16
#define BSS_SSE_SL_EPI16 _mm_sll_epi16
#define BSS_SSE_SLI_EPI16 _mm_slli_epi16
#define BSS_SSE_SR_EPI16 _mm_srl_epi16
#define BSS_SSE_SRI_EPI16 _mm_srli_epi16
#define BSS_SSE_CMPEQ_EPI16 _mm_cmpeq_epi16
#define BSS_SSE_CMPLT_EPI16 _mm_cmplt_epi16
#define BSS_SSE_CMPGT_EPI16 _mm_cmpgt_epi16
#define BSS_SSE_SI128_SI16 (short)_mm_cvtsi128_si32

#define BSS_SSE_SET_EPI64 _mm_set_epi64
#define BSS_SSE_SET1_EPI64 _mm_set1_epi64
#define BSS_SSE_ADD_EPI64 _mm_add_epi64
#define BSS_SSE_SUB_EPI64 _mm_sub_epi64
#define BSS_SSE_SL_EPI64 _mm_sll_epi64
#define BSS_SSE_SLI_EPI64 _mm_slli_epi64
#define BSS_SSE_SR_EPI64 _mm_srl_epi64
#define BSS_SSE_SRI_EPI64 _mm_srli_epi64
#define BSS_SSE_SI128_SI64 _mm_cvtsi128_si64 //64-bit architectures only

#define BSS_SSE_SHUFFLEHI_EPI16 _mm_shufflehi_epi16
#define BSS_SSE_SHUFFLELO_EPI16 _mm_shufflelo_epi16
#define BSS_SSE_SHUFFLE_PD _mm_shuffle_pd
#define BSS_SSE_SHUFFLE_PS _mm_shuffle_ps
#define BSS_SSE_SETZERO_PD _mm_setzero_pd
#define BSS_SSE_SETZERO_PS _mm_setzero_ps
#define BSS_SSE_SETZERO_SI128 _mm_setzero_si128

#define BSS_SSE_TPS_EPI32 _mm_cvttps_epi32 //converts using truncation
#define BSS_SSE_TPD_EPI32 _mm_cvttpd_epi32 //converts using truncation
#define BSS_SSE_PS_EPI32 _mm_cvtps_epi32
#define BSS_SSE_PD_EPI32 _mm_cvtpd_epi32
#define BSS_SSE_EPI32_PS _mm_cvtepi32_ps
#define BSS_SSE_PD_PS _mm_cvtpd_ps 
#define BSS_SSE_M128 __m128
#define BSS_SSE_M128i8 __m128i
#define BSS_SSE_M128i16 __m128i
#define BSS_SSE_M128i __m128i
#define BSS_SSE_M128i64 __m128i
#define BSS_SSE_M128d __m128d

#define BSS_SSE_SHUFFLE(x,y,z,w) ((w<<6) | (z<<4) | (y<<2) | x)

#ifdef BSS_COMPILER_CLANG
#define BSS_SSE_CMPLT_PS 
#define BSS_SSE_CMPLTE_PS 
#define BSS_SSE_CMPGT_PS 
#define BSS_SSE_CMPGTE_PS 
#define BSS_SSE_SAR_EPI32
#define BSS_SSE_EPI32_PS
#define BSS_SSE_TPS_EPI32
#endif

// SSE2 does not have min or max, so we use this manual implementation of the instruction
static BSS_FORCEINLINE BSS_SSE_M128i BSS_FASTCALL bss_mm_min_epi32(BSS_SSE_M128i a, BSS_SSE_M128i b)
{
  BSS_SSE_M128i mask = BSS_SSE_CMPLT_EPI32(a, b);
  a = BSS_SSE_AND(a, mask);
  b = BSS_SSE_ANDNOT(mask, b);
  return BSS_SSE_OR(a, b);
}
 
static BSS_FORCEINLINE BSS_SSE_M128i BSS_FASTCALL bss_mm_max_epi32(BSS_SSE_M128i a, BSS_SSE_M128i b)
{
  BSS_SSE_M128i mask = BSS_SSE_CMPGT_EPI32(a, b);
  a = BSS_SSE_AND(a, mask);
  b = BSS_SSE_ANDNOT(mask, b);
  return BSS_SSE_OR(a, b);
}

static BSS_FORCEINLINE BSS_SSE_M128 BSS_FASTCALL bss_mm_min_ps(BSS_SSE_M128 a, BSS_SSE_M128 b)
{
  BSS_SSE_M128i mask = _mm_castps_si128(BSS_SSE_CMPLT_PS(a, b));
  BSS_SSE_M128i c = BSS_SSE_AND(_mm_castps_si128(a), mask);
  BSS_SSE_M128i d = BSS_SSE_ANDNOT(mask, _mm_castps_si128(b));
  return _mm_castsi128_ps(BSS_SSE_OR(c, d));
}
 
static BSS_FORCEINLINE BSS_SSE_M128 BSS_FASTCALL bss_mm_max_ps(BSS_SSE_M128 a, BSS_SSE_M128 b)
{
  BSS_SSE_M128i mask = _mm_castps_si128(BSS_SSE_CMPGT_PS(a, b));
  BSS_SSE_M128i c = BSS_SSE_AND(_mm_castps_si128(a), mask);
  BSS_SSE_M128i d = BSS_SSE_ANDNOT(mask, _mm_castps_si128(b));
  return _mm_castsi128_ps(BSS_SSE_OR(c, d));
}

static BSS_FORCEINLINE BSS_SSE_M128d BSS_FASTCALL bss_mm_min_pd(BSS_SSE_M128d a, BSS_SSE_M128d b)
{
  BSS_SSE_M128i mask = _mm_castpd_si128(BSS_SSE_CMPLT_PD(a, b));
  BSS_SSE_M128i c = BSS_SSE_AND(_mm_castpd_si128(a), mask);
  BSS_SSE_M128i d = BSS_SSE_ANDNOT(mask, _mm_castpd_si128(b));
  return _mm_castsi128_pd(BSS_SSE_OR(c, d));
}
 
static BSS_FORCEINLINE BSS_SSE_M128d BSS_FASTCALL bss_mm_max_pd(BSS_SSE_M128d a, BSS_SSE_M128d b)
{
  BSS_SSE_M128i mask = _mm_castpd_si128(BSS_SSE_CMPGT_PD(a, b));
  BSS_SSE_M128i c = BSS_SSE_AND(_mm_castpd_si128(a), mask);
  BSS_SSE_M128i d = BSS_SSE_ANDNOT(mask, _mm_castpd_si128(b));
  return _mm_castsi128_pd(BSS_SSE_OR(c, d));
}

// SSE2 does not have any way to multiply all four 32bit integer components
static BSS_FORCEINLINE BSS_SSE_M128i bss_mm_mul_epi32(BSS_SSE_M128i a, BSS_SSE_M128i b)
{
  __m128i tmp1 = _mm_mul_epu32(a, b); /* mul 2,0*/
  __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4)); /* mul 3,1 */
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))); /* shuffle results to [63..0] and pack */
}

// Struct that wraps around a pointer to signify that it is not aligned
template<typename T>
struct BSS_UNALIGNED {
  inline explicit BSS_UNALIGNED(T* p) : _p(p) { } 
  T* _p;
};

// Struct that abstracts out a lot of common SSE operations
template<typename T> struct sseVecT {};

// 32-bit floating point operations
template<>
BSS_ALIGNED_STRUCT(16) sseVecT<float>
{
  BSS_FORCEINLINE sseVecT<float>(BSS_SSE_M128 v) : xmm(v) {}
  BSS_FORCEINLINE explicit sseVecT<float>(BSS_SSE_M128i v) : xmm(BSS_SSE_EPI32_PS(v)) {}
  BSS_FORCEINLINE explicit sseVecT<float>(BSS_SSE_M128d v) : xmm(BSS_SSE_PD_PS(v)) {}
  BSS_FORCEINLINE sseVecT<float>(float v) : xmm(BSS_SSE_SET1_PS(v)) {}
  //BSS_FORCEINLINE sseVecT<float>(const float*BSS_RESTRICT v) : xmm(BSS_SSE_LOAD_APS(v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<float>(const float(&v)[4]) : xmm(BSS_SSE_LOAD_APS(v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<float>(BSS_UNALIGNED<const float> v) : xmm(BSS_SSE_LOAD_UPS(v._p)) {}
  BSS_FORCEINLINE sseVecT<float>(float x, float y, float z, float w) : xmm(BSS_SSE_SET_PS(w, z, y, x)) {}
  BSS_FORCEINLINE sseVecT<float> operator+(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_ADD_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator-(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_SUB_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator*(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_MUL_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator/(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_DIV_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float>& operator+=(const sseVecT<float>& r) { xmm=BSS_SSE_ADD_PS(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<float>& operator-=(const sseVecT<float>& r) { xmm=BSS_SSE_SUB_PS(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<float>& operator*=(const sseVecT<float>& r) { xmm=BSS_SSE_MUL_PS(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<float>& operator/=(const sseVecT<float>& r) { xmm=BSS_SSE_DIV_PS(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE operator BSS_SSE_M128() const { return xmm; }
  //BSS_FORCEINLINE void operator>>(float*BSS_RESTRICT v) { assert(!(((size_t)v)%16)); BSS_SSE_STORE_APS(v, xmm); }
  BSS_FORCEINLINE void operator>>(float(&v)[4]) const { assert(!(((size_t)v)%16)); BSS_SSE_STORE_APS(v, xmm); }
  BSS_FORCEINLINE void operator>>(BSS_UNALIGNED<float> v) const { BSS_SSE_STORE_UPS(v._p, xmm); }
  BSS_FORCEINLINE void operator>>(float& v) const { v = BSS_SSE_SS_F32(xmm); }
  
  BSS_FORCEINLINE sseVecT<float> min(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_MIN_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> max(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_MAX_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator==(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_CMPEQ_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator!=(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_CMPNEQ_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator<(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_CMPLT_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator<=(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_CMPLTE_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator>(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_CMPGT_PS(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<float> operator>=(const sseVecT<float>& r) const { return sseVecT<float>(BSS_SSE_CMPGTE_PS(xmm, r.xmm)); }
  template<int x, int y, int z, int w>
  BSS_FORCEINLINE sseVecT<float> Shuffle() const { return Shuffle<BSS_SSE_SHUFFLE(x, y, z, w)>(*this); }
  template<int shuffle>
  static sseVecT<float> Shuffle(const sseVecT<float>& x) { return _mm_castsi128_ps(BSS_SSE_SHUFFLE_EPI32(_mm_castps_si128(x), shuffle)); }
  static sseVecT<float> ZeroVector() { return sseVecT<float>(BSS_SSE_SETZERO_PS()); }
  BSS_FORCEINLINE float Sum() const { sseVecT<float> x = BSS_SSE_ADD_PS(xmm, Shuffle<1, 0, 3, 2>()); return BSS_SSE_SS_F32(BSS_SSE_ADD_PS(x, x.Shuffle<2, 3, 0, 1>())); }

  BSS_SSE_M128 xmm;
};

// 64-bit double operations
template<>
BSS_ALIGNED_STRUCT(16) sseVecT<double>
{
  BSS_FORCEINLINE sseVecT<double>(BSS_SSE_M128d v) : xmm(v) {}
  BSS_FORCEINLINE sseVecT<double>(double v) : xmm(BSS_SSE_SET1_PD(v)) {}
  //BSS_FORCEINLINE sseVec(const double*BSS_RESTRICT v) : xmm(BSS_SSE_LOAD_APD(v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<double>(const double(&v)[2]) : xmm(BSS_SSE_LOAD_APD(v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<double>(BSS_UNALIGNED<const double> v) : xmm(BSS_SSE_LOAD_UPD(v._p)) {}
  BSS_FORCEINLINE sseVecT<double>(double x, double y) : xmm(BSS_SSE_SET_PD(y, x)) {}
  BSS_FORCEINLINE sseVecT<double> operator+(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_ADD_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator-(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_SUB_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator*(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_MUL_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator/(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_DIV_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double>& operator+=(const sseVecT<double>& r) { xmm=BSS_SSE_ADD_PD(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<double>& operator-=(const sseVecT<double>& r) { xmm=BSS_SSE_SUB_PD(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<double>& operator*=(const sseVecT<double>& r) { xmm=BSS_SSE_MUL_PD(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<double>& operator/=(const sseVecT<double>& r) { xmm=BSS_SSE_DIV_PD(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE operator BSS_SSE_M128d() const { return xmm; }
  //BSS_FORCEINLINE void operator>>(double*BSS_RESTRICT v) { assert(!(((size_t)v)%16)); BSS_SSE_STORE_APD(v, xmm); }
  BSS_FORCEINLINE void operator>>(double(&v)[2]) const { assert(!(((size_t)v)%16)); BSS_SSE_STORE_APD(v, xmm); }
  BSS_FORCEINLINE void operator>>(BSS_UNALIGNED<double> v) const { BSS_SSE_STORE_UPD(v._p, xmm); }
  BSS_FORCEINLINE void operator>>(double& v) const { v = BSS_SSE_SD_F64(xmm); }

  BSS_FORCEINLINE sseVecT<double> min(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_MIN_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> max(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_MAX_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator==(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_CMPEQ_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator!=(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_CMPNEQ_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator<(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_CMPLT_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator<=(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_CMPLTE_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator>(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_CMPGT_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> operator>=(const sseVecT<double>& r) const { return sseVecT<double>(BSS_SSE_CMPGTE_PD(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<double> Swap() const { return BSS_SSE_SHUFFLE_PD(xmm, xmm, 1); }

  template<int shuffle>
  static sseVecT<double> Shuffle(const sseVecT<double>& x) { return _mm_castsi128_pd(BSS_SSE_SHUFFLE_EPI32(_mm_castpd_si128(x), shuffle)); }
  static sseVecT<double> ZeroVector() { return sseVecT<double>(BSS_SSE_SETZERO_PD()); }
  BSS_FORCEINLINE double Sum() const { return BSS_SSE_SD_F64(BSS_SSE_ADD_PD(xmm, Swap())); }

  BSS_SSE_M128d xmm;
};

// 8-bit signed integer operations
template<>
BSS_ALIGNED_STRUCT(16) sseVecT<char>
{
  BSS_FORCEINLINE sseVecT<char>(BSS_SSE_M128i16 v) : xmm(v) {} //__fastcall is obviously useless here since we're dealing with xmm registers
  BSS_FORCEINLINE sseVecT<char>(char v) : xmm(BSS_SSE_SET1_EPI8(v)) {}
  //BSS_FORCEINLINE sseVecT<char>(const int*BSS_RESTRICT v) : xmm(BSS_SSE_LOAD_ASI128(v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<char>(const char(&v)[16]) : xmm(BSS_SSE_LOAD_ASI128((BSS_SSE_M128i16*)v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<char>(BSS_UNALIGNED<const char> v) : xmm(BSS_SSE_LOAD_USI128((BSS_SSE_M128i16*)v._p)) {}
  BSS_FORCEINLINE sseVecT<char>(char b1, char b2, char b3, char b4, char b5, char b6, char b7, char b8, char b9, char b10,
    char b11, char b12, char b13, char b14, char b15, char b16) : // This is the craziest constructor definition ever
    xmm(BSS_SSE_SET_EPI8(b16, b15, b14, b13, b12, b11, b10, b9, b8, b7, b6, b5, b4, b3, b2, b1)) {}
  BSS_FORCEINLINE sseVecT<char> operator+(const sseVecT<char>& r) const { return sseVecT<char>(BSS_SSE_ADD_EPI8(xmm, r.xmm)); } // These don't return const sseVecT<char> because it makes things messy.
  BSS_FORCEINLINE sseVecT<char> operator-(const sseVecT<char>& r) const { return sseVecT<char>(BSS_SSE_SUB_EPI8(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<char> operator&(const sseVecT<char>& r) const { return sseVecT<char>(BSS_SSE_AND(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<char> operator|(const sseVecT<char>& r) const { return sseVecT<char>(BSS_SSE_OR(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<char> operator^(const sseVecT<char>& r) const { return sseVecT<char>(BSS_SSE_XOR(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<char>& operator+=(const sseVecT<char>& r) { xmm=BSS_SSE_ADD_EPI8(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<char>& operator-=(const sseVecT<char>& r) { xmm=BSS_SSE_SUB_EPI8(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<char>& operator&=(const sseVecT<char>& r) { xmm=BSS_SSE_AND(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<char>& operator|=(const sseVecT<char>& r) { xmm=BSS_SSE_OR(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<char>& operator^=(const sseVecT<char>& r) { xmm=BSS_SSE_XOR(xmm, r.xmm); return *this; }

  BSS_FORCEINLINE operator BSS_SSE_M128i16() const { return xmm; }
  BSS_FORCEINLINE sseVecT<char> operator==(const sseVecT<char>& r) const { return sseVecT<char>(BSS_SSE_CMPEQ_EPI8(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<char> operator!=(const sseVecT<char>& r) const { return !operator==(r); }
  BSS_FORCEINLINE sseVecT<char> operator<(const sseVecT<char>& r) const { return sseVecT<char>(BSS_SSE_CMPLT_EPI8(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<char> operator<=(const sseVecT<char>& r) const { return !operator>(r); }
  BSS_FORCEINLINE sseVecT<char> operator>(const sseVecT<char>& r) const { return sseVecT<char>(BSS_SSE_CMPGT_EPI8(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<char> operator>=(const sseVecT<char>& r) const { return !operator<(r); }
  BSS_FORCEINLINE sseVecT<char> operator~() const { return sseVecT<char>(BSS_SSE_XOR(xmm, sseVecT<char>(-1))); }
  BSS_FORCEINLINE sseVecT<char> operator!() const { return sseVecT<char>(BSS_SSE_CMPEQ_EPI8(xmm, ZeroVector())); }
  BSS_FORCEINLINE sseVecT<char>& operator=(BSS_SSE_M128i16 r) { xmm=r; return *this; }
  BSS_FORCEINLINE void operator>>(char(&v)[16]) const { assert(!(((size_t)v)%16)); BSS_SSE_STORE_ASI128((BSS_SSE_M128i16*)v, xmm); }
  BSS_FORCEINLINE void operator>>(BSS_UNALIGNED<char> v) const { BSS_SSE_STORE_USI128((BSS_SSE_M128i16*)v._p, xmm); }
  BSS_FORCEINLINE void operator>>(char& v) const { v = BSS_SSE_SI128_SI8(xmm); }
  static sseVecT<char> ZeroVector() { return sseVecT<char>(_mm_setzero_si128()); }
  template<int shuffle>
  static sseVecT<char> Shuffle(const sseVecT<char>& x) { return BSS_SSE_SHUFFLE_EPI32(x, shuffle); }
  template<int shuffle>
  static sseVecT<char> ShuffleHi(const sseVecT<char>& x) { return BSS_SSE_SHUFFLEHI_EPI16(x, shuffle); }
  template<int shuffle>
  static sseVecT<char> ShuffleLo(const sseVecT<char>& x) { return BSS_SSE_SHUFFLELO_EPI16(x, shuffle); }

  BSS_SSE_M128i8 xmm;
};

// 16-bit signed integer operations
template<>
BSS_ALIGNED_STRUCT(16) sseVecT<int16_t>
{
  BSS_FORCEINLINE sseVecT<int16_t>(BSS_SSE_M128i16 v) : xmm(v) {} //__fastcall is obviously useless here since we're dealing with xmm registers
  BSS_FORCEINLINE sseVecT<int16_t>(int16_t v) : xmm(BSS_SSE_SET1_EPI16(v)) {}
  //BSS_FORCEINLINE sseVecT<int16_t>(const int*BSS_RESTRICT v) : xmm(BSS_SSE_LOAD_ASI128(v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<int16_t>(const int16_t(&v)[8]) : xmm(BSS_SSE_LOAD_ASI128((BSS_SSE_M128i16*)v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<int16_t>(BSS_UNALIGNED<const int16_t> v) : xmm(BSS_SSE_LOAD_USI128((BSS_SSE_M128i16*)v._p)) {}
  BSS_FORCEINLINE sseVecT<int16_t>(int16_t s, int16_t t, int16_t u, int16_t v, int16_t w, int16_t x, int16_t y, int16_t z) : xmm(BSS_SSE_SET_EPI16(z, y, x, w, v, u, t, s)) {}
  BSS_FORCEINLINE sseVecT<int16_t> operator+(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_ADD_EPI16(xmm, r.xmm)); } // These don't return const sseVecT<int16_t> because it makes things messy.
  BSS_FORCEINLINE sseVecT<int16_t> operator-(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_SUB_EPI16(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator&(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_AND(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator|(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_OR(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator^(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_XOR(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator>>(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_SR_EPI16(xmm, r.xmm)); } // Amazingly enough this does not conflict with the extraction operator
  BSS_FORCEINLINE sseVecT<int16_t> operator>>(int r) const { return sseVecT<int16_t>(BSS_SSE_SRI_EPI16(xmm, r)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator<<(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_SL_EPI16(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator<<(int r) const { return sseVecT<int16_t>(BSS_SSE_SLI_EPI16(xmm, r)); }
  BSS_FORCEINLINE sseVecT<int16_t>& operator+=(const sseVecT<int16_t>& r) { xmm=BSS_SSE_ADD_EPI16(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int16_t>& operator-=(const sseVecT<int16_t>& r) { xmm=BSS_SSE_SUB_EPI16(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int16_t>& operator&=(const sseVecT<int16_t>& r) { xmm=BSS_SSE_AND(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int16_t>& operator|=(const sseVecT<int16_t>& r) { xmm=BSS_SSE_OR(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int16_t>& operator^=(const sseVecT<int16_t>& r) { xmm=BSS_SSE_XOR(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int16_t>& operator>>=(const sseVecT<int16_t>& r) { xmm=BSS_SSE_SR_EPI16(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int16_t>& operator>>=(int r) { xmm=BSS_SSE_SRI_EPI16(xmm, r); return *this; }
  BSS_FORCEINLINE sseVecT<int16_t>& operator<<=(const sseVecT<int16_t>& r) { xmm=BSS_SSE_SL_EPI16(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int16_t>& operator<<=(int r) { xmm=BSS_SSE_SLI_EPI16(xmm, r); return *this; }

  BSS_FORCEINLINE operator BSS_SSE_M128i16() const { return xmm; }
  BSS_FORCEINLINE sseVecT<int16_t> operator==(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_CMPEQ_EPI16(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator!=(const sseVecT<int16_t>& r) const { return !operator==(r); }
  BSS_FORCEINLINE sseVecT<int16_t> operator<(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_CMPLT_EPI16(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator<=(const sseVecT<int16_t>& r) const { return !operator>(r); }
  BSS_FORCEINLINE sseVecT<int16_t> operator>(const sseVecT<int16_t>& r) const { return sseVecT<int16_t>(BSS_SSE_CMPGT_EPI16(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int16_t> operator>=(const sseVecT<int16_t>& r) const { return !operator<(r); }
  BSS_FORCEINLINE sseVecT<int16_t> operator~() const { return sseVecT<int16_t>(BSS_SSE_XOR(xmm, sseVecT<int16_t>(-1))); }
  BSS_FORCEINLINE sseVecT<int16_t> operator!() const { return sseVecT<int16_t>(BSS_SSE_CMPEQ_EPI16(xmm, ZeroVector())); }
  BSS_FORCEINLINE sseVecT<int16_t>& operator=(BSS_SSE_M128i16 r) { xmm=r; return *this; }
  BSS_FORCEINLINE void operator>>(int16_t(&v)[8]) const { assert(!(((size_t)v)%16)); BSS_SSE_STORE_ASI128((BSS_SSE_M128i16*)v, xmm); }
  BSS_FORCEINLINE void operator>>(BSS_UNALIGNED<int16_t> v) const { BSS_SSE_STORE_USI128((BSS_SSE_M128i16*)v._p, xmm); }
  BSS_FORCEINLINE void operator>>(int16_t& v) const { v = BSS_SSE_SI128_SI16(xmm); }
  static sseVecT<int16_t> ZeroVector() { return sseVecT<int16_t>(_mm_setzero_si128()); }
  template<int shuffle>
  static sseVecT<int16_t> Shuffle(const sseVecT<int16_t>& x) { return BSS_SSE_SHUFFLE_EPI32(x, shuffle); }
  template<int shuffle>
  static sseVecT<int16_t> ShuffleHi(const sseVecT<int16_t>& x) { return BSS_SSE_SHUFFLEHI_EPI16(x, shuffle); }
  template<int shuffle>
  static sseVecT<int16_t> ShuffleLo(const sseVecT<int16_t>& x) { return BSS_SSE_SHUFFLELO_EPI16(x, shuffle); }

  BSS_SSE_M128i16 xmm;
};

// 32-bit signed integer operations
template<>
BSS_ALIGNED_STRUCT(16) sseVecT<int32_t>
{
  BSS_FORCEINLINE sseVecT<int>(BSS_SSE_M128i v) : xmm(v) {} //__fastcall is obviously useless here since we're dealing with xmm registers
  BSS_FORCEINLINE explicit sseVecT<int>(BSS_SSE_M128 v) : xmm(BSS_SSE_TPS_EPI32(v)) {}
  BSS_FORCEINLINE explicit sseVecT<int>(BSS_SSE_M128d v) : xmm(BSS_SSE_TPD_EPI32(v)) {}
  BSS_FORCEINLINE sseVecT<int>(BSS_SSE_M128 v, char round) : xmm(BSS_SSE_PS_EPI32(v)) {}
  BSS_FORCEINLINE sseVecT<int>(int v) : xmm(BSS_SSE_SET1_EPI32(v)) {}
  //BSS_FORCEINLINE sseVecT<int>(const int*BSS_RESTRICT v) : xmm(BSS_SSE_LOAD_ASI128(v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<int>(const int(&v)[4]) : xmm(BSS_SSE_LOAD_ASI128((BSS_SSE_M128i*)v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<int>(BSS_UNALIGNED<const int> v) : xmm(BSS_SSE_LOAD_USI128((BSS_SSE_M128i*)v._p)) {}
  BSS_FORCEINLINE sseVecT<int>(int x, int y, int z, int w) : xmm(BSS_SSE_SET_EPI32(w, z, y, x)) {}
  BSS_FORCEINLINE sseVecT<int> operator+(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_ADD_EPI32(xmm, r.xmm)); } // These don't return const sseVecT<int> because it makes things messy.
  BSS_FORCEINLINE sseVecT<int> operator-(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_SUB_EPI32(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator*(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_MUL_EPI32(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator&(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_AND(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator|(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_OR(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator^(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_XOR(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator>>(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_SAR_EPI32(xmm, r.xmm)); } // Amazingly enough this does not conflict with the extraction operator
  BSS_FORCEINLINE sseVecT<int> operator>>(int r) const { return sseVecT<int>(BSS_SSE_SARI_EPI32(xmm, r)); }
  BSS_FORCEINLINE sseVecT<int> operator<<(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_SL_EPI32(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator<<(int r) const { return sseVecT<int>(BSS_SSE_SLI_EPI32(xmm, r)); }
  BSS_FORCEINLINE sseVecT<int>& operator+=(const sseVecT<int>& r) { xmm=BSS_SSE_ADD_EPI32(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator-=(const sseVecT<int>& r) { xmm=BSS_SSE_SUB_EPI32(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator*=(const sseVecT<int>& r) { xmm=BSS_SSE_MUL_EPI32(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator&=(const sseVecT<int>& r) { xmm=BSS_SSE_AND(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator|=(const sseVecT<int>& r) { xmm=BSS_SSE_OR(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator^=(const sseVecT<int>& r) { xmm=BSS_SSE_XOR(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator>>=(const sseVecT<int>& r) { xmm=BSS_SSE_SAR_EPI32(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator>>=(int r) { xmm=BSS_SSE_SARI_EPI32(xmm, r); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator<<=(const sseVecT<int>& r) { xmm=BSS_SSE_SL_EPI32(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator<<=(int r) { xmm=BSS_SSE_SLI_EPI32(xmm, r); return *this; }

  BSS_FORCEINLINE sseVecT<int> min(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_MIN_EPI32(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> max(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_MAX_EPI32(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator==(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_CMPEQ_EPI32(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator!=(const sseVecT<int>& r) const { return !operator==(r); }
  BSS_FORCEINLINE sseVecT<int> operator<(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_CMPLT_EPI32(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator<=(const sseVecT<int>& r) const { return !operator>(r); }
  BSS_FORCEINLINE sseVecT<int> operator>(const sseVecT<int>& r) const { return sseVecT<int>(BSS_SSE_CMPGT_EPI32(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int> operator>=(const sseVecT<int>& r) const { return !operator<(r); }
  BSS_FORCEINLINE operator BSS_SSE_M128i() const { return xmm; }
  BSS_FORCEINLINE sseVecT<int> operator!() const { return sseVecT<int>(BSS_SSE_CMPEQ_EPI32(xmm, ZeroVector())); }
  BSS_FORCEINLINE sseVecT<int> operator~() const { return sseVecT<int>(BSS_SSE_XOR(xmm, sseVecT<int>(-1))); }
  BSS_FORCEINLINE sseVecT<int>& operator=(BSS_SSE_M128i r) { xmm=r; return *this; }
  BSS_FORCEINLINE sseVecT<int>& operator=(BSS_SSE_M128 r) { xmm=BSS_SSE_TPS_EPI32(r); return *this; }
  //BSS_FORCEINLINE void operator>>(int*BSS_RESTRICT v) { assert(!(((size_t)v)%16)); BSS_SSE_STORE_APS(v, xmm); }
  BSS_FORCEINLINE void operator>>(int(&v)[4]) const { assert(!(((size_t)v)%16)); BSS_SSE_STORE_ASI128((BSS_SSE_M128i*)v, xmm); }
  BSS_FORCEINLINE void operator>>(BSS_UNALIGNED<int> v) const { BSS_SSE_STORE_USI128((BSS_SSE_M128i*)v._p, xmm); }
  BSS_FORCEINLINE void operator>>(int& v) const { v = BSS_SSE_SI128_SI32(xmm); }
  static sseVecT<int> ZeroVector() { return sseVecT<int>(BSS_SSE_SETZERO_SI128()); }
  template<int x, int y, int z, int w>
  BSS_FORCEINLINE sseVecT<int> Shuffle() const { return Shuffle<BSS_SSE_SHUFFLE(x, y, z, w)>(*this); }
  BSS_FORCEINLINE int Sum() const { sseVecT<int> x = BSS_SSE_ADD_EPI32(xmm, Shuffle<1, 0, 3, 2>()); return BSS_SSE_SI128_SI32(BSS_SSE_ADD_EPI32(x, x.Shuffle<2, 3, 0, 1>())); }

  template<int shuffle>
  static sseVecT<int> Shuffle(const sseVecT<int>& x) { return BSS_SSE_SHUFFLE_EPI32(x, shuffle); }
  template<int shuffle>
  static sseVecT<int> ShuffleHi(const sseVecT<int>& x) { return BSS_SSE_SHUFFLEHI_EPI16(x, shuffle); }
  template<int shuffle>
  static sseVecT<int> ShuffleLo(const sseVecT<int>& x) { return BSS_SSE_SHUFFLELO_EPI16(x, shuffle); }

  BSS_SSE_M128i xmm;
};

// 64-bit signed integer operations
template<>
BSS_ALIGNED_STRUCT(16) sseVecT<int64_t>
{
  BSS_FORCEINLINE sseVecT<int64_t>(BSS_SSE_M128i64 v) : xmm(v) {} //__fastcall is obviously useless here since we're dealing with xmm registers
  BSS_FORCEINLINE sseVecT<int64_t>(int64_t v) { BSS_ALIGN(16) int64_t vv[2]={ v, v }; xmm=BSS_SSE_LOAD_ASI128((BSS_SSE_M128i64*)vv); }
  //BSS_FORCEINLINE sseVecT<int64_t>(const int*BSS_RESTRICT v) : xmm(BSS_SSE_LOAD_ASI128(v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<int64_t>(const int64_t(&v)[2]) : xmm(BSS_SSE_LOAD_ASI128((BSS_SSE_M128i64*)v)) { assert(!(((size_t)v)%16)); }
  BSS_FORCEINLINE explicit sseVecT<int64_t>(BSS_UNALIGNED<const int64_t> v) : xmm(BSS_SSE_LOAD_USI128((BSS_SSE_M128i64*)v._p)) {}
  BSS_FORCEINLINE sseVecT<int64_t>(int64_t x, int64_t y) { BSS_ALIGN(16) int64_t xy[2]={ x, y }; xmm=BSS_SSE_LOAD_ASI128((BSS_SSE_M128i64*)xy); }
  BSS_FORCEINLINE sseVecT<int64_t> operator+(const sseVecT<int64_t>& r) const { return sseVecT<int64_t>(BSS_SSE_ADD_EPI64(xmm, r.xmm)); } // These don't return const sseVecT<int64_t> because it makes things messy.
  BSS_FORCEINLINE sseVecT<int64_t> operator-(const sseVecT<int64_t>& r) const { return sseVecT<int64_t>(BSS_SSE_SUB_EPI64(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int64_t> operator&(const sseVecT<int64_t>& r) const { return sseVecT<int64_t>(BSS_SSE_AND(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int64_t> operator|(const sseVecT<int64_t>& r) const { return sseVecT<int64_t>(BSS_SSE_OR(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int64_t> operator^(const sseVecT<int64_t>& r) const { return sseVecT<int64_t>(BSS_SSE_XOR(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int64_t> operator>>(const sseVecT<int64_t>& r) const { return sseVecT<int64_t>(BSS_SSE_SR_EPI64(xmm, r.xmm)); } // Amazingly enough this does not conflict with the extraction operator
  BSS_FORCEINLINE sseVecT<int64_t> operator>>(int r) const { return sseVecT<int64_t>(BSS_SSE_SRI_EPI64(xmm, r)); }
  BSS_FORCEINLINE sseVecT<int64_t> operator<<(const sseVecT<int64_t>& r) const { return sseVecT<int64_t>(BSS_SSE_SL_EPI64(xmm, r.xmm)); }
  BSS_FORCEINLINE sseVecT<int64_t> operator<<(int r) const { return sseVecT<int64_t>(BSS_SSE_SLI_EPI64(xmm, r)); }
  BSS_FORCEINLINE sseVecT<int64_t>& operator+=(const sseVecT<int64_t>& r) { xmm=BSS_SSE_ADD_EPI64(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int64_t>& operator-=(const sseVecT<int64_t>& r) { xmm=BSS_SSE_SUB_EPI64(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int64_t>& operator&=(const sseVecT<int64_t>& r) { xmm=BSS_SSE_AND(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int64_t>& operator|=(const sseVecT<int64_t>& r) { xmm=BSS_SSE_OR(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int64_t>& operator^=(const sseVecT<int64_t>& r) { xmm=BSS_SSE_XOR(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int64_t>& operator>>=(const sseVecT<int64_t>& r) { xmm=BSS_SSE_SR_EPI64(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int64_t>& operator>>=(int r) { xmm=BSS_SSE_SRI_EPI64(xmm, r); return *this; }
  BSS_FORCEINLINE sseVecT<int64_t>& operator<<=(const sseVecT<int64_t>& r) { xmm=BSS_SSE_SL_EPI64(xmm, r.xmm); return *this; }
  BSS_FORCEINLINE sseVecT<int64_t>& operator<<=(int r) { xmm=BSS_SSE_SLI_EPI64(xmm, r); return *this; }

  BSS_FORCEINLINE operator BSS_SSE_M128i64() const { return xmm; }
  BSS_FORCEINLINE sseVecT<int64_t> operator~() const { return sseVecT<int64_t>(BSS_SSE_XOR(xmm, sseVecT<int64_t>(-1))); }
  BSS_FORCEINLINE sseVecT<int64_t>& operator=(BSS_SSE_M128i64 r) { xmm=r; return *this; }
  BSS_FORCEINLINE void operator>>(int64_t(&v)[2]) const { assert(!(((size_t)v)%16)); BSS_SSE_STORE_ASI128((BSS_SSE_M128i64*)v, xmm); }
  BSS_FORCEINLINE void operator>>(BSS_UNALIGNED<int64_t> v) const { BSS_SSE_STORE_USI128((BSS_SSE_M128i64*)v._p, xmm); }
#ifdef BSS_64BIT
  BSS_FORCEINLINE void operator>>(int64_t& v) const { v = _mm_cvtsi128_si64(xmm); }
#else
  BSS_FORCEINLINE void operator>>(int64_t& v) const { BSS_ALIGN(16) int64_t store[2]; BSS_SSE_STORE_ASI128((BSS_SSE_M128i64*)store, xmm); v = store[0]; }
#endif
  BSS_FORCEINLINE int64_t Sum() const { BSS_SSE_M128i64 x = BSS_SSE_ADD_EPI64(xmm, Swap()); BSS_ALIGN(16) int64_t store[2]; BSS_SSE_STORE_ASI128((BSS_SSE_M128i64*)store, x); return store[0]; }
  BSS_FORCEINLINE sseVecT<int64_t> Swap() const { return BSS_SSE_SHUFFLE_EPI32(xmm, BSS_SSE_SHUFFLE(2, 3, 0, 1)); }

  static sseVecT<int64_t> ZeroVector() { return sseVecT<int64_t>(_mm_setzero_si128()); }
  template<int shuffle>
  static sseVecT<int64_t> Shuffle(const sseVecT<int64_t>& x) { return BSS_SSE_SHUFFLE_EPI32(x, shuffle); }
  template<int shuffle>
  static sseVecT<int64_t> ShuffleHi(const sseVecT<int64_t>& x) { return BSS_SSE_SHUFFLEHI_EPI16(x, shuffle); }
  template<int shuffle>
  static sseVecT<int64_t> ShuffleLo(const sseVecT<int64_t>& x) { return BSS_SSE_SHUFFLELO_EPI16(x, shuffle); }

  BSS_SSE_M128i64 xmm;
};

typedef sseVecT<float> sseVec;
typedef sseVecT<double> sseVecd;
typedef sseVecT<char> sseVeci8;
typedef sseVecT<int16_t> sseVeci16;
typedef sseVecT<int32_t> sseVeci;
typedef sseVecT<int64_t> sseVeci64;

#endif
