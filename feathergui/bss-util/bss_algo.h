// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __BSS_ALGO_H__
#define __BSS_ALGO_H__

#include "bss_util.h"
#include "bss_compare.h"
#include "bss_sse.h"
#include "bss_vector.h"
#include "cDynArray.h"
#include "cDisjointSet.h"
#include "delegate.h"
#include <algorithm>
#include <random>
#include <array>
#ifdef BSS_COMPILER_GCC
#include <alloca.h>
#endif

namespace bss_util {
  // Performs a binary search on "arr" between first and last. if CEQ=NEQ and char CVAL=-1, uses an upper bound, otherwise uses lower bound.
  template<typename T, typename D, typename CT_, char(*CFunc)(const D&, const T&), char(*CEQ)(const char&, const char&), char CVAL>
  inline static CT_ BSS_FASTCALL binsearch_near(const T* arr, const D& data, CT_ first, CT_ last)
  {
    typename std::make_signed<CT_>::type c = last-first; // Must be a signed version of whatever CT_ is
    CT_ c2; //No possible operation can make this negative so we leave it as possibly unsigned.
    CT_ m;
    while(c>0)
    {
      c2 = (c>>1);
      m = first+c2;

      //if(!(_Val < *_Mid))
      if((*CEQ)((*CFunc)(data, arr[m]), CVAL))
      {	// try top half
        first = m+1;
        c -= c2+1;
      } else
        c = c2;
    }
    return first;
  }
  // Either gets the element that matches the value in question or one immediately before the closest match. Could return an invalid -1 value.
  template<typename T, typename CT_, char(*CFunc)(const T&, const T&)>
  BSS_FORCEINLINE static CT_ BSS_FASTCALL binsearch_before(const T* arr, const T& data, CT_ first, CT_ last) { return binsearch_near<T, T, CT_, CFunc, CompT_NEQ<char>, -1>(arr, data, first, last)-1; }

  template<typename T, typename CT_, char(*CFunc)(const T&, const T&)>
  BSS_FORCEINLINE static CT_ BSS_FASTCALL binsearch_before(const T* arr, CT_ length, const T& data) { return binsearch_near<T, T, CT_, CFunc, CompT_NEQ<char>, -1>(arr, data, 0, length)-1; }

  template<typename T, typename CT_, char(*CFunc)(const T&, const T&), CT_ I>
  BSS_FORCEINLINE static CT_ BSS_FASTCALL binsearch_before(const T(&arr)[I], const T& data) { return binsearch_before<T, CT_, CFunc>(arr, I, data); }

  // Either gets the element that matches the value in question or one immediately after the closest match.
  template<typename T, typename CT_, char(*CFunc)(const T&, const T&)>
  BSS_FORCEINLINE static CT_ BSS_FASTCALL binsearch_after(const T* arr, const T& data, CT_ first, CT_ last) { return binsearch_near<T, T, CT_, CFunc, CompT_EQ<char>, 1>(arr, data, first, last); }

  template<typename T, typename CT_, char(*CFunc)(const T&, const T&)>
  BSS_FORCEINLINE static CT_ BSS_FASTCALL binsearch_after(const T* arr, CT_ length, const T& data) { return binsearch_near<T, T, CT_, CFunc, CompT_EQ<char>, 1>(arr, data, 0, length); }

  template<typename T, typename CT_, char(*CFunc)(const T&, const T&), CT_ I>
  BSS_FORCEINLINE static CT_ BSS_FASTCALL binsearch_after(const T(&arr)[I], const T& data) { return binsearch_after<T, CT_, CFunc>(arr, I, data); }

  // Returns index of the item, if it exists, or -1
  template<typename T, typename D, typename CT_, char(*CFunc)(const T&, const D&)>
  inline static CT_ BSS_FASTCALL binsearch_exact(const T* arr, const D& data, typename std::make_signed<CT_>::type f, typename std::make_signed<CT_>::type l)
  {
    --l; // Done so l can be an exclusive size parameter even though the algorithm is inclusive.
    CT_ m; // While f and l must be signed ints or the algorithm breaks, m does not.
    char r;
    while(l>=f) // This only works when l is an inclusive max indice
    {
      m=f+((l-f)>>1); // Done to avoid overflow on large numbers

      if((r=(*CFunc)(arr[m], data))<0) // This is faster than a switch statement
        f=m+1;
      else if(r>0)
        l=m-1;
      else
        return m;
    }
    return (CT_)-1;
  }

  template<typename T, typename CT_, char(*CFunc)(const T&, const T&), CT_ I>
  BSS_FORCEINLINE static CT_ BSS_FASTCALL binsearch_exact(const T(&arr)[I], const T& data) { return binsearch_exact<T, T, CT_, CFunc>(arr, data, 0, I); }

  // Implementation of an xorshift64star generator. x serves as the generator state, which should initially be set to the RNG seed.
  uint64_t static xorshift64star(uint64_t& x)
  {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * UINT64_C(2685821657736338717);
  }

  // Implementation of 2^1024-1 period xorshift generator. x is the 16*64 bit state, plus 1 extra integer for counting indices.
  uint64_t static xorshift1024star(uint64_t(&x)[17])
  {
    uint64_t x0 = x[x[16]];
    uint64_t x1 = x[x[16] = (x[16] + 1) & 15];
    x1 ^= x1 << 31; // a
    x1 ^= x1 >> 11; // b
    x0 ^= x0 >> 30; // c
    return (x[x[16]] = x0 ^ x1) * UINT64_C(1181783497276652981);
  }

  // Generates a seed for xorshift1024star from a 64-bit value
  void static genxor1024seed(uint64_t x, uint64_t(&seed)[17])
  {
    xorshift64star(x);
    for(uint8_t i = 0; i < 16; ++i)
      seed[i] = xorshift64star(x);
    seed[16]=0;
  }

  template<typename T>
  class xorshift_engine_base
  {
  public:
    BSS_FORCEINLINE static T base_min() { return std::numeric_limits<T>::min(); }
    BSS_FORCEINLINE static T base_max() { return std::numeric_limits<T>::max(); }
    BSS_FORCEINLINE static T base_transform(uint64_t x) { return (T)x; }
  };

  template<>
  class xorshift_engine_base<float>
  {
  public:
    BSS_FORCEINLINE static float base_min() { return base_transform(0xFFFFFFFFFFFFFFFF); }
    BSS_FORCEINLINE static float base_max() { return base_transform(0x7FFFFFFFFFFFFFFF); }
    BSS_FORCEINLINE static float base_transform(uint64_t x)
    { 
      uint32_t y=(uint32_t)x;
      y = (y&0xBFFFFFFF)+0x1F800000; // Mask out the top exponent bit to force exponent to 0-127 range, then add 63 to the exponent to get it in [63,190] range ([-64,63] when biased)
      return *(float*)(&y); // convert our integer into a float, assuming IEEE format
    }
  };

  template<>
  class BSS_COMPILER_DLLEXPORT xorshift_engine_base<double>
  {
  public:
    BSS_FORCEINLINE static double base_min() { return base_transform(0xFFFFFFFFFFFFFFFF); }
    BSS_FORCEINLINE static double base_max() { return base_transform(0x7FFFFFFFFFFFFFFF); }
    BSS_FORCEINLINE static double base_transform(uint64_t x)
    {
      x = (x&0xBFFFFFFFFFFFFFFF)+0x1FF0000000000000; // Mask out the top exponent bit to force exponent to 0-1023 range, then add 511 to the exponent to get it in [511,1534] range ([-512,511] when biased)
      return *(double*)(&x); // convert our integer into a double, assuming IEEE format
    }
  };

  template<typename T = uint64_t>
  class BSS_COMPILER_DLLEXPORT xorshift_engine : protected xorshift_engine_base<T>
  {
  public:
    xorshift_engine() { seed(); }
    explicit xorshift_engine(uint64_t s) { seed(s); }
    explicit xorshift_engine(uint64_t s[16]) { seed(s); }
    void seed() { std::random_device rd; genxor1024seed(rd(), _state); }
    void seed(uint64_t s) { genxor1024seed(s, _state); }
    void seed(uint64_t s[16]) { for(int i = 0; i < 16; ++i) _state[i]=s[i]; _state[16]=0; }
    void discard(unsigned long long z) { for(int i = 0; i < z; ++i) xorshift1024star(_state); }

    inline static T min() { return xorshift_engine_base<T>::base_min(); }
    inline static T max() { return xorshift_engine_base<T>::base_max(); }

    inline T operator()() { return xorshift_engine_base<T>::base_transform(xorshift1024star(_state)); } // Truncate to return_value size.
    bool operator ==(const xorshift_engine& r) const { for(int i = 0; i < 17; ++i) if(_state[i]!=r._state[i]) return false; return true; }
    bool operator !=(const xorshift_engine& r) const { return !operator==(r); }

    typedef T result_type;

  protected:
    uint64_t _state[17];
  };

  inline static uint64_t xorshiftrand(uint64_t seed=0) {
    static uint64_t state[17];
    if(seed) genxor1024seed(seed, state);
    return xorshift1024star(state);
  }
  typedef xorshift_engine<uint64_t> xorshift_engine64;

  template<typename T, class ENGINE, typename ET>
  T __bss_gencanonical(ENGINE& e, ET _Emin)
  {	// scale random value to [0, 1), integer engine
      return ((e() - _Emin)
        / ((T)e.max() - (T)_Emin + (T)1));
  }

  template<typename T, typename ENGINE>
  T __bss_gencanonical(ENGINE& e, float _Emin)
  {	// scale random value to [0, 1), float engine
      return ((e() - _Emin) / (e.max() - _Emin));
  }

  template<typename T, typename ENGINE>
  T __bss_gencanonical(ENGINE& e, double _Emin)
  {	// scale random value to [0, 1), double engine
      return ((e() - _Emin) / (e.max() - _Emin));
  }

  // VC++ has a broken implementation of std::generate_canonical so we get to reimplement it ourselves.
  template<typename T, typename ENGINE>
  T bss_gencanonical(ENGINE& e)
  {
    return __bss_gencanonical<T,ENGINE>(e, (typename ENGINE::result_type)e.min());
  }

  inline static xorshift_engine<uint64_t>& bss_getdefaultengine()
  {
    static xorshift_engine<uint64_t> e;
    return e;
  }

  // Generates a number in the range [min,max) using the given engine.
  template<typename T, typename ENGINE = xorshift_engine<uint64_t>>
  inline static T bssrand(T min, T max, ENGINE& e = bss_getdefaultengine())
  {
    return min+static_cast<T>(bss_gencanonical<double, ENGINE>(e)*static_cast<double>(max-min));
  }
  inline static double bssrandreal(double min, double max) { return bssrand<double>(min, max); }
  inline static int64_t bssrandint(int64_t min, int64_t max) { return bssrand<int64_t>(min, max); }
  inline static void bssrandseed(uint64_t s) { bss_getdefaultengine().seed(s); }

  // Shuffler using Fisher-Yates/Knuth Shuffle algorithm based on Durstenfeld's implementation.
  template<typename T, typename CT, typename ENGINE>
  inline static void BSS_FASTCALL shuffle(T* p, CT size, ENGINE& e)
  {
    for(CT i=size; i>0; --i)
      rswap<T>(p[i-1], p[bssrand<CT,ENGINE>(0, i, e)]);
  }
  template<typename T, typename CT, typename ENGINE, CT size>
  inline static void BSS_FASTCALL shuffle(T(&p)[size], ENGINE& e) { shuffle<T, CT, ENGINE>(p, size, e); }

  /* Shuffler using default random number generator.*/
  template<typename T>
  BSS_FORCEINLINE static void BSS_FASTCALL shuffle(T* p, int size)
  {
    xorshift_engine<uint64_t> e; 
    shuffle<T, int, xorshift_engine<uint64_t>>(p, size, e);
  }
  template<typename T, int size>
  BSS_FORCEINLINE static void BSS_FASTCALL shuffle(T(&p)[size]) { shuffle<T>(p, size); }

  template<class F, typename T, size_t SIZE>
  BSS_FORCEINLINE static void transform(T(&t)[SIZE], T(&result)[SIZE], F func) { std::transform(std::begin(t), std::end(t), result, func); }
  template<class F, typename T, size_t SIZE>
  BSS_FORCEINLINE static void transform(T(&t)[SIZE], F func) { std::transform(std::begin(t), std::end(t), t, func); }
  template<class F, typename T, size_t SIZE>
  BSS_FORCEINLINE static void for_each(T(&t)[SIZE], F func) { std::for_each(std::begin(t), std::end(t), func); }

  // Random queue that pops a random item instead of the last item.
  template<typename T, typename CType = uint32_t, typename ENGINE = xorshift_engine<uint64_t>, ARRAY_TYPE ArrayType = CARRAY_SIMPLE, typename Alloc = StaticAllocPolicy<T>>
  class BSS_COMPILER_DLLEXPORT cRandomQueue : protected cDynArray<T, CType, ArrayType, Alloc>
  {
  protected:
    typedef CType CT_;
    typedef cDynArray<T, CType, ArrayType, Alloc> AT_;
    using AT_::_array;
    using AT_::_length;

  public:
    cRandomQueue(const cRandomQueue& copy) : AT_(copy) {}
    cRandomQueue(cRandomQueue&& mov) : AT_(std::move(mov)) {}
    explicit cRandomQueue(CT_ size = 0, ENGINE& e = bss_getdefaultengine()) : AT_(size), _e(e) {}
    inline void Push(const T& t) { AT_::Add(t); }
    inline void Push(T&& t) { AT_::Add(std::move(t)); }
    inline T Pop() { CT_ i=bssrand<CT_, ENGINE>(0, _length, _e); T r = std::move(_array[i]); Remove(i); return r; }
    inline void Remove(CT_ index) { _array[index]=std::move(_array[--_length]); }
    inline bool Empty() const { return !_length; }
    inline void Clear() { _length=0; }
    inline void SetLength(CT_ length) { AT_::SetLength(length); }
    inline CT_ Length() const { return _length; }
    inline const T* begin() const { return _array; }
    inline const T* end() const { return _array+_length; }
    inline T* begin() { return _array; }
    inline T* end() { return _array+_length; }

    inline operator T*() { return _array; }
    inline operator const T*() const { return _array; }
    inline cRandomQueue& operator=(const cRandomQueue& copy) { AT_::operator=(copy); return *this; }
    inline cRandomQueue& operator=(cRandomQueue&& mov) { AT_::operator=(std::move(mov)); return *this; }
    inline cRandomQueue& operator +=(const cRandomQueue& add) { AT_::operator+=(add); return *this; }
    inline const cRandomQueue operator +(const cRandomQueue& add) const { cRandomQueue r(*this); return (r+=add); }

  protected:
    ENGINE& _e;
  };

  static const double ZIGNOR_R = 3.442619855899;

  // Instance for generating random samples from a normal distribution.
  template<int ZIGNOR_C=128, typename T=double, typename ENGINE = xorshift_engine<uint64_t>>
  struct NormalZig
  {
    T s_adZigX[ZIGNOR_C + 1], s_adZigR[ZIGNOR_C];

    NormalZig(int iC=ZIGNOR_C, T dV=9.91256303526217e-3, T dR=ZIGNOR_R, ENGINE& e = bss_getdefaultengine()) :  // (R * phi(R) + Pr(X>=R)) * sqrt(2\pi)
      _e(e), _dist(0, 0x7F), _rdist(0, 1.0)
    {
      int i; T f;
      f = exp(T(-0.5) * (dR * dR));
      s_adZigX[0] = dV / f; /* [0] is bottom block: V / f(R) */
      s_adZigX[1] = dR;
      s_adZigX[iC] = 0;
      for(i = 2; i < iC; ++i)
      {
        s_adZigX[i] = sqrt(-2 * log(dV / s_adZigX[i - 1] + f));
        f = exp(T(-0.5) * (s_adZigX[i] * s_adZigX[i]));
      }
      for(i = 0; i < iC; ++i)
        s_adZigR[i] = s_adZigX[i + 1] / s_adZigX[i];
    }

    T DRanNormalTail(T dMin, int iNegative)
    {
      T x, y;
      do
      {
        x = log(_rdist(_e)) / dMin;
        y = log(_rdist(_e));
      } while(-2 * y < x * x);
      return iNegative ? x - dMin : dMin - x;
    }

    // Returns a random normally distributed number using the ziggurat method
    T Get()
    {
      uint32_t i;
      T x, u, f0, f1;
      for(;;)
      {
        u = 2 * _rdist(_e) - 1;
        i = _dist(_e); // get random number between 0 and 0x7F
        /* first try the rectangular boxes */
        if(fabs(u) < s_adZigR[i])
          return u * s_adZigX[i];
        /* bottom box: sample from the tail */
        if(i == 0)
          return DRanNormalTail(ZIGNOR_R, u < 0);
        /* is this a sample from the wedges? */
        x = u * s_adZigX[i];
        f0 = exp(T(-0.5) * (s_adZigX[i] * s_adZigX[i] - x * x));
        f1 = exp(T(-0.5) * (s_adZigX[i+1] * s_adZigX[i+1] - x * x));
        if(f1 + _rdist(_e) * (f0 - f1) < 1.0)
          return x;
      }
    }

    T operator()() { return Get(); }
    ENGINE& _e;
    std::uniform_int_distribution<uint32_t> _dist;
    std::uniform_real_distribution<double> _rdist;
  };

  // Randomly subdivides a rectangular area into smaller rects of varying size. F1 takes (depth,rect) and returns how likely it is that a branch will terminate.
  template<typename T, typename F1, typename F2, typename F3> // F2 takes (const float (&rect)[4]) and is called when a branch terminates on a rect.
  static void StochasticSubdivider(const T(&rect)[4], const F1& f1, const F2& f2, const F3& f3, uint32_t depth=0) // F3 returns a random number from [0,1]
  {
    if(bssrandreal(0, 1.0)<f1(depth, rect))
    {
      f2(rect);
      return;
    }
    uint8_t axis=depth%2;
    T div=lerp(rect[axis], rect[2+axis], f3(depth, rect));
    T r1[4] ={ rect[0], rect[1], rect[2], rect[3] };
    T r2[4] ={ rect[0], rect[1], rect[2], rect[3] };
    r1[axis]=div;
    r2[2+axis]=div;
    StochasticSubdivider(r1, f1, f2, f3, ++depth);
    StochasticSubdivider(r2, f1, f2, f3, depth);
  }

  template<typename T>
  BSS_FORCEINLINE static size_t BSS_FASTCALL _PDS_imageToGrid(const std::array<T, 2>& pt, T cell, size_t gw, T(&rect)[4])
  {
    return (size_t)((pt[0]-rect[0]) / cell) + gw*(size_t)((pt[1]-rect[1]) / cell) + 2 + gw + gw;
  }

  // Implementation of Fast Poisson Disk Sampling by Robert Bridson
  template<typename T, typename F>
  static void PoissonDiskSample(T(&rect)[4], T mindist, F && f, uint32_t pointsPerIteration=30)
  {
    //Create the grid
    T cell = mindist/(T)SQRT_TWO;
    T w = rect[2]-rect[0];
    T h = rect[3]-rect[1];
    size_t gw = ((size_t)ceil(w/cell))+4; //gives us buffer room so we don't have to worry about going outside the grid
    size_t gh = ((size_t)ceil(h/cell))+4;
    std::array<T, 2>* grid = new std::array<T, 2>[gw*gh];      //grid height
    uint64_t* ig = (uint64_t*)grid;
    memset(grid, 0xFFFFFFFF, gw*gh*sizeof(std::array<T, 2>));
    assert(!(~ig[0]));

    cRandomQueue<std::array<T, 2>> list;
    std::array<T, 2> pt ={ (T)bssrandreal(rect[0], rect[2]), (T)bssrandreal(rect[1], rect[3]) };

    //update containers 
    list.Push(pt);
    f(pt.data());
    grid[_PDS_imageToGrid<T>(pt, cell, gw, rect)] = pt;

    T mindistsq = mindist*mindist;
    T radius, angle;
    size_t center, edge;
    //generate other points from points in queue.
    while(!list.Empty())
    {
      auto point = list.Pop();
      for(uint32_t i = 0; i < pointsPerIteration; i++)
      {
        radius = mindist*((T)bssrandreal(1, 2)); //random point between mindist and 2*mindist
        angle = (T)bssrandreal(0, PI_DOUBLE);
        pt[0] = point[0] + radius * cos(angle); //the new point is generated around the point (x, y)
        pt[1] = point[1] + radius * sin(angle);

        if(pt[0]>rect[0] && pt[0]<rect[2] && pt[1]>rect[1] && pt[1]<rect[3]) //Ensure point is inside recT
        {
          center = _PDS_imageToGrid<T>(pt, cell, gw, rect); // If another point is in the neighborhood, abort this point.
          edge=center-gw-gw;
          assert(edge>0);
#define POISSONSAMPLE_CHECK(edge) if((~ig[edge])!=0 && distsqr(grid[edge][0],grid[edge][1],pt[0],pt[1])<mindistsq) continue
          POISSONSAMPLE_CHECK(edge-1);
          POISSONSAMPLE_CHECK(edge);
          POISSONSAMPLE_CHECK(edge+1);
          edge+=gw;
          if(~(ig[edge-1]&ig[edge]&ig[edge+1])) continue;
          POISSONSAMPLE_CHECK(edge-2);
          POISSONSAMPLE_CHECK(edge+2);
          edge+=gw;
          if(~(ig[edge-1]&ig[edge]&ig[edge+1])) continue;
          POISSONSAMPLE_CHECK(edge-2);
          POISSONSAMPLE_CHECK(edge+2);
          edge+=gw;
          if(~(ig[edge-1]&ig[edge]&ig[edge+1])) continue;
          POISSONSAMPLE_CHECK(edge-2);
          POISSONSAMPLE_CHECK(edge+2);
          edge+=gw;
          POISSONSAMPLE_CHECK(edge-1);
          POISSONSAMPLE_CHECK(edge);
          POISSONSAMPLE_CHECK(edge+1);
          list.Push(pt);
          f(pt.data());
          grid[center] = pt;
        }
      }
    }
  }

  // Implementation of a uniform quadratic B-spline interpolation
  template<typename T, typename D>
  inline static T BSS_FASTCALL UniformQuadraticBSpline(D t, const T& p1, const T& p2, const T& p3)
  {
    D t2=t*t;
    return (p1*(1 - 2*t + t2) + p2*(1 + 2*t - 2*t2) + p3*t2)/((D)2.0);
  }

  // Implementation of a uniform cubic B-spline interpolation. A uniform cubic B-spline matrix is:
  //                / -1  3 -3  1 \         / p1 \
  // [t^3,t²,t,1] * |  3 -6  3  0 | * 1/6 * | p2 |
  //                | -3  0  3  0 |         | p3 |
  //                \  1  4  1  0 /         \ p4 /
  template<typename T, typename D>
  inline static T BSS_FASTCALL UniformCubicBSpline(D t, const T& p1, const T& p2, const T& p3, const T& p4)
  {
    D t2=t*t;
    D t3=t2*t;
    return (p1*(1-3*t+3*t2-t3) + p2*(4-6*t2+3*t3)+p3*(1+3*t+3*t2-3*t3)+(p4*t3))/((D)6.0);
  }

  // Implementation of a basic cubic interpolation. The B-spline matrix for this is
  //                / -1  3 -3  1 \       / p1 \
  // [t^3,t²,t,1] * |  2 -5  4 -1 | * ½ * | p2 |
  //                | -1  0  1  0 |       | p3 |
  //                \  0  2  0  0 /       \ p4 /
  template<typename T, typename D>
  inline static T BSS_FASTCALL CubicBSpline(D t, const T& p1, const T& p2, const T& p3, const T& p4)
  {
    D t2=t*t;
    D t3=t2*t;
    return (p1*(-t3+2*t2-t)+p2*(3*t3-5*t2+2)+p3*(-3*t3+4*t2+t)+p4*(t3-t2))/((D)2.0);
  }

  // This implements all possible B-spline functions, but does it statically without optimizations (so it can be used with any type)
  template<typename T, typename D, const T(&m)[4][4]>
  BSS_FORCEINLINE static T BSS_FASTCALL StaticGenericSpline(D t, const T(&p)[4])
  {
    D t2 = t*t;
    D t3 = t2*t;
    return p[0]*(m[0][0]*t3+m[1][0]*t2+m[2][0]*t+m[3][0]) +
      p[1]*(m[0][1]*t3+m[1][1]*t2+m[2][1]*t+m[3][1]) +
      p[2]*(m[0][2]*t3+m[1][2]*t2+m[2][2]*t+m[3][2]) +
      p[3]*(m[0][3]*t3+m[1][3]*t2+m[2][3]*t+m[3][3]);
  }

  // This implements all possible B-spline functions using a given matrix m, optimized for floats
  inline static float BSS_FASTCALL GenericBSpline(float t, const float(&p)[4], const float(&m)[4][4])
  {
    float a[4] ={ t*t*t, t*t, t, 1 };
    sseVec r = MatrixMultiply1x4<float>(a, m);
    r *= sseVec(p);
    sseVec r2 = r;
    sseVec::Shuffle<0xB1>(r2);
    r += r2;
    sseVec::Shuffle<0x1B>(r2);
    return BSS_SSE_SS_F32(r + r2);
  }

  // Implementation of a bezier curve. The B-spline matrix for this is
  //                / -1  3 -3  1 \   / p1 \
      // [t^3,t²,t,1] * |  3 -6  3  0 | * | p2 |
  //                | -3  3  0  0 |   | p3 |
  //                \  1  0  0  0 /   \ p4 /
  template<typename T, typename D>
  inline static T BSS_FASTCALL BezierCurve(D t, const T& p1, const T& p2, const T& p3, const T& p4)
  {
    static constexpr float m[4][4] ={-1.0f, 3.0f, -3.0f, 1.0f, 3.0f, -6.0f, 3.0f, 0, -3.0f, 3, 0, 0, 1.0f, 0, 0, 0};
    const float p[4] ={ p1, p2, p3, p4 };
    return StaticGenericSpline<T, D, m>(t, p);
  }

  // Generic breadth-first search for a binary tree. Don't use this on a min/maxheap - it's internal array already IS in breadth-first order.
  template<typename T, bool(*FACTION)(T*), T* (*LCHILD)(T*), T* (*RCHILD)(T*)> // return true to quit
  inline static void BSS_FASTCALL BreadthFirstTree(T* root, size_t n)
  {
    n=(n/2)+1;
    DYNARRAY(T*, queue, n);
    queue[0]=root;
    size_t l=1;
    for(size_t i=0; i!=l; i=(i+1)%n)
    {
      if(FACTION(queue[i])) return;
      queue[l]=LCHILD(queue[i]); //Enqueue the children
      l=(l+(size_t)(queue[l]!=0))%n;
      queue[l]=RCHILD(queue[i]);
      l=(l+(size_t)(queue[l]!=0))%n;
    }
  }

  template<typename T, typename D>
  inline T QuadraticBezierCurve(D t, const T& p0, const T& p1, const T& p2)
  {
    D inv = D(1) - t;
    return (inv*inv*p0) + (D(2) * inv*t*p1) + (t*t*p2);
  }

  // Find a quadratic curve (A,B,C) that passes through 3 points
  template<typename T>
  inline T QuadraticFit(T t, T x1, T y1, T x2, T y2, T x3, T y3)
  {
    T x1_2 = x1*x1;
    T x2_2 = x2*x2;
    T x3_2 = x3*x3;
    T A = ((y2 - y1)(x1 - x3) + (y3 - y1)(x2 - x1)) / ((x1 - x3)(x2_2 - x1_2) + (x2 - x1)(x3_2 - x1_2));
    T B = ((y2 - y1) - A(x2_2 - x1_2)) / (x2 - x1);
    T C = y1 - A*x1_2 - B*x1;
    return A*t*t + B*t + C;
  }

  // Find a quadratic curve (A,B,C) that passes through the points (x1, 0), (x2, 0.5), (x3, 1)
  template<typename T>
  inline T QuadraticCanonicalFit(T t, T x1, T x2, T x3)
  {
    t = (t - x1) / (x3 - x1); // We transform all points with (x - x1) / x3, which yields:
    // x1 = (x1 - x1) / (x3 - x1) = 0
    T x = (x2 - x1) / (x3 - x1);
    // x3 = (x3 - x1) / (x3 - x1) = 1
    //T x1_2 = 0;
    //T x2_2 = x2*x2;
    //T x3_2 = 1;
    T H = ((T)1 / (T)2);
    //T A = ((H - 0)(0 - 1) + (1 - 0)(x2 - 0)) / ((0 - 1)(x2_2 - 0) + (x2 - 0)(1 - 0));
    T A = (x - H) / (x - (x*x));
    //T B = ((H - 0) - A(x2_2 - 0)) / (x2 - 0);
    T B = (H / x) - (A * x);
    //T C = 0 - A*0 - B*0;

    return (A*t*t + B*t)*(x3 - x1) + x1; // reverse our transformation
  }

  // Splits a cubic (P0,P1,P2,P3) into two cubics: (P0, N1, N2, N3) and (N3, R1, R2, P3) using De Casteljau's Algorithm
  template<typename T, int I>
  inline static void SplitCubic(T t, const T(&P0)[I], const T(&P1)[I], const T(&P2)[I], const T(&P3)[I], T(&N1)[I], T(&N2)[I], T(&N3)[I], T(&R1)[I], T(&R2)[I])
  {
    T F[I]; // A = P0, B = P1, C = P2, D = P3, E = N1, F, G = R2, H = N2, J = R1, K = N3
    for(int i = 0; i < I; ++i)
    {
      N1[i] = lerp<T>(P0[i], P1[i], t); // lerp(A+B)
      F[i] = lerp<T>(P1[i], P2[i], t); // lerp(B+C)
      R2[i] = lerp<T>(P2[i], P3[i], t); // lerp(C+D)
      N2[i] = lerp<T>(N1[i], F[i], t); // lerp(E+F)
      R1[i] = lerp<T>(F[i], R2[i], t); // lerp(F+G)
      N3[i] = lerp<T>(N2[i], R1[i], t); // lerp(H+J)
    }
  }

  // Splits a quadratic (P0, P1, P2) into two quadratics: (P0, N1, N2) and (N2, R1, P2) using De Casteljau's Algorithm
  template<typename T, int I>
  inline static void SplitQuadratic(T t, const T(&P0)[I], const T(&P1)[I], const T(&P2)[I], T(&N1)[I], T(&N2)[I], T(&R1)[I])
  {
    for(int i = 0; i < I; ++i)
    {
      N1[i] = lerp<T>(P0[i], P1[i], t);
      R1[i] = lerp<T>(P1[i], P2[i], t);
      N2[i] = lerp<T>(N1[i], R1[i], t);
    }
  }

  // Solves a quadratic equation of the form at² + bt + c
  template<typename T>
  inline static void SolveQuadratic(T a, T b, T c, T(&r)[2])
  {
    T d = FastSqrt<T>(b*b - 4*a*c);
    r[0] = (-b - d) / (2 * a);
    r[1] = (-b + d) / (2 * a);
  }

  // See: http://www.caffeineowl.com/graphics/2d/vectorial/cubic-inflexion.html
  template<typename T>
  inline static void CubicInflectionPoints(const T(&P0)[2], const T(&P1)[2], const T(&P2)[2], const T(&P3)[2], T(&r)[2])
  {
    T a[2];
    T b[2];
    T c[2];
    for(int i = 0; i < 2; ++i)
    {
      a[i] = P1[i] - P0[i];
      b[i] = P2[i] - P1[i] - a[i];
      c[i] = P3[i] - P2[i] - a[i] - 2 * b[i];
    }
    SolveQuadratic<T>(b[0] * c[1] - b[1] * c[0], a[0] * c[1] - a[1] * c[0], a[0] * b[1] - a[1] * b[0], r);
  }

  // Uses modified formulas from: http://www.caffeineowl.com/graphics/2d/vectorial/cubic2quad01.html
  template<typename T, int I>
  inline static T ApproxCubicError(const T(&P0)[I], const T(&P1)[I], const T(&P2)[I], const T(&P3)[I])
  {
    // The error for a quadratic approximating a cubic (sharing the anchor points) is: t·(1 - t)·|2·C - 3·C1 + P1 + 3·t·(P2 - 3·C2 + 3·C1 - P1)|    where |v| is the modulus: sqrt(v[0]² + v[1]²)
    // If we choose C = (3·C2 - P2 + 3·C1 - P1)/4 we get f(t) = t·(1 - t)·½(6·t - 1)|(3·C_1 - 3·C_2 - P_1 + P_2)|
    // f'(t) = -½(2·t(9·t - 7) + 1) |(3·C_1 - 3·C_2 - P_1 + P_2)| = 0 -> -½(2·t(9·t - 7) + 1) = 0 -> 2·t(9·t - 7) = -1
    // Solving the derivative for 0 to maximize the error value, we get t = (1/18)(7±sqrt(31)). Only the + result is inside [0,1], so we plug that into t·(1 - t)·½(6·t - 1) = 77/486+(31 sqrt(31))/972 ~ 0.336008945728118
    const double term = 0.336008945728118;

    T r = 0;
    T M; 
    for(int i = 0; i < I; ++i) // 3·C_1 - 3·C_2 - P_1 + P_2
    {
      M = 3 * P1[i] - 3 * P2[i] - P0[i] + P3[i];
      r += M*M;
    }
    return term * FastSqrt<T>(r);
  }

  template<typename T, typename FN>
  inline static void ApproxCubicR(T(&t)[3], const T(&P0)[2], const T(&P1)[2], const T(&P2)[2], const T(&P3)[2], FN fn, T maxerror)
  {
    T N1[2];
    T N2[2];
    T N3[2];
    T R1[2];
    T R2[2];

    SplitCubic(t[1], P0, P1, P2, P3, N1, N2, N3, R1, R2);

    // Check first section: P0, N1, N2, N3
    if(ApproxCubicError<T, 2>(P0, N1, N2, N3) > maxerror)
    {
      T tfirst[3] = { t[0], (t[1] + t[0]) / 2, t[1] };
      ApproxCubicR<T, FN>(tfirst, P0, N1, N2, N3, fn, maxerror);
    }
    else
    {
      T C[2]; // (3·(P2 + P1) - P3 - P0) / 4
      for(int i = 0; i < 2; ++i)
        C[i] = (3*(N2[i] + N1[i]) - N3[i] - P0[i]) / 4;
      fn(P0, C, N3);
    }
    // Check second section: N3, R1, R2, P3
    if(ApproxCubicError<T, 2>(N3, R1, R2, P3) > maxerror)
    {
      T tsecond[3] = { t[1], (t[2] + t[1]) / 2, t[2] };
      ApproxCubicR<T, FN>(tsecond, N3, R1, R2, P3, fn, maxerror);
    }
    else
    {
      T C[2]; // (3·(P2 + P1) - P3 - P0) / 4
      for(int i = 0; i < 2; ++i)
        C[i] = (3 * (R2[i] + R1[i]) - P3[i] - N3[i]) / 4;
      fn(N3, C, P3);
    }
  }

  template<typename T, typename FN>
  inline static void ApproxCubic(const T(&P0)[2], const T(&P1)[2], const T(&P2)[2], const T(&P3)[2], FN fn, T maxerror = FLT_EPSILON)
  {
    T r[2];
    CubicInflectionPoints<T>(P0, P1, P2, P3, r);
    T t[3] = { 0.0, (r[0] >= 0.0 && r[0] <= 1.0) ? r[0] : ((r[1] >= 0.0 && r[1] <= 1.0) ? r[1] : 0.5), 1.0 };
    ApproxCubicR<T, FN>(t, P0, P1, P2, P3, fn, maxerror); // Call recursive function that does the actual splitting
  }

  // Solves a cubic equation of the form at^3 + bt² + ct + d by normalizing it (dividing everything by a).
  template<typename T>
  inline int solveCubic(T at, T bt, T ct, T dt, T (&r)[3])
  {
    T a = bt / at;
    T b = ct / at;
    T c = dt / at;
    T p = b - a*a / 3;
    T q = a * (2 * a*a - 9 * b) / 27 + c;
    T p3 = p*p*p;
    T d = q*q + 4 * p3 / 27;
    T offset = -a / 3;
    if(d >= 0) { // Single solution
      T z = FastSqrt<T>(d);
      T u = (-q + z) / 2;
      T v = (-q - z) / 2;
      u = cbrt(u);
      v = cbrt(v);
      r[0] = offset + u + v;
      return 1;
    }
    T u = FastSqrt<T>(-p / 3);
    T v = acos(-FastSqrt<T>(-27 / p3) * q / 2) / 3;
    T m = cos(v), n = sin(v)*((T)1.732050808);
    r[0] = offset + u * (m + m);
    r[1] = offset - u * (n + m);
    r[2] = offset + u * (n - m);
    return 3;
  }

  // Uses Newton's method to find the root of F given it's derivative dF. Returns false if it fails to converge within the given error range.
  template<typename T, typename TF, typename TDF>
  inline static bool NewtonRaphson(T& result, T estimate, TF F, TDF dF, T epsilon, uint32_t maxiterations = 20)
  {
    T x = estimate;
    for(uint32_t i = 0; i < maxiterations; ++i)
    {
      T f = F(x);
      if(fsmall(f, epsilon)) // If we're close enough to zero, return our value
      {
        result = x;
        return true;
      }
      x = x - (f / dF(x));
    }
    return false; // We failed to converge to the required error bound within the given number of iterations
  }

  // A hybrid method that combines both Newton's method and the bisection method, using the bisection method to improve the guess until Newton's method finally starts to converge.
  template<typename T, typename TF, typename TDF>
  inline static T NewtonRaphsonBisection(T estimate, T min, T max, TF F, TDF dF, T epsilon, uint32_t maxiterations = 50)
  {
    T x = estimate;
    for(uint32_t i = 0; i < maxiterations; ++i)
    {
      T f = F(x);
      if(fsmall(f, epsilon)) // If we're close enough to zero, return x
        return x;

      if(f > 0)
        max = x;
      else
        min = x;

      x = x - f / dF(x);

      if(x <= min || x >= max) // If candidate is out of range, use bisection instead
        x = 0.5*(min + max);
    }
    return x; // Return a good-enough value. Because we're using bisection, it has to be at least reasonably close to the root.
  }

  inline static size_t BSS_FASTCALL Base64Encode(const uint8_t* src, size_t cnt, char* out)
  {
    size_t cn = ((cnt / 3) << 2) + (cnt % 3) + (cnt % 3 != 0);
    if(!out) return cn;

    /*const uint32_t* ints = (const uint32_t*)src;
    size_t s = (cnt - (cnt % 12))/4;
    size_t c = 0;
    size_t i;
    for(i = 0; i < s; i += 3)
    {
      sseVeci x(ints[i], ints[i], ints[i + 1], ints[i + 2]);
      sseVeci y(ints[i], ints[i + 1], ints[i + 2], ints[i + 2]);
      sseVeci a(0b00111111001111110011111100111111, 0b11000000110000001100000011000000, 0b11110000111100001111000011110000, 0b11111100111111001111110011111100);
      sseVeci b(0b00000000000000000000000000000000, 0b00001111000011110000111100001111, 0b00000011000000110000001100000011, 0b00000000000000000000000000000000);
      sseVeci n(0, 6, 4, 2);
      sseVeci m(0, 2, 4, 0);

      sseVeci res = (sseVeci(BSS_SSE_SR_EPI32((x&a), n)) | ((y&b) << m));
      res += sseVeci('A')&(res < sseVeci(26));
      res += sseVeci('a' - 26)&(res < sseVeci(52));
      res += sseVeci('0' - 52)&(res < sseVeci(62)); // this works because up until this point, all previous blocks were moved past the 62 mark
      res += sseVeci('-' - 62)&(res == sseVeci(62));
      res += sseVeci('_' - 63)&(res == sseVeci(63));

      BSS_SSE_STORE_USI128((BSS_SSE_M128i16*)(out + c), res);
      c += 16;
    }*/

    static const char* code = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    size_t c = 0;
    size_t s = cnt - (cnt % 3);
    size_t i;
    for(i = 0; i < s; i+=3)
    {
      out[c++] = code[((src[i + 0] & 0b11111100) >> 2)];
      out[c++] = code[((src[i + 0] & 0b00000011) << 4) | ((src[i + 1] & 0b11110000) >> 4)];
      out[c++] = code[((src[i + 1] & 0b00001111) << 2) | ((src[i + 2] & 0b11000000) >> 6)];
      out[c++] = code[((src[i + 2] & 0b00111111))];
      //out[c++] = code[src[i] & 0b00111111];
      //out[c++] = code[((src[i] & 0b11000000) >> 6) | ((src[i + 1] & 0b00001111) << 2)];
      //out[c++] = code[((src[i + 1] & 0b11110000) >> 4) | ((src[i + 2] & 0b00000011) << 4)];
      //out[c++] = code[((src[i + 2] & 0b11111100) >> 2)];
    }
    if(i < cnt)
    {
      out[c++] = code[((src[i] & 0b11111100) >> 2)];
      if((i + 1) >= cnt) out[c++] = code[((src[i + 0] & 0b00000011) << 4)];
      else
      {
        out[c++] = code[((src[i + 0] & 0b00000011) << 4) | ((src[i + 1] & 0b11110000) >> 4)];
        out[c++] = code[((src[i + 1] & 0b00001111) << 2)];
      }
    }
    return c;
  }

  inline static size_t BSS_FASTCALL Base64Decode(const char* src, size_t cnt, uint8_t* out)
  {
    if((cnt & 0b11) == 1) return 0; // You cannot have a legal base64 encode of this length.
    size_t cn = ((cnt >> 2) * 3) + (cnt & 0b11) - ((cnt & 0b11) != 0);
    if(!out) return cn;

    size_t i = 0;
    size_t c = 0;
    uint8_t map[78] = { 62,0,0,52,53,54,55,56,57,58,59,60,61,0,0,0,0,0,0,0,
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,0,0,0,0,63,0,
    26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51 };
    size_t s = cnt & (~0b11);
    for(; i < s; i += 4)
    {
      out[c++] = (map[src[i + 0] - '-'] << 2) | (map[src[i + 1] - '-'] >> 4);
      out[c++] = (map[src[i + 1] - '-'] << 4) | (map[src[i + 2] - '-'] >> 2);
      out[c++] = (map[src[i + 2] - '-'] << 6) | (map[src[i + 3] - '-']);
    }
    if(++i < cnt) out[c++] = (map[src[i - 1] - '-'] << 2) | (map[src[i] - '-'] >> 4); // we do ++i here even though i was already valid because the first character requires TWO source characters, not 1.
    if(++i < cnt) out[c++] = (map[src[i - 1] - '-'] << 4) | (map[src[i] - '-'] >> 2);
    return c;
  }
}


#endif
