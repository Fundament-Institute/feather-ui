// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_HASH_H__BSS__
#define __C_HASH_H__BSS__

#include "khash.h"
#include "bss_util.h"
#include "cArray.h"
#include "cStr.h"
#include <wchar.h>
#include <iterator>

namespace bss_util {
  template<class T, ARRAY_TYPE ArrayType, typename Alloc>
  struct __cHashBaseAlloc
  {
    inline static T* _realloc(khint32_t* flags, T* src, khint_t new_n_buckets, khint_t n_buckets) noexcept { return (T*)Alloc::allocate(new_n_buckets * sizeof(T), (char*)src); }
  };

  template<class T, typename Alloc>
  struct __cHashBaseAlloc<T, CARRAY_CONSTRUCT, Alloc>
  {
    inline static T* _realloc(khint32_t* flags, T* src, khint_t new_n_buckets, khint_t n_buckets) noexcept { return (T*)Alloc::allocate(new_n_buckets * sizeof(T), (char*)src); }
  };

  template<class T, typename Alloc>
  struct __cHashBaseAlloc<T, CARRAY_SAFE, Alloc>
  {
    inline static T* _realloc(khint32_t* flags, T* src, khint_t new_n_buckets, khint_t n_buckets) noexcept
    {
      T* n = (T*)Alloc::allocate(new_n_buckets * sizeof(T), 0);
      if(n != nullptr)
      {
        for(khint_t i = 0; i < n_buckets; ++i)
        {
          if(!__ac_iseither(flags, i))
            new(n + i) T(std::move(src[i]));
        }
      }
      if(src) Alloc::deallocate((char*)src);
      return n;
    }
  };

  template<class T, typename Alloc>
  struct __cHashBaseAlloc<T, CARRAY_MOVE, Alloc>
  {
    inline static T* _realloc(khint32_t* flags, T* src, khint_t new_n_buckets, khint_t n_buckets) noexcept { return __cHashBaseAlloc<T, CARRAY_SAFE, Alloc>(flags, src, new_n_buckets, n_buckets); }
  };

  template<class T, bool integral>
  struct __cHashBaseInvalid { static const T INVALID() { return (T)0; } };

  template<class T>
  struct __cHashBaseInvalid<T, true> { static const T INVALID() { return (T)~0; } };

  template<class T, bool value>
  struct __cHashBaseGET { typedef T GET; static BSS_FORCEINLINE GET F(T& s) { return s; } };

  template<class T>
  struct __cHashBaseGET<T, false> { typedef T* GET; static BSS_FORCEINLINE GET F(T& s) { return &s; } };

  template<>
  struct __cHashBaseGET<cStr, false> { typedef const char* GET; static BSS_FORCEINLINE GET F(cStr& s) { return s.c_str(); } };

#ifdef BSS_PLATFORM_WIN32
  template<>
  struct __cHashBaseGET<cStrW, false> { typedef const wchar_t* GET; static BSS_FORCEINLINE GET F(cStrW& s) { return s.c_str(); } };
#endif

  template<class Key, class Data, bool IsMap, khint_t(*__hash_func)(const Key&), bool(*__hash_equal)(const Key&, const Key&), ARRAY_TYPE ArrayType = CARRAY_SIMPLE, typename Alloc = StaticAllocPolicy<char>>
  class cHashBase
  {
  public:
    typedef Key KEY;
    typedef Data DATA;
    typedef typename __cHashBaseGET<Data, std::is_integral<Data>::value | std::is_pointer<Data>::value>::GET GET;

    cHashBase(const cHashBase& copy)
    {
      memset(this, 0, sizeof(cHashBase));
      operator=(copy);
    }
    cHashBase(cHashBase&& mov)
    {
      memcpy(this, &mov, sizeof(cHashBase));
      memset(&mov, 0, sizeof(cHashBase));
    }
    explicit cHashBase(khint_t n_buckets = 0)
    {
      memset(this, 0, sizeof(cHashBase));
      if(n_buckets > 0)
        _resize(n_buckets);
    }
    ~cHashBase()
    {
      Clear(); // calls all destructors.
      if(flags) Alloc::deallocate((char*)flags);
      if(keys) Alloc::deallocate((char*)keys);
      if(vals) Alloc::deallocate((char*)vals);
    }

    template<bool U = IsMap>
    inline typename std::enable_if<U, khiter_t>::type Insert(const Key& key, const Data& value) { return _insert<const Key&, const Data&>(key, value); }
    template<bool U = IsMap>
    inline typename std::enable_if<U, khiter_t>::type Insert(const Key& key, Data&& value) { return _insert<const Key&, Data&&>(key, std::move(value)); }
    template<bool U = IsMap>
    inline typename std::enable_if<U, khiter_t>::type Insert(Key&& key, const Data& value) { return _insert<Key&&, const Data&>(std::move(key), value); }
    template<bool U = IsMap>
    inline typename std::enable_if<U, khiter_t>::type Insert(Key&& key, Data&& value) { return _insert<Key&&, Data&&>(std::move(key), std::move(value)); }
    template<bool U = IsMap>
    inline typename std::enable_if<!U, khiter_t>::type Insert(const Key& key) { int r; return _put<const Key&>(key, &r); }
    template<bool U = IsMap>
    inline typename std::enable_if<!U, khiter_t>::type Insert(Key&& key) { int r; return _put<Key&&>(std::move(key), &r); }

    void Clear()
    {
      if(flags)
      {
        for(khint_t i = 0; i < n_buckets; ++i)
        {
          if(!__ac_iseither(flags, i))
          {
            keys[i].~Key();
            if(IsMap)
              vals[i].~Data();
          }
        }
        memset(flags, 0xaa, ((n_buckets >> 4) + 1) * sizeof(khint32_t));
        size = n_occupied = 0;
      }
    }
    inline khiter_t Iterator(const Key& key) const { return _get(key); }
    inline const Key& GetKey(khiter_t i) { return keys[i]; }
    template<bool U = IsMap>
    inline typename std::enable_if<U, GET>::type GetValue(khiter_t i) const {
      static const GET INVALID = __cHashBaseInvalid<GET, std::is_integral<GET>::value>::INVALID();
      if(!ExistsIter(i))
        return INVALID;
      return __cHashBaseGET<Data, std::is_integral<Data>::value | std::is_pointer<Data>::value>::F(vals[i]);
    }
    template<bool U = IsMap>
    inline typename std::enable_if<U, GET>::type Get(const Key& key) const { return GetValue(Iterator(key)); }
    template<bool U = IsMap>
    inline typename std::enable_if<U, Data&>::type UnsafeValue(khiter_t i) const { return vals[i]; }
    inline bool SetValue(khiter_t iterator, const Data& newvalue) { return _setvalue<const Data&>(iterator, newvalue); }
    inline bool SetValue(khiter_t iterator, Data&& newvalue) { return _setvalue<Data&&>(iterator, std::move(newvalue)); }
    inline bool Set(const Key& key, const Data& newvalue) { return _setvalue<const Data&>(Iterator(key), newvalue); }
    inline bool Set(const Key& key, Data&& newvalue) { return _setvalue<Data&&>(Iterator(key), std::move(newvalue)); }
    inline void SetCapacity(uint32_t size) { if(n_buckets < size) _resize(size); }
    inline bool Remove(const Key& key)
    {
      khiter_t iterator = Iterator(key);
      if(n_buckets == iterator) return false; // This isn't ExistsIter because _get will return n_buckets if key doesn't exist
      _delete(iterator);
      return true;
    }
    inline bool RemoveIter(khiter_t iterator)
    {
      if(!ExistsIter(iterator)) return false;
      _delete(iterator);
      return true;
    }
    inline uint32_t Length() const { return size; }
    inline uint32_t Capacity() const { return n_buckets; }
    inline khiter_t Start() const { return 0; }
    inline khiter_t End() const { return n_buckets; }
    inline bool ExistsIter(khiter_t iterator) const { return (iterator < n_buckets) && !__ac_iseither(flags, iterator); }
    inline bool Exists(const Key& key) const { return ExistsIter(Iterator(key)); }
    template<bool U = IsMap>
    inline typename std::enable_if<U, GET>::type operator[](const Key& key) const { return Get(key); }
    inline bool operator()(const Key& key) const { return Exists(key); }
    template<bool U = IsMap>
    inline typename std::enable_if<U, bool>::type operator()(const Key& key, Data& v) const { khiter_t i = Iterator(key); if(!ExistsIter(i)) return false; v = vals[i]; return true; }

    cHashBase& operator =(const cHashBase& copy)
    {
      Clear();
      _resize(copy.n_buckets - 1); // _resize adds one to the chosen prime number so this will result in the correct assignment.
      memcpy(flags, copy.flags, ((n_buckets >> 4) + 1) * sizeof(khint32_t));

      for(khint_t i = 0; i < n_buckets; ++i)
      {
        if(!__ac_iseither(flags, i))
        {
          new(keys + i) Key(copy.keys[i]);
          if(IsMap)
            new(vals + i) Data(copy.vals[i]);
        }
      }

      assert(n_buckets == copy.n_buckets);
      size = copy.size;
      n_occupied = copy.n_occupied;
      upper_bound = copy.upper_bound;
      return *this;
    }
    cHashBase& operator =(cHashBase&& mov)
    {
      Clear();
      if(flags) Alloc::deallocate((char*)flags);
      if(keys) Alloc::deallocate((char*)keys);
      if(vals) Alloc::deallocate((char*)vals);
      memcpy(this, &mov, sizeof(cHashBase));
      memset(&mov, 0, sizeof(cHashBase));
      return *this;
    }

    class BSS_TEMPLATE_DLLEXPORT cHash_Iter : public std::iterator<std::bidirectional_iterator_tag, khiter_t>
    {
    public:
      inline explicit cHash_Iter(const cHashBase& src) : _src(&src), cur(0) { _chknext(); }
      inline cHash_Iter(const cHashBase& src, khiter_t start) : _src(&src), cur(start) { _chknext(); }
      inline khiter_t operator*() const { return cur; }
      inline cHash_Iter& operator++() { ++cur; _chknext(); return *this; } //prefix
      inline cHash_Iter operator++(int) { cHash_Iter r(*this); ++*this; return r; } //postfix
      inline cHash_Iter& operator--() { while((--cur)<_src->End() && !_src->ExistsIter(cur)); return *this; } //prefix
      inline cHash_Iter operator--(int) { cHash_Iter r(*this); --*this; return r; } //postfix
      inline bool operator==(const cHash_Iter& _Right) const { return (cur == _Right.cur); }
      inline bool operator!=(const cHash_Iter& _Right) const { return (cur != _Right.cur); }
      inline bool operator!() const { return !IsValid(); }
      inline bool IsValid() { return cur<_src->End(); }

      khiter_t cur; //khiter_t is unsigned (this is why operator--() works)

    protected:
      inline void _chknext() { while(cur<_src->End() && !_src->ExistsIter(cur)) ++cur; }

      const cHashBase* _src;
    };

    inline cHash_Iter begin() const { return cHash_Iter(*this, Start()); }
    inline cHash_Iter end() const { return cHash_Iter(*this, End()); }

  protected:
    template<typename U, typename V>
    inline khiter_t _insert(U && key, V && value)
    {
      int r;
      khiter_t i = _put<const Key&>(std::forward<U>(key), &r);
      if(!r) // If r is 0, this key was already present, so we need to assign, not initialize
        vals[i] = std::forward<V>(value);
      else
        new(vals + i) Data(std::forward<V>(value));
      return i;
    }
    template<typename U>
    inline bool _setvalue(khiter_t i, U && newvalue) { if(!ExistsIter(i)) return false; vals[i] = std::forward<U>(newvalue); return true; }
    void _resize(khint_t new_n_buckets)
    {
      khint32_t *new_flags = 0;
      khint_t j = 1;
      {
        khint_t t = __ac_HASH_PRIME_SIZE - 1;
        while(__ac_prime_list[t] > new_n_buckets) --t;
        new_n_buckets = __ac_prime_list[t + 1];
        if(size >= (khint_t)(new_n_buckets * __ac_HASH_UPPER + 0.5)) j = 0;
        else
        {
          new_flags = (khint32_t*)Alloc::allocate(((new_n_buckets >> 4) + 1) * sizeof(khint32_t), 0);
          memset(new_flags, 0xaa, ((new_n_buckets >> 4) + 1) * sizeof(khint32_t));
          if(new_n_buckets > n_buckets)
          {
            keys = __cHashBaseAlloc<Key, ArrayType, Alloc>::_realloc(flags, keys, new_n_buckets, n_buckets);
            if(IsMap)
              vals = __cHashBaseAlloc<Data, ArrayType, Alloc>::_realloc(flags, vals, new_n_buckets, n_buckets);
          }
        }
      }
      if(j)
      {
        for(j = 0; j != n_buckets; ++j)
        {
          if(__ac_iseither(flags, j) == 0)
          {
            Key key(std::move(keys[j]));
            Data val;
            if(IsMap) val = std::move(vals[j]);
            __ac_set_isdel_true(flags, j);
            while(1)
            {
              khint_t inc, k, i;
              k = __hash_func(key);
              i = k % new_n_buckets;
              inc = 1 + k % (new_n_buckets - 1);
              while(!__ac_isempty(new_flags, i))
              {
                if(i + inc >= new_n_buckets) i = i + inc - new_n_buckets;
                else i += inc;
              }
              __ac_set_isempty_false(new_flags, i);
              if(i < n_buckets && __ac_iseither(flags, i) == 0)
              {
                rswap(keys[i], key);
                if(IsMap) rswap(vals[i], val);
                __ac_set_isdel_true(flags, i);
              }
              else // this code only runs if this bucket doesn't exist, so initialize
              {
                new(keys + i) Key(std::move(key));
                if(IsMap) new(vals + i) Data(std::move(val));
                break;
              }
            }
          }
        }
        if(n_buckets > new_n_buckets)
        {
          keys = __cHashBaseAlloc<Key, ArrayType, Alloc>::_realloc(flags, keys, new_n_buckets, n_buckets);
          if(IsMap)
            vals = __cHashBaseAlloc<Data, ArrayType, Alloc>::_realloc(flags, vals, new_n_buckets, n_buckets);
        }
        Alloc::deallocate((char*)flags);
        flags = new_flags;
        n_buckets = new_n_buckets;
        n_occupied = size;
        upper_bound = (khint_t)(n_buckets * __ac_HASH_UPPER + 0.5);
      }
    }

    template<typename U>
    khint_t _put(U && key, int* ret)
    {
      khint_t x;
      if(n_occupied >= upper_bound)
      {
        if(n_buckets > (size << 1)) _resize(n_buckets - 1);
        else _resize(n_buckets + 1);
      }
      {
        khint_t inc, k, i, site, last;
        x = site = n_buckets; k = __hash_func(key); i = k % n_buckets;
        if(__ac_isempty(flags, i)) x = i;
        else
        {
          inc = 1 + k % (n_buckets - 1); last = i;
          while(!__ac_isempty(flags, i) && (__ac_isdel(flags, i) || !__hash_equal(keys[i], key)))
          {
            if(__ac_isdel(flags, i)) site = i;
            if(i + inc >= n_buckets) i = i + inc - n_buckets;
            else i += inc;
            if(i == last) { x = site; break; }
          }
          if(x == n_buckets)
          {
            if(__ac_isempty(flags, i) && site != n_buckets) x = site;
            else x = i;
          }
        }
      }
      if(__ac_isempty(flags, x))
      {
        new(keys + x) Key(std::move(key));
        __ac_set_isboth_false(flags, x);
        ++size; ++n_occupied;
        *ret = 1;
      }
      else if(__ac_isdel(flags, x))
      {
        new(keys + x) Key(std::move(key));
        __ac_set_isboth_false(flags, x);
        ++size;
        *ret = 2;
      }
      else *ret = 0;
      return x;
    }
    khint_t _get(const Key& key) const
    {
      if(n_buckets)
      {
        khint_t inc, k, i, last;
        k = __hash_func(key); i = k % n_buckets;
        inc = 1 + k % (n_buckets - 1); last = i;
        while(!__ac_isempty(flags, i) && (__ac_isdel(flags, i) || !__hash_equal(keys[i], key)))
        {
          if(i + inc >= n_buckets) i = i + inc - n_buckets;
          else i += inc;
          if(i == last) return n_buckets;
        }
        return __ac_iseither(flags, i) ? n_buckets : i;
      }
      else return 0;
    }
    inline void _delete(khint_t x)
    {
      if(x != n_buckets && !__ac_iseither(flags, x))
      {
        keys[x].~Key();
        if(IsMap)
          vals[x].~Data();
        __ac_set_isdel_true(flags, x);
        --size;
      }
    }

    khint_t n_buckets, size, n_occupied, upper_bound;
    khint32_t* flags;
    Key* keys;
    Data* vals;
  };

  template<typename T>
  inline khint_t khint_hashfunc(const T& in) { return (khint_t)in; }
  template<typename T>
  inline bool khint_equalfunc(const T& a, const T& b) { return a == b; }

  inline static khint_t KH_INT64_HASHFUNC(int64_t key) { return (khint32_t)((key) >> 33 ^ (key) ^ (key) << 11); }
  template<class T>
  BSS_FORCEINLINE khint_t KH_INT_HASHFUNC(T key) { return (khint32_t)key; }
  inline static khint_t KH_STR_HASHFUNC(const char * s) { khint_t h = *s; if(h) for(++s; *s; ++s) h = (h << 5) - h + *s; return h; }
  inline static khint_t KH_STRINS_HASHFUNC(const char *s) { khint_t h = ((*s)>64 && (*s)<91) ? (*s) + 32 : *s;	if(h) for(++s; *s; ++s) h = (h << 5) - h + (((*s)>64 && (*s)<91) ? (*s) + 32 : *s); return h; }
  inline static khint_t KH_STRW_HASHFUNC(const wchar_t * s) { khint_t h = *s; if(h) for(++s; *s; ++s) h = (h << 5) - h + *s; return h; }
  inline static khint_t KH_STRWINS_HASHFUNC(const wchar_t *s) { khint_t h = towlower(*s); if(h) for(++s; *s; ++s) h = (h << 5) - h + towlower(*s); return h; }
  template<class T>
  BSS_FORCEINLINE khint_t KH_POINTER_HASHFUNC(T p) {
#ifdef BSS_64BIT
    return KH_INT64_HASHFUNC((int64_t)p);
#else
    return (khint32_t)p;
#endif
  }
  template<typename T, int I> struct KH_AUTO_HELPER { };
  template<typename T> struct KH_AUTO_HELPER<T, 1> { BSS_FORCEINLINE static khint_t hash(T k) { return KH_POINTER_HASHFUNC<T>(k); } };
  template<typename T> struct KH_AUTO_HELPER<T, 2> { BSS_FORCEINLINE static khint_t hash(T k) { return KH_INT_HASHFUNC<T>((int32_t)k); } };
  template<typename T> struct KH_AUTO_HELPER<T, 4> { BSS_FORCEINLINE static khint_t hash(T k) { return KH_INT64_HASHFUNC((int64_t)k); } };

  template<typename T>
  static BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC(const T& k) { return KH_AUTO_HELPER<T, std::is_pointer<T>::value + (std::is_integral<T>::value * 2 * (1 + (sizeof(T) == 8)))>::hash(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<cStr>(const cStr& k) { return KH_STR_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<const char*>(const char* const& k) { return KH_STR_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<char*>(char* const& k) { return KH_STR_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<const wchar_t*>(const wchar_t* const& k) { return KH_STRW_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<wchar_t*>(wchar_t* const& k) { return KH_STRW_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<double>(const double& k) { return KH_INT64_HASHFUNC(*(int64_t*)&k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<float>(const float& k) { return *(khint32_t*)&k; }
  template<typename T> static BSS_FORCEINLINE khint_t KH_AUTOINS_HASHFUNC(const T& k) { return KH_STRINS_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTOINS_HASHFUNC<cStr>(const cStr& k) { return KH_STRINS_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTOINS_HASHFUNC<const wchar_t*>(const wchar_t* const& k) { return KH_STRWINS_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTOINS_HASHFUNC<wchar_t*>(wchar_t* const& k) { return KH_STRWINS_HASHFUNC(k); }

  template<typename U, typename V> struct KH_AUTO_HELPER<std::pair<U, V>, 0> { BSS_FORCEINLINE static khint_t hash(std::pair<U, V> k) { return KH_INT64_HASHFUNC(uint64_t(KH_AUTO_HASHFUNC<U>(k.first)) | (uint64_t(KH_AUTO_HASHFUNC<V>(k.second)) << 32)); } };

  template<typename T>
  static BSS_FORCEINLINE bool KH_AUTO_EQUALFUNC(T const& a, T const& b) { return a == b; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTO_EQUALFUNC<cStr>(cStr const& a, cStr const& b) { return strcmp(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTO_EQUALFUNC<const char*>(const char* const& a, const char* const& b) { return strcmp(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTO_EQUALFUNC<const wchar_t*>(const wchar_t* const& a, const wchar_t* const& b) { return wcscmp(a, b) == 0; }
  template<typename T> static BSS_FORCEINLINE bool KH_AUTOINS_EQUALFUNC(T const& a, T const& b) { return STRICMP(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTOINS_EQUALFUNC<cStr>(cStr const& a, cStr const& b) { return STRICMP(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTOINS_EQUALFUNC<const wchar_t*>(const wchar_t* const& a, const wchar_t* const& b) { return WCSICMP(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTOINS_EQUALFUNC<wchar_t*>(wchar_t* const& a, wchar_t* const& b) { return WCSICMP(a, b) == 0; }

#ifdef BSS_PLATFORM_WIN32
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<cStrW>(const cStrW& k) { return KH_STRW_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTOINS_HASHFUNC<cStrW>(const cStrW& k) { return KH_STRWINS_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTO_EQUALFUNC<cStrW>(cStrW const& a, cStrW const& b) { return wcscmp(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTOINS_EQUALFUNC<cStrW>(cStrW const& a, cStrW const& b) { return WCSICMP(a, b) == 0; }
#endif

  template<typename T, bool I> struct __cKh_KHGET { typedef typename std::remove_pointer<T>::type* KHGET; static const uint32_t INV = 0; };
  template<typename T> struct __cKh_KHGET<T, true> { typedef T KHGET; static const KHGET INV = (KHGET)-1; };
  template<typename T> struct KHGET_cKh : __cKh_KHGET<T, std::is_integral<T>::value> { typedef typename __cKh_KHGET<T, std::is_integral<T>::value>::KHGET KHGET; };
  template<> struct KHGET_cKh<void> { typedef char KHGET; static const KHGET INV = (KHGET)-1; };
  template<typename T, bool ins> struct CHASH_HELPER {
    BSS_FORCEINLINE static khint_t hash(const T& k) { return KH_AUTO_HASHFUNC<T>(k); }
    BSS_FORCEINLINE static bool equal(const T& a, const T& b) { return KH_AUTO_EQUALFUNC<T>(a, b); }
  };
  template<typename T> struct CHASH_HELPER<T, true> {
    BSS_FORCEINLINE static khint_t hash(const T& k) { return KH_AUTOINS_HASHFUNC<T>(k); }
    BSS_FORCEINLINE static bool equal(const T& a, const T& b) { return KH_AUTOINS_EQUALFUNC<T>(a, b); }
  };

  // Generic hash definition
  template<typename K, typename T = void, bool ins = false, ARRAY_TYPE ArrayType = CARRAY_SIMPLE, typename Alloc = StaticAllocPolicy<char>>
  class BSS_COMPILER_DLLEXPORT cHash : public cHashBase<K, typename std::conditional<std::is_void<T>::value, char, T>::type, !std::is_void<T>::value, &CHASH_HELPER<K, ins>::hash, &CHASH_HELPER<K, ins>::equal, ArrayType, Alloc>
  {
    typedef cHashBase<K, typename std::conditional<std::is_void<T>::value, char, T>::type, !std::is_void<T>::value, &CHASH_HELPER<K, ins>::hash, &CHASH_HELPER<K, ins>::equal, ArrayType, Alloc> BASE;

  public:
    inline cHash(const cHash& copy) : BASE(copy) {}
    inline cHash(cHash&& mov) : BASE(std::move(mov)) {}
    inline cHash(uint32_t size = 0) : BASE(size) {}

    inline cHash& operator =(const cHash& right) { BASE::operator=(right); return *this; }
    inline cHash& operator =(cHash&& mov) { BASE::operator=(std::move(mov)); return *this; }
    inline bool operator()(const K& key) const { return BASE::operator()(key); }
    template<bool U = !std::is_void<T>::value>
    inline typename std::enable_if<U, bool>::type operator()(const K& key, typename BASE::DATA& v) const { return BASE::operator()(key, v); }
    template<bool U = !std::is_void<T>::value>
    inline typename std::enable_if<U, typename BASE::GET>::type operator[](const K& key) const { return BASE::operator[](key); }
  };
}

#endif
