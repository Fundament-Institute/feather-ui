// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_HASH_H__BSS__
#define __C_HASH_H__BSS__

#include "khash.h"
#include "bss_defines.h"
#include <wchar.h>
#include <iterator>

namespace bss_util {
  template<typename T>
  inline khint_t khint_hashfunc(T in) { return (khint_t)in; }
  template<typename T>
  inline bool khint_equalfunc(T a, T b) { return a == b; }

  template<typename khkey_t, typename khval_t, bool kh_is_map, khint_t(*__hash_func)(khkey_t), bool(*__hash_equal)(khkey_t, khkey_t)>
  class BSS_TEMPLATE_DLLEXPORT kh_template_t
  {
  public:
    khint_t n_buckets, size, n_occupied, upper_bound;
    khint32_t *flags;
    khkey_t *keys;
    khval_t *vals;

    kh_template_t(const kh_template_t& copy) { memset(this, 0, sizeof(kh_template_t)); operator=(copy); }
    kh_template_t(kh_template_t&& mov)
    {
      memcpy(this, &mov, sizeof(kh_template_t));
      memset(&mov, 0, sizeof(kh_template_t));
    }
    kh_template_t() { memset(this, 0, sizeof(kh_template_t)); }
    ~kh_template_t()
    {
      if(keys) free(keys);
      if(flags) free(flags);
      if(vals) free(vals);
    }
    kh_template_t& operator =(const kh_template_t& copy)
    {
      kh_clear_template();
      kh_resize_template(copy.n_buckets<4 ? copy.n_buckets : (copy.n_buckets - 1));
      memcpy(flags, copy.flags, ((copy.n_buckets >> 4) + 1) * sizeof(khint32_t));
      memcpy(keys, copy.keys, copy.n_buckets * sizeof(khkey_t));
      if(copy.vals) memcpy(vals, copy.vals, copy.n_buckets * sizeof(khval_t));
      else vals = 0;
      return *this;
    }
    kh_template_t& operator =(kh_template_t&& mov)
    {
      this->~kh_template_t();
      memcpy(this, &mov, sizeof(kh_template_t));
      memset(&mov, 0, sizeof(kh_template_t));
      return *this;
    }


    inline void kh_clear_template()
    {
      if(flags) {
        memset(flags, 0xaa, ((n_buckets >> 4) + 1) * sizeof(khint32_t));
        size = n_occupied = 0;
      }
    }
    static inline kh_template_t* kh_init_template() {
      return (kh_template_t*)calloc(1, sizeof(kh_template_t));
    }

    static inline void kh_destroy_template(kh_template_t* h)
    {
      if(h) {
        free(h->keys); free(h->flags);
        if(h->vals) free(h->vals);
        free(h);
      }
    }
    khint_t kh_get_template(khkey_t key) const
    {
      if(n_buckets) {
        khint_t inc, k, i, last;
        k = __hash_func(key); i = k % n_buckets;
        inc = 1 + k % (n_buckets - 1); last = i;
        while(!__ac_isempty(flags, i) && (__ac_isdel(flags, i) || !__hash_equal(keys[i], key))) {
          if(i + inc >= n_buckets) i = i + inc - n_buckets;
          else i += inc;
          if(i == last) return n_buckets;
        }
        return __ac_iseither(flags, i) ? n_buckets : i;
      }
      else return 0;
    }
    void kh_resize_template(khint_t new_n_buckets)
    {
      khint32_t *new_flags = 0;
      khint_t j = 1;
      {
        khint_t t = __ac_HASH_PRIME_SIZE - 1;
        while(__ac_prime_list[t] > new_n_buckets) --t;
        new_n_buckets = __ac_prime_list[t + 1];
        if(size >= (khint_t)(new_n_buckets * __ac_HASH_UPPER + 0.5)) j = 0;
        else {
          new_flags = (khint32_t*)malloc(((new_n_buckets >> 4) + 1) * sizeof(khint32_t));
          memset(new_flags, 0xaa, ((new_n_buckets >> 4) + 1) * sizeof(khint32_t));
          if(n_buckets < new_n_buckets) {
            keys = (khkey_t*)realloc(keys, new_n_buckets * sizeof(khkey_t));
            if(kh_is_map)
              vals = (khval_t*)realloc(vals, new_n_buckets * sizeof(khval_t));
          }
        }
      }
      if(j) {
        for(j = 0; j != n_buckets; ++j) {
          if(__ac_iseither(flags, j) == 0) {
            khkey_t key = keys[j];
            khval_t val;
            if(kh_is_map) val = vals[j];
            __ac_set_isdel_true(flags, j);
            while(1) {
              khint_t inc, k, i;
              k = __hash_func(key);
              i = k % new_n_buckets;
              inc = 1 + k % (new_n_buckets - 1);
              while(!__ac_isempty(new_flags, i)) {
                if(i + inc >= new_n_buckets) i = i + inc - new_n_buckets;
                else i += inc;
              }
              __ac_set_isempty_false(new_flags, i);
              if(i < n_buckets && __ac_iseither(flags, i) == 0) {
                { khkey_t tmp = keys[i]; keys[i] = key; key = tmp; }
                if(kh_is_map) { khval_t tmp = vals[i]; vals[i] = val; val = tmp; }
                __ac_set_isdel_true(flags, i);
              }
              else {
                keys[i] = key;
                if(kh_is_map) vals[i] = val;
                break;
              }
            }
          }
        }
        if(n_buckets > new_n_buckets) {
          keys = (khkey_t*)realloc(keys, new_n_buckets * sizeof(khkey_t));
          if(kh_is_map)
            vals = (khval_t*)realloc(vals, new_n_buckets * sizeof(khval_t));
        }
        free(flags);
        flags = new_flags;
        n_buckets = new_n_buckets;
        n_occupied = size;
        upper_bound = (khint_t)(n_buckets * __ac_HASH_UPPER + 0.5);
      }
    }
    khint_t kh_put_template(khkey_t key, int *ret)
    {
      khint_t x;
      if(n_occupied >= upper_bound) {
        if(n_buckets > (size << 1)) kh_resize_template(n_buckets - 1);
        else kh_resize_template(n_buckets + 1);
      }
      {
        khint_t inc, k, i, site, last;
        x = site = n_buckets; k = __hash_func(key); i = k % n_buckets;
        if(__ac_isempty(flags, i)) x = i;
        else {
          inc = 1 + k % (n_buckets - 1); last = i;
          while(!__ac_isempty(flags, i) && (__ac_isdel(flags, i) || !__hash_equal(keys[i], key))) {
            if(__ac_isdel(flags, i)) site = i;
            if(i + inc >= n_buckets) i = i + inc - n_buckets;
            else i += inc;
            if(i == last) { x = site; break; }
          }
          if(x == n_buckets) {
            if(__ac_isempty(flags, i) && site != n_buckets) x = site;
            else x = i;
          }
        }
      }
      if(__ac_isempty(flags, x)) {
        keys[x] = key;
        __ac_set_isboth_false(flags, x);
        ++size; ++n_occupied;
        *ret = 1;
      }
      else if(__ac_isdel(flags, x)) {
        keys[x] = key;
        __ac_set_isboth_false(flags, x);
        ++size;
        *ret = 2;
      }
      else *ret = 0;
      return x;
    }
    inline void kh_del_template(khint_t x)
    {
      if(x != n_buckets && !__ac_iseither(flags, x)) {
        __ac_set_isdel_true(flags, x);
        --size;
      }
    }
  };

  template<typename khkey_t, typename khval_t, bool kh_is_map, khint_t(*__hash_func)(khkey_t), bool(*__hash_equal)(khkey_t, khkey_t)>
  class BSS_TEMPLATE_DLLEXPORT kh_insert_template_t : protected kh_template_t<khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal>
  {
    template<bool CHK, typename KHGET, typename KHTYPE> friend struct __getval_KHASH;

  public:
    inline void Insert(khkey_t key, const khval_t& value) { _insert<const khval_t&>(key, value); }
    inline void Insert(khkey_t key, khval_t&& value) { _insert<khval_t&&>(key, std::move(value)); }

  protected:
    template<typename U>
    inline void _insert(khkey_t key, U && value)
    {
      //if(kh_size(this) >= kh_end(this)) _resize(); // Not needed, kh_put_template resizes as necessary
      int r;
      khiter_t retval = kh_template_t<khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal>::kh_put_template(key, &r);
      kh_val(this, retval) = std::forward<U>(value); // we can always assume this is a map because otherwise we'd use the specialization below
    }
  };

  template<typename khkey_t, typename khval_t, khint_t(*__hash_func)(khkey_t), bool(*__hash_equal)(khkey_t, khkey_t)>
  class BSS_TEMPLATE_DLLEXPORT kh_insert_template_t<khkey_t, khval_t, false, __hash_func, __hash_equal> : protected kh_template_t<khkey_t, khval_t, false, __hash_func, __hash_equal>
  {
    template<bool CHK, typename KHGET, typename KHTYPE> friend struct __getval_KHASH;

  public:
    inline void Insert(khkey_t key) { int r; kh_template_t<khkey_t, khval_t, false, __hash_func, __hash_equal>::kh_put_template(key, &r); }
  };

  template<bool CHK, typename KHGET, typename KHTYPE> // GCC doesn't like it when you try to embed a templatized function inside a class
  struct __getval_KHASH { static inline KHGET BSS_FASTCALL _getval(const KHTYPE* _h, khiter_t i) { return kh_val(_h, i); } };
  template<typename KHGET, typename KHTYPE>
  struct __getval_KHASH<true, KHGET, KHTYPE> { static inline KHGET BSS_FASTCALL _getval(const KHTYPE* _h, khiter_t i) { return &kh_val(_h, i); } };

  // Base template for cKhash. If you want a set instead of a map, set khval_t to 'char' and kh_is_map to 'false'.
  template<typename khkey_t, typename khval_t, bool kh_is_map, khint_t(*__hash_func)(khkey_t), bool(*__hash_equal)(khkey_t, khkey_t), typename khget_t = khval_t*, khget_t INVALID = 0>
  class BSS_TEMPLATE_DLLEXPORT cKhash : public kh_insert_template_t<khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal>
  {
  protected:
    typedef khkey_t KHKEY;
    typedef khval_t KHVAL;
    typedef kh_insert_template_t<KHKEY, KHVAL, kh_is_map, __hash_func, __hash_equal> KHTYPE;
    typedef khget_t KHGET;

  public:
    inline cKhash(const cKhash& copy) : KHTYPE(copy) { }
    inline cKhash(cKhash&& mov) : KHTYPE(std::move(mov)) { }
    inline cKhash(unsigned int size = 0) { KHTYPE::kh_resize_template(size); };
    inline ~cKhash() { }
    inline khiter_t Iterator(KHKEY key) const { return KHTYPE::kh_get_template(key); }
    //inline KHKEY GetIterKey(khiter_t iterator) { return kh_key(iterator); }
    inline KHKEY GetKey(khiter_t iterator) { return kh_key(this, iterator); }
    inline bool SetKey(khiter_t iterator, KHKEY key) { if(kh_end(this) == iterator) return false; kh_key(this, iterator) = key; return true; }
    inline KHGET Get(KHKEY key) const { return GetValue(Iterator(key)); }
    inline KHGET GetValue(khiter_t iter) const { if(!ExistsIter(iter)) return INVALID; return __getval_KHASH<std::is_same<KHGET, khval_t*>::value, KHGET, KHTYPE>::_getval(this, iter); }
    inline const KHVAL& UnsafeValue(khiter_t iterator) const { return kh_val(this, iterator); }
    inline bool SetValue(khiter_t iterator, const KHVAL& newvalue) { return _setvalue<const KHVAL&>(iterator, newvalue); }
    inline bool SetValue(khiter_t iterator, KHVAL&& newvalue) { return _setvalue<KHVAL&&>(iterator, std::move(newvalue)); }
    inline bool Set(KHKEY key, const KHVAL& newvalue) { return _setvalue<const KHVAL&>(Iterator(key), newvalue); }
    inline bool Set(KHKEY key, KHVAL&& newvalue) { return _setvalue<KHVAL&&>(Iterator(key), std::move(newvalue)); }
    inline void SetCapacity(unsigned int size) { if(kh_end(this) < size) KHTYPE::kh_resize_template(size); }
    inline bool Remove(KHKEY key)
    {
      khiter_t iterator = Iterator(key);
      if(kh_end(this) == iterator) return false; // This isn't ExistsIter because kh_get_template will return kh_end(this) if key doesn't exist
      KHTYPE::kh_del_template(iterator);
      return true;
    }
    inline bool RemoveIter(khiter_t iterator)
    {
      if(!ExistsIter(iterator)) return false;
      KHTYPE::kh_del_template(iterator);
      return true;
    }
    inline void Clear() { KHTYPE::kh_clear_template(); }
    inline unsigned int Length() const { return kh_size(this); }
    inline unsigned int Capacity() const { return kh_end(this); }
    inline khiter_t Start() const { return kh_begin(this); }
    inline khiter_t End() const { return kh_end(this); }
    inline bool ExistsIter(khiter_t iterator) const { return iterator<kh_end(this) && kh_exist(this, iterator) != 0; }
    inline bool Exists(KHKEY key) const { return ExistsIter(Iterator(key)); }
    inline cKhash& operator =(const cKhash& right)
    {
      if(&right == this) return *this;
      KHTYPE::operator=(right);
      return *this;
    }
    inline cKhash& operator =(cKhash&& mov)
    {
      if(&mov == this) return *this;
      KHTYPE::operator=(std::move(mov));
      return *this;
    }
    inline KHGET operator[](KHKEY key) const { return Get(key); }
    //inline KHVAL& operator[](khiter_t iterator) { return kh_val(iterator); }
    //inline const KHVAL& operator[](khiter_t iterator) const { return kh_val(iterator); }

    class BSS_TEMPLATE_DLLEXPORT cKhash_Iter : public std::iterator<std::bidirectional_iterator_tag, khiter_t>
    {
    public:
      inline explicit cKhash_Iter(const cKhash& src) : _src(&src), cur(0) { _chknext(); }
      inline cKhash_Iter(const cKhash& src, khiter_t start) : _src(&src), cur(start) { _chknext(); }
      inline khiter_t operator*() const { return cur; }
      inline cKhash_Iter& operator++() { ++cur; _chknext(); return *this; } //prefix
      inline cKhash_Iter operator++(int) { cKhash_Iter r(*this); ++*this; return r; } //postfix
      inline cKhash_Iter& operator--() { while((--cur)<_src->End() && !_src->ExistsIter(cur)); return *this; } //prefix
      inline cKhash_Iter operator--(int) { cKhash_Iter r(*this); --*this; return r; } //postfix
      inline bool operator==(const cKhash_Iter& _Right) const { return (cur == _Right.cur); }
      inline bool operator!=(const cKhash_Iter& _Right) const { return (cur != _Right.cur); }
      inline bool operator!() const { return !IsValid(); }
      inline bool IsValid() { return cur<_src->End(); }

      khiter_t cur; //khiter_t is unsigned (this is why operator--() works)

    protected:
      inline void _chknext() { while(cur<_src->End() && !_src->ExistsIter(cur)) ++cur; }

      const cKhash* _src;
    };

    inline cKhash_Iter begin() const { return cKhash_Iter(*this, Start()); }
    inline cKhash_Iter end() const { return cKhash_Iter(*this, End()); }

  protected:
    template<typename U>
    inline bool _setvalue(khiter_t i, U && newvalue) { if(i >= kh_end(this)) return false; kh_val(this, i) = std::forward<U>(newvalue); return true; }
  };

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
  static BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC(T k) { return KH_AUTO_HELPER<T, std::is_pointer<T>::value + (std::is_integral<T>::value * 2 * (1 + (sizeof(T) == 8)))>::hash(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<const char*>(const char* k) { return KH_STR_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<char*>(char* k) { return KH_STR_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<const wchar_t*>(const wchar_t* k) { return KH_STRW_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<wchar_t*>(wchar_t* k) { return KH_STRW_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<double>(double k) { return KH_INT64_HASHFUNC(*(int64_t*)&k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTO_HASHFUNC<float>(float k) { return *(khint32_t*)&k; }
  template<typename T> static BSS_FORCEINLINE khint_t KH_AUTOINS_HASHFUNC(T k) { return KH_STRINS_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTOINS_HASHFUNC<const wchar_t*>(const wchar_t* k) { return KH_STRWINS_HASHFUNC(k); }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE khint_t KH_AUTOINS_HASHFUNC<wchar_t*>(wchar_t* k) { return KH_STRWINS_HASHFUNC(k); }

  template<typename U, typename V> struct KH_AUTO_HELPER<std::pair<U, V>, 0> { BSS_FORCEINLINE static khint_t hash(std::pair<U, V> k) { return KH_INT64_HASHFUNC(uint64_t(KH_AUTO_HASHFUNC<U>(k.first)) | (uint64_t(KH_AUTO_HASHFUNC<V>(k.second))<<32)); } };

  template<typename T>
  static BSS_FORCEINLINE bool KH_AUTO_EQUALFUNC(T a, T b) { return a == b; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTO_EQUALFUNC<const char*>(const char* a, const char* b) { return strcmp(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTO_EQUALFUNC<const wchar_t*>(const wchar_t* a, const wchar_t* b) { return wcscmp(a, b) == 0; }
  template<typename T> static BSS_FORCEINLINE bool KH_AUTOINS_EQUALFUNC(T a, T b) { return STRICMP(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTOINS_EQUALFUNC<const wchar_t*>(const wchar_t* a, const wchar_t* b) { return WCSICMP(a, b) == 0; }
  template<> BSS_EXPLICITSTATIC BSS_FORCEINLINE bool KH_AUTOINS_EQUALFUNC<wchar_t*>(wchar_t* a, wchar_t* b) { return WCSICMP(a, b) == 0; }

  template<typename T, bool I> struct __cKh_KHGET { typedef typename std::remove_pointer<T>::type* KHGET; static const unsigned int INV = 0; };
  template<typename T> struct __cKh_KHGET<T, true> { typedef T KHGET; static const KHGET INV = (KHGET)-1; };
  template<typename T> struct KHGET_cKh : __cKh_KHGET<T, std::is_integral<T>::value> { typedef typename __cKh_KHGET<T, std::is_integral<T>::value>::KHGET KHGET; };
  template<> struct KHGET_cKh<void> { typedef char KHGET; static const KHGET INV = (KHGET)-1; };
  template<typename T, bool ins> struct CHASH_HELPER {
    BSS_FORCEINLINE static khint_t hash(T k) { return KH_AUTO_HASHFUNC<T>(k); }
    BSS_FORCEINLINE static bool equal(T a, T b) { return KH_AUTO_EQUALFUNC<T>(a, b); }
  };
  template<typename T> struct CHASH_HELPER<T, true> {
    BSS_FORCEINLINE static khint_t hash(T k) { return KH_AUTOINS_HASHFUNC<T>(k); }
    BSS_FORCEINLINE static bool equal(T a, T b) { return KH_AUTOINS_EQUALFUNC<T>(a, b); }
  };

  // Generic hash definition
  template<typename K, typename T = void, bool ins = false>
  class BSS_COMPILER_DLLEXPORT cHash : public cKhash<K, typename std::conditional<std::is_void<T>::value, char, T>::type, !std::is_void<T>::value, &CHASH_HELPER<K, ins>::hash, &CHASH_HELPER<K, ins>::equal, typename KHGET_cKh<T>::KHGET, (typename KHGET_cKh<T>::KHGET)KHGET_cKh<T>::INV>
  {
    typedef cKhash<K, typename std::conditional<std::is_void<T>::value, char, T>::type, !std::is_void<T>::value, &CHASH_HELPER<K, ins>::hash, &CHASH_HELPER<K, ins>::equal, typename KHGET_cKh<T>::KHGET, (typename KHGET_cKh<T>::KHGET)KHGET_cKh<T>::INV> BASE;

  public:
    inline cHash(const cHash& copy) : BASE(copy) {}
    inline cHash(cHash&& mov) : BASE(std::move(mov)) {}
    inline cHash(unsigned int size = 0) : BASE(size) {}

    inline cHash& operator =(const cHash& right) { BASE::operator=(right); return *this; }
    inline cHash& operator =(cHash&& mov) { BASE::operator=(std::move(mov)); return *this; }
    inline typename BASE::KHGET operator[](typename BASE::KHKEY key) const { return BASE::operator[](key); }
  };

  template<typename _Ty>
  struct cDict
  {
    typedef cHash<const char*, _Ty, false> T;
    typedef cHash<const wchar_t*, _Ty, false> Tw;
    typedef cHash<const char*, _Ty, true> INS;
    typedef cHash<const wchar_t*, _Ty, true> INSw;
  };
}

#endif
