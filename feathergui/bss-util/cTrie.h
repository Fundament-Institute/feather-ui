// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "bss_util.h"

#ifndef __C_TRIE_H__
#define __C_TRIE_H__

#include "cBinaryHeap.h"
#include "bss_algo.h"
#include <stdarg.h>

namespace bss_util {
  // Trie node
  template<typename T=uint8_t>
  struct BSS_COMPILER_DLLEXPORT TRIE_NODE__
  {
    char chr; // letter that this node has
    T child; // Pointer to this node's child, if it has one
    T clen; // Number of siblings this node has
    T word; // This stores the original index of the word that this node corresponds to, but only if chr is nullptr (indicating the end of a word)
  };

  // A static trie optimized for looking up small collections of words.
  template<typename T=uint8_t>
  class BSS_COMPILER_DLLEXPORT cTrie : protected cArrayBase<TRIE_NODE__<T>, T>
  {
    typedef cArrayBase<TRIE_NODE__<T>, T> BASE;
    typedef TRIE_NODE__<T> TNODE;
    using BASE::_array;
    using BASE::_capacity;
    typedef std::pair<T, const char*> PAIR;

  public:
    inline cTrie(cTrie&& mov) : BASE(std::move(mov)) {}
    inline cTrie(const cTrie& copy) : BASE(copy) {}
    inline cTrie(T num, ...) : BASE(num)
    {
      _fill(0, num);
      DYNARRAY(PAIR, s, num);
      va_list vl;
      va_start(vl, num);
      for(T i = 0; i < num; ++i) { s[i].first=i; s[i].second=va_arg(vl, const char*); }
      va_end(vl);
      cBinaryHeap<PAIR, T, CompTSecond<PAIR, CompStr<const char*>>>::HeapSort(s, num); // sort into alphabetical order
      _init(num, s, 0, 0); // Put into our recursive initializer
    }
    inline cTrie(T num, const char* const* initstr) : BASE(num) { _construct(num, initstr); }
    template<int SZ>
    inline cTrie(const char* const (&initstr)[SZ]) : BASE(SZ) { _construct(SZ, initstr); }
    inline ~cTrie() {}
    T BSS_FASTCALL Get(const char* word) const
    {
      assert(word!=0);
      TNODE* cur=_array; // root is always 0
      T r=0;
      char c;
      while((c=*(word++)))
      {
        if(cur->clen>1) // This is faster than a switch statement
          r=binsearch_exact<TNODE, char, T, &cTrie::_CompTNode>(cur, c, 0, cur->clen);
        else if(cur->clen==1)
          r=(T)-(cur->chr!=c);
        else
          return (T)-1;
        if(r==(T)-1) return (T)-1;
        cur=_array+cur[r].child;
      }
      return cur->word;
    }
    T BSS_FASTCALL Get(const char* word, T len) const
    {
      assert(word!=0);
      TNODE* cur=_array; // root is always 0
      T r=0;
      char c;
      while((len--)>0)
      {
        c=*(word++);
        if(cur->clen>1) // This is faster than a switch statement
          r=binsearch_exact<TNODE, char, T, &cTrie::_CompTNode>(cur, c, 0, cur->clen);
        else if(cur->clen==1)
          r=(T)-(cur->chr!=c);
        else
          return (T)-1;
        if(r==(T)-1) return (T)-1;
        cur=_array+cur[r].child;
      }
      return cur->word;
    }
    inline const TNODE* Internal() const { return _array; }
    inline T operator[](const char* word) const { return Get(word); }
    inline cTrie& operator=(const cTrie& copy) { BASE::operator=(copy); return *this; }
    inline cTrie& operator=(cTrie&& mov) { BASE::operator=(std::move(mov)); return *this; }
    static inline char _CompTNode(const TNODE& t, const char& c) { return SGNCOMPARE(t.chr, c); }

  protected:
    void BSS_FASTCALL _construct(T num, const char* const* initstr)
    {
      _fill(0, num);
      DYNARRAY(PAIR, s, num);
      for(T i = 0; i < num; ++i) { s[i].first=i; s[i].second=initstr[i]; }
      cBinaryHeap<PAIR, T, CompTSecond<PAIR, CompStr<const char*>>>::HeapSort(s, num); // sort into alphabetical order
      _init(num, s, 0, 0); // Put into our recursive initializer
    }
    BSS_FORCEINLINE void BSS_FASTCALL _fill(T s, T e) // Zeros out a range of nodes
    {
      for(T i = s; i < e; ++i)
      {
        _array[i].word=(T)-1;
        _array[i].child=(T)-1;
        _array[i].clen=0;
        _array[i].chr=0;
      }
    }
    BSS_FORCEINLINE void BSS_FASTCALL _checksize(T r) { assert(r<(std::numeric_limits<T>::max()-2)); if(r>=_capacity) { T s=_capacity; BASE::SetCapacity(_capacity<<1); _fill(s, _capacity); } }
    T BSS_FASTCALL _init(T len, PAIR const* str, T cnt, T level)
    {
      T r=cnt-1;
      char c=str[0].second[level];
      if(!c) { _checksize(r+1); _array[r+1].word=str[0].first; ++str; --len; } // The only place we'll recieve the end of the word is in our starting position due to alphabetical order

      char l=0;
      for(T i = 0; i < len; ++i) //first pass so we can assemble top level nodes here
      {
        c=str[i].second[level];
        if(l!=c) { _checksize(++r); _array[r].chr=(l=c); }
        assert(_array[r].clen<(std::numeric_limits<T>::max()-2));
        ++_array[r].clen;
      }
      len=(++r)-cnt;
      ++level;
      T last=r;
      for(T i = cnt; i < last; ++i) // Second pass that generates children
      {
        _array[i].child=r;
        r=_init(_array[i].clen, str, r, level);
        str+=_array[i].clen;
      }
      _array[cnt].clen=len;
      return !len?1+r:r;
    }
  };
}

#endif