// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__UTIL_H
#define FG__UTIL_H

#include "feather/feather.h"
#include <unordered_map>
#include <algorithm>
#include <initializer_list>

inline fgVec& operator+=(fgVec& l, const fgVec& r) { l.x += r.x; l.y += r.y; return l; }
inline fgVec& operator+=(fgVec& l, float f) { l.x += f; l.y += f; return l; }
inline fgVec operator+(const fgVec& l, const fgVec& r) { return fgVec{ l.x + r.x, l.y + r.y }; }
inline fgVec operator+(float f, const fgVec& r) { return fgVec{ f + r.x, f + r.y }; }
inline fgVec operator+(const fgVec& l, float f) { return fgVec{ l.x + f, l.y + f }; }

inline fgVec& operator-=(fgVec& l, const fgVec& r) { l.x -= r.x; l.y -= r.y; return l; }
inline fgVec& operator-=(fgVec& l, float f) { l.x -= f; l.y -= f; return l; }
inline fgVec operator-(const fgVec& l, const fgVec& r) { return fgVec{ l.x - r.x, l.y - r.y }; }
inline fgVec operator-(float f, const fgVec& r) { return fgVec{ f - r.x, f - r.y }; }
inline fgVec operator-(const fgVec& l, float f) { return fgVec{ l.x - f, l.y - f }; }
inline fgVec operator-(const fgVec& l) { return fgVec{ -l.x, -l.y }; }

inline fgRect& operator+=(fgRect& l, const fgVec& r) { l.left += r.x; l.top += r.y; l.right += r.x; l.bottom += r.y; return l; }
inline fgRect& operator+=(fgRect& l, const fgRect& r) { return l += r.topleft; }
inline fgRect operator+(const fgRect& l, const fgVec& r) { return fgRect{ l.left + r.x, l.top + r.y, l.right + r.x, l.bottom + r.y }; }
inline fgRect operator+(const fgRect& l, const fgRect& r) { return l + r.topleft; }

inline fgRect& operator-=(fgRect& l, const fgVec& r) { l.left -= r.x; l.top -= r.y; l.right -= r.x; l.bottom -= r.y; return l; }
inline fgRect& operator-=(fgRect& l, const fgRect& r) { return l -= r.topleft; }
inline fgRect operator-(const fgRect& l, const fgVec& r) { return fgRect{ l.left - r.x, l.top - r.y, l.right - r.x, l.bottom - r.y }; }
inline fgRect operator-(const fgRect& l, const fgRect& r) { return l - r.topleft; }
inline fgRect operator-(const fgRect& l) { return fgRect{ -l.left, -l.top, -l.right, -l.bottom }; }

// Inserts a linked list node before the root and re-assigns it
template<typename T>
inline void LLAdd(T* node, T*& root) noexcept
{
  node->prev = 0;
  node->next = root;
  if(root) root->prev = node;
  root = node;
}

// Remove a doubly-linked list node
template<typename T>
inline void LLRemove(T* node, T*& root) noexcept
{
  if(node->prev != 0) node->prev->next = node->next;
  else if(root == node) root = node->next;
  if(node->next != 0) node->next->prev = node->prev;
}

/*template<class T>
struct ResolverInfo
{
  FG_EVENTS e;
  FG_CALC_NODES type;
  T* t;
};

unsigned int fgGenericResolverInner(unsigned int index, fgCalcNode* out)
{
  return ~0;
}

template<class Arg, class... Args>
unsigned int fgGenericResolverInner(unsigned int index, fgCalcNode* out, const ResolverInfo<Arg>& arg, const ResolverInfo<Args>&... args)
{
  if(index == arg.e)
  {
    if(out->type == FG_CALC_NONE)
    {
      switch(arg.type)
      {
      case FG_CALC_STRING: out->value.s = (const char*)*arg.t;
      case FG_CALC_INT: out->value.i = (int)*arg.t;
      case FG_CALC_FLOAT: out->value.d = (double)*arg.t;
      }
      return out->type = arg.type;
    }
    if(out->type != arg.type)
      break;
    switch(arg.type)
    {
    case FG_CALC_STRING: *arg.t = (T)&out->value.s;
    case FG_CALC_INT: *arg.t = (T)&out->value.i;
    case FG_CALC_FLOAT: *arg.t = (T)&out->value.d;
    }
    data->title = out->value.s;
    return i->type;
  }
  return fgGenericResolverInner<Args...>(args...);
}

template<int N, class T, class... Args>
unsigned int fgGenericResolver(std::unordered_map<std::string, unsigned int>& map, unsigned int (&fields)[N], void* ptr, unsigned int index, fgCalcNode* out, const char* id, const ResolverInfo<Args>&... args)
{
  if(id)
  {
    std::string raw(id);
    std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);

    if(auto i = map.find(raw.c_str()); i != map.end())
      return i->second;
    return ~0;
  }
  if(!ptr)
  {
    out->value.p = fields;
    return N;
  }

  auto data = reinterpret_cast<T*>(ptr);
  return fgGenericResolverInner(index, out, args...)
}*/

#define RESOLVER_CASE(out, field, e, t, d) \
  case e##: \
    if(out->type == FG_CALC_NONE) \
    { \
      out->value.##field = d; \
      return out->type = t; \
    } \
    if(out->type != t) \
      break; \
    d = out->value.##field; \
    return t;

#endif
