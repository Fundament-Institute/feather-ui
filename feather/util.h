// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__UTIL_H
#define FG__UTIL_H

#include "feather/feather.h"

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

#endif
