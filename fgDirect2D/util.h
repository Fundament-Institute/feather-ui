// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__UTIL_H
#define D2D__UTIL_H

#include "win32_includes.h"
#include "feather.h"
#include <d2d1_1.h>

inline static D2D1_COLOR_F ToD2Color(unsigned int color)
{
  fgColor c = { color };
  return D2D1::ColorF((c.r << 16) | (c.g << 8) | c.b, c.a / 255.0f);
}

#ifdef FG_DEBUG
#define fgassert(x) if(!(x)) { int* p = nullptr; *p = 1; }
#else
#define fgassert(x) 
#endif

#endif
