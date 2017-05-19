// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_UTIL_D2D_H__
#define __FG_UTIL_D2D_H__

#include "win32_includes.h"
#include "feathergui.h"
#include <d2d1_1.h>

inline static D2D1_COLOR_F ToD2Color(unsigned int color)
{
  fgColor c = { color };
  return D2D1::ColorF((c.r << 16) | (c.g << 8) | c.b, c.a / 255.0f);
}

#ifdef BSS_DEBUG
#define fgassert(x) if(!(x)) { int* p = nullptr; *p = 1; }
#else
#define fgassert(x) 
#endif

#endif