// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__UTIL_H
#define D2D__UTIL_H

#include "win32_includes.h"
#include "backend.h"
#include "compiler.h"
#include <d2d1_1.h>
#include <xmmintrin.h>

namespace D2D {
  inline static D2D1_COLOR_F ToD2Color(unsigned int color)
  {
    FG_Color c = { color };
    return D2D1::ColorF((c.r << 16) | (c.g << 8) | c.b, c.a / 255.0f);
  }

  FG_COMPILER_DLLEXPORT size_t UTF8toUTF16(const char* FG_RESTRICT input, ptrdiff_t srclen, wchar_t* FG_RESTRICT output, size_t buflen);
  FG_COMPILER_DLLEXPORT size_t UTF16toUTF8(const wchar_t* FG_RESTRICT input, ptrdiff_t srclen, char* FG_RESTRICT output, size_t buflen);

  inline int FastTruncate(float f) noexcept
  {
    return _mm_cvtt_ss2si(_mm_load_ss(&f));
  }
}

#endif
