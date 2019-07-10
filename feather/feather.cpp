// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/feather.h"
#include <intrin.h>

extern "C" void fgScaleRectDPI(fgRect* rect, float dpix, float dpiy)
{
  FG_ALIGN(16) float scale[4];
  scale[0] = (!dpix) ? 1.0f : (dpix / 96.0f);
  scale[1] = (!dpiy) ? 1.0f : (dpiy / 96.0f);
  scale[2] = scale[0];
  scale[3] = scale[1];
  _mm_storeu_ps(rect->ltrb, _mm_mul_ps(_mm_loadu_ps(rect->ltrb), _mm_load_ps(scale)));
  if(rect->right < rect->left)
    rect->right = rect->left;
  if(rect->bottom < rect->top)
    rect->bottom = rect->top;
}

extern "C" void fgInvScaleRectDPI(fgRect* rect, float dpix, float dpiy)
{
  FG_ALIGN(16) float scale[4];
  scale[0] = (!dpix) ? 1.0f : (96.0f / dpix);
  scale[1] = (!dpiy) ? 1.0f : (96.0f / dpiy);
  scale[2] = scale[0];
  scale[3] = scale[1];
  _mm_storeu_ps(rect->ltrb, _mm_mul_ps(_mm_loadu_ps(rect->ltrb), _mm_load_ps(scale)));
}

extern "C" void fgScaleVecDPI(fgVec* v, float dpix, float dpiy)
{
  v->x *= (!dpix) ? 1.0f : (dpix / 96.0f);
  v->y *= (!dpiy) ? 1.0f : (dpiy / 96.0f);
}

extern "C" void fgInvScaleVecDPI(fgVec* v, float dpix, float dpiy)
{
  v->x *= (!dpix) ? 1.0f : (96.0f / dpix);
  v->y *= (!dpiy) ? 1.0f : (96.0f / dpiy);
}

extern "C" void fgMergeRect(fgRect* target, const fgRect* src)
{
  FG_ALIGN(16) const float invert[4] = { 0, 0, -1.0f, -1.0f };
  auto inv = _mm_load_ps(invert);

  _mm_storeu_ps(
    target->ltrb,
    _mm_mul_ps(
      _mm_min_ps(
        _mm_mul_ps(_mm_loadu_ps(target->ltrb), inv),
        _mm_mul_ps(_mm_loadu_ps(src->ltrb), inv)
      ),
      inv
    )
  );
}
