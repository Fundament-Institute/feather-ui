// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_CURVE_H__
#define __FG_CURVE_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

  enum FGCURVE_FLAGS
  {
    FGCURVE_LINE = 0,
    FGCURVE_QUADRATIC = (1 << 8),
    FGCURVE_CUBIC = (2 << 8),
    FGCURVE_BSPLINE = (3 << 8),
    FGCURVE_CURVEMASK = (3 << 8),
  };

  typedef fgDeclareVector(AbsVec, Point) fgVectorPoint;

  // fgCurve stores either a series of lines, or a curve that is tesselated into a strip of lines.
  typedef struct {
    fgElement element;
    fgColor color;
    fgVectorPoint points; // use ADDITEM or REMOVEITEM to add or remove points
    fgVectorPoint cache; // curves are subdivided into lines
    float factor; // subdivision factor, set using SETSTATE
  } fgCurve;

  FG_EXTERN fgElement* FG_FASTCALL fgCurve_Create(const AbsVec* points, size_t npoints, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
  FG_EXTERN void FG_FASTCALL fgCurve_Init(fgCurve* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
  FG_EXTERN void FG_FASTCALL fgCurve_Destroy(fgCurve* self);
  FG_EXTERN size_t FG_FASTCALL fgCurve_Message(fgCurve* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif