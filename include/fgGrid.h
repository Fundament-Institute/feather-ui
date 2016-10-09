// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_GRID_H__
#define _FG_GRID_H__

#include "fgBox.h"

#ifdef  __cplusplus
extern "C" {
#endif
  // Represents a grid of columns (or rows) with labels.
  typedef struct _FG_GRID {
    fgBox box;
    fgElement header; // The header is where the labels are
#ifdef  __cplusplus
    inline operator fgElement*() { return &box.window.control.element; }
    inline fgElement* operator->() { return operator fgElement*(); }
#endif
  } fgGrid;

  FG_EXTERN void FG_FASTCALL fgGrid_Init(fgGrid* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
  FG_EXTERN void FG_FASTCALL fgGrid_Destroy(fgGrid* self);
  FG_EXTERN size_t FG_FASTCALL fgGrid_Message(fgGrid* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
