// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_PROGRESSBAR_H__
#define __FG_PROGRESSBAR_H__

#include "fgControl.h"
#include "fgText.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A slider can be dragged along either a smooth value or between integer increments
typedef struct _FG_PROGRESSBAR {
  fgControl control;
  fgText text; // Text displayed in the center of the progress bar
  fgElement bar; // bar object whose [left,right] coordinates are set to [0.0,value] by default
  FREL value;
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgProgressbar;

FG_EXTERN void FG_FASTCALL fgProgressbar_Init(fgProgressbar* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void FG_FASTCALL fgProgressbar_Destroy(fgProgressbar* self);
FG_EXTERN size_t FG_FASTCALL fgProgressbar_Message(fgProgressbar* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif