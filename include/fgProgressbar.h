// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_PROGRESSBAR_H__
#define __FG_PROGRESSBAR_H__

#include "fgWindow.h"
#include "fgText.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A slider can be dragged along either a smooth value or between integer increments
typedef struct _FG_PROGRESSBAR {
  fgWindow window;
  fgText text; // Text displayed in the center of the progress bar
  fgChild bar; // bar object whose [left,right] coordinates are set to [0.0,value] by default
  FREL value;
} fgProgressbar;

FG_EXTERN fgChild* FG_FASTCALL fgProgressbar_Create(FREL value, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgProgressbar_Init(fgProgressbar* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgProgressbar_Destroy(fgProgressbar* self);
FG_EXTERN size_t FG_FASTCALL fgProgressbar_Message(fgProgressbar* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif