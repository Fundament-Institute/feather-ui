// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_PROGRESSBAR_H__
#define __FG_PROGRESSBAR_H__

#include "fgWindow.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A slider can be dragged along either a smooth value or between integer increments
typedef struct _FG_PROGRESSBAR {
  fgWindow window;
  fgChild slider; // Slider object centered around the current value
  FREL value;
} fgProgressbar;

FG_EXTERN fgChild* FG_FASTCALL fgProgressbar_Create(fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgProgressbar_Init(fgProgressbar* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgProgressbar_Destroy(fgProgressbar* self);
FG_EXTERN size_t FG_FASTCALL fgProgressbar_Message(fgProgressbar* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif