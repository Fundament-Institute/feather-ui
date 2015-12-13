// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SLIDER_H__
#define __FG_SLIDER_H__

#include "fgWindow.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A slider can be dragged along either a smooth value or between integer increments
typedef struct _FG_SLIDER {
  fgWindow window;
  fgChild slider; //Slider object centered around the current value
  size_t range; // Represents a range from [0,range], INCLUSIVE. Set with FG_SETSTATE and a second argument set to 1
  size_t value;
} fgSlider;

FG_EXTERN fgChild* FG_FASTCALL fgSlider_Create(size_t range, fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgSlider_Init(fgSlider* BSS_RESTRICT self, size_t range, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgSlider_Destroy(fgSlider* self);
FG_EXTERN size_t FG_FASTCALL fgSlider_Message(fgSlider* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif