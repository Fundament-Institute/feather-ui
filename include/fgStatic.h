// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_RENDERABLE_H__
#define __FG_RENDERABLE_H__

#include "fgChild.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FG_TEXTFLAGS
{
  FGTEXT_CHARWRAP=16, // Wraps lines that go past the edge of the container by character
  FGTEXT_WORDWRAP=32, // Wraps lines that go past the edge of the container by word (the definition of a "word" is implementation specific)
  FGTEXT_ELLIPSES=64, // Lines that go past the bounderies of the text object are cut off with an ellipses (...)
  FGTEXT_RTL=128, // Forces right-to-left text rendering.
  FGTEXT_RIGHTALIGN=256,
  FGTEXT_CENTER=512, // Text horizontal centering behaves differently, because it centers each individual line.
  FGTEXT_STRETCH=1024, // Stretches the text to fill the area.
};

enum FG_IMAGEFLAGS
{
  FGIMAGE_STRETCHX=16, // Stretches the image instead of tiling it
  FGIMAGE_STRETCHY=32,
};

FG_EXTERN size_t FG_FASTCALL fgStatic_Message(fgChild* self, const FG_Msg* msg);

FG_EXTERN void* FG_FASTCALL fgCreateImageDef(fgFlag flags, const char* data, size_t length, unsigned int color, const CRect* uv);
FG_EXTERN void* FG_FASTCALL fgDefaultTextDef(fgFlag flags, const char* text);
FG_EXTERN void* FG_FASTCALL fgCreateTextDef(fgFlag flags, const char* text, const char* font, unsigned short fontsize, unsigned short lineheight, unsigned int color);
//FG_EXTERN void* FG_FASTCALL fgCreateVectorDef(const void* data, size_t length);
FG_EXTERN fgChild* FG_FASTCALL fgLoadDef(void* def, const fgElement* element, int order);
FG_EXTERN void* FG_FASTCALL fgCloneDef(void* def);
FG_EXTERN void FG_FASTCALL fgDestroyDef(void* def);

#ifdef  __cplusplus
}
#endif

#endif
