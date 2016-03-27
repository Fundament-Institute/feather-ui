// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_TEXT_H__
#define _FG_TEXT_H__

#include "fgChild.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGTEXT_FLAGS
{
  FGTEXT_CHARWRAP = (1 << 6), // Wraps lines that go past the edge of the container by character
  FGTEXT_WORDWRAP = (1 << 7), // Wraps lines that go past the edge of the container by word (the definition of a "word" is implementation specific)
  FGTEXT_ELLIPSES = (1 << 8), // Lines that go past the bounderies of the text object are cut off with an ellipses (...)
  FGTEXT_RTL = (1 << 9), // Forces right-to-left text rendering.
  FGTEXT_RIGHTALIGN = (1 << 10),
  FGTEXT_CENTER = (1 << 11), // Text horizontal centering behaves differently, because it centers each individual line.
  FGTEXT_STRETCH = (1 << 12), // Stretches the text to fill the area.
  FGTEXT_SUBPIXEL = (1 << 13), // Indicates this text should try to render with LCD subpixel hinting.
};

// fgText stores a string and renders it according to the font and fontcolor that it has.
typedef struct {
  fgChild element;
  char* text;
  void* font;
  fgColor color;
#ifdef  __cplusplus
  inline operator fgChild*() { return &element; }
  inline fgChild* operator->() { return operator fgChild*(); }
#endif
} fgText;

FG_EXTERN fgChild* FG_FASTCALL fgText_Create(char* text, void* font, unsigned int color, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgText_Init(fgText* self, char* text, void* font, unsigned int color, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgText_Destroy(fgText* self);
FG_EXTERN size_t FG_FASTCALL fgText_Message(fgText* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgText_Recalc(fgText* self);

FG_EXTERN void* FG_FASTCALL fgCreateFont(fgFlag flags, const char* font, unsigned int fontsize, float lineheight, float letterspacing);
FG_EXTERN void* FG_FASTCALL fgCloneFont(void* font);
FG_EXTERN void FG_FASTCALL fgDestroyFont(void* font);
FG_EXTERN void FG_FASTCALL fgDrawFont(void* font, const char* text, unsigned int color, const AbsRect* area, FABS rotation, AbsVec* center, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgFontSize(void* font, const char* text, AbsRect* area, fgFlag flags);

#ifdef  __cplusplus
}
#endif

#endif