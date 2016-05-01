// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_TEXT_H__
#define _FG_TEXT_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGTEXT_FLAGS
{
  FGTEXT_CHARWRAP = (1 << 8), // Wraps lines that go past the edge of the container by character
  FGTEXT_WORDWRAP = (1 << 9), // Wraps lines that go past the edge of the container by word (the definition of a "word" is implementation specific)
  FGTEXT_ELLIPSES = (1 << 10), // Lines that go past the bounderies of the text object are cut off with an ellipses (...)
  FGTEXT_RTL = (1 << 11), // Forces right-to-left text rendering.
  FGTEXT_RIGHTALIGN = (1 << 12),
  FGTEXT_CENTER = (1 << 13), // Text horizontal centering behaves differently, because it centers each individual line.
  FGTEXT_SUBPIXEL = (1 << 14), // Indicates this text should try to render with LCD subpixel hinting.
};

// fgText stores a string and renders it according to the font and fontcolor that it has.
typedef struct {
  fgElement element;
  char* text;
  void* font;
  void* cache;
  fgColor color;
  float lineheight;
  float letterspacing;
#ifdef  __cplusplus
  inline operator fgElement*() { return &element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgText;

FG_EXTERN fgElement* FG_FASTCALL fgText_Create(char* text, void* font, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgText_Init(fgText* self, char* text, void* font, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgText_Destroy(fgText* self);
FG_EXTERN size_t FG_FASTCALL fgText_Message(fgText* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgText_Recalc(fgText* self);

FG_EXTERN void* FG_FASTCALL fgCreateFont(fgFlag flags, const char* font, unsigned int fontsize, unsigned int dpi);
FG_EXTERN void* FG_FASTCALL fgCloneFontDPI(void* font, unsigned int dpi);
FG_EXTERN void* FG_FASTCALL fgCloneFont(void* font);
FG_EXTERN void FG_FASTCALL fgDestroyFont(void* font);
FG_EXTERN void* FG_FASTCALL fgDrawFont(void* font, const char* text, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, AbsVec* center, fgFlag flags, void* cache);
FG_EXTERN void FG_FASTCALL fgFontSize(void* font, const char* text, float lineheight, float letterspacing, AbsRect* area, fgFlag flags); // this should return EXPECTED TEXT AREA that is calculated by lineheight - it should discard excessive vertical space caused by weird unicode modifiers.

#ifdef  __cplusplus
}
#endif

#endif