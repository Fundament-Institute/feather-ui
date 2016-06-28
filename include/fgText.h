// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_TEXT_H__
#define _FG_TEXT_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGTEXT_FLAGS // these start at (1 << 13) so they don't intersect the scrollbar flags, due to fgTextbox
{
  FGTEXT_CHARWRAP = (1 << 13), // Wraps lines that go past the edge of the container by character
  FGTEXT_WORDWRAP = (1 << 14), // Wraps lines that go past the edge of the container by word (the definition of a "word" is implementation specific)
  FGTEXT_ELLIPSES = (1 << 15), // Lines that go past the bounderies of the text object are cut off with an ellipses (...)
  FGTEXT_RTL = (1 << 16), // Forces right-to-left text rendering.
  FGTEXT_RIGHTALIGN = (1 << 17),
  FGTEXT_CENTER = (1 << 18), // Text horizontal centering behaves differently, because it centers each individual line.
  FGTEXT_SUBPIXEL = (1 << 19), // Indicates this text should try to render with LCD subpixel hinting.
};

typedef fgDeclareVector(char, String) fgVectorString;
typedef fgDeclareVector(int, UTF32) fgVectorUTF32;

// fgText stores a string and renders it according to the font and fontcolor that it has.
typedef struct {
  fgElement element;
  fgVectorUTF32 text;
  fgVectorString buf;
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
FG_EXTERN void FG_FASTCALL fgText_Init(fgText* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgText_Destroy(fgText* self);
FG_EXTERN size_t FG_FASTCALL fgText_Message(fgText* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgText_Recalc(fgText* self);

FG_EXTERN void* FG_FASTCALL fgCreateFont(fgFlag flags, const char* font, unsigned int fontsize, unsigned int dpi);
FG_EXTERN void* FG_FASTCALL fgCopyFont(void* font, unsigned int fontsize, unsigned int dpi);
FG_EXTERN void* FG_FASTCALL fgCloneFont(void* font);
FG_EXTERN void FG_FASTCALL fgDestroyFont(void* font);
FG_EXTERN void* FG_FASTCALL fgDrawFont(void* font, const int* text, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, AbsVec* center, fgFlag flags, void* cache);
FG_EXTERN void FG_FASTCALL fgFontSize(void* font, const int* text, float lineheight, float letterspacing, AbsRect* area, fgFlag flags); // this should return EXPECTED TEXT AREA that is calculated by lineheight - it should discard excessive vertical space caused by weird unicode modifiers.
FG_EXTERN size_t FG_FASTCALL fgFontIndex(void* font, const int* text, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, void* cache);
FG_EXTERN AbsVec FG_FASTCALL fgFontPos(void* font, const int* text, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, void* cache);
FG_EXTERN void FG_FASTCALL fgFontGet(void* font, float* lineheight, unsigned int* size, unsigned int* dpi);

#ifdef  __cplusplus
}
#endif

#endif