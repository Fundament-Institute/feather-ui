// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TEXT_H__
#define __FG_TEXT_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGTEXT_FLAGS // these start at (1 << 14) so they don't intersect the scrollbar flags, due to fgTextbox
{
  FGTEXT_CHARWRAP = (1 << 14), // Wraps lines that go past the edge of the container by character
  FGTEXT_WORDWRAP = (1 << 15), // Wraps lines that go past the edge of the container by word (the definition of a "word" is implementation specific)
  FGTEXT_ELLIPSES = (1 << 16), // Lines that go past the bounderies of the text object are cut off with an ellipses (...)
  FGTEXT_RTL = (1 << 17), // Forces right-to-left text rendering.
  FGTEXT_RIGHTALIGN = (1 << 18),
  FGTEXT_CENTER = (1 << 19), // Text horizontal centering behaves differently, because it centers each individual line.
  FGTEXT_SUBPIXEL = (1 << 20), // Indicates this text should try to render with LCD subpixel hinting.
};

typedef fgDeclareVector(char, UTF8) fgVectorUTF8;
typedef fgDeclareVector(wchar_t, UTF16) fgVectorUTF16;
typedef fgDeclareVector(int, UTF32) fgVectorUTF32;

typedef struct _FG_FONT_DESC {
  FABS ascender;
  FABS descender;
  FABS lineheight;
  unsigned int pt;
  fgIntVec dpi;
} fgFontDesc;

// fgText stores a string and renders it according to the font and fontcolor that it has.
typedef struct _FG_TEXT {
  fgElement element;
  fgVectorUTF32 text32;
  fgVectorUTF16 text16;
  fgVectorUTF8 text8;
  fgFont font;
  void* layout;
  fgColor color;
  float lineheight;
  float letterspacing;
#ifdef  __cplusplus
  inline operator fgElement*() { return &element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgText;

FG_EXTERN fgElement* fgText_Create(char* text, fgFont font, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgText_Init(fgText* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgText_Destroy(fgText* self);
FG_EXTERN size_t fgText_Message(fgText* self, const FG_Msg* msg);
FG_EXTERN void fgText_Recalc(fgText* self);

#ifdef  __cplusplus
}
#endif

#endif