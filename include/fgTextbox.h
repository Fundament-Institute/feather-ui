// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TEXTBOX_H__
#define __FG_TEXTBOX_H__

#include "fgScrollbar.h"
#include "fgText.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGTEXTBOX_FLAGS
{
  FGTEXTBOX_ACTION = (FGTEXT_SUBPIXEL << 1),
  FGTEXTBOX_SINGLELINE = (FGTEXTBOX_ACTION << 1),
};

enum FGTEXTBOX_ACTIONS
{
  FGTEXTBOX_SELECTALL = FGSCROLLBAR_NUM,
  FGTEXTBOX_CUT,
  FGTEXTBOX_COPY,
  FGTEXTBOX_PASTE,
  FGTEXTBOX_GOTOSTART,
  FGTEXTBOX_GOTOEND,
  FGTEXTBOX_GOTOLINESTART,
  FGTEXTBOX_GOTOLINEEND,
  FGTEXTBOX_TOGGLEINSERT,
};

// A Textbox is really just a text static inside an optional Scrollbar. It can be single or multi-line with an optional validation regex.
// The textbox only understands single UTF codepoints, so an external library should be used to perform unicode normalization before setting it.
typedef struct {
  fgScrollbar scroll;
  char* validation; // validation regex
  char* formatting; // printf formatting string matched to capture groups in the validation regex
  int mask; // If not zero, stores a unicode character for password masking. 
  fgVectorUTF32 text;
  fgVectorString buf;
  fgVectorUTF32 placeholder; // placeholder text displayed when textbox is empty.
  fgColor placecolor; // placeholder text color. Use SETCOLOR with the subtype set to 1.
  fgColor cursorcolor; // cursor color. Use SETCOLOR with the subtype set to 2.
  fgColor selector; // Color of the selector rectangle. Use SETCOLOR with the subtype set to 3.
  size_t start; // current cursor
  AbsVec startpos;
  size_t end; // end of selection
  AbsVec endpos;
  float lastx; // stores the x coordinate while we are hitting up or down keys.
  AbsRect areacache; // Stores a cache of the last area we knew about. If this changes, startpos and endpos must be recalculated.
  char inserting;
  void* font;
  void* cache;
  fgColor color;
  float lineheight;
  float letterspacing;
  double lastclick; // determines the starting point of the cursor blink animation
#ifdef  __cplusplus
  inline operator fgElement*() { return &scroll.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgTextbox;

FG_EXTERN void FG_FASTCALL fgTextbox_Init(fgTextbox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void FG_FASTCALL fgTextbox_Destroy(fgTextbox* self);
FG_EXTERN size_t FG_FASTCALL fgTextbox_Message(fgTextbox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif