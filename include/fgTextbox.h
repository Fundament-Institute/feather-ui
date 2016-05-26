// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_TEXTBOX_H__
#define _FG_TEXTBOX_H__

#include "fgScrollbar.h"
#include "fgText.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGTEXTBOX_ACTIONS
{
  FGTEXTBOX_SELECTALL = 0,
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
  fgScrollbar window;
  char* validation; // validation regex
  int mask; // If not zero, stores a unicode character for password masking. 
  fgVectorUTF32 text;
  fgVectorString buf;
  const int* placeholder; // placeholder text displayed when textbox is empty. Use SETTEXT or GETTEXT with the second argument set to 1.
  fgColor selector; // Color of the selector rectangle
  fgColor placecolor; // placeholder text color. Use SETCOLOR with the second argument set to 1.
  size_t start; // start of text selection
  size_t end; // end of text selection (or just where the cursor is)
  char inserting;
  void* font;
  void* cache;
  fgColor color;
  float lineheight;
  float letterspacing;
#ifdef  __cplusplus
  inline operator fgElement*() { return &window.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgTextbox;

FG_EXTERN void FG_FASTCALL fgTextbox_Init(fgTextbox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN size_t FG_FASTCALL fgTextbox_Message(fgTextbox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif