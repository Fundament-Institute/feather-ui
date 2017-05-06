// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_DEBUG_H__
#define __FG_DEBUG_H__

#include "fgTreeview.h"
#include "fgMenu.h"
#include "fgGrid.h"
#include "fgTabcontrol.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGDEBUG_FLAGS
{
  FGDEBUG_CLEARONHIDE = (FGELEMENT_SNAPY << 1), // If set, clears the message log when the debug view is hidden
  FGDEBUG_OVERLAY = (FGDEBUG_CLEARONHIDE << 1),
};

union _FG_DEBUG_MESSAGE_STORAGE {
  void* p;
  float f;
  size_t u;
  ptrdiff_t i;
  fgTransform transform;
  CRect crect;
  AbsRect rect;
  CVec cvec;
  AbsVec vec;
  struct { fgElement* element; const char* name; };
  FG_Msg* message;
  const char* s;
};

typedef struct _FG_DEBUG_MESSAGE {
  union {
    struct {
      union _FG_DEBUG_MESSAGE_STORAGE arg1;
      union _FG_DEBUG_MESSAGE_STORAGE arg2;
    };
    struct {
      float x; float y; // Mouse and touch events
      union {
        struct { unsigned char button; unsigned char allbtn; };
        struct { short scrolldelta; short scrollhdelta; }; // MOUSESCROLL
        short touchindex; // Touch events
      };
    } mouse;
    struct {  // Keys
      int keychar; //Only used by KEYCHAR, represents a utf32 character
      unsigned char keycode; //only used by KEYDOWN/KEYUP, represents an actual keycode, not a character
      char sigkeys; // 1: shift, 2: ctrl, 4: alt, 8: held
    } keys;
    struct { float joyvalue; short joyaxis; }; // JOYAXIS
    struct { char joydown; short joybutton; }; // JOYBUTTON
  };
  unsigned short type;
  unsigned short subtype;
  unsigned long long time;
  union { size_t value; float valuef; void* valuep; };
  size_t depth;
} fgDebugMessage;

// The debug view is designed as an overlay to help you inspect the current state of the GUI
typedef struct _FG_DEBUG {
  fgTabcontrol tabs;
  fgElement overlay;
  fgElement* tablayout;
  fgElement* tabmessages;
  fgTreeview elements; // TreeView of the elements, minus the debug view itself.
  fgGrid properties; // element properties
  fgText contents; // message contents
  fgMenu context;
  fgTextbox editbox;
  size_t(*behaviorhook)(struct _FG_ELEMENT* self, const FG_Msg* msg);
  fgDeclareVector(fgDebugMessage, DebugMessage) messagelog;
  fgDeclareVector(char*, strings) messagestrings;
  size_t depth;
  fgElement* hover;
  int ignore;
  void* font;
  fgColor color;
  float lineheight;
  float letterspacing;
  AbsRect oldpadding;
  fgVectorUTF32 text32;
  fgVectorUTF16 text16;
  fgVectorUTF8 text8;
  AbsVec lastscroll;

#ifdef  __cplusplus
  inline operator fgElement*() { return &tabs.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgDebug;

FG_EXTERN void fgDebug_Init(fgDebug* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgDebug_Destroy(fgDebug* self);
FG_EXTERN size_t fgDebug_Message(fgDebug* self, const FG_Msg* msg);
FG_EXTERN void fgDebug_ClearLog(fgDebug* self);
FG_EXTERN void fgDebug_Show(const fgTransform* tf, char overlay);
FG_EXTERN void fgDebug_Hide();
FG_EXTERN fgDebug* fgDebug_Get();
FG_EXTERN size_t fgDebug_LogMessage(fgDebug* self, const FG_Msg* msg, unsigned long long time, size_t depth);
FG_EXTERN ptrdiff_t fgDebug_WriteMessage(fgDebugMessage* msg, char* buf, size_t bufsize);
FG_EXTERN void fgDebug_DumpMessages(const char* file);
FG_EXTERN void fgDebug_BuildTree(fgElement* treeview);

FG_EXTERN size_t fgRoot_BehaviorDebug(fgElement* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif