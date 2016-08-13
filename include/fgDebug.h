// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_DEBUG_H__
#define _FG_DEBUG_H__

#include "fgTreeView.h"

enum FGDEBUG_FLAGS
{
  FGDEBUG_CLEARONHIDE = (FGELEMENT_SNAPY << 1), // If set, clears the message log when the debug view is hidden
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
      int x; int y; // Mouse and touch events
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
  unsigned char subtype;
  unsigned long long time;
  union { size_t value; float valuef; void* valuep; };
  size_t depth;
} fgDebugMessage;

// The debug view is designed as an overlay to help you inspect the current state of the GUI
typedef struct _FG_DEBUG {
  fgElement element;
  fgTreeView elements; // TreeView of the elements, minus the debug view itself.
  fgTreeView messages; // Log of all messages passing through the GUI
  fgText properties; // element properties
  fgText contents; // message contents
  size_t(FG_FASTCALL *behaviorhook)(struct _FG_ELEMENT* self, const FG_Msg* msg);
  fgDeclareVector(fgDebugMessage, DebugMessage) messagelog;
  fgDeclareVector(char*, strings) messagestrings;
  size_t depth;
  fgElement* depthelement;
  fgElement* hover;
  int ignore;

#ifdef  __cplusplus
  inline operator fgElement*() { return &element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgDebug;

FG_EXTERN void FG_FASTCALL fgDebug_Init(fgDebug* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgDebug_Destroy(fgDebug* self);
FG_EXTERN size_t FG_FASTCALL fgDebug_Message(fgDebug* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgDebug_ClearLog(fgDebug* self);
FG_EXTERN void FG_FASTCALL fgDebug_Show(float left, float right);
FG_EXTERN void FG_FASTCALL fgDebug_Hide();
FG_EXTERN fgDebug* FG_FASTCALL fgDebug_Get();
FG_EXTERN size_t FG_FASTCALL fgDebug_LogMessage(fgDebug* self, const FG_Msg* msg, unsigned long long time, size_t depth);
FG_EXTERN ptrdiff_t FG_FASTCALL fgDebug_WriteMessage(char* buf, size_t bufsize, fgDebugMessage* msg);
FG_EXTERN void FG_FASTCALL fgDebug_BuildTree(fgElement* treeview);

FG_EXTERN size_t FG_FASTCALL fgRoot_BehaviorDebug(fgElement* self, const FG_Msg* msg);

#endif