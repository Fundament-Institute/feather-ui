// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_ROOT_H__
#define __FG_ROOT_H__

#include "fgControl.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct __kh_fgRadioGroup_t;
struct __kh_fgFunctionMap_t;
struct _FG_MONITOR;

typedef struct _FG_DEFER_ACTION {
  struct _FG_DEFER_ACTION* next; // It's crucial that this is the first element
  struct _FG_DEFER_ACTION* prev;
  char (FG_FASTCALL *action)(void*); // If this returns nonzero, this node is deallocated after this returns.
  void* arg; // Argument passed into the function
  double time; // Time when the action should be triggered
} fgDeferAction;

// Defines the root interface to the GUI. This object should be returned by the implementation at some point
typedef struct _FG_ROOT {
  fgControl gui;
  size_t (FG_FASTCALL *behaviorhook)(struct _FG_ELEMENT* self, const FG_Msg* msg);
  struct _FG_MONITOR* monitors;
  fgDeferAction* updateroot;
  struct __kh_fgRadioGroup_t* radiohash;
  struct __kh_fgFunctionMap_t* functionhash;
  size_t dpi;
  float lineheight;
  float fontscale;
  double time; // In seconds
  double cursorblink; // In seconds
  fgMouseState mouse;
  char lastcursor; // FG_CURSOR
  void* lastcursordata;
  char nextcursor; // FG_CURSOR
  void* nextcursordata;
  char overridecursor; // FG_CURSOR
  void* overridecursordata;
  char dragtype; // FG_CLIPBOARD
  void* dragdata;
  fgElement* dragdraw;
  fgElement* topmost;
  unsigned int keys[8]; // 8*4*8 = 256
#ifdef  __cplusplus
  inline bool GetKey(unsigned char key) const { return (keys[key / 32] & (1 << (key % 32))) != 0; }
  inline operator fgElement*() { return &gui.element; }
#endif
} fgRoot;

FG_EXTERN fgRoot* FG_FASTCALL fgInitialize();
FG_EXTERN fgRoot* FG_FASTCALL fgSingleton();
FG_EXTERN char FG_FASTCALL fgLoadExtension(const char* extname, void* fg, size_t sz);
FG_EXTERN void FG_FASTCALL fgTerminate(fgRoot* root);
FG_EXTERN char FG_FASTCALL fgMessageLoop(fgRoot* root);
FG_EXTERN void FG_FASTCALL fgRoot_Init(fgRoot* self, const AbsRect* area, size_t dpi);
FG_EXTERN void FG_FASTCALL fgRoot_Destroy(fgRoot* self);
FG_EXTERN size_t FG_FASTCALL fgRoot_Message(fgRoot* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgRoot_Inject(fgRoot* self, const FG_Msg* msg); // Returns 0 if handled, 1 otherwise
FG_EXTERN size_t FG_FASTCALL fgRoot_BehaviorDefault(fgElement* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgRoot_BehaviorListener(fgElement* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgRoot_Update(fgRoot* self, double delta);
FG_EXTERN void FG_FASTCALL fgRoot_CheckMouseMove(fgRoot* self);
FG_EXTERN void FG_FASTCALL fgRoot_SetCursor(char cursor, void* data);
FG_EXTERN fgDeferAction* FG_FASTCALL fgRoot_AllocAction(char (FG_FASTCALL *action)(void*), void* arg, double time);
FG_EXTERN void FG_FASTCALL fgRoot_DeallocAction(fgRoot* self, fgDeferAction* action); // Removes action from the list if necessary
FG_EXTERN void FG_FASTCALL fgRoot_AddAction(fgRoot* self, fgDeferAction* action); // Adds an action. Action can't already be in list.
FG_EXTERN void FG_FASTCALL fgRoot_RemoveAction(fgRoot* self, fgDeferAction* action); // Removes an action. Action must be in list.
FG_EXTERN void FG_FASTCALL fgRoot_ModifyAction(fgRoot* self, fgDeferAction* action); // Moves action if it needs to be moved, or inserts it if it isn't already in the list.
FG_EXTERN struct _FG_MONITOR* FG_FASTCALL fgRoot_GetMonitor(const fgRoot* self, const AbsRect* rect);
FG_EXTERN size_t FG_FASTCALL fgStandardInject(fgElement* self, const FG_Msg* msg, const AbsRect* area);
FG_EXTERN size_t FG_FASTCALL fgOrderedInject(fgElement* self, const FG_Msg* msg, const AbsRect* area, fgElement* skip, fgElement* (*fn)(fgElement*, const FG_Msg*));
FG_EXTERN void FG_FASTCALL fgStandardDraw(fgElement* self, const AbsRect* area, size_t dpi, char culled);
FG_EXTERN void FG_FASTCALL fgOrderedDraw(fgElement* self, const AbsRect* area, size_t dpi, char culled, fgElement* skip, fgElement* (*fn)(fgElement*, const AbsRect*));
FG_EXTERN void fgPushClipRect(const AbsRect* clip);
FG_EXTERN AbsRect fgPeekClipRect();
FG_EXTERN void fgPopClipRect();
FG_EXTERN void fgDirtyElement(fgElement* elem);
FG_EXTERN void fgDragStart(char type, void* data, fgElement* draw);
FG_EXTERN void fgSetCursor(unsigned int type, void* custom); // What custom actually is depends on the implemention
FG_EXTERN void fgClipboardCopy(unsigned int type, const void* data, size_t length); // passing in NULL will erase whatever what was in the clipboard.
FG_EXTERN char fgClipboardExists(unsigned int type);
FG_EXTERN const void* fgClipboardPaste(unsigned int type, size_t* length); // The pointer returned to this MUST BE FREED by calling fgClipboardFree() once you are done with it.
FG_EXTERN void fgClipboardFree(const void* mem);
FG_EXTERN fgElement* FG_FASTCALL fgCreateDefault(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN short FG_FASTCALL fgMessageMapDefault(const char* name);
FG_EXTERN void BSS_FORCEINLINE fgStandardApplyClipping(fgElement* hold, const AbsRect* area, bool& clipping);

#ifdef  __cplusplus
}

template<FG_MSGTYPE type, typename... Args>
inline size_t fgSendMsg(fgElement* self, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgSingleton()->behaviorhook)(self, &msg);
}

template<FG_MSGTYPE type, typename... Args>
inline size_t fgSendSubMsg(fgElement* self, unsigned char sub, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = sub;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgSingleton()->behaviorhook)(self, &msg);
}
#endif

#endif
