// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_ROOT_H__
#define __FG_ROOT_H__

#include "fgControl.h"
#include "fgBackend.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct kh_fgRadioGroup_s;
struct kh_fgFunctionMap_s;
struct kh_fgIDMap_s;
struct kh_fgCursorMap_s;
struct kh_fgIDHash_s;
struct _FG_MONITOR;
struct _FG_SKIN;
struct _FG_MESSAGEQUEUE;
struct _FG_ROOT;
typedef void(*fgInitializer)(fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, const char*, fgFlag, const fgTransform*, unsigned short);
typedef size_t(*fgInject)(struct _FG_ROOT* self, const FG_Msg* msg);
typedef fgElement* (*fgOrderedDrawGet)(fgElement*, const AbsRect*, const AbsRect*);
typedef void(*fgAuxDrawFunction)(fgElement*, const AbsRect*, const fgDrawAuxData*, fgElement*);

typedef struct _FG_DEFER_ACTION {
  struct _FG_DEFER_ACTION* next; // It's crucial that this is the first element
  struct _FG_DEFER_ACTION* prev;
  char (*action)(void*); // If this returns nonzero, this node is deallocated after this returns.
  void* arg; // Argument passed into the function
  double time; // Time when the action should be triggered
} fgDeferAction;

// Defines the root interface to the GUI. This object should be returned by the implementation at some point
typedef struct _FG_ROOT {
  fgControl gui;
  fgBackend backend;
  struct _FG_MONITOR* monitors;
  fgDeferAction* updateroot;
  struct kh_fgRadioGroup_s* radiohash;
  struct kh_fgFunctionMap_s* functionhash;
  struct kh_fgIDMap_s* idmap;
  struct kh_fgIDHash_s* idhash; // reverse ID lookup
  struct kh_fgInitMap_s* initmap;
  struct kh_fgCursorMap_s* cursormap;
  fgIntVec dpi;
  float lineheight;
  float fontscale;
  double time; // In seconds
  unsigned int cursor;
  double cursorblink; // In seconds
  fgMouseState mouse;
  char dragtype; // FG_CLIPBOARD
  void* dragdata;
  fgElement* dragdraw;
  fgElement* topmost;
  unsigned int keys[8]; // 8*4*8 = 256
  void* aniroot;
  struct _FG_MESSAGEQUEUE* queue;
  struct _FG_MESSAGEQUEUE* aux;
  fgInject inject;
  fgElement* fgPendingFocusedWindow;
  fgElement* fgFocusedWindow;
  fgElement* fgLastHover; // Last window the mouse moved over, used to generate MOUSEON and MOUSEOFF events
  fgElement* fgCaptureWindow;
  size_t(*fgBehaviorHook)(struct _FG_ELEMENT* self, const FG_Msg* msg);
#ifdef  __cplusplus
  inline bool GetKey(unsigned char key) const { return (keys[key / 32] & (1 << (key % 32))) != 0; }
  inline operator fgElement*() { return &gui.element; }
#endif
} fgRoot;

FG_EXTERN fgRoot* fgSingleton();
FG_EXTERN char fgLoadExtension(const char* extname, void* fg, size_t sz);
FG_EXTERN void fgRoot_Init(fgRoot* self, const AbsRect* area, const fgIntVec* dpi, const fgBackend* backend);
FG_EXTERN void fgRoot_Destroy(fgRoot* self);
FG_EXTERN size_t fgRoot_Message(fgRoot* self, const FG_Msg* msg);
FG_EXTERN size_t fgRoot_DefaultInject(fgRoot* self, const FG_Msg* msg); // Returns 0 if handled, 1 otherwise
FG_EXTERN void fgRoot_Update(fgRoot* self, double delta);
FG_EXTERN void fgRoot_CheckMouseMove(fgRoot* self);
FG_EXTERN fgDeferAction* fgRoot_AllocAction(char (*action)(void*), void* arg, double time);
FG_EXTERN void fgRoot_DeallocAction(fgRoot* self, fgDeferAction* action); // Removes action from the list if necessary
FG_EXTERN void fgRoot_AddAction(fgRoot* self, fgDeferAction* action); // Adds an action. Action can't already be in list.
FG_EXTERN void fgRoot_RemoveAction(fgRoot* self, fgDeferAction* action); // Removes an action. Action must be in list.
FG_EXTERN void fgRoot_ModifyAction(fgRoot* self, fgDeferAction* action); // Moves action if it needs to be moved, or inserts it if it isn't already in the list.
FG_EXTERN struct _FG_MONITOR* fgRoot_GetMonitor(const fgRoot* self, const AbsRect* rect);
FG_EXTERN size_t fgStandardInject(fgElement* self, const FG_Msg* msg, const AbsRect* area);
FG_EXTERN size_t fgOrderedInject(fgElement* self, const FG_Msg* msg, const AbsRect* area, fgElement* skip, fgElement* (*fn)(fgElement*, const FG_Msg*), fgElement* selected);
FG_EXTERN void fgStandardDraw(fgElement* self, const AbsRect* area, const fgDrawAuxData* aux, char culled, fgAuxDrawFunction draw);
FG_EXTERN void fgOrderedDraw(fgElement* self, const AbsRect* area, const fgDrawAuxData* aux, char culled, fgElement* skip, fgOrderedDrawGet fn, fgAuxDrawFunction draw, fgElement* selected);
FG_EXTERN char fgDrawSkin(fgVector* skinstyle, AbsRect* padding, const struct _FG_SKIN* skin, const AbsRect* area, const fgDrawAuxData* aux, char culled, char foreground, char clipping);
FG_EXTERN fgElement* fgCreate(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN int fgRegisterCursor(int cursor, const void* data, size_t sz);
FG_EXTERN int fgRegisterFunction(const char* name, fgListener fn);
FG_EXTERN int fgRegisterDelegate(const char* name, void* p, void(*fn)(void*, struct _FG_ELEMENT*, const FG_Msg*));
FG_EXTERN fgElement* fgSetTopmost(fgElement* target);
FG_EXTERN char fgClearTopmost(fgElement* target);
FG_EXTERN void fgRegisterControl(const char* name, fgInitializer fn, size_t sz, fgFlag flags);
FG_EXTERN void fgIterateControls(void* p, void(*fn)(void*, const char*));
FG_EXTERN void fgIterateGroups(void* p, void(*fn)(void*, fgStyleIndex));
FG_EXTERN void fgSendMessageAsync(fgElement* element, const FG_Msg* msg, unsigned int arg1size, unsigned int arg2size);
FG_EXTERN size_t fgGetTypeSize(const char* type);
FG_EXTERN fgFlag fgGetTypeFlags(const char* type);
FG_EXTERN fgInject fgSetInjectFunc(fgInject inject); // Sets the injection function and returns the previous one
FG_EXTERN fgElement* fgGetID(const char* id);
FG_EXTERN void fgAddID(const char* id, fgElement* element);
FG_EXTERN char fgRemoveID(fgElement* element);
FG_EXTERN int fgLog(const char* format, ...);

#ifdef  __cplusplus
}

template<class T, void(T::*F)(struct _FG_ELEMENT*, const FG_Msg*)>
inline void fgRegisterDelegate(const char* name, T* src) { fgRegisterDelegate(name, src, &fgDelegateListener::stub<T, F>); }
template<class T, void(T::*F)(struct _FG_ELEMENT*, const FG_Msg*) const>
inline void fgRegisterDelegate(const char* name, const T* src) { fgRegisterDelegate(name, const_cast<T*>(src), &fgDelegateListener::stubconst<T, F>); }

template<FG_MSGTYPE type, typename... Args>
inline size_t fgSendMsg(fgElement* self, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgSingleton()->fgBehaviorHook)(self, &msg);
}

template<FG_MSGTYPE type, typename... Args>
inline size_t fgSendSubMsg(fgElement* self, unsigned short sub, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = sub;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgSingleton()->fgBehaviorHook)(self, &msg);
}
#endif

#endif
