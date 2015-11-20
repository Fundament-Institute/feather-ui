// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_ROOT_H__
#define __FG_ROOT_H__

#include "fgWindow.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct FG_DEFER_ACTION {
  struct FG_DEFER_ACTION* next; // It's crucial that this is the first element
  struct FG_DEFER_ACTION* prev;
  char (FG_FASTCALL *action)(void*); // If this returns nonzero, this node is deallocated after this returns.
  void* arg; // Argument passed into the function
  double time; // Time when the action should be triggered
} fgDeferAction;

// Defines the root interface to the GUI. This object should be returned by the implementation at some point
typedef struct _FG_ROOT {
  fgWindow gui;
  size_t (FG_FASTCALL *keymsghook)(const FG_Msg* msg);
  size_t (FG_FASTCALL *behaviorhook)(struct _FG_CHILD* self, const FG_Msg* msg);
  void (FG_FASTCALL *update)(struct _FG_ROOT* self, double delta);
  fgChild mouse; // Add children to this to have them follow the mouse around.
  fgDeferAction* updateroot;
  double time; // In seconds
} fgRoot;

FG_EXTERN fgRoot* FG_FASTCALL fgInitialize();
FG_EXTERN fgRoot* FG_FASTCALL fgSingleton();
FG_EXTERN char FG_FASTCALL fgLoadExtension(void* fg, const char* extname);
FG_EXTERN void FG_FASTCALL fgTerminate(fgRoot* root);
FG_EXTERN char FG_FASTCALL fgMessageLoop(fgRoot* root);
FG_EXTERN void FG_FASTCALL fgRoot_Init(fgRoot* self);
FG_EXTERN void FG_FASTCALL fgRoot_Destroy(fgRoot* self);
FG_EXTERN size_t FG_FASTCALL fgRoot_Message(fgRoot* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgRoot_Inject(fgRoot* self, const FG_Msg* msg); // Returns 0 if handled, 1 otherwise
FG_EXTERN size_t FG_FASTCALL fgRoot_BehaviorDefault(fgChild* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgRoot_CallBehavior(fgChild* self, const FG_Msg* msg); // Calls the appropriate fgroot behavior function
FG_EXTERN size_t FG_FASTCALL fgRoot_KeyMsgHook(const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgRoot_Update(fgRoot* self, double delta);
FG_EXTERN fgDeferAction* FG_FASTCALL fgRoot_AllocAction(char (FG_FASTCALL *action)(void*), void* arg, double time);
FG_EXTERN void FG_FASTCALL fgRoot_DeallocAction(fgRoot* self, fgDeferAction* action); // Removes action from the list if necessary
FG_EXTERN void FG_FASTCALL fgRoot_AddAction(fgRoot* self, fgDeferAction* action); // Adds an action. Action can't already be in list.
FG_EXTERN void FG_FASTCALL fgRoot_RemoveAction(fgRoot* self, fgDeferAction* action); // Removes an action. Action must be in list.
FG_EXTERN void FG_FASTCALL fgRoot_ModifyAction(fgRoot* self, fgDeferAction* action); // Moves action if it needs to be moved, or inserts it if it isn't already in the list.
fgChild* fgLayoutLoadMapping(const char* name, fgFlag flags, fgChild* parent, fgElement* element);

#ifdef  __cplusplus
}
#endif

#endif
