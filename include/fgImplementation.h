// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_IMPLEMENTATION_H__
#define __FG_IMPLEMENTATION_H__

#include "fgRoot.h"
#include "fgTextbox.h"
#include "fgTopWindow.h"
#include "fgMenu.h"

#ifdef  __cplusplus
extern "C" {
#endif

// You can pass this to allow a library to work with an arbitrary feather implementation by using the function pointers instead of a static link.
typedef struct FG_IMPLEMENTATION {
  void (FG_FASTCALL *fgChild_Init)(fgChild* self);
  void (FG_FASTCALL *fgChild_Destroy)(fgChild* self);
  void (FG_FASTCALL *fgChild_SetParent)(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent, fgFlag flag);
  AbsVec (FG_FASTCALL *ResolveVec)(const CVec* v, const AbsRect* last);
  void (FG_FASTCALL *ResolveRect)(const fgChild* self, AbsRect* out);
  void (FG_FASTCALL *ResolveRectCache)(AbsRect* BSS_RESTRICT r, const fgChild* elem, const AbsRect* BSS_RESTRICT last);
  char (FG_FASTCALL *CompareCRects)(const CRect* l, const CRect* r); // Returns 0 if both are the same or 1 otherwise
  void (FG_FASTCALL *MoveCRect)(AbsVec v, CRect* r);
  char (FG_FASTCALL *HitAbsRect)(const AbsRect* r, FABS x, FABS y);
  void (FG_FASTCALL *ToIntAbsRect)(const AbsRect* r, int target[4]);
  void (FG_FASTCALL *ToLongAbsRect)(const AbsRect* r, long target[4]);
  char (FG_FASTCALL *MsgHitAbsRect)(const FG_Msg* msg, const AbsRect* r);
  char (FG_FASTCALL *MsgHitCRect)(const FG_Msg* msg, const fgChild* child);
  void (FG_FASTCALL *LList_Remove)(fgChild* self, fgChild** root, fgChild** last);
  void (FG_FASTCALL *LList_Add)(fgChild* self, fgChild** root, fgChild** last, fgFlag flag);
  void (FG_FASTCALL *LList_Insert)(fgChild* self, fgChild* target, fgChild** last);
  void (FG_FASTCALL *LList_ChangeOrder)(fgChild* self, fgChild** root, fgChild** last, fgFlag flag);
  void (FG_FASTCALL *VirtualFreeChild)(fgChild* self);
  void (FG_FASTCALL *fgWindow_Init)(fgWindow* BSS_RESTRICT self, fgWindow* BSS_RESTRICT parent, const fgElement* element, FG_UINT id, fgFlag flags);
  void (FG_FASTCALL *fgWindow_Destroy)(fgWindow* self);
  char (FG_FASTCALL *fgWindow_Message)(fgWindow* self, const FG_Msg* msg);
  void (FG_FASTCALL *fgWindow_SetElement)(fgWindow* self, const fgElement* element);
  void (FG_FASTCALL *fgWindow_SetArea)(fgWindow* self, const CRect* area);
  char (FG_FASTCALL *fgWindow_BasicMessage)(fgWindow* self, unsigned char type); // Shortcut for sending type messages with no data
  char (FG_FASTCALL *fgWindow_VoidMessage)(fgWindow* self, unsigned char type, void* data); // Shortcut for sending void* messages
  char (FG_FASTCALL *fgWindow_VoidAuxMessage)(fgWindow* self, unsigned char type, void* data, int aux);
  char (FG_FASTCALL *fgWindow_IntMessage)(fgWindow* self, unsigned char type, int data); // Shortcut for sending int messages
  void (FG_FASTCALL *fgWindow_SetParent)(fgWindow* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent);
  char (FG_FASTCALL *fgWindow_HoverProcess)(fgWindow* self, const FG_Msg* msg);

  void (FG_FASTCALL *fgStatic_Init)(fgStatic* self);
  void (FG_FASTCALL *fgStatic_Destroy)(fgStatic* self);
  void (FG_FASTCALL *fgStatic_Message)(fgStatic* self, unsigned char type, void* arg, int other);
  void (FG_FASTCALL *fgStatic_RemoveParent)(fgStatic* self);
  void (FG_FASTCALL *fgStatic_NotifyParent)(fgStatic* self);
  void (FG_FASTCALL *fgStatic_Clone)(fgStatic* self, fgStatic* from); // Clones information and all the children from "from" to "self"
  void (FG_FASTCALL *fgStatic_SetParent)(fgStatic* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent);

  fgStatic* (FG_FASTCALL *fgLoadImage)(const char* path);
  fgStatic* (FG_FASTCALL *fgLoadImageData)(const void* data, size_t length, fgFlag flags);
  void* (FG_FASTCALL *fgLoadImageDef)(fgFlag flags, const char* data, size_t length, unsigned int color, const CRect* uv);
  fgStatic* (FG_FASTCALL *fgLoadText)(const char* text, fgFlag flags, const char* font, unsigned short fontsize, unsigned short lineheight);
  void* (FG_FASTCALL *fgLoadTextDef)(fgFlag flags, const char* text, const char* font, unsigned short fontsize, unsigned short lineheight, unsigned int color);
  fgStatic* (FG_FASTCALL *fgLoadDefaultText)(const char* text);
  fgStatic* (FG_FASTCALL *fgEmptyStatic)(fgFlag flags);
  fgStatic* (FG_FASTCALL *fgLoadDef)(void* def, const fgElement* element, int order);
  void (FG_FASTCALL *fgDestroyDef)(void* def);

  fgRoot* (FG_FASTCALL *fgInitialize)();
  fgRoot* (FG_FASTCALL *fgSingleton)();
  void (FG_FASTCALL *fgTerminate)(fgRoot* root);
  void (FG_FASTCALL *fgRoot_Init)(fgRoot* self);
  void (FG_FASTCALL *fgRoot_Destroy)(fgRoot* self);
  char (FG_FASTCALL *fgRoot_Message)(fgWindow* self, const FG_Msg* msg);
  char (FG_FASTCALL *fgRoot_Inject)(fgRoot* self, const FG_Msg* msg); // Returns 0 if handled, 1 otherwise
  char (FG_FASTCALL *fgRoot_BehaviorDefault)(fgWindow* self, const FG_Msg* msg);
  char (FG_FASTCALL *fgRoot_CallBehavior)(fgWindow* self, const FG_Msg* msg); // Calls the appropriate fgroot behavior function
  char (FG_FASTCALL *fgRoot_KeyMsgHook)(const FG_Msg* msg);
  void (FG_FASTCALL *fgRoot_Render)(fgRoot* self);
  void (FG_FASTCALL *fgRoot_RListRender)(fgStatic* self, AbsRect* area);
  void (FG_FASTCALL *fgRoot_WinRender)(fgWindow* self, AbsRect* area);
  void (FG_FASTCALL *fgRoot_Update)(fgRoot* self, double delta);
  fgDeferAction* (FG_FASTCALL *fgRoot_AllocAction)(char (FG_FASTCALL *action)(void*), void* arg, double time);
  void (FG_FASTCALL *fgRoot_DeallocAction)(fgRoot* self, fgDeferAction* action); // Removes action from the list if necessary
  void (FG_FASTCALL *fgRoot_AddAction)(fgRoot* self, fgDeferAction* action); // Adds an action. Action can't already be in list.
  void (FG_FASTCALL *fgRoot_RemoveAction)(fgRoot* self, fgDeferAction* action); // Removes an action. Action must be in list.
  void (FG_FASTCALL *fgRoot_ModifyAction)(fgRoot* self, fgDeferAction* action); // Moves action if it needs to be moved, or inserts it if it isn't already in the list.

  fgWindow* (FG_FASTCALL *fgButton_Create)(fgStatic* item, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
  void (FG_FASTCALL *fgButton_Init)(fgButton* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
  char (FG_FASTCALL *fgButton_Message)(fgButton* self, const FG_Msg* msg);
  void (FG_FASTCALL *fgGrid_Init)(fgGrid* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
  void (FG_FASTCALL *fgGrid_Destroy)(fgGrid* self);
  char (FG_FASTCALL *fgGrid_Message)(fgGrid* self, const FG_Msg* msg);
  fgWindow* (FG_FASTCALL *fgList_Create)(fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
  void (FG_FASTCALL *fgList_Init)(fgList* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
  char (FG_FASTCALL *fgList_Message)(fgList* self, const FG_Msg* msg);
  fgWindow* (FG_FASTCALL *fgMenu_Create)(fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
  void (FG_FASTCALL *fgMenu_Init)(fgMenu* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
  void (FG_FASTCALL *fgMenu_Destroy)(fgMenu* self);
  fgWindow* (FG_FASTCALL *fgTopWindow_Create)(const char* caption, const fgElement* element, FG_UINT id, fgFlag flags);
  void (FG_FASTCALL *fgTopWindow_Init)(fgTopWindow* self, const fgElement* element, FG_UINT id, fgFlag flags);
  void (FG_FASTCALL *fgTopWindow_Destroy)(fgTopWindow* self);

  struct __kh_skin_t* (FG_FASTCALL *fgSkin_Create)();
  void (FG_FASTCALL *fgSkin_Destroy)(struct __kh_skin_t* self);
  void (FG_FASTCALL *fgSkin_Insert)(struct __kh_skin_t* self, struct FG_WINDOWSKIN* skin, const char* id);
  struct FG_WINDOWSKIN* (FG_FASTCALL *fgSkin_Get)(struct __kh_skin_t* self, const char* id);
  struct FG_WINDOWSKIN* (FG_FASTCALL *fgSkin_Remove)(struct __kh_skin_t* self, const char* id);
  char (FG_FASTCALL *fgSkin_Apply)(struct __kh_skin_t* self, fgWindow* p);
} fgImplementation;

FG_EXTERN void FG_FASTCALL fgLoadImplementation(fgImplementation* fg);
FG_EXTERN char FG_FASTCALL fgLoadExtension(void* fg, const char* extname);

#ifdef  __cplusplus
}
#endif

#endif
