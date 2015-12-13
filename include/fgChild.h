// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_CHILD_H__
#define __FG_CHILD_H__

#include "feathergui.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGCHILD_FLAGS
{
  FGCHILD_BACKGROUND = (1 << 0), // "Background" children do not have padding applied to them. They are still sorted along with all non-background elements, however.
  FGCHILD_NOCLIP = (1 << 1), // By default, children are clipped by their parents. When set on a control, it will not be clipped by its parent.
  FGCHILD_IGNORE = (1 << 2), // When this flag is set, no mouse events will be sent to the control.
  FGCHILD_HIDDEN = (1 << 3), // Signals that this control and it's children should not be rendered. However, they will still be laid out as if they existed.
  FGCHILD_EXPANDX = (1 << 4), // Signals to the layout that the control should expand to include all it's elements
  FGCHILD_EXPANDY = (1 << 5),
  FGCHILD_EXPAND = FGCHILD_EXPANDX | FGCHILD_EXPANDY,
};

typedef void (FG_FASTCALL *FN_DESTROY)(void*);
typedef size_t(FG_FASTCALL *FN_MESSAGE)(void*, const FG_Msg*);

typedef struct _FG_CHILD {
  fgElement element;
  FN_DESTROY destroy;
  void (*free)(void* self); // pointer to deallocation function
  FN_MESSAGE message;
  struct _FG_CHILD* parent;
  struct _FG_CHILD* root; // children list root
  struct _FG_CHILD* last; // children list last
  struct _FG_CHILD* next;
  struct _FG_CHILD* prev;
  struct _FG_CHILD* rootclip; // children with clipping list root
  struct _FG_CHILD* lastclip; // children with clipping list last
  struct _FG_CHILD* rootnoclip; // children with clipping list root
  struct _FG_CHILD* lastnoclip; // children with clipping list last
  struct _FG_CHILD* nextclip;
  struct _FG_CHILD* prevclip;
  struct _FG_CHILD* lastfocus; // Stores the last child that had focus, if any. This never points to the child that CURRENTLY has focus, only to the child that HAD focus.
  AbsRect margin; // defines the amount of external margin.
  AbsRect padding; // Defines the amount of internal padding. Only affects children that DON'T have FGCHILD_BACKGROUND set.
  int order; // order relative to other windows or statics
  fgFlag flags;
  const struct __FG_SKIN* skin; // skin reference
  fgVector skinrefs; // Type: fgChild* - References to skin children or subcontrols.
  int prechild; // stores where the last preallocated subcontrol reference is in skinrefs.
  FG_UINT style; // Set to -1 if no style has been assigned, in which case the style from its parent will be used.
  FG_UINT userid;
  void* userdata;
} fgChild;

FG_EXTERN void FG_FASTCALL fgChild_InternalSetup(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element, void (FG_FASTCALL *destroy)(void*), size_t(FG_FASTCALL *message)(void*, const FG_Msg*));
FG_EXTERN void FG_FASTCALL fgChild_Init(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgChild_Destroy(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_SetParent(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent);
FG_EXTERN size_t FG_FASTCALL fgChild_Message(fgChild* self, const FG_Msg* msg);

FG_EXTERN size_t FG_FASTCALL fgLayout_Default(fgChild* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgLayout_Distribute(fgChild* self, const FG_Msg* msg, char axis);
FG_EXTERN size_t FG_FASTCALL fgLayout_Tile(fgChild* self, const FG_Msg* msg, char axes);
FG_EXTERN size_t FG_FASTCALL fgChild_IntMessage(fgChild* self, unsigned char type, ptrdiff_t data, ptrdiff_t aux);
FG_EXTERN size_t FG_FASTCALL fgChild_VoidAuxMessage(fgChild* self, unsigned char type, void* data, ptrdiff_t aux);
FG_EXTERN size_t FG_FASTCALL fgChild_VoidMessage(fgChild* self, unsigned char type, void* data);
FG_EXTERN size_t FG_FASTCALL fgChild_PassMessage(fgChild* self, const FG_Msg* msg);

FG_EXTERN void FG_FASTCALL ResolveRect(const fgChild* self, AbsRect* out);
FG_EXTERN void FG_FASTCALL ResolveRectCache(AbsRect* BSS_RESTRICT r, const fgChild* elem, const AbsRect* BSS_RESTRICT last);
//FG_EXTERN char FG_FASTCALL CompChildOrder(const fgChild* l, const fgChild* r);
FG_EXTERN char FG_FASTCALL MsgHitCRect(const FG_Msg* msg, const fgChild* child);
FG_EXTERN void FG_FASTCALL LList_RemoveAll(fgChild* self);
FG_EXTERN void FG_FASTCALL LList_AddAll(fgChild* self);
FG_EXTERN char FG_FASTCALL LList_ChangeOrderAll(fgChild* self);
FG_EXTERN void FG_FASTCALL VirtualFreeChild(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_Clear(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_AddPreChild(fgChild* self, fgChild* child);

#ifdef  __cplusplus
}
#endif

#endif