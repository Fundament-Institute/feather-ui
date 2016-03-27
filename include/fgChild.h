// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_CHILD_H__
#define _FG_CHILD_H__

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
  FGCHILD_LAYOUTRESIZE = 1, // Called when the element is resized
  FGCHILD_LAYOUTADD = 2, // Called when any child is added that needs to have the layout applied to it.
  FGCHILD_LAYOUTREMOVE = 3, // Called when any child is removed that needs to have the layout applied to it.
  FGCHILD_LAYOUTMOVE = 4, // Called when any child is moved so the layout can adjust as necessary.
  FGCHILD_LAYOUTREORDER = 5, // Called when any child is reordered
  FGCHILD_LAYOUTRESET = 6, // Called when something invalidates the entire layout (like adding an EXPAND flag)
};

typedef void (FG_FASTCALL *FN_DESTROY)(void*);
typedef size_t(FG_FASTCALL *FN_MESSAGE)(void*, const FG_Msg*);
struct _FG_CHILD;
struct _FG_STYLE;
struct _FG_SKIN;
struct _FG_LAYOUT;
typedef fgDeclareVector(struct _FG_CHILD*, Child) fgVectorChild;
typedef struct _FG_CHILD* (*FN_MAPPING)(const char*, fgFlag, struct _FG_CHILD*, struct _FG_CHILD*, const fgElement*);
typedef void(FG_FASTCALL *FN_LISTENER)(struct _FG_CHILD*, const FG_Msg*);

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
  int index; // Internal index used for layout ordering or vector mapping.
  fgFlag flags;
  const struct _FG_SKIN* skin; // skin reference
  fgVectorChild skinrefs; // Type: fgChild* - References to skin children or subcontrols.
  int prechild; // stores where the last preallocated subcontrol reference is in skinrefs.
  FG_UINT style; // Set to -1 if no style has been assigned, in which case the style from its parent will be used.
  FG_UINT userid;
  void* userdata;

#ifdef  __cplusplus
  FG_DLLEXPORT void Construct();
  FG_DLLEXPORT void FG_FASTCALL Move(unsigned char subtype, struct _FG_CHILD* child, unsigned long long diff);
  FG_DLLEXPORT size_t FG_FASTCALL SetAlpha(float alpha);
  FG_DLLEXPORT size_t FG_FASTCALL SetArea(const CRect& area);
  FG_DLLEXPORT size_t FG_FASTCALL SetElement(const fgElement& element);
  FG_DLLEXPORT void FG_FASTCALL SetFlag(fgFlag flag, bool value);
  FG_DLLEXPORT void FG_FASTCALL SetFlags(fgFlag flags);
  FG_DLLEXPORT size_t FG_FASTCALL SetMargin(const AbsRect& margin);
  FG_DLLEXPORT size_t FG_FASTCALL SetPadding(const AbsRect& padding);
  FG_DLLEXPORT void FG_FASTCALL SetParent(struct _FG_CHILD* parent, struct _FG_CHILD* prev);
  FG_DLLEXPORT  size_t FG_FASTCALL AddChild(struct _FG_CHILD* child);
  FG_DLLEXPORT size_t FG_FASTCALL RemoveChild(struct _FG_CHILD* child);
  FG_DLLEXPORT size_t FG_FASTCALL LayoutFunction(const FG_Msg& msg, const CRect& area);
  FG_DLLEXPORT void FG_FASTCALL LayoutChange(unsigned char subtype, struct _FG_CHILD* target, struct _FG_CHILD* old);
  FG_DLLEXPORT ptrdiff_t FG_FASTCALL LayoutIndex(struct _FG_CHILD* child);
  FG_DLLEXPORT size_t FG_FASTCALL LayoutLoad(struct _FG_LAYOUT* layout, FN_MAPPING mapping);
  FG_DLLEXPORT size_t Drag(struct _FG_CHILD* target, const FG_Msg& msg);
  FG_DLLEXPORT size_t Dragging(int x, int y);
  FG_DLLEXPORT size_t Drop(struct _FG_CHILD* target);
  FG_DLLEXPORT void Draw(AbsRect* area);
  FG_DLLEXPORT struct _FG_CHILD* FG_FASTCALL Clone(struct _FG_CHILD* from);
  FG_DLLEXPORT size_t FG_FASTCALL SetSkin(struct _FG_SKIN* skin, FN_MAPPING mapping);
  FG_DLLEXPORT struct _FG_SKIN* FG_FASTCALL GetSkin(struct _FG_CHILD* child);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(const char* name);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(struct _FG_STYLE* style);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(size_t index);
  FG_DLLEXPORT struct _FG_STYLE* GetStyle();
  FG_DLLEXPORT const char* GetClassName();
  FG_DLLEXPORT size_t FG_FASTCALL MouseDown(int x, int y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t FG_FASTCALL MouseDblClick(int x, int y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t FG_FASTCALL MouseUp(int x, int y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t FG_FASTCALL MouseOn(int x, int y);
  FG_DLLEXPORT size_t FG_FASTCALL MouseOff(int x, int y);
  FG_DLLEXPORT size_t FG_FASTCALL MouseMove(int x, int y);
  FG_DLLEXPORT size_t FG_FASTCALL MouseScroll(int x, int y, unsigned short delta);
  FG_DLLEXPORT size_t FG_FASTCALL MouseLeave(int x, int y);
  FG_DLLEXPORT size_t FG_FASTCALL KeyUp(unsigned char keycode, char sigkeys);
  FG_DLLEXPORT size_t FG_FASTCALL KeyDown(unsigned char keycode, char sigkeys);
  FG_DLLEXPORT size_t FG_FASTCALL KeyChar(int keychar, char sigkeys);
  FG_DLLEXPORT size_t FG_FASTCALL JoyButtonDown(short joybutton);
  FG_DLLEXPORT size_t FG_FASTCALL JoyButtonUp(short joybutton);
  FG_DLLEXPORT size_t FG_FASTCALL JoyAxis(float joyvalue, short joyaxis);
  FG_DLLEXPORT size_t GotFocus();
  FG_DLLEXPORT void LostFocus();
  FG_DLLEXPORT size_t FG_FASTCALL SetName(const char* name);
  FG_DLLEXPORT const char* GetName();
  FG_DLLEXPORT void Nuetral();
  FG_DLLEXPORT void Hover();
  FG_DLLEXPORT void Active();
  FG_DLLEXPORT void Action();
  FG_DLLEXPORT struct _FG_CHILD* GetSelectedItem();
  FG_DLLEXPORT size_t FG_FASTCALL SetResource(void* res);
  FG_DLLEXPORT size_t FG_FASTCALL SetUV(const CRect& uv);
  FG_DLLEXPORT size_t FG_FASTCALL SetColor(unsigned int color, int index);
  FG_DLLEXPORT size_t FG_FASTCALL SetOutline(float outline);
  FG_DLLEXPORT size_t FG_FASTCALL SetFont(void* font);
  FG_DLLEXPORT size_t FG_FASTCALL SetText(const char* text);
  FG_DLLEXPORT void* GetResource();
  FG_DLLEXPORT const CRect* GetUV();
  FG_DLLEXPORT unsigned int FG_FASTCALL GetColor(int index);
  FG_DLLEXPORT float GetOutline();
  FG_DLLEXPORT void* GetFont();
  FG_DLLEXPORT const char* GetText();
  FG_DLLEXPORT void AddListener(unsigned short type, FN_LISTENER listener);
#endif
} fgChild;

FG_EXTERN void FG_FASTCALL fgChild_InternalSetup(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element, void (FG_FASTCALL *destroy)(void*), size_t(FG_FASTCALL *message)(void*, const FG_Msg*));
FG_EXTERN void FG_FASTCALL fgChild_Init(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgChild_Destroy(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_SetParent(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev);
FG_EXTERN size_t FG_FASTCALL fgChild_Message(fgChild* self, const FG_Msg* msg);

FG_EXTERN size_t FG_FASTCALL fgLayout_Default(fgChild* self, const FG_Msg* msg, CRect* area);
FG_EXTERN size_t FG_FASTCALL fgLayout_Distribute(fgChild* self, const FG_Msg* msg, char axis);
FG_EXTERN size_t FG_FASTCALL fgLayout_Tile(fgChild* self, const FG_Msg* msg, char axes);
FG_EXTERN size_t FG_FASTCALL fgChild_IntMessage(fgChild* self, unsigned char type, ptrdiff_t data, size_t aux);
FG_EXTERN size_t FG_FASTCALL fgChild_VoidMessage(fgChild* self, unsigned char type, void* data, ptrdiff_t aux);
FG_EXTERN size_t FG_FASTCALL fgChild_PassMessage(fgChild* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgChild_SubMessage(fgChild* self, unsigned char type, unsigned char subtype, void* data, ptrdiff_t aux);
FG_EXTERN fgChild* FG_FASTCALL fgChild_GetChildUnderMouse(fgChild* self, int x, int y, AbsRect* cache);
FG_EXTERN void FG_FASTCALL fgChild_ClearListeners(fgChild* self);

FG_EXTERN void FG_FASTCALL ResolveRect(const fgChild* self, AbsRect* out);
FG_EXTERN void FG_FASTCALL ResolveRectCache(const fgChild* self, AbsRect* BSS_RESTRICT out, const AbsRect* BSS_RESTRICT last, const AbsRect* BSS_RESTRICT padding);
FG_EXTERN char FG_FASTCALL MsgHitCRect(const FG_Msg* msg, const fgChild* child);
FG_EXTERN void FG_FASTCALL LList_RemoveAll(fgChild* self);
FG_EXTERN void FG_FASTCALL LList_InsertAll(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT prev);
FG_EXTERN void FG_FASTCALL VirtualFreeChild(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_Clear(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_AddPreChild(fgChild* self, fgChild* child);
FG_EXTERN void FG_FASTCALL fgChild_MouseMoveCheck(fgChild* self);
FG_EXTERN void FG_FASTCALL fgChild_AddListener(fgChild* self, unsigned short type, FN_LISTENER listener);

#ifdef  __cplusplus
}
#endif

#endif