// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_ELEMENT_H__
#define _FG_ELEMENT_H__

#include "feathergui.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGELEMENT_FLAGS
{
  FGELEMENT_BACKGROUND = (1 << 0), // "Background" children do not have padding applied to them. They are still sorted along with all non-background elements, however.
  FGELEMENT_NOCLIP = (1 << 1), // By default, children are clipped by their parents. When set on a control, it will not be clipped by its parent.
  FGELEMENT_IGNORE = (1 << 2), // When this flag is set, no mouse events will be sent to the control.
  FGELEMENT_HIDDEN = (1 << 3), // Signals that this control and it's children should not be rendered. However, they will still be laid out as if they existed.
  FGELEMENT_EXPANDX = (1 << 4), // Signals to the layout that the control should expand to include all it's elements
  FGELEMENT_EXPANDY = (1 << 5),
  FGELEMENT_EXPAND = FGELEMENT_EXPANDX | FGELEMENT_EXPANDY,
  FGELEMENT_SNAPX = (1 << 6), // If true, the rect resolution will snap this element to the nearest pixel on this axis.
  FGELEMENT_SNAPY = (1 << 7),
  FGELEMENT_SNAP = FGELEMENT_SNAPX | FGELEMENT_SNAPY,
  FGELEMENT_LAYOUTRESIZE = 1, // Called when the element is resized
  FGELEMENT_LAYOUTADD = 2, // Called when any child is added that needs to have the layout applied to it.
  FGELEMENT_LAYOUTREMOVE = 3, // Called when any child is removed that needs to have the layout applied to it.
  FGELEMENT_LAYOUTMOVE = 4, // Called when any child is moved so the layout can adjust as necessary.
  FGELEMENT_LAYOUTREORDER = 5, // Called when any child is reordered
  FGELEMENT_LAYOUTRESET = 6, // Called when something invalidates the entire layout (like adding an EXPAND flag)
};

typedef void (FG_FASTCALL *FN_DESTROY)(void*);
typedef size_t(FG_FASTCALL *FN_MESSAGE)(void*, const FG_Msg*);
struct _FG_ELEMENT;
struct _FG_STYLE;
struct _FG_SKIN;
struct _FG_LAYOUT;
typedef fgDeclareVector(struct _FG_ELEMENT*, Element) fgVectorElement;
typedef struct _FG_ELEMENT* (*FN_MAPPING)(const char*, struct _FG_ELEMENT*, struct _FG_ELEMENT*, const char*, fgFlag, const fgTransform*);

typedef void(FG_FASTCALL *FN_LISTENER)(struct _FG_ELEMENT*, const FG_Msg*);

typedef struct _FG_ELEMENT {
  fgTransform transform;
  FN_DESTROY destroy;
  void (*free)(void* self); // pointer to deallocation function
  FN_MESSAGE message;
  struct _FG_ELEMENT* parent;
  struct _FG_ELEMENT* root; // children list root
  struct _FG_ELEMENT* last; // children list last
  struct _FG_ELEMENT* next;
  struct _FG_ELEMENT* prev;
  struct _FG_ELEMENT* rootclip; // children with clipping list root
  struct _FG_ELEMENT* lastclip; // children with clipping list last
  struct _FG_ELEMENT* rootnoclip; // children with clipping list root
  struct _FG_ELEMENT* lastnoclip; // children with clipping list last
  struct _FG_ELEMENT* nextclip;
  struct _FG_ELEMENT* prevclip;
  struct _FG_ELEMENT* lastfocus; // Stores the last child that had focus, if any. This never points to the child that CURRENTLY has focus, only to the child that HAD focus.
  AbsRect margin; // defines the amount of external margin.
  AbsRect padding; // Defines the amount of internal padding. Only affects children that DON'T have FGELEMENT_BACKGROUND set.
  fgFlag flags;
  const struct _FG_SKIN* skin; // skin reference
  fgVectorElement skinrefs; // Type: fgElement* - References to skin children or subcontrols.
  char* name; // Optional name used for mapping to skin collections
  FG_UINT style; // Set to -1 if no style has been assigned, in which case the style from its parent will be used.
  FG_UINT userid;
  void* userdata;

#ifdef  __cplusplus
  FG_DLLEXPORT void Construct();
  FG_DLLEXPORT void FG_FASTCALL Move(unsigned char subtype, struct _FG_ELEMENT* child, unsigned long long diff);
  FG_DLLEXPORT size_t FG_FASTCALL SetAlpha(float alpha);
  FG_DLLEXPORT size_t FG_FASTCALL SetArea(const CRect& area);
  FG_DLLEXPORT size_t FG_FASTCALL SetElement(const fgTransform& element);
  FG_DLLEXPORT void FG_FASTCALL SetFlag(fgFlag flag, bool value);
  FG_DLLEXPORT void FG_FASTCALL SetFlags(fgFlag flags);
  FG_DLLEXPORT size_t FG_FASTCALL SetMargin(const AbsRect& margin);
  FG_DLLEXPORT size_t FG_FASTCALL SetPadding(const AbsRect& padding);
  FG_DLLEXPORT void FG_FASTCALL SetParent(struct _FG_ELEMENT* parent, struct _FG_ELEMENT* prev);
  FG_DLLEXPORT  size_t FG_FASTCALL AddChild(struct _FG_ELEMENT* child);
  FG_DLLEXPORT size_t FG_FASTCALL RemoveChild(struct _FG_ELEMENT* child);
  FG_DLLEXPORT size_t FG_FASTCALL LayoutFunction(const FG_Msg& msg, const CRect& area);
  FG_DLLEXPORT void FG_FASTCALL LayoutChange(unsigned char subtype, struct _FG_ELEMENT* target, struct _FG_ELEMENT* old);
  FG_DLLEXPORT size_t FG_FASTCALL LayoutLoad(struct _FG_LAYOUT* layout, FN_MAPPING mapping);
  FG_DLLEXPORT size_t Drag(struct _FG_ELEMENT* target, const FG_Msg& msg);
  FG_DLLEXPORT size_t Dragging(int x, int y);
  FG_DLLEXPORT size_t Drop(struct _FG_ELEMENT* target);
  FG_DLLEXPORT void Draw(AbsRect* area, int dpi);
  FG_DLLEXPORT struct _FG_ELEMENT* FG_FASTCALL Clone(struct _FG_ELEMENT* from);
  FG_DLLEXPORT size_t FG_FASTCALL SetSkin(struct _FG_SKIN* skin, FN_MAPPING mapping);
  FG_DLLEXPORT struct _FG_SKIN* FG_FASTCALL GetSkin(struct _FG_ELEMENT* child);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(const char* name, FG_UINT mask);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(struct _FG_STYLE* style);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(FG_UINT index, FG_UINT mask);
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
  FG_DLLEXPORT struct _FG_ELEMENT* GetSelectedItem();
  FG_DLLEXPORT size_t GetState(ptrdiff_t aux);
  FG_DLLEXPORT float GetStatef(ptrdiff_t aux);
  FG_DLLEXPORT size_t SetState(ptrdiff_t state, size_t aux);
  FG_DLLEXPORT size_t SetStatef(float state, size_t aux);
  FG_DLLEXPORT size_t FG_FASTCALL SetResource(void* res);
  FG_DLLEXPORT size_t FG_FASTCALL SetUV(const CRect& uv);
  FG_DLLEXPORT size_t FG_FASTCALL SetColor(unsigned int color, int index);
  FG_DLLEXPORT size_t FG_FASTCALL SetOutline(float outline);
  FG_DLLEXPORT size_t FG_FASTCALL SetFont(void* font);
  FG_DLLEXPORT size_t FG_FASTCALL SetLineHeight(float lineheight);
  FG_DLLEXPORT size_t FG_FASTCALL SetLetterSpacing(float letterspacing);
  FG_DLLEXPORT size_t FG_FASTCALL SetText(const char* text);
  FG_DLLEXPORT void* GetResource();
  FG_DLLEXPORT const CRect* GetUV();
  FG_DLLEXPORT unsigned int FG_FASTCALL GetColor(int index);
  FG_DLLEXPORT float GetOutline();
  FG_DLLEXPORT void* GetFont();
  FG_DLLEXPORT float GetLineHeight();
  FG_DLLEXPORT float GetLetterSpacing();
  FG_DLLEXPORT const char* GetText();
  FG_DLLEXPORT void AddListener(unsigned short type, FN_LISTENER listener);
#endif
} fgElement;

FG_EXTERN void FG_FASTCALL fgElement_InternalSetup(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, void (FG_FASTCALL *destroy)(void*), size_t(FG_FASTCALL *message)(void*, const FG_Msg*));
FG_EXTERN void FG_FASTCALL fgElement_Init(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgElement_Destroy(fgElement* self);
FG_EXTERN size_t FG_FASTCALL fgElement_Message(fgElement* self, const FG_Msg* msg);
FG_EXTERN fgElement* FG_FASTCALL fgElement_GetChildUnderMouse(fgElement* self, int x, int y, AbsRect* cache);
FG_EXTERN void FG_FASTCALL fgElement_ClearListeners(fgElement* self);

FG_EXTERN size_t FG_FASTCALL fgIntMessage(fgElement* self, unsigned char type, ptrdiff_t data, size_t aux);
FG_EXTERN size_t FG_FASTCALL fgVoidMessage(fgElement* self, unsigned char type, void* data, ptrdiff_t aux);
FG_EXTERN size_t FG_FASTCALL fgPassMessage(fgElement* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgSubMessage(fgElement* self, unsigned char type, unsigned char subtype, void* data, ptrdiff_t aux);

FG_EXTERN void FG_FASTCALL ResolveRect(const fgElement* self, AbsRect* out);
FG_EXTERN void FG_FASTCALL ResolveRectCache(const fgElement* self, AbsRect* BSS_RESTRICT out, const AbsRect* BSS_RESTRICT last, const AbsRect* BSS_RESTRICT padding);
FG_EXTERN char FG_FASTCALL MsgHitCRect(const FG_Msg* msg, const fgElement* child);
FG_EXTERN void FG_FASTCALL LList_RemoveAll(fgElement* self);
FG_EXTERN void FG_FASTCALL LList_InsertAll(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT next);
FG_EXTERN void FG_FASTCALL VirtualFreeChild(fgElement* self);
FG_EXTERN void FG_FASTCALL fgElement_Clear(fgElement* self);
FG_EXTERN void FG_FASTCALL fgElement_MouseMoveCheck(fgElement* self);
FG_EXTERN void FG_FASTCALL fgElement_AddListener(fgElement* self, unsigned short type, FN_LISTENER listener);

#ifdef  __cplusplus
}
#endif

#endif