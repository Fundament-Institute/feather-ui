// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_ELEMENT_H__
#define __FG_ELEMENT_H__

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

typedef void (FG_FASTCALL *fgDestroy)(void*);
typedef size_t(FG_FASTCALL *fgMessage)(void*, const FG_Msg*);
struct _FG_ELEMENT;
struct _FG_STYLE;
struct _FG_SKIN;
struct _FG_LAYOUT;
struct __kh_fgUserdata_t;
typedef fgDeclareVector(struct _FG_ELEMENT*, Element) fgVectorElement;

typedef void(FG_FASTCALL *fgListener)(struct _FG_ELEMENT*, const FG_Msg*);

// Defines the base GUI element
typedef struct _FG_ELEMENT {
  fgTransform transform;
  fgDestroy destroy;
  void (*free)(void* self); // pointer to deallocation function
  fgMessage message;
  struct _FG_ELEMENT* parent;
  struct _FG_ELEMENT* root; // children list root
  struct _FG_ELEMENT* last; // children list last
  struct _FG_ELEMENT* next;
  struct _FG_ELEMENT* prev;
  struct _FG_ELEMENT* rootinject; // children without FGELEMENT_IGNORE
  struct _FG_ELEMENT* lastinject;
  struct _FG_ELEMENT* nextinject;
  struct _FG_ELEMENT* previnject;
  struct _FG_ELEMENT* rootnoclip; // children with FGELEMENT_NOCLIP
  struct _FG_ELEMENT* lastnoclip;
  struct _FG_ELEMENT* nextnoclip;
  struct _FG_ELEMENT* prevnoclip;
  struct _FG_ELEMENT* lastfocus; // Stores the last child that had focus, if any. This never points to the child that CURRENTLY has focus, only to the child that HAD focus.
  AbsRect margin; // defines the amount of external margin.
  AbsRect padding; // Defines the amount of internal padding. Only affects children that DON'T have FGELEMENT_BACKGROUND set.
  AbsVec maxdim;
  AbsVec mindim;
  fgFlag flags;
  const struct _FG_SKIN* skin; // skin reference
  fgVectorElement skinrefs; // Type: fgElement* - References to skin children or subcontrols.
  char* name; // Optional name used for mapping to skin collections
  FG_UINT style; // Set to -1 if no style has been assigned, in which case the style from its parent will be used.
  FG_UINT userid;
  void* userdata;
  struct __kh_fgUserdata_t* userhash;

#ifdef  __cplusplus
  FG_DLLEXPORT void Construct();
  FG_DLLEXPORT void FG_FASTCALL Move(unsigned char subtype, struct _FG_ELEMENT* child, size_t diff);
  FG_DLLEXPORT size_t FG_FASTCALL SetAlpha(float alpha);
  FG_DLLEXPORT size_t FG_FASTCALL SetArea(const CRect& area);
  FG_DLLEXPORT size_t FG_FASTCALL SetTransform(const fgTransform& transform);
  FG_DLLEXPORT void FG_FASTCALL SetFlag(fgFlag flag, bool value);
  FG_DLLEXPORT void FG_FASTCALL SetFlags(fgFlag flags);
  FG_DLLEXPORT size_t FG_FASTCALL SetMargin(const AbsRect& margin);
  FG_DLLEXPORT size_t FG_FASTCALL SetPadding(const AbsRect& padding);
  FG_DLLEXPORT void FG_FASTCALL SetParent(struct _FG_ELEMENT* parent, struct _FG_ELEMENT* next = 0);
  FG_DLLEXPORT size_t FG_FASTCALL AddChild(struct _FG_ELEMENT* child, struct _FG_ELEMENT* next = 0);
  FG_DLLEXPORT struct _FG_ELEMENT* FG_FASTCALL AddItem(void* item);
  FG_DLLEXPORT struct _FG_ELEMENT* FG_FASTCALL AddItemText(const char* item);
  FG_DLLEXPORT struct _FG_ELEMENT* FG_FASTCALL AddItemElement(struct _FG_ELEMENT* item);
  FG_DLLEXPORT size_t FG_FASTCALL RemoveChild(struct _FG_ELEMENT* child);
  FG_DLLEXPORT void FG_FASTCALL LayoutChange(unsigned char subtype, struct _FG_ELEMENT* target, struct _FG_ELEMENT* old);
  FG_DLLEXPORT size_t FG_FASTCALL LayoutFunction(const FG_Msg& msg, const CRect& area, bool scrollbar = false);
  FG_DLLEXPORT size_t FG_FASTCALL LayoutLoad(struct _FG_LAYOUT* layout);
  FG_DLLEXPORT size_t DragOver(int x, int y);
  FG_DLLEXPORT size_t Drop(int x, int y, unsigned char allbtn);
  FG_DLLEXPORT void Draw(AbsRect* area, int dpi);
  FG_DLLEXPORT size_t FG_FASTCALL Inject(const FG_Msg* msg, const AbsRect* area);
  FG_DLLEXPORT struct _FG_ELEMENT* FG_FASTCALL Clone(struct _FG_ELEMENT* from);
  FG_DLLEXPORT size_t FG_FASTCALL SetSkin(struct _FG_SKIN* skin);
  FG_DLLEXPORT struct _FG_SKIN* FG_FASTCALL GetSkin(struct _FG_ELEMENT* child = 0);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(const char* name, FG_UINT mask);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(struct _FG_STYLE* style);
  FG_DLLEXPORT size_t FG_FASTCALL SetStyle(FG_UINT index, FG_UINT mask);
  FG_DLLEXPORT struct _FG_STYLE* GetStyle();
  FG_DLLEXPORT size_t FG_FASTCALL GetDPI();
  FG_DLLEXPORT void FG_FASTCALL SetDPI(int dpi);
  FG_DLLEXPORT const char* GetClassName();
  FG_DLLEXPORT void* FG_FASTCALL GetUserdata(const char* name = 0);
  FG_DLLEXPORT void FG_FASTCALL SetUserdata(void* data, const char* name = 0);
  FG_DLLEXPORT size_t FG_FASTCALL MouseDown(int x, int y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t FG_FASTCALL MouseDblClick(int x, int y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t FG_FASTCALL MouseUp(int x, int y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t FG_FASTCALL MouseOn(int x, int y);
  FG_DLLEXPORT size_t FG_FASTCALL MouseOff(int x, int y);
  FG_DLLEXPORT size_t FG_FASTCALL MouseMove(int x, int y);
  FG_DLLEXPORT size_t FG_FASTCALL MouseScroll(int x, int y, unsigned short delta, unsigned short hdelta);
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
  FG_DLLEXPORT void FG_FASTCALL SetDim(float x, float y, FGDIM type = FGDIM_MAX);
  FG_DLLEXPORT const AbsVec* GetDim(FGDIM type = FGDIM_MAX);
  FG_DLLEXPORT struct _FG_ELEMENT* GetSelectedItem(ptrdiff_t index = 0);
  FG_DLLEXPORT size_t GetValue(ptrdiff_t aux = 0);
  FG_DLLEXPORT float GetValueF(ptrdiff_t aux = 0);
  FG_DLLEXPORT void* GetValueP(ptrdiff_t aux = 0);
  FG_DLLEXPORT size_t SetValue(ptrdiff_t value, size_t aux = 0);
  FG_DLLEXPORT size_t SetValueF(float value, size_t aux = 0);
  FG_DLLEXPORT size_t SetValueP(void* ptr, size_t aux = 0);
  FG_DLLEXPORT size_t FG_FASTCALL SetResource(void* res);
  FG_DLLEXPORT size_t FG_FASTCALL SetUV(const CRect& uv);
  FG_DLLEXPORT size_t FG_FASTCALL SetColor(unsigned int color, FGSETCOLOR index);
  FG_DLLEXPORT size_t FG_FASTCALL SetOutline(float outline);
  FG_DLLEXPORT size_t FG_FASTCALL SetFont(void* font);
  FG_DLLEXPORT size_t FG_FASTCALL SetLineHeight(float lineheight);
  FG_DLLEXPORT size_t FG_FASTCALL SetLetterSpacing(float letterspacing);
  FG_DLLEXPORT size_t FG_FASTCALL SetText(const char* text, FGSETTEXT mode = FGSETTEXT_UTF8);
  FG_DLLEXPORT void* GetResource();
  FG_DLLEXPORT const CRect* GetUV();
  FG_DLLEXPORT unsigned int FG_FASTCALL GetColor(FGSETCOLOR index);
  FG_DLLEXPORT float GetOutline();
  FG_DLLEXPORT void* GetFont();
  FG_DLLEXPORT float GetLineHeight();
  FG_DLLEXPORT float GetLetterSpacing();
  FG_DLLEXPORT const int* GetText(FGSETTEXT mode = FGSETTEXT_UTF8);
  FG_DLLEXPORT void AddListener(unsigned short type, fgListener listener);
#endif
} fgElement;

FG_EXTERN void FG_FASTCALL fgElement_InternalSetup(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, void (FG_FASTCALL *destroy)(void*), size_t(FG_FASTCALL *message)(void*, const FG_Msg*));
FG_EXTERN void FG_FASTCALL fgElement_Init(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgElement_Destroy(fgElement* self);
FG_EXTERN size_t FG_FASTCALL fgElement_Message(fgElement* self, const FG_Msg* msg);
FG_EXTERN fgElement* FG_FASTCALL fgElement_GetChildUnderMouse(fgElement* self, int x, int y, AbsRect* cache);
FG_EXTERN void FG_FASTCALL fgElement_ClearListeners(fgElement* self);
FG_EXTERN size_t FG_FASTCALL fgElement_CheckLastFocus(fgElement* self);
FG_EXTERN fgElement* FG_FASTCALL fgCreate(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN short FG_FASTCALL fgMessageMap(const char* name);

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
FG_EXTERN void FG_FASTCALL fgElement_AddListener(fgElement* self, unsigned short type, fgListener listener);

#ifdef  __cplusplus
}
#endif

#endif