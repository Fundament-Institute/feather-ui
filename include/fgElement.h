// Copyright ©2017 Black Sphere Studios
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
  FGELEMENT_SNAPX = (1 << 6), // If true, any rendering this element performs will be snapped to the nearest pixel.
  FGELEMENT_SNAPY = (1 << 7),
  FGELEMENT_SNAP = FGELEMENT_SNAPX | FGELEMENT_SNAPY,
  FGELEMENT_LAYOUTRESIZE = 1, // Called when the element is resized
  FGELEMENT_LAYOUTADD = 2, // Called when any child is added that needs to have the layout applied to it.
  FGELEMENT_LAYOUTREMOVE = 3, // Called when any child is removed that needs to have the layout applied to it.
  FGELEMENT_LAYOUTMOVE = 4, // Called when any child is moved so the layout can adjust as necessary.
  FGELEMENT_LAYOUTREORDER = 5, // Called when any child is reordered
  FGELEMENT_LAYOUTRESET = 6, // Called when something invalidates the entire layout (like adding an EXPAND flag)
  FGELEMENT_USEDEFAULTS = (1 << ((sizeof(fgFlag)<<3) - 1)),
};

typedef void (*fgDestroy)(void*);
typedef size_t(*fgMessage)(void*, const FG_Msg*);
struct _FG_ELEMENT;
struct _FG_STYLE;
struct _FG_SKIN;
struct _FG_LAYOUT;
struct __kh_fgUserdata_t;
struct __kh_fgSkinElements_t;
typedef fgDeclareVector(struct _FG_ELEMENT*, Element) fgVectorElement;

typedef void(*fgListener)(struct _FG_ELEMENT*, const FG_Msg*);

// Defines the base GUI element
typedef struct _FG_ELEMENT {
  fgTransform transform;
  AbsRect margin; // defines the amount of external margin.
  AbsRect padding; // Defines the amount of internal padding. Only affects children that DON'T have FGELEMENT_BACKGROUND set.
  AbsVec maxdim;
  AbsVec mindim;
  AbsVec layoutdim;
  AbsVec scaling;
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
  fgFlag flags;
  const struct _FG_SKIN* skin; // skin reference
  struct __kh_fgSkinElements_t* skinelements; // child elements that are part of the skin
  char* name; // Optional name used for mapping to skin collections
  FG_UINT style; // Set to -1 if no style has been assigned, in which case the style from its parent will be used.
  FG_UINT userid;
  void* userdata;
  struct __kh_fgUserdata_t* userhash;

#ifdef  __cplusplus
  FG_DLLEXPORT void Construct();
  FG_DLLEXPORT void Move(unsigned short subtype, struct _FG_ELEMENT* child, size_t diff);
  FG_DLLEXPORT size_t SetAlpha(float alpha);
  FG_DLLEXPORT size_t SetArea(const CRect& area);
  FG_DLLEXPORT size_t SetTransform(const fgTransform& transform);
  FG_DLLEXPORT void SetFlag(fgFlag flag, bool value);
  FG_DLLEXPORT void SetFlags(fgFlag flags);
  FG_DLLEXPORT size_t SetMargin(const AbsRect& margin);
  FG_DLLEXPORT size_t SetPadding(const AbsRect& padding);
  FG_DLLEXPORT void SetParent(struct _FG_ELEMENT* parent, struct _FG_ELEMENT* next = 0);
  FG_DLLEXPORT size_t AddChild(struct _FG_ELEMENT* child, struct _FG_ELEMENT* next = 0);
  FG_DLLEXPORT struct _FG_ELEMENT* AddItem(void* item, size_t index = (size_t)~0);
  FG_DLLEXPORT struct _FG_ELEMENT* AddItemText(const char* item, FGTEXTFMT fmt = FGTEXTFMT_UTF8);
  FG_DLLEXPORT struct _FG_ELEMENT* AddItemElement(struct _FG_ELEMENT* item, size_t index = (size_t)~0);
  FG_DLLEXPORT size_t RemoveChild(struct _FG_ELEMENT* child);
  FG_DLLEXPORT size_t RemoveItem(size_t item);
  FG_DLLEXPORT void LayoutChange(unsigned short subtype, struct _FG_ELEMENT* target, struct _FG_ELEMENT* old);
  FG_DLLEXPORT size_t LayoutFunction(const FG_Msg& msg, const CRect& area, bool scrollbar = false);
  FG_DLLEXPORT size_t LayoutLoad(struct _FG_LAYOUT* layout);
  FG_DLLEXPORT size_t DragOver(float x, float y);
  FG_DLLEXPORT size_t Drop(float x, float y, unsigned char allbtn);
  FG_DLLEXPORT void Draw(const AbsRect* area, const fgDrawAuxData* aux);
  FG_DLLEXPORT size_t Inject(const FG_Msg* msg, const AbsRect* area);
  FG_DLLEXPORT size_t SetSkin(struct _FG_SKIN* skin);
  FG_DLLEXPORT struct _FG_SKIN* GetSkin(struct _FG_ELEMENT* child = 0);
  FG_DLLEXPORT size_t SetStyle(const char* name, FG_UINT mask);
  FG_DLLEXPORT size_t SetStyle(struct _FG_STYLE* style);
  FG_DLLEXPORT size_t SetStyle(FG_UINT index, FG_UINT mask);
  FG_DLLEXPORT struct _FG_STYLE* GetStyle();
  FG_DLLEXPORT fgIntVec& GetDPI();
  FG_DLLEXPORT void SetDPI(int x, int y);
  FG_DLLEXPORT const char* GetClassName();
  FG_DLLEXPORT void* GetUserdata(const char* name = 0);
  FG_DLLEXPORT void SetUserdata(void* data, const char* name = 0);
  FG_DLLEXPORT size_t MouseDown(float x, float y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t MouseDblClick(float x, float y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t MouseUp(float x, float y, unsigned char button, unsigned char allbtn);
  FG_DLLEXPORT size_t MouseOn(float x, float y);
  FG_DLLEXPORT size_t MouseOff(float x, float y);
  FG_DLLEXPORT size_t MouseMove(float x, float y);
  FG_DLLEXPORT size_t MouseScroll(float x, float y, unsigned short delta, unsigned short hdelta);
  FG_DLLEXPORT size_t KeyUp(unsigned char keycode, char sigkeys);
  FG_DLLEXPORT size_t KeyDown(unsigned char keycode, char sigkeys);
  FG_DLLEXPORT size_t KeyChar(int keychar, char sigkeys);
  FG_DLLEXPORT size_t JoyButtonDown(short joybutton);
  FG_DLLEXPORT size_t JoyButtonUp(short joybutton);
  FG_DLLEXPORT size_t JoyAxis(float joyvalue, short joyaxis);
  FG_DLLEXPORT size_t GotFocus();
  FG_DLLEXPORT void LostFocus();
  FG_DLLEXPORT size_t SetName(const char* name);
  FG_DLLEXPORT const char* GetName();
  FG_DLLEXPORT void SetContextMenu(struct _FG_ELEMENT* menu);
  FG_DLLEXPORT struct _FG_ELEMENT* GetContextMenu();
  FG_DLLEXPORT void Neutral();
  FG_DLLEXPORT void Hover();
  FG_DLLEXPORT void Active();
  FG_DLLEXPORT void Action();
  FG_DLLEXPORT void SetDim(float x, float y, FGDIM type = FGDIM_MAX);
  FG_DLLEXPORT const AbsVec* GetDim(FGDIM type = FGDIM_MAX);
  FG_DLLEXPORT struct _FG_ELEMENT* GetItem(size_t index);
  FG_DLLEXPORT struct _FG_ELEMENT* GetItemAt(float x, float y);
  FG_DLLEXPORT size_t GetNumItems();
  FG_DLLEXPORT struct _FG_ELEMENT* GetSelectedItem(size_t index = 0);
  FG_DLLEXPORT size_t GetValue(ptrdiff_t aux = 0);
  FG_DLLEXPORT float GetValueF(ptrdiff_t aux = 0);
  FG_DLLEXPORT void* GetValueP(ptrdiff_t aux = 0);
  FG_DLLEXPORT size_t SetValue(ptrdiff_t value, size_t aux = 0);
  FG_DLLEXPORT size_t SetValueF(float value, size_t aux = 0);
  FG_DLLEXPORT size_t SetValueP(void* ptr, size_t aux = 0);
  FG_DLLEXPORT size_t SetAsset(fgAsset asset);
  FG_DLLEXPORT size_t SetUV(const CRect& uv);
  FG_DLLEXPORT size_t SetColor(unsigned int color, FGSETCOLOR index);
  FG_DLLEXPORT size_t SetOutline(float outline);
  FG_DLLEXPORT size_t SetFont(void* font);
  FG_DLLEXPORT size_t SetLineHeight(float lineheight);
  FG_DLLEXPORT size_t SetLetterSpacing(float letterspacing);
  FG_DLLEXPORT size_t SetText(const char* text, FGTEXTFMT fmt = FGTEXTFMT_UTF8);
  FG_DLLEXPORT size_t SetTextW(const wchar_t* text);
  FG_DLLEXPORT size_t SetTextU(const int* text);
  FG_DLLEXPORT size_t SetPlaceholder(const char* text);
  FG_DLLEXPORT size_t SetPlaceholderW(const wchar_t* text);
  FG_DLLEXPORT size_t SetPlaceholderU(const int* text);
  FG_DLLEXPORT size_t SetMask(int mask);
  FG_DLLEXPORT fgAsset GetAsset();
  FG_DLLEXPORT const CRect* GetUV();
  FG_DLLEXPORT unsigned int GetColor(FGSETCOLOR index);
  FG_DLLEXPORT float GetOutline();
  FG_DLLEXPORT void* GetFont();
  FG_DLLEXPORT float GetLineHeight();
  FG_DLLEXPORT float GetLetterSpacing();
  FG_DLLEXPORT const char* GetText(FGTEXTFMT fmt = FGTEXTFMT_UTF8);
  FG_DLLEXPORT const wchar_t* GetTextW();
  FG_DLLEXPORT const int* GetTextU();
  FG_DLLEXPORT const char* GetPlaceholder();
  FG_DLLEXPORT const wchar_t* GetPlaceholderW();
  FG_DLLEXPORT const int* GetPlaceholderU();
  FG_DLLEXPORT int GetMask();
  FG_DLLEXPORT void AddListener(unsigned short type, fgListener listener);
#endif
} fgElement;

FG_EXTERN void fgElement_InternalSetup(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units, void (*destroy)(void*), size_t(*message)(void*, const FG_Msg*));
FG_EXTERN void fgElement_Init(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgElement_Destroy(fgElement* self);
FG_EXTERN size_t fgElement_Message(fgElement* self, const FG_Msg* msg);
FG_EXTERN fgElement* fgElement_GetChildUnderMouse(fgElement* self, float x, float y, AbsRect* cache);
FG_EXTERN void fgElement_ClearListeners(fgElement* self);
FG_EXTERN size_t fgElement_CheckLastFocus(fgElement* self);

FG_EXTERN size_t fgDimMessage(fgElement* self, unsigned short type, unsigned short subtype, float x, float y);
FG_EXTERN size_t fgFloatMessage(fgElement* self, unsigned short type, unsigned short subtype, float data, ptrdiff_t aux);
FG_EXTERN float fgGetFloatMessage(fgElement* self, unsigned short type, unsigned short subtype, ptrdiff_t aux);
FG_EXTERN size_t fgIntMessage(fgElement* self, unsigned short type, ptrdiff_t data, size_t aux);
FG_EXTERN size_t fgVoidMessage(fgElement* self, unsigned short type, void* data, ptrdiff_t aux);
FG_EXTERN size_t fgPassMessage(fgElement* self, const FG_Msg* msg);
FG_EXTERN size_t fgSubMessage(fgElement* self, unsigned short type, unsigned short subtype, void* data, ptrdiff_t aux);
FG_EXTERN void* fgGetPtrMessage(fgElement* self, unsigned short type, unsigned short subtype, size_t data, size_t aux);

// The outer (layout) rect does not have margins or padding applied. This is the rect used by the layout.
FG_EXTERN void ResolveOuterRect(const fgElement* self, AbsRect* out);
FG_EXTERN void ResolveOuterRectCache(const fgElement* self, AbsRect* BSS_RESTRICT out, const AbsRect* BSS_RESTRICT last, const AbsRect* BSS_RESTRICT padding);
// The standard (clipping) rect has margins applied. This is used by background elements, mouse injection and drawing.
FG_EXTERN void ResolveRect(const fgElement* self, AbsRect* out);
FG_EXTERN void ResolveRectCache(const fgElement* self, AbsRect* BSS_RESTRICT out, const AbsRect* BSS_RESTRICT last, const AbsRect* BSS_RESTRICT padding);
// The inner (child) rect has margins and padding applied. This is used when resolving foreground elements.
FG_EXTERN void ResolveInnerRect(const fgElement* self, AbsRect* out);
FG_EXTERN void GetInnerRect(const fgElement* self, AbsRect* inner, const AbsRect* standard);
// Gets the total area covered by the element's standard rect and all its nonclipping children's standard rects.
FG_EXTERN void ResolveNoClipRect(const fgElement* self, AbsRect* out);
// Tests if a MOUSEMOVE message (or other valid mouse message) hits the given element
FG_EXTERN char MsgHitElement(const FG_Msg* msg, const fgElement* element);
FG_EXTERN void VirtualFreeChild(fgElement* self);
FG_EXTERN void fgElement_Clear(fgElement* self);
FG_EXTERN void fgElement_MouseMoveCheck(fgElement* self);
FG_EXTERN void fgElement_AddListener(fgElement* self, unsigned short type, fgListener listener);

#ifdef  __cplusplus
}
#endif

#endif