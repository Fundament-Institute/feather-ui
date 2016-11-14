// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgBackend.h"
#include "fgRoot.h"
#include "fgResource.h"
#include "fgWindow.h"
#include "fgRadiobutton.h"
#include "fgProgressbar.h"
#include "fgSlider.h"
#include "fgTextbox.h"
#include "fgTreeview.h"
#include "fgDebug.h"
#include "fgList.h"
#include "fgCurve.h"
#include "fgDropdown.h"
#include "fgTabcontrol.h"
#include "fgMenu.h"
#include "feathercpp.h"
#include "bss-util/cTrie.h"

using namespace bss_util;

void* FG_FASTCALL fgCreateFontDefault(fgFlag flags, const char* font, unsigned int fontsize, unsigned int dpi) { return (void*)~0; }
void* FG_FASTCALL fgCopyFontDefault(void* font, unsigned int fontsize, unsigned int dpi) { return (void*)~0; }
void* FG_FASTCALL fgCloneFontDefault(void* font) { return (void*)~0; }
void FG_FASTCALL fgDestroyFontDefault(void* font) { }
void* FG_FASTCALL fgDrawFontDefault(void* font, const int* text, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, void* cache) { return 0; }
void FG_FASTCALL fgFontSizeDefault(void* font, const int* text, float lineheight, float letterspacing, AbsRect* area, fgFlag flags) { }
void FG_FASTCALL fgFontGetDefault(void* font, float* lineheight, unsigned int* size, unsigned int* dpi)
{
  if(lineheight) *lineheight = 0.0f;
  if(size) *size = 0;
  if(dpi) *dpi = 0;
}

void* FG_FASTCALL fgCreateResourceDefault(fgFlag flags, const char* data, size_t length) { return (void*)~0; }
void* FG_FASTCALL fgCloneResourceDefault(void* res) { return (void*)~0; }
void FG_FASTCALL fgDestroyResourceDefault(void* res) { }
void FG_FASTCALL fgDrawResourceDefault(void* res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags) {}
void FG_FASTCALL fgResourceSizeDefault(void* res, const CRect* uv, AbsVec* dim, fgFlag flags) { }

size_t FG_FASTCALL fgFontIndexDefault(void* font, const int* text, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, void* cache) { return 0; }
AbsVec FG_FASTCALL fgFontPosDefault(void* font, const int* text, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, void* cache) { AbsVec a = { 0,0 }; return a; }

void FG_FASTCALL fgDrawLinesDefault(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center) {}

AbsRect* clipstack = 0;
size_t clipcapacity = 0;
size_t clipnum = 0;

void FG_FASTCALL fgPushClipRectDefault(const AbsRect* clip)
{
  if(clipcapacity >= clipnum)
  {
    clipcapacity = clipcapacity * 2;
    clipstack = (AbsRect*)realloc(clipstack, sizeof(AbsRect)*clipcapacity);
  }
  clipstack[clipnum++] = *clip;
}
AbsRect FG_FASTCALL fgPeekClipRectDefault()
{
  static const AbsRect BLANK = { 0,0,0,0 };
  return !clipnum ? BLANK : clipstack[clipnum - 1];
}
void FG_FASTCALL fgPopClipRectDefault()
{
  --clipnum;
}

void FG_FASTCALL fgDragStartDefault(char type, void* data, fgElement* draw)
{
  fgroot_instance->dragtype = type;
  fgroot_instance->dragdata = data;
  fgroot_instance->dragdraw = draw;
  fgCaptureWindow = 0;
}

char FG_FASTCALL fgLoadExtensionDefault(const char* extname, void* fg, size_t sz) { return -1; }
void FG_FASTCALL fgSetCursorDefault(unsigned int type, void* custom) {}
void FG_FASTCALL fgClipboardCopyDefault(unsigned int type, const void* data, size_t length) {}
char FG_FASTCALL fgClipboardExistsDefault(unsigned int type) { return 0; }
const void* FG_FASTCALL fgClipboardPasteDefault(unsigned int type, size_t* length) { return 0; }
void FG_FASTCALL fgClipboardFreeDefault(const void* mem) {}
void FG_FASTCALL fgDirtyElementDefault(fgElement* elem) {}

template<class T, void (FG_FASTCALL *INIT)(T* BSS_RESTRICT, fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, const char*, fgFlag, const fgTransform*, unsigned short)>
fgElement* _create_default(fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  T* r = fgmalloc<T>(1, __FILE__, __LINE__);
  INIT(r, parent, next, name, flags, transform, units);
#ifdef BSS_DEBUG
  ((fgElement*)r)->free = &fgfreeblank;
#else
  ((fgElement*)r)->free = &free; // We do this because the compiler can't figure out the inlining weirdness going on here
#endif
  return (fgElement*)r;
}

fgElement* FG_FASTCALL fgCreateDefault(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  static bss_util::cTrie<uint16_t, true> t(25, "element", "control", "resource", "text", "box", "scrollbar", "button", "window", "checkbox",
    "radiobutton", "progressbar", "slider", "textbox", "treeview", "treeitem", "list", "listitem", "curve", "dropdown", "tabcontrol", "tab",
    "menu", "submenu", "menuitem", "debug");

  switch(t[type])
  {
  case 0:
    return _create_default<fgElement, fgElement_Init>(parent, next, name, flags, transform, units);
  case 1:
    return _create_default<fgControl, fgControl_Init>(parent, next, name, flags, transform, units);
  case 2:
    return _create_default<fgResource, fgResource_Init>(parent, next, name, flags, transform, units);
  case 3:
    return _create_default<fgText, fgText_Init>(parent, next, name, flags, transform, units);
  case 4:
    return _create_default<fgBox, fgBox_Init>(parent, next, name, flags, transform, units);
  case 5:
    return _create_default<fgScrollbar, fgScrollbar_Init>(parent, next, name, flags, transform, units);
  case 6:
    return _create_default<fgButton, fgButton_Init>(parent, next, name, flags, transform, units);
  case 7:
    return _create_default<fgWindow, fgWindow_Init>(parent, next, name, flags, transform, units);
  case 8:
    return _create_default<fgCheckbox, fgCheckbox_Init>(parent, next, name, flags, transform, units);
  case 9:
    return _create_default<fgRadiobutton, fgRadiobutton_Init>(parent, next, name, flags, transform, units);
  case 10:
    return _create_default<fgProgressbar, fgProgressbar_Init>(parent, next, name, flags, transform, units);
  case 11:
    return _create_default<fgSlider, fgSlider_Init>(parent, next, name, flags, transform, units);
  case 12:
    return _create_default<fgTextbox, fgTextbox_Init>(parent, next, name, flags, transform, units);
  case 13:
    return _create_default<fgTreeview, fgTreeview_Init>(parent, next, name, flags, transform, units);
  case 14:
    return _create_default<fgTreeItem, fgTreeItem_Init>(parent, next, name, flags, transform, units);
  case 15:
    return _create_default<fgList, fgList_Init>(parent, next, name, flags, transform, units);
  case 16:
    return _create_default<fgControl, fgListItem_Init>(parent, next, name, flags, transform, units);
  case 17:
    return _create_default<fgCurve, fgCurve_Init>(parent, next, name, flags, transform, units);
  case 18:
    return _create_default<fgDropdown, fgDropdown_Init>(parent, next, name, flags, transform, units);
  case 19:
    return _create_default<fgTabcontrol, fgTabcontrol_Init>(parent, next, name, flags, transform, units);
  case 20: //Tab
  {
    fgElement* e = parent->AddItemText(name);
    e->SetFlags(flags);
    return e;
  }
  case 21:
    return _create_default<fgMenu, fgMenu_Init>(parent, next, name, flags, transform, units);
  case 22:
    return _create_default<fgMenu, fgSubmenu_Init>(parent, next, name, flags, transform, units);
  case 23:
    return _create_default<fgMenuItem, fgMenuItem_Init>(parent, next, name, flags, transform, units);
  case 24:
    return _create_default<fgDebug, fgDebug_Init>(parent, next, name, flags, transform, units);
  }

  return 0;
}

short FG_FASTCALL fgMessageMapDefault(const char* name)
{
  static bss_util::cTrie<uint16_t, true> t(FG_CUSTOMEVENT, "CONSTRUCT", "DESTROY", "MOVE", "SETALPHA", "SETAREA", "SETTRANSFORM", "SETFLAG", "SETFLAGS", "SETMARGIN", "SETPADDING",
    "SETPARENT", "ADDCHILD", "REMOVECHILD", "LAYOUTCHANGE", "LAYOUTFUNCTION", "LAYOUTLOAD", "DRAG", "DRAGGING", "DROP", "DRAW", "INJECT", "CLONE", "SETSKIN", "GETSKIN",
    "SETSTYLE", "GETSTYLE", "GETCLASSNAME", "GETDPI", "SETDPI", "SETUSERDATA", "GETUSERDATA", "MOUSEDOWN", "MOUSEDBLCLICK", "MOUSEUP", "MOUSEON", "MOUSEOFF", "MOUSEMOVE",
    "MOUSESCROLL", "TOUCHBEGIN", "TOUCHEND", "TOUCHMOVE", "KEYUP", "KEYDOWN", "KEYCHAR", "JOYBUTTONDOWN", "JOYBUTTONUP", "JOYAXIS", "GOTFOCUS", "LOSTFOCUS",
    "SETNAME", "GETNAME", "NEUTRAL", "HOVER", "ACTIVE", "ACTION", "SETDIM", "GETDIM", "GETITEM", "ADDITEM", "REMOVEITEM", "GETSELECTEDITEM", "GETSTATE", "SETSTATE",
    "SETRESOURCE", "SETUV", "SETCOLOR", "SETOUTLINE", "SETFONT", "SETLINEHEIGHT", "SETLETTERSPACING", "SETTEXT", "GETRESOURCE", "GETUV", "GETCOLOR", "GETOUTLINE", "GETFONT",
    "GETLINEHEIGHT", "GETLETTERSPACING", "GETTEXT");

  return t[name];
}