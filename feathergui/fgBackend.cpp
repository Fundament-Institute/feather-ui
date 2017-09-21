// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgBackend.h"
#include "fgAll.h"
#include "feathercpp.h"
#include "bss-util/Trie.h"
#ifdef BSS_PLATFORM_WIN32
#include "bss-util/win32_includes.h"
#else
#include <dlfcn.h>
#endif

KHASH_INIT(fgFunctionMap, const char*, fgDelegateListener, 1, kh_str_hash_func, kh_str_hash_equal);

using namespace bss;

fgFont fgCreateFontDefault(fgFlag flags, const char* family, short weight, char italic, unsigned int size, const AbsVec* dpi) { return (void*)~0; }
fgFont fgCloneFontDefault(fgFont font, const struct _FG_FONT_DESC* desc) { return (void*)~0; }
void fgDestroyFontDefault(fgFont font) { }
void fgDrawFontDefault(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data, void* layout) { }
fgFont fgFontLayoutDefault(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, AbsRect* area, fgFlag flags, const AbsVec* dpi, void* prevlayout) { return 0; }
void fgFontGetDefault(fgFont font, struct _FG_FONT_DESC* desc) { if(desc) memset(desc, 0, sizeof(struct _FG_FONT_DESC)); }
size_t fgFontIndexDefault(void* font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, const AbsVec* dpi, void* cache) { return 0; }
AbsVec fgFontPosDefault(void* font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, const AbsVec* dpi, void* cache) { AbsVec a = { 0,0 }; return a; }

fgAsset fgCreateAssetFileDefault(fgFlag flags, const char* file, const AbsVec* dpi)
{
  FILE* f;
  FOPEN(f, file, "rb");
  if(!f) return 0;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  VARARRAY(char, buf, len);
  fread(buf, 1, len, f);
  fclose(f);
  void* r = fgroot_instance->backend.fgCreateAsset(flags, buf, len, dpi);
  return r;
}
fgAsset fgCreateAssetDefault(fgFlag flags, const char* data, size_t length, const AbsVec* dpi) { return (void*)~0; }
fgAsset fgCloneAssetDefault(fgAsset res, fgElement* src) { return (void*)~0; }
void fgDestroyAssetDefault(fgAsset res) { }
void fgDrawAssetDefault(fgAsset res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data) {}
void fgAssetSizeDefault(fgAsset res, const CRect* uv, AbsVec* dim, fgFlag flags, const AbsVec* dpi) { }

void fgDrawLinesDefault(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center, const fgDrawAuxData* data) {}

static std::unique_ptr<AbsRect, decltype(&free)> clipstack(nullptr, &free);
static size_t clipcapacity = 0;
static size_t clipnum = 0;

void fgPushClipRectDefault(const AbsRect* clip, const fgDrawAuxData* data)
{
  if(clipcapacity >= clipnum)
  {
    clipcapacity = clipcapacity * 2;
    AbsRect* p = clipstack.release();
    clipstack.reset((AbsRect*)realloc(p, sizeof(AbsRect)*clipcapacity));
  }
  clipstack.get()[clipnum++] = *clip;
}
AbsRect fgPeekClipRectDefault(const fgDrawAuxData* data)
{
  static const AbsRect BLANK = { 0,0,0,0 };
  return !clipnum ? BLANK : clipstack.get()[clipnum - 1];
}
void fgPopClipRectDefault(const fgDrawAuxData* data)
{
  --clipnum;
}

void fgDragStartDefault(char type, void* data, fgElement* draw)
{
  fgroot_instance->dragtype = type;
  fgroot_instance->dragdata = data;
  fgroot_instance->dragdraw = draw;
  fgroot_instance->fgCaptureWindow = 0;
}

size_t fgLoadExtensionDefault(const char* extname, void* fg, size_t sz) { return -1; }
void fgSetCursorDefault(unsigned int type, void* custom) {}
void fgDirtyElementDefault(fgElement* e) { if(e->flags&FGELEMENT_SILENT) return; }

static unsigned int __fgClipboardType = 0;
static std::unique_ptr<uint8_t[]> __fgClipboardData = 0;
static size_t __fgClipboardSize = 0;

void fgClipboardCopyDefault(unsigned int type, const void* data, size_t length)
{
  __fgClipboardType = type;
  __fgClipboardSize = length;
  __fgClipboardData = std::unique_ptr<uint8_t[]>(new uint8_t[length]);
  MEMCPY(__fgClipboardData.get(), length, data, length);
}
char fgClipboardExistsDefault(unsigned int type) { return type == __fgClipboardType && __fgClipboardSize != 0; }
const void* fgClipboardPasteDefault(unsigned int type, size_t* length)
{
  *length = __fgClipboardSize;
  void* data = malloc(__fgClipboardSize);
  MEMCPY(data, __fgClipboardSize, __fgClipboardData.get(), __fgClipboardSize);
  return data;
}
void fgClipboardFreeDefault(const void* mem)
{
  assert(mem != 0);
  free(const_cast<void*>(mem));
}

template<class T, void (*INIT)(T* BSS_RESTRICT, fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, const char*, fgFlag, const fgTransform*, fgMsgType)>
BSS_FORCEINLINE fgElement* _create_default(fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units, const char* file, size_t line)
{
  T* r = fgmalloc<T>(1, file, line);
  INIT(r, parent, next, name, flags, transform, units);
#ifdef BSS_DEBUG
  ((fgElement*)r)->free = &fgfreeblank;
#else
  ((fgElement*)r)->free = &free; // We do this because the compiler can't figure out the inlining weirdness going on here
#endif
  return (fgElement*)r;
}

fgMsgType fgMessageMapDefault(const char* name)
{
  static bss::Trie<fgMsgType, true> t(FG_CUSTOMEVENT+1, "UNKNOWN", "CONSTRUCT", "DESTROY", "CLONE", "MOVE", "SETALPHA", "SETAREA", "SETTRANSFORM", "SETFLAG", "SETFLAGS", "SETMARGIN", "SETPADDING",
    "SETPARENT", "ADDCHILD", "REMOVECHILD", "REORDERCHILD", "PARENTCHANGE", "LAYOUTCHANGE", "LAYOUTFUNCTION", "LAYOUTLOAD", "DRAGOVER", "DROP", "DRAW", "INJECT", "SETSKIN", "GETSKIN", "SETSTYLE", "GETSTYLE",
    "GETCLASSNAME", "GETDPI", "SETDPI", "SETUSERDATA", "GETUSERDATA", "SETDIM", "GETDIM", "SETSCALING", "GETSCALING", "MOUSEDOWN", "MOUSEDBLCLICK", "MOUSEUP", "MOUSEON", "MOUSEOFF", "MOUSEMOVE",
    "MOUSESCROLL", "TOUCHBEGIN", "TOUCHEND", "TOUCHMOVE", "KEYUP", "KEYDOWN", "KEYCHAR", "JOYBUTTONDOWN", "JOYBUTTONUP", "JOYAXIS", "GOTFOCUS", "LOSTFOCUS", "SETNAME", "GETNAME", "SETCONTEXTMENU",
    "GETCONTEXTMENU", "SETTOOLTIP", "GETTOOLTIP", "NEUTRAL", "HOVER", "ACTIVE", "ACTION", "GETITEM", "ADDITEM", "REMOVEITEM", "SETITEM", "CLEAR", "SELECTION", "GETSELECTEDITEM", "GETVALUE", "SETVALUE",
    "GETRANGE", "SETRANGE", "SETASSET", "SETUV", "SETCOLOR", "SETOUTLINE", "SETFONT", "SETLINEHEIGHT", "SETLETTERSPACING", "SETTEXT", "GETASSET", "GETUV", "GETCOLOR", "GETOUTLINE", "GETFONT", "GETLINEHEIGHT",
    "GETLETTERSPACING", "GETTEXT", "DEBUGMESSAGE", "$FG_CUSTOMEVENT");

  assert(t["$FG_CUSTOMEVENT"] == FG_CUSTOMEVENT); // Ensure this count is correct
  return t[name];
}

const char* fgFlagMapDefault(const char* type, fgFlag flag)
{
  static Trie<uint16_t, true> t(30, "element", "control", "scrollbar", "box", "list", "grid", "resource", "text", "button", "window", "checkbox", "radiobutton",
    "progressbar", "slider", "textbox", "treeview", "treeitem", "listitem", "curve", "dropdown", "tabcontrol", "menu", "submenu",
    "menuitem", "gridrow", "workspace", "toolbar", "toolgroup", "combobox", "debug");

  switch(flag)
  {
  case FGELEMENT_BACKGROUND: return "BACKGROUND";
  case FGELEMENT_NOCLIP: return "NOCLIP";
  case FGELEMENT_IGNORE: return "IGNORE";
  case FGELEMENT_HIDDEN: return "HIDDEN";
  case FGELEMENT_SILENT: return "SILENT";
  case FGELEMENT_EXPANDX: return "EXPANDX";
  case FGELEMENT_EXPANDY: return "EXPANDY";
  case FGELEMENT_EXPANDX | FGELEMENT_EXPANDY: return "EXPAND";
  case FGELEMENT_SNAPX: return "SNAPX";
  case FGELEMENT_SNAPY: return "SNAPY";
  case FGELEMENT_SNAPX | FGELEMENT_SNAPY: return "SNAP";
  }

  uint16_t ty = t[type];

  // Top level flags (no other class inherits these)
  switch(ty)
  {
  case 5:  // grid
    switch(flag)
    {
    case FGGRID_AUTOEDIT: return "AUTOEDIT";
    }
    break;
  case 6: // resource
    switch(flag)
    {
    case FGRESOURCE_UVTILE: return "BACKGROUND";
    case FGRESOURCE_RECT: return "RECT";
    case FGRESOURCE_CIRCLE: return "CIRCLE";
    case FGRESOURCE_TRIANGLE: return "TRIANGLE";
    }
    break;
  case 8: //button
    switch(flag)
    {
    case FGBUTTON_NOFOCUS: return "NOFOCUS";
    }
    break;
  case 9: // window
    switch(flag)
    {
    case FGWINDOW_MINIMIZABLE: return "MINIMIZABLE";
    case FGWINDOW_MAXIMIZABLE: return "MAXIMIZABLE";
    case FGWINDOW_RESIZABLE: return "RESIZABLE";
    case FGWINDOW_NOCAPTION: return "NOCAPTION";
    case FGWINDOW_NOBORDER: return "NOBORDER";
    }
    break;
  case 18: // curve
    switch(flag)
    {
    case FGCURVE_QUADRATIC: return "QUADRATIC";
    case FGCURVE_CUBIC: return "CUBIC";
    case FGCURVE_BSPLINE: return "BSPLINE";
    }
    break;
  case 25: // workspace
    switch(flag)
    {
    case FGWORKSPACE_RULERX: return "RULERX";
    case FGWORKSPACE_RULERY: return "RULERY";
    case FGWORKSPACE_SNAPTOX: return "SNAPTOX";
    case FGWORKSPACE_SNAPTOY: return "SNAPTOY";
    }
    break;
  case 26: // toolbar
    switch(flag)
    {
    case FGTOOLBAR_LOCKED: return "LOCKED";
    }
    break;
  case 29:  // debug
    switch(flag)
    {
    case FGDEBUG_CLEARONHIDE: return "CLEARONHIDE";
    case FGDEBUG_OVERLAY: return "OVERLAY";
    }
    break;
  }

  switch(ty) // list flags
  {
  case 4: // list
  case 5: // grid
  case 26: // toolbar
    switch(flag)
    {
    case FGLIST_SELECT: return "SELECT";
    case FGLIST_MULTISELECT: return "MULTISELECT";
    case FGLIST_DRAGGABLE: return "DRAGGABLE";
    }
    break;
  }

  switch(ty) // box flags
  {
  case 3: // box
  case 4: // list
  case 5: // grid
  case 19: // dropdown
  case 21: // menu
  case 26: // toolbar
  case 27: // toolgroup
  case 28: // combobox
    switch(flag)
    {
    case FGBOX_IGNOREMARGINEDGEX: return "IGNOREMARGINEDGEX";
    case FGBOX_IGNOREMARGINEDGEY: return "IGNOREMARGINEDGEY";
    case FGBOX_IGNOREMARGINEDGEX|FGBOX_IGNOREMARGINEDGEY: return "IGNOREMARGINEDGE";
    case FGBOX_TILEX: return "TILEX";
    case FGBOX_TILEY: return "TILEY";
    case FGBOX_TILEX|FGBOX_TILEY: return "TILE";
    case FGBOX_GROWY: return "GROWY";
    case FGBOX_REVERSE: return "REVERSE";
    case FGBOX_DISTRIBUTE: return "DISTRIBUTE";
    }
    break;
  }

  switch(ty) // scrollbar flags
  {
  case 2: // scrollbar
  case 3: // box
  case 4: // list
  case 5: // grid
  case 14: // textbox
  case 15: // treeview
  case 19: // dropdown
  case 21: // menu
  case 25: // workspace
  case 26: // toolbar
  case 27: // toolgroup
  case 28: // combobox
    switch(flag)
    {
    case FGLIST_SELECT: return "SELECT";
    case FGLIST_MULTISELECT: return "MULTISELECT";
    case FGLIST_DRAGGABLE: return "DRAGGABLE";
    }
    break;
  }

  switch(ty) // text flags
  {
  case 7: // text
  case 14: // textbox
  case 27: // toolgroup
  //case 28: // combobox
    switch(flag)
    {
    case FGTEXT_CHARWRAP: return "CHARWRAP";
    case FGTEXT_WORDWRAP: return "WORDWRAP";
    case FGTEXT_ELLIPSES: return "ELLIPSES";
    case FGTEXT_RTL: return "RTL";
    case FGTEXT_RIGHTALIGN: return "RIGHTALIGN";
    case FGTEXT_CENTER: return "CENTER";
    case FGTEXT_SUBPIXEL: return "SUBPIXEL";
    }
    break;
  }

  switch(ty) // textbox flags
  {
  case 14: // textbox
  //case 28: // combobox
    switch(flag)
    {
    case FGTEXTBOX_ACTION: return "ACTION";
    case FGTEXTBOX_SINGLELINE: return "SINGLELINE";
    }
    break;
  }

  switch(ty) // Control flags
  {
  case 1:
  case 2:
  case 3:
  case 4:
  case 5: // grid
  case 8: //button
  case 9:
  case 10:
  case 11:
  case 12: //progressbar
  case 13:
  case 14:
  case 15:
  case 16:
  case 17: //listitem
  case 19:
  case 20:
  case 21:
  case 22: //submenu
  case 23:
  case 24:
  case 25:
  case 26:
  case 27:
  case 28: // combobox
    switch(flag)
    {
    case FGCONTROL_DISABLE: return "DISABLE";
    }
  }

  return 0;
}
void fgUserDataMapDefaultProcess(fgElement* self, struct _FG_KEY_VALUE* pair)
{
  if(!STRNICMP(pair->key, "contextmenu", 11))
  {
    fgElement* menu = fgGetID(pair->value);
    if(menu != 0)
      self->SetContextMenu(menu);
  }
  else
    self->SetUserdata((char*)pair->value, pair->key);
}

void fgUserDataMapDefault(fgElement* self, struct __VECTOR__KeyValue* pairs)
{
  for(size_t i = 0; i < pairs->l; ++i)
    fgUserDataMapDefaultProcess(self, pairs->p + i);
}

void fgUserDataMapCallbacksProcess(fgElement* self, struct _FG_KEY_VALUE* pair)
{
  if(!STRNICMP(pair->key, "on", 2))
  {
    fgMsgType type = (*fgroot_instance->backend.fgMessageMap)(pair->key + 2);
    if(type != (fgMsgType)~0)
    {
      khint_t i = kh_get_fgFunctionMap(fgroot_instance->functionhash, (char*)pair->value);
      if(i != kh_end(fgroot_instance->functionhash))
      {
        auto& d = kh_val(fgroot_instance->functionhash, i);
        fgElement_AddDelegateListener(self, type, d.RawSource(), d.RawFunc());
      }
    }
  }
  else
    fgUserDataMapDefaultProcess(self, pair);
}
void fgUserDataMapCallbacks(fgElement* self, struct __VECTOR__KeyValue* pairs)
{
  for(size_t i = 0; i < pairs->l; ++i)
    fgUserDataMapCallbacksProcess(self, pairs->p + i);
}

size_t fgBehaviorHookSimple(fgElement* self, const FG_Msg* msg)
{
  assert(self != 0);
  return (*self->message)(self, msg);
}

size_t fgBehaviorHookDefault(fgElement* self, const FG_Msg* msg)
{
  assert(self != 0);
  size_t ret = (*self->message)(self, msg);
  khiter_t iter = fgListenerHash.Iterator(std::pair<fgElement*, fgMsgType>(self, msg->type));
  if(fgListenerHash.ExistsIter(iter))
    (*fgListenerHash.GetValue(iter))(self, msg);
  return ret;
}

struct kh_fgFunctionMap_s* fgFunctionMap_init()
{
  return kh_init_fgFunctionMap();
}
void fgFunctionMap_destroy(struct kh_fgFunctionMap_s* h)
{
  for(khiter_t i = 0; i != kh_end(h); ++i)
    if(kh_exist(h, i))
      fgFreeText(kh_key(h, i), __FILE__, __LINE__);

  kh_destroy_fgFunctionMap(h);
}

int fgRegisterFunction(const char* name, fgListener fn)
{
  return fgRegisterDelegate(name, (void*)fn, &fgDelegateListener::stubembed);
}

int fgRegisterDelegate(const char* name, void* p, void(*fn)(void*, struct _FG_ELEMENT*, const FG_Msg*))
{
  assert(fgroot_instance->fgBehaviorHook != fgBehaviorHookSimple); // You must have this set to fgBehaviorHookListener or an equivelent for this to do anything!
  int r;
  khint_t iter = kh_put_fgFunctionMap(fgroot_instance->functionhash, fgCopyText(name, __FILE__, __LINE__), &r);
  kh_val(fgroot_instance->functionhash, iter) = fgDelegateListener(p, fn);
  return r;
}

void fgTerminateDefault()
{
  VirtualFreeChild(&fgroot_instance->gui.element);
  fgroot_instance = 0;
}

char fgProcessMessagesDefault()
{
  _FG_MESSAGEQUEUE* q = ((std::atomic<_FG_MESSAGEQUEUE*>&)fgroot_instance->queue).load(std::memory_order_acquire);
  ((std::atomic<_FG_MESSAGEQUEUE*>&)fgroot_instance->queue).exchange(fgroot_instance->aux, std::memory_order_release);
  fgroot_instance->aux = q;
  q->lock.Lock();
  size_t cur = 0;
  size_t l = q->length.load(std::memory_order_relaxed);
  char* mem = (char*)q->mem.load(std::memory_order_relaxed);
  while(cur < l)
  {
    QUEUEDMESSAGE* m = (QUEUEDMESSAGE*)(mem + cur);
    fgSendMessage(m->e, &m->msg);
    cur += m->sz;
  }
  q->length.store(0, std::memory_order_relaxed);
#ifdef BSS_DEBUG
  memset(mem, 0xcd, l);
#endif
  q->lock.Unlock();
  return 1; // return 1 as long as the program should continue running. Return 0 when it should quit.
}

typedef struct _FG_ROOT*(*FN_INITIALIZE)();
static void* fgDynLibP = 0;

struct _FG_ROOT* fgLoadBackend(const char* dll)
{
  if(fgDynLibP != 0)
    fgUnloadBackend();
  fgDynLibP = LOADDYNLIB(dll);
  if(!fgDynLibP)
    return 0;
  FN_INITIALIZE init = (FN_INITIALIZE)GETDYNFUNC(fgDynLibP, fgInitialize);
  if(!init)
  {
    FREEDYNLIB(fgDynLibP);
    fgDynLibP = 0;
    return 0;
  }
  return init();
}

void fgUnloadBackend()
{
  if(fgroot_instance != 0)
    fgroot_instance->backend.fgTerminate();
  fgroot_instance = 0;
  FREEDYNLIB(fgDynLibP);
  fgDynLibP = 0;
}

int fgLogHookDefault(char level, const char* format, va_list args)
{
  static const char* LEVELS[] = { "FATAL: ", "ERROR: ", "WARNING: ", "NOTICE: ", "INFO: ", "DEBUG: " };
  static std::unique_ptr<FILE, decltype(&fclose)> f(fopen("feathergui.log", "wb"), &fclose);
  if(level != -1)
    fprintf(f.get(), LEVELS[level]);
  vfprintf(f.get(), format, args);
  fwrite("\n", 1, 1, f.get());
  fflush(f.get());
  if(level != -1)
    printf(LEVELS[level]);
  int n = vprintf(format, args);
  printf("\n");
  return n;
}