// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgBackend.h"
#include "fgRoot.h"
#include "feathercpp.h"
#include "bss-util/cTrie.h"

KHASH_INIT(fgFunctionMap, char*, fgListener, 1, kh_str_hash_func, kh_str_hash_equal);

using namespace bss_util;

size_t FG_FASTCALL fgBehaviorHookDefault(fgElement* self, const FG_Msg* msg)
{
  assert(self != 0);
  return (*self->message)(self, msg);
}

size_t FG_FASTCALL fgBehaviorHookListener(fgElement* self, const FG_Msg* msg)
{
  assert(self != 0);
  size_t ret = (*self->message)(self, msg);
  khiter_t iter = fgListenerHash.Iterator(std::pair<fgElement*, unsigned short>(self, msg->type));
  if(fgListenerHash.ExistsIter(iter))
    fgListenerHash.GetValue(iter)(self, msg);
  return ret;
}

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

template<class T, void (MSC_FASTCALL *GCC_FASTCALL INIT)(T* BSS_RESTRICT, fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, const char*, fgFlag, const fgTransform*, unsigned short)>
BSS_FORCEINLINE fgElement* _create_default(fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units, const char* file, size_t line)
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

short FG_FASTCALL fgMessageMapDefault(const char* name)
{
  static bss_util::cTrie<uint16_t, true> t(FG_CUSTOMEVENT+1, "UNKNOWN", "CONSTRUCT", "DESTROY", "MOVE", "SETALPHA", "SETAREA", "SETTRANSFORM", "SETFLAG", "SETFLAGS", "SETMARGIN", "SETPADDING",
    "SETPARENT", "ADDCHILD", "REMOVECHILD", "LAYOUTCHANGE", "LAYOUTFUNCTION", "LAYOUTLOAD", "DRAGOVER", "DROP", "DRAW", "INJECT", "SETSKIN", "GETSKIN", "SETSTYLE", "GETSTYLE", "GETCLASSNAME",
    "GETDPI", "SETDPI", "SETUSERDATA", "GETUSERDATA", "SETDIM", "GETDIM", "MOUSEDOWN", "MOUSEDBLCLICK", "MOUSEUP", "MOUSEON", "MOUSEOFF", "MOUSEMOVE", "MOUSESCROLL", "TOUCHBEGIN", "TOUCHEND",
    "TOUCHMOVE", "KEYUP", "KEYDOWN", "KEYCHAR", "JOYBUTTONDOWN", "JOYBUTTONUP", "JOYAXIS", "GOTFOCUS", "LOSTFOCUS", "SETNAME", "GETNAME", "SETCONTEXTMENU", "GETCONTEXTMENU", "NEUTRAL",
    "HOVER", "ACTIVE", "ACTION", "GETITEM", "ADDITEM", "REMOVEITEM", "SETITEM", "GETSELECTEDITEM", "GETVALUE", "SETVALUE", "SETRESOURCE", "SETUV", "SETCOLOR", "SETOUTLINE", "SETFONT",
    "SETLINEHEIGHT", "SETLETTERSPACING", "SETTEXT", "GETRESOURCE", "GETUV", "GETCOLOR", "GETOUTLINE", "GETFONT", "GETLINEHEIGHT", "GETLETTERSPACING", "GETTEXT", "$FG_CUSTOMEVENT");

  assert(t["$FG_CUSTOMEVENT"] == FG_CUSTOMEVENT); // Ensure this count is correct
  return t[name];
}

void FG_FASTCALL fgUserDataMapDefaultProcess(fgElement* self, struct _FG_KEY_VALUE* pair)
{
  if(!STRNICMP(pair->key, "contextmenu", 11))
  {
    fgElement* menu = fgRoot_GetID(fgroot_instance, pair->value);
    if(menu != 0)
      self->SetContextMenu(menu);
  }
  else
    self->SetUserdata(pair->value, pair->key);
}

void FG_FASTCALL fgUserDataMapDefault(fgElement* self, struct __VECTOR__KeyValue* pairs)
{
  for(size_t i = 0; i < pairs->l; ++i)
    fgUserDataMapDefaultProcess(self, pairs->p + i);
}

void FG_FASTCALL fgUserDataMapCallbacksProcess(fgElement* self, struct _FG_KEY_VALUE* pair)
{
  if(!STRNICMP(pair->key, "on", 2))
  {
    short type = (*fgroot_instance->backend.fgMessageMap)(pair->key + 2);
    if(type != -1)
    {
      khint_t i = kh_get_fgFunctionMap(fgroot_instance->functionhash, pair->value);
      if(i != kh_end(fgroot_instance->functionhash))
        fgElement_AddListener(self, type, kh_val(fgroot_instance->functionhash, i));
    }
  }
  else
    fgUserDataMapDefaultProcess(self, pair);
}
void FG_FASTCALL fgUserDataMapCallbacks(fgElement* self, struct __VECTOR__KeyValue* pairs)
{
  for(size_t i = 0; i < pairs->l; ++i)
    fgUserDataMapCallbacksProcess(self, pairs->p + i);
}

__inline struct __kh_fgFunctionMap_t* fgFunctionMap_init()
{
  return kh_init_fgFunctionMap();
}
void fgFunctionMap_destroy(struct __kh_fgFunctionMap_t* h)
{
  for(khiter_t i = 0; i != kh_end(h); ++i)
    if(kh_exist(h, i))
      fgfree(kh_key(h, i), __FILE__, __LINE__);

  kh_destroy_fgFunctionMap(h);
}

int FG_FASTCALL fgRegisterFunction(const char* name, fgListener fn)
{
  int r;
  khint_t iter = kh_put_fgFunctionMap(fgroot_instance->functionhash, fgCopyText(name, __FILE__, __LINE__), &r);
  kh_val(fgroot_instance->functionhash, iter) = fn;
  return r;
}