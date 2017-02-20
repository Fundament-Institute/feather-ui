// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgBackend.h"
#include "fgRoot.h"
#include "feathercpp.h"
#include "bss-util/cTrie.h"
#ifdef BSS_PLATFORM_WIN32
#include "bss-util/bss_win32_includes.h"
#else
#include <dlfcn.h>
#endif

KHASH_INIT(fgFunctionMap, const char*, fgListener, 1, kh_str_hash_func, kh_str_hash_equal);

using namespace bss_util;

fgFont fgCreateFontDefault(fgFlag flags, const char* font, unsigned int fontsize, const fgIntVec* dpi) { return (void*)~0; }
fgFont fgCloneFontDefault(fgFont font, const struct _FG_FONT_DESC* desc) { return (void*)~0; }
void fgDestroyFontDefault(fgFont font) { }
void fgDrawFontDefault(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data, void* layout) { }
fgFont fgFontLayoutDefault(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, AbsRect* area, fgFlag flags, void* prevlayout) { return 0; }
void fgFontGetDefault(fgFont font, struct _FG_FONT_DESC* desc) { if(desc) memset(desc, 0, sizeof(struct _FG_FONT_DESC)); }
size_t fgFontIndexDefault(void* font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, void* cache) { return 0; }
AbsVec fgFontPosDefault(void* font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, void* cache) { AbsVec a = { 0,0 }; return a; }

fgAsset fgCreateAssetDefault(fgFlag flags, const char* data, size_t length) { return (void*)~0; }
fgAsset fgCloneAssetDefault(fgAsset res, fgElement* src) { return (void*)~0; }
void fgDestroyAssetDefault(fgAsset res) { }
void fgDrawAssetDefault(fgAsset res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data) {}
void fgAssetSizeDefault(fgAsset res, const CRect* uv, AbsVec* dim, fgFlag flags) { }

void fgDrawLinesDefault(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center, const fgDrawAuxData* data) {}

AbsRect* clipstack = 0;
size_t clipcapacity = 0;
size_t clipnum = 0;

void fgPushClipRectDefault(const AbsRect* clip, const fgDrawAuxData* data)
{
  if(clipcapacity >= clipnum)
  {
    clipcapacity = clipcapacity * 2;
    clipstack = (AbsRect*)realloc(clipstack, sizeof(AbsRect)*clipcapacity);
  }
  clipstack[clipnum++] = *clip;
}
AbsRect fgPeekClipRectDefault(const fgDrawAuxData* data)
{
  static const AbsRect BLANK = { 0,0,0,0 };
  return !clipnum ? BLANK : clipstack[clipnum - 1];
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
  fgCaptureWindow = 0;
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

template<class T, void (*INIT)(T* BSS_RESTRICT, fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, const char*, fgFlag, const fgTransform*, unsigned short)>
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

short fgMessageMapDefault(const char* name)
{
  static bss_util::cTrie<uint16_t, true> t(FG_CUSTOMEVENT+1, "UNKNOWN", "CONSTRUCT", "DESTROY", "CLONE", "MOVE", "SETALPHA", "SETAREA", "SETTRANSFORM", "SETFLAG", "SETFLAGS", "SETMARGIN", "SETPADDING",
    "SETPARENT", "ADDCHILD", "REMOVECHILD", "PARENTCHANGE", "LAYOUTCHANGE", "LAYOUTFUNCTION", "LAYOUTLOAD", "DRAGOVER", "DROP", "DRAW", "INJECT", "SETSKIN", "GETSKIN", "SETSTYLE", "GETSTYLE",
    "GETCLASSNAME", "GETDPI", "SETDPI", "SETUSERDATA", "GETUSERDATA", "SETDIM", "GETDIM", "SETSCALING", "GETSCALING", "MOUSEDOWN", "MOUSEDBLCLICK", "MOUSEUP", "MOUSEON", "MOUSEOFF", "MOUSEMOVE",
    "MOUSESCROLL", "TOUCHBEGIN", "TOUCHEND", "TOUCHMOVE", "KEYUP", "KEYDOWN", "KEYCHAR", "JOYBUTTONDOWN", "JOYBUTTONUP", "JOYAXIS", "GOTFOCUS", "LOSTFOCUS", "SETNAME", "GETNAME", "SETCONTEXTMENU",
    "GETCONTEXTMENU", "NEUTRAL", "HOVER", "ACTIVE", "ACTION", "GETITEM", "ADDITEM", "REMOVEITEM", "SETITEM", "GETSELECTEDITEM", "GETVALUE", "SETVALUE", "GETRANGE", "SETRANGE", "SETASSET", "SETUV",
    "SETCOLOR", "SETOUTLINE", "SETFONT", "SETLINEHEIGHT", "SETLETTERSPACING", "SETTEXT", "GETASSET", "GETUV", "GETCOLOR", "GETOUTLINE", "GETFONT", "GETLINEHEIGHT", "GETLETTERSPACING", "GETTEXT",
    "$FG_CUSTOMEVENT");

  assert(t["$FG_CUSTOMEVENT"] == FG_CUSTOMEVENT); // Ensure this count is correct
  return t[name];
}

void fgUserDataMapDefaultProcess(fgElement* self, struct _FG_KEY_VALUE* pair)
{
  if(!STRNICMP(pair->key, "contextmenu", 11))
  {
    fgElement* menu = fgRoot_GetID(fgroot_instance, pair->value);
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
    short type = (*fgroot_instance->backend.fgMessageMap)(pair->key + 2);
    if(type != -1)
    {
      khint_t i = kh_get_fgFunctionMap(fgroot_instance->functionhash, (char*)pair->value);
      if(i != kh_end(fgroot_instance->functionhash))
        fgElement_AddListener(self, type, kh_val(fgroot_instance->functionhash, i));
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

size_t fgBehaviorHookDefault(fgElement* self, const FG_Msg* msg)
{
  assert(self != 0);
  return (*self->message)(self, msg);
}

size_t fgBehaviorHookListener(fgElement* self, const FG_Msg* msg)
{
  assert(self != 0);
  size_t ret = (*self->message)(self, msg);
  khiter_t iter = fgListenerHash.Iterator(std::pair<fgElement*, unsigned short>(self, msg->type));
  if(fgListenerHash.ExistsIter(iter))
    fgListenerHash.GetValue(iter)(self, msg);
  return ret;
}

__inline struct __kh_fgFunctionMap_t* fgFunctionMap_init()
{
  return kh_init_fgFunctionMap();
}
void fgFunctionMap_destroy(struct __kh_fgFunctionMap_t* h)
{
  for(khiter_t i = 0; i != kh_end(h); ++i)
    if(kh_exist(h, i))
      fgFreeText(kh_key(h, i), __FILE__, __LINE__);

  kh_destroy_fgFunctionMap(h);
}

int fgRegisterFunction(const char* name, fgListener fn)
{
  int r;
  khint_t iter = kh_put_fgFunctionMap(fgroot_instance->functionhash, fgCopyText(name, __FILE__, __LINE__), &r);
  kh_val(fgroot_instance->functionhash, iter) = fn;
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
  FN_INITIALIZE init = (FN_INITIALIZE)GETDYNFUNC(fgDynLibP, "fgInitialize");
  return !init ? 0 : init();
}

void fgUnloadBackend()
{
  if(fgroot_instance != 0)
    fgroot_instance->backend.fgTerminate();
  fgroot_instance = 0;
  FREEDYNLIB(fgDynLibP);
  fgDynLibP = 0;
}

int fgLogHookDefault(const char* format, va_list args)
{
  static std::unique_ptr<FILE, void(*)(FILE*)> f(fopen("feathergui.log", "wb"), [](FILE* x) { fclose(x); });
  vfprintf(f.get(), format, args);
  fwrite("\n", 1, 1, f.get());
  int n = vprintf(format, args);
  printf("\n");
  return n;
}
