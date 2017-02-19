// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FEATHER_CPP_H__
#define __FEATHER_CPP_H__

#include "fgResource.h"
#include "fgText.h"
#include "fgRoot.h"
#include "bss-util/cArraySort.h"
#include "fgLayout.h"
#include "bss-util/cHash.h"
#include "bss-util/cAVLtree.h"
#include "bss-util/rwlock.h"

#ifdef BSS_64BIT
#define kh_ptr_hash_func(key) kh_int64_hash_func((uint64_t)key)
#else
#define kh_ptr_hash_func kh_int_hash_func
#endif

//#define FG_NO_TEXT_CACHE // Uncomment this if text is leaking somewhere and you don't know where.

// This is a template implementation of malloc that we can overload for memory leak detection
#ifdef BSS_DEBUG

extern bool fgTextFreeCheck(void* p);
extern void fgTextLeakDump(FILE* f);

class fgLeakTracker
{
  struct LEAKINFO
  {
    void* ptr;
    size_t size;
    const char* file;
    size_t line;
    const char* freefile;
    size_t freeline;
    size_t freecount;
  };

public:
  fgLeakTracker()
  {
    FOPEN(f, "memleaks.txt", "w");
  }
  ~fgLeakTracker()
  {
    fputs("--- Memory leaks---\n", f);
    LEAKINFO* pinfo;
    for(auto curiter = _leakinfo.begin(); curiter.IsValid(); ++curiter)
    {
      pinfo = _leakinfo.GetValue(*curiter);
      if(!pinfo->freecount)
        fprintf(f, "%p (Size: %zi) leaked at %s:%zi\n", pinfo->ptr, pinfo->size, pinfo->file, pinfo->line);

      free((char*)pinfo->file);
      free(pinfo);
    }

    fgTextLeakDump(f);
    fclose(f);
  }

  void Add(void* ptr, size_t size, const char* file, size_t line)
  {
    const char* hold = strrchr(file, '\\');
    const char* hold2 = strrchr(file, '/');
    file = bssmax(bssmax(hold, hold2), file);

    if(_leakinfo.Exists(ptr))
    {
      if(!_leakinfo[ptr]->freecount) // If we just allocated memory that hasn't actually been freed something really bad happened.
        throw "DUPLICATE ASSIGNMENT ERROR";
      _leakinfo[ptr]->freecount = 0;
    }

    LEAKINFO* pinfo = (LEAKINFO*)malloc(sizeof(LEAKINFO));
    pinfo->size = size;
    size_t sz = strlen(file) + 1;
    pinfo->file = (const char*)malloc(sz);
    MEMCPY((char*)pinfo->file, sz, file, sz);
    pinfo->line = line;
    pinfo->ptr = ptr;
    pinfo->freecount = 0;
    pinfo->freefile = 0;
    pinfo->freeline = 0;
    _leakinfo.Insert(ptr, pinfo);
  }
  bool Remove(void* ptr, const char* file, size_t line)
  {
    auto i = _leakinfo.Iterator(ptr);
    if(!_leakinfo.ExistsIter(i))
      fprintf(f, "Attempt to delete unassigned memory location (%p) at %s:%zi\n", ptr, file, line);
    else
    {
      LEAKINFO* info = _leakinfo.GetValue(i);
      ++info->freecount;

      if(info->freecount > 1)
        fprintf(f, "Attempted to delete memory location (%p) at least %zi times at %s:%zi. First free location at %s:%zi\n", ptr, info->freecount, file, line, info->freefile, info->freeline);
      else
      {
        info->freefile = file;
        info->freeline = line;
        memset(ptr, 0xFD, info->size);
        return true;
      }
    }

    return false;
  }
  bool Verify(void* ptr)
  {
    return _leakinfo.Exists(ptr);
  }

  static fgLeakTracker Tracker;
protected:
  bss_util::cHash<void*, LEAKINFO*> _leakinfo;
  FILE* f;
};

template<typename T>
static BSS_FORCEINLINE T* fgmalloc(size_t sz, const char* file, size_t line)
{
  T* p = reinterpret_cast<T*>(malloc(sz * sizeof(T)));
  fgLeakTracker::Tracker.Add(p, sz * sizeof(T), file, line);
  return p;
}
static BSS_FORCEINLINE void fgfree(void* p, const char* file, size_t line)
{
  assert(!fgTextFreeCheck(p)); // In debug mode, make sure you are freeing text allocated with fgCopyText with fgFreeText
  if(fgLeakTracker::Tracker.Remove(p, file, line))
    free(p);
}

#else
template<typename T>
BSS_FORCEINLINE static T* fgmalloc(size_t sz, const char*, size_t) { return reinterpret_cast<T*>(malloc(sz * sizeof(T))); }
BSS_FORCEINLINE static void fgfree(void* p, const char*, size_t) { free(p); }
#endif

BSS_FORCEINLINE static void fgfreeblank(void* p)
{
  fgfree(p, "", 0);
}

template<void* (*CLONE)(void*), void (*DESTROY)(void*)>
struct fgArbitraryRef
{
  fgArbitraryRef() : ref(0) {}
  fgArbitraryRef(const fgArbitraryRef& copy) : ref(!copy.ref ? 0 : CLONE(copy.ref)) {}
  fgArbitraryRef(fgArbitraryRef&& mov) : ref(mov.ref) { mov.ref = 0; }
  fgArbitraryRef(void* r) : ref(!r ? 0 : CLONE(r)) {}
  ~fgArbitraryRef() { if(ref) DESTROY(ref); }

  fgArbitraryRef& operator=(const fgArbitraryRef& copy) { if(ref) DESTROY(ref); ref = !copy.ref ? 0 : CLONE(copy.ref); return *this; }
  fgArbitraryRef& operator=(fgArbitraryRef&& mov) { if(ref) DESTROY(ref); ref = mov.ref; mov.ref = 0; return *this; }

  void* ref;
};

template<class T, typename... Args>
struct fgConstruct
{
  template<void (*DESTROY)(T*), void (*CONSTRUCT)(T*, Args...)>
  struct fgConstructor : public T
  {
    fgConstructor(fgConstructor&& mov) { memcpy(this, &mov, sizeof(T)); memset(&mov, 0, sizeof(T)); }
    fgConstructor() {}
    fgConstructor(Args... args) { CONSTRUCT((T*)this, args...); }
    ~fgConstructor() { DESTROY((T*)this); }
    operator T&() { return *this; }
    operator const T&() const { return *this; }
    fgConstructor& operator=(fgConstructor&&) = delete;
    fgConstructor& operator=(const fgConstructor&) = delete;
  };
};

template<class T>
struct fgConstruct<T>
{
  template<void (*DESTROY)(T*), void (*CONSTRUCT)(T*)>
  struct fgConstructor : public T
  {
    fgConstructor(fgConstructor&& mov) { memcpy(this, &mov, sizeof(T)); memset(&mov, 0, sizeof(T)); }
    fgConstructor() { CONSTRUCT((T*)this); }
    ~fgConstructor() { DESTROY((T*)this); }
    operator T&() { return *this; }
    operator const T&() const { return *this; }
    fgConstructor& operator=(fgConstructor&&) = delete;
    fgConstructor& operator=(const fgConstructor&) = delete;
  };
};

struct _FG_ROOT;
extern struct _FG_ROOT* fgroot_instance;
struct _FG_DEBUG;
extern struct _FG_DEBUG* fgdebug_instance;
extern FG_UINT fgStyleFlagMask;

typedef typename fgConstruct<fgSkinLayout, const char*, fgFlag, const fgTransform*, short, int>::fgConstructor<fgSkinLayout_Destroy, fgSkinLayout_Init> fgSkinLayoutConstruct;
typedef typename fgConstruct<fgClassLayout, const char*, const char*, fgFlag, const fgTransform*, short, int>::fgConstructor<fgClassLayout_Destroy, fgClassLayout_Init> fgClassLayoutConstruct;

// For some batshit insane reason, putting static on the functions themselves makes VC++ explode unless they are static members of a class
struct fgStaticFn
{
  BSS_FORCEINLINE static void* fgCloneResourceCpp(void* r) { return fgroot_instance->backend.fgCloneAsset(r, 0); }
  BSS_FORCEINLINE static void fgDestroyResourceCpp(void* r) { return fgroot_instance->backend.fgDestroyAsset(r); }
  BSS_FORCEINLINE static void* fgCloneFontCpp(void* r) { return fgroot_instance->backend.fgCloneFont(r, 0); }
  BSS_FORCEINLINE static void fgDestroyFontCpp(void* r) { return fgroot_instance->backend.fgDestroyFont(r); }
  BSS_FORCEINLINE static void fgConstructKeyValue(_FG_KEY_VALUE*) {}
  BSS_FORCEINLINE static void fgDestructKeyValue(_FG_KEY_VALUE* p)
  {
    fgFreeText(p->key, __FILE__, __LINE__);
    if(p->value)
      fgFreeText(p->value, __FILE__, __LINE__);
  }
  BSS_FORCEINLINE static char fgSortStyleLayout(const fgSkinLayoutConstruct& l, const fgSkinLayoutConstruct& r) { return -SGNCOMPARE(l.layout.order, r.layout.order); }
  BSS_FORCEINLINE static char fgSortClassLayout(const fgClassLayoutConstruct& l, const fgClassLayoutConstruct& r) { return -SGNCOMPARE(l.layout.order, r.layout.order); }
};

typedef bss_util::cDynArray<fgArbitraryRef<fgStaticFn::fgCloneResourceCpp, fgStaticFn::fgDestroyResourceCpp>, FG_UINT, bss_util::CARRAY_CONSTRUCT> fgResourceArray;
typedef bss_util::cDynArray<fgArbitraryRef<fgStaticFn::fgCloneFontCpp, fgStaticFn::fgDestroyFontCpp>, FG_UINT, bss_util::CARRAY_CONSTRUCT> fgFontArray;

template<class T>
typename T::T_ DynGet(const fgVector& r, FG_UINT i) { return ((T&)r)[i]; }

template<class T>
char DynArrayRemove(T& a, FG_UINT index)
{
  if(index >= a.Length())
    return 0;
  a.Remove(index);
  return 1;
}

typedef bss_util::cArraySort<fgSkinLayoutConstruct, fgStaticFn::fgSortStyleLayout, size_t, bss_util::CARRAY_CONSTRUCT> fgSkinLayoutArray;
typedef bss_util::cDynArray<typename fgConstruct<fgStyle>::fgConstructor<fgStyle_Destroy, fgStyle_Init>, size_t, bss_util::CARRAY_CONSTRUCT> fgStyleArray;
typedef bss_util::cDynArray<typename fgConstruct<struct _FG_KEY_VALUE>::fgConstructor<fgStaticFn::fgDestructKeyValue, fgStaticFn::fgConstructKeyValue>, size_t, bss_util::CARRAY_CONSTRUCT> fgKeyValueArray;
typedef bss_util::cArraySort<fgClassLayoutConstruct, fgStaticFn::fgSortClassLayout, size_t, bss_util::CARRAY_CONSTRUCT> fgClassLayoutArray;

struct __kh_fgRadioGroup_t;
extern __inline struct __kh_fgRadioGroup_t* fgRadioGroup_init();
extern void fgRadioGroup_destroy(struct __kh_fgRadioGroup_t*);

struct __kh_fgFunctionMap_t;
extern __inline struct __kh_fgFunctionMap_t* fgFunctionMap_init();
extern void fgFunctionMap_destroy(struct __kh_fgFunctionMap_t*);

template<FG_MSGTYPE type, typename... Args>
static inline size_t _sendmsg(fgElement* self, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgroot_instance->backend.fgBehaviorHook)(self, &msg);
}

template<FG_MSGTYPE type, typename... Args>
static inline size_t _sendsubmsg(fgElement* self, unsigned short sub, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = sub;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgroot_instance->backend.fgBehaviorHook)(self, &msg);
}

FG_EXTERN bss_util::cHash<std::pair<fgElement*, unsigned short>, fgListener> fgListenerHash;

static inline FG_UINT fgStyleGetMask() { return 0; }

template<typename Arg, typename... Args>
static inline FG_UINT fgStyleGetMask(Arg arg, Args... args)
{
  return fgStyle_GetName(arg, false) | fgStyleGetMask(args...);
}

static BSS_FORCEINLINE size_t fgStandardNeutralSetStyle(fgElement* self, const char* style, unsigned short sub = FGSETSTYLE_NAME)
{
  return _sendsubmsg<FG_SETSTYLE, const void*, size_t>(self, sub, style, fgStyleGetMask("neutral", "hover", "active", "disable"));
}

static BSS_FORCEINLINE size_t fgMaskSetStyle(fgElement* self, const char* style, FG_UINT mask)
{
  return _sendsubmsg<FG_SETSTYLE, const void*, size_t>(self, FGSETSTYLE_NAME, style, mask);
}

static BSS_FORCEINLINE FABS fgResolveUnit(fgElement* self, FABS x, size_t unit, bool axis)
{
  switch(unit)
  {
  default:
  case FGUNIT_DP:
    return x;
  case FGUNIT_SP:
    return x*fgroot_instance->fontscale;
  case FGUNIT_EM:
    return x * self->GetLineHeight();
  case FGUNIT_PX:
    return axis ? (x * ((FABS)self->GetDPI().y / (FABS)fgroot_instance->dpi.y)) : (x * ((FABS)self->GetDPI().x / (FABS)fgroot_instance->dpi.x));
  }
}

static BSS_FORCEINLINE void fgSnapAbsRect(AbsRect& r, fgFlag flags)
{
  if(flags&FGELEMENT_SNAPX)
  {
    r.left = floor(r.left);
    r.right = floor(r.right);
  }
  if(flags&FGELEMENT_SNAPY)
  {
    r.top = floor(r.top);
    r.bottom = floor(r.bottom);
  }
}

static BSS_FORCEINLINE void fgResolveRectUnit(fgElement* self, AbsRect& r, size_t subtype)
{
  r.left = fgResolveUnit(self, r.left, (subtype & FGUNIT_LEFT_MASK) >> FGUNIT_LEFT, false);
  r.top = fgResolveUnit(self, r.top, (subtype & FGUNIT_TOP_MASK) >> FGUNIT_TOP, true);
  r.right = fgResolveUnit(self, r.right, (subtype & FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT, false);
  r.bottom = fgResolveUnit(self, r.bottom, (subtype & FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM, true);
}

static BSS_FORCEINLINE void fgResolveVecUnit(fgElement* self, AbsVec& v, size_t subtype)
{
  v.x = fgResolveUnit(self, v.x, (subtype & FGUNIT_X_MASK) >> FGUNIT_X, false);
  v.y = fgResolveUnit(self, v.y, (subtype & FGUNIT_Y_MASK) >> FGUNIT_Y, true);
}

static BSS_FORCEINLINE void fgResolveCRectUnit(fgElement* self, CRect& r, size_t subtype)
{
  r.left.abs = fgResolveUnit(self, r.left.abs, (subtype & FGUNIT_LEFT_MASK) >> FGUNIT_LEFT, false);
  r.top.abs = fgResolveUnit(self, r.top.abs, (subtype & FGUNIT_TOP_MASK) >> FGUNIT_TOP, true);
  r.right.abs = fgResolveUnit(self, r.right.abs, (subtype & FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT, false);
  if(subtype & FGUNIT_RIGHT_WIDTH) r.right.abs += r.left.abs;
  r.bottom.abs = fgResolveUnit(self, r.bottom.abs, (subtype & FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM, true);
  if(subtype & FGUNIT_BOTTOM_HEIGHT) r.bottom.abs += r.top.abs;
}

static BSS_FORCEINLINE void fgResolveCVecUnit(fgElement* self, CVec& v, size_t subtype)
{
  v.x.abs = fgResolveUnit(self, v.x.abs, (subtype & FGUNIT_X_MASK) >> FGUNIT_X, false);
  v.y.abs = fgResolveUnit(self, v.y.abs, (subtype & FGUNIT_Y_MASK) >> FGUNIT_Y, true);
}

extern "C" {
  FG_EXTERN size_t fgUTF8toUTF32(const char*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t fgUTF32toUTF8(const int*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t fgUTF32toUTF16(const int*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t fgUTF16toUTF32(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
}

namespace bss_util { struct cXMLNode; }
struct __VECTOR__UTF8;
struct __VECTOR__UTF16;
struct __VECTOR__UTF32;

extern fgSkin* fgSkinBase_LoadNodeXML(fgSkinBase* self, const bss_util::cXMLNode* root);
extern inline __kh_fgSkins_t *kh_init_fgSkins();
extern void fgStyle_LoadAttributesXML(struct _FG_STYLE* self, const bss_util::cXMLNode* cur, int flags, struct _FG_SKIN_BASE* root, const char* path, const char** id, fgKeyValueArray* userdata);
extern int fgStyle_NodeEvalTransform(const bss_util::cXMLNode* node, fgTransform& t);
extern fgElement* fgLayout_GetNext(fgElement* cur);
extern fgElement* fgLayout_GetPrev(fgElement* cur);
BSS_FORCEINLINE FABS fgLayout_GetElementWidth(fgElement* child);
BSS_FORCEINLINE FABS fgLayout_GetElementHeight(fgElement* child);
extern inline fgVector* fgText_Conversion(int type, struct __VECTOR__UTF8* text8, struct __VECTOR__UTF16* text16, struct __VECTOR__UTF32* text32);
extern void fgMenu_Show(struct _FG_MENU* self, bool show);

struct _FG_BOX_ORDERED_ELEMENTS_;

template<fgFlag FLAGS>
inline fgElement* fgOrderedGet(struct _FG_BOX_ORDERED_ELEMENTS_* self, const AbsRect* area, const AbsRect* cache);
template<fgFlag FLAGS>
inline fgElement* fgOrderedVec(struct _FG_BOX_ORDERED_ELEMENTS_* order, AbsVec v);

// Set a bunch of memory to zero after a given subclass
template<class T, class SUBCLASS>
void memsubset(T* p, int v) { memset(reinterpret_cast<char*>(p) + sizeof(SUBCLASS), v, sizeof(T) - sizeof(SUBCLASS)); }
template<class T, class SUBCLASS>
void memsubcpy(T* dest, const T* src)
{
  MEMCPY(((char*)dest) + sizeof(SUBCLASS), sizeof(T) - sizeof(SUBCLASS), ((char*)src) + sizeof(SUBCLASS), sizeof(T) - sizeof(SUBCLASS));
}
template<class T>
T* memcpyalloc(const T* src, size_t len)
{
  T* r = (T*)malloc(len * sizeof(T));
  MEMCPY(r, len * sizeof(T), src, len * sizeof(T));
  return r;
}

struct _FG_MESSAGEQUEUE {
  bss_util::RWLock lock;
  std::atomic<size_t> length;
  std::atomic<size_t> capacity;
  std::atomic<void*> mem;
};

struct QUEUEDMESSAGE
{
  uint32_t sz;
  fgElement* e;
  FG_Msg msg;
};
#endif
