// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FEATHER_CPP_H__
#define __FEATHER_CPP_H__

#include "fgResource.h"
#include "fgText.h"
#include "fgRoot.h"
#include "bss-util/ArraySort.h"
#include "fgLayout.h"
#include "bss-util/Hash.h"
#include "bss-util/AVLTree.h"
#include "bss-util/RWLock.h"
#include "bss-util/DisjointSet.h"

#ifdef BSS_64BIT
#define kh_ptr_hash_func(key) kh_int64_hash_func((uint64_t)key)
#else
#define kh_ptr_hash_func kh_int_hash_func
#endif

//#define FG_NO_TEXT_CACHE // Uncomment this if text is leaking somewhere and you don't know where.

struct _FG_ROOT;
extern struct _FG_ROOT* fgroot_instance;

// This is a template implementation of malloc that we can overload for memory leak detection
#ifdef BSS_DEBUG

extern bool fgTextFreeCheck(void* p);
extern void fgTextLeakDump();

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
  fgLeakTracker() { }
  ~fgLeakTracker() { }

  void Dump()
  {
    assert(fgroot_instance != 0);
    fgLog("--- Memory leaks---\n");
    LEAKINFO* pinfo;
    for(auto curiter = _leakinfo.begin(); curiter.IsValid(); ++curiter)
    {
      pinfo = _leakinfo.GetValue(*curiter);
      if(!pinfo->freecount)
        fgLog("%p (Size: %zi) leaked at %s:%zi\n", pinfo->ptr, pinfo->size, pinfo->file, pinfo->line);

      free((char*)pinfo->file);
      free(pinfo);
    }

    fgTextLeakDump();
    _leakinfo.Clear();
  }
  void Add(void* ptr, size_t size, const char* file, size_t line)
  {
    assert(fgroot_instance != 0);
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
    assert(fgroot_instance != 0);
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
  bss::Hash<void*, LEAKINFO*> _leakinfo;
  FILE* f;
};

template<typename T>
BSS_FORCEINLINE T* fgmalloc(size_t sz, const char* file, size_t line)
{
  T* p = reinterpret_cast<T*>(malloc(sz * sizeof(T)));
  fgLeakTracker::Tracker.Add(p, sz * sizeof(T), file, line);
  return p;
}
BSS_FORCEINLINE void fgfree(void* p, const char* file, size_t line)
{
  assert(!fgTextFreeCheck(p)); // In debug mode, make sure you are freeing text allocated with fgCopyText with fgFreeText
  if(fgLeakTracker::Tracker.Remove(p, file, line))
    free(p);
}

#else
template<typename T>
BSS_FORCEINLINE T* fgmalloc(size_t sz, const char*, size_t) { return reinterpret_cast<T*>(malloc(sz * sizeof(T))); }
BSS_FORCEINLINE void fgfree(void* p, const char*, size_t) { free(p); }
#endif

BSS_FORCEINLINE void fgfreeblank(void* p)
{
  fgfree(p, "", 0);
}

template<class T, typename... Args>
struct fgConstruct
{
  template<void (*DESTROY)(T*), void (*CONSTRUCT)(T*, Args...)>
  struct fgConstructor : public T
  {
    fgConstructor(fgConstructor&& mov) { memcpy(this, &mov, sizeof(T)); bss::bssFill(mov, 0); }
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
    fgConstructor(fgConstructor&& mov) { memcpy(this, &mov, sizeof(T)); bss::bssFill(mov, 0); }
    fgConstructor() { CONSTRUCT((T*)this); }
    ~fgConstructor() { DESTROY((T*)this); }
    operator T&() { return *this; }
    operator const T&() const { return *this; }
    fgConstructor& operator=(fgConstructor&&) = delete;
    fgConstructor& operator=(const fgConstructor&) = delete;
  };
};

struct _FG_DEBUG;
extern struct _FG_DEBUG* fgdebug_instance;

typedef typename fgConstruct<fgSkinLayout, const char*, fgFlag, const fgTransform*, short, int>::fgConstructor<fgSkinLayout_Destroy, fgSkinLayout_Init> fgSkinLayoutConstruct;
typedef typename fgConstruct<fgClassLayout, const char*, const char*, fgFlag, const fgTransform*, short, int>::fgConstructor<fgClassLayout_Destroy, fgClassLayout_Init> fgClassLayoutConstruct;

BSS_FORCEINLINE void fgConstructKeyValue(_FG_KEY_VALUE*) {}
BSS_FORCEINLINE void fgDestructKeyValue(_FG_KEY_VALUE* p)
{
  fgFreeText(p->key, __FILE__, __LINE__);
  if(p->value)
    fgFreeText(p->value, __FILE__, __LINE__);
}
BSS_FORCEINLINE char fgSortStyleLayout(const fgSkinLayoutConstruct& l, const fgSkinLayoutConstruct& r) { return -SGNCOMPARE(l.element.order, r.element.order); }
BSS_FORCEINLINE char fgSortClassLayout(const fgClassLayoutConstruct& l, const fgClassLayoutConstruct& r) { return -SGNCOMPARE(l.element.order, r.element.order); }

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

typedef bss::ArraySort<fgSkinLayoutConstruct, fgSortStyleLayout, size_t, bss::ARRAY_CONSTRUCT> fgSkinLayoutArray;
typedef bss::DynArray<typename fgConstruct<struct _FG_KEY_VALUE>::fgConstructor<fgDestructKeyValue, fgConstructKeyValue>, size_t, bss::ARRAY_CONSTRUCT> fgKeyValueArray;
typedef bss::ArraySort<fgClassLayoutConstruct, fgSortClassLayout, size_t, bss::ARRAY_CONSTRUCT> fgClassLayoutArray;
typedef bss::ArraySort<struct _FG_ELEMENT*> fgElementArray;

struct kh_fgRadioGroup_s;
extern struct kh_fgRadioGroup_s* fgRadioGroup_init();
extern void fgRadioGroup_destroy(struct kh_fgRadioGroup_s*);

struct kh_fgFunctionMap_s;
extern struct kh_fgFunctionMap_s* fgFunctionMap_init();
extern void fgFunctionMap_destroy(struct kh_fgFunctionMap_s*);

template<FG_MSGTYPE type, typename... Args>
inline size_t _sendmsg(fgElement* self, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgroot_instance->fgBehaviorHook)(self, &msg);
}

template<FG_MSGTYPE type, typename... Args>
inline size_t _sendsubmsg(fgElement* self, unsigned short sub, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = sub;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgroot_instance->fgBehaviorHook)(self, &msg);
}

FG_EXTERN bss::Hash<std::pair<fgElement*, unsigned short>, fgDelegateListener> fgListenerHash;

BSS_FORCEINLINE size_t fgSetFlagStyle(fgElement* self, const char* style, bool value = true)
{
  size_t f = fgStyle_GetName(style);
  return _sendsubmsg<FG_SETSTYLE, size_t, size_t>(self, FGSETSTYLE_INDEX, !value ? 0 : f, f);
}

template<FABS(*F)(FABS)>
BSS_FORCEINLINE FABS fgSnapDPI(FABS x, int dpi)
{
  if(dpi != 96 && dpi != 0)
  {
    FABS s = dpi / 96.0f;
    return F(x * s) / s;
  }
  return x;
}
template<FABS(*F)(FABS)>
BSS_FORCEINLINE FABS fgSnapAll(FABS x, int dpi)
{
  if(dpi != 96 && dpi != 0)
  {
    FABS s = dpi / 96.0f;
    return F(x * s) / s;
  }
  return F(x);
}

BSS_FORCEINLINE FABS fgResolveUnit(FABS x, size_t unit, int dpi, FABS lineheight, bool snap)
{
  switch(unit)
  {
  default:
  case FGUNIT_DP:
    break;
  case FGUNIT_SP:
    x *= fgroot_instance->fontscale;
    break;
  case FGUNIT_EM:
    x *= lineheight;
    break;
  case FGUNIT_PX:
    return x * (dpi / 96.0f);
  }

  if(snap)
    x = fgSnapDPI<roundf>(x, dpi);
  assert(!std::isnan(x));
  return x;
}

BSS_FORCEINLINE void fgSnapAbsRect(AbsRect& r, fgFlag flags)
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

BSS_FORCEINLINE void fgResolveRectUnit(AbsRect& r, const fgIntVec& dpi, FABS lineheight, size_t subtype)
{
  r.left = fgResolveUnit(r.left, (subtype & FGUNIT_LEFT_MASK) >> FGUNIT_LEFT, dpi.x, lineheight, (subtype & FGUNIT_SNAP) != 0);
  r.top = fgResolveUnit(r.top, (subtype & FGUNIT_TOP_MASK) >> FGUNIT_TOP, dpi.y, lineheight, (subtype & FGUNIT_SNAP) != 0);
  r.right = fgResolveUnit(r.right, (subtype & FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT, dpi.x, lineheight, (subtype & FGUNIT_SNAP) != 0);
  r.bottom = fgResolveUnit(r.bottom, (subtype & FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM, dpi.y, lineheight, (subtype & FGUNIT_SNAP) != 0);
}

BSS_FORCEINLINE void fgResolveVecUnit(AbsVec& v, const fgIntVec& dpi, FABS lineheight, size_t subtype)
{
  v.x = fgResolveUnit(v.x, (subtype & FGUNIT_X_MASK) >> FGUNIT_X, dpi.x, lineheight, (subtype & FGUNIT_SNAP) != 0);
  v.y = fgResolveUnit(v.y, (subtype & FGUNIT_Y_MASK) >> FGUNIT_Y, dpi.y, lineheight, (subtype & FGUNIT_SNAP) != 0);
}

BSS_FORCEINLINE void fgResolveCRectUnit(CRect& r, const fgIntVec& dpi, FABS lineheight, size_t subtype)
{
  r.left.abs = fgResolveUnit(r.left.abs, (subtype & FGUNIT_LEFT_MASK) >> FGUNIT_LEFT, dpi.x, lineheight, (subtype & FGUNIT_SNAP) != 0);
  r.top.abs = fgResolveUnit(r.top.abs, (subtype & FGUNIT_TOP_MASK) >> FGUNIT_TOP, dpi.y, lineheight, (subtype & FGUNIT_SNAP) != 0);
  r.right.abs = fgResolveUnit(r.right.abs,(subtype & FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT, dpi.x, lineheight, (subtype & FGUNIT_SNAP) != 0);
  if(subtype & FGUNIT_RIGHT_WIDTH)
    r.right.abs += r.left.abs;
  r.bottom.abs = fgResolveUnit(r.bottom.abs, (subtype & FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM, dpi.y, lineheight, (subtype & FGUNIT_SNAP) != 0);
  if(subtype & FGUNIT_BOTTOM_HEIGHT)
    r.bottom.abs += r.top.abs;
}

BSS_FORCEINLINE void fgResolveCVecUnit(CVec& v, const fgIntVec& dpi, FABS lineheight, size_t subtype)
{
  v.x.abs = fgResolveUnit(v.x.abs, (subtype & FGUNIT_X_MASK) >> FGUNIT_X, dpi.x, lineheight, (subtype & FGUNIT_SNAP) != 0);
  v.y.abs = fgResolveUnit(v.y.abs, (subtype & FGUNIT_Y_MASK) >> FGUNIT_Y, dpi.y, lineheight, (subtype & FGUNIT_SNAP) != 0);
}

extern "C" {
  FG_EXTERN size_t fgUTF8toUTF32(const char*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t fgUTF32toUTF8(const int*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t fgUTF32toUTF16(const int*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t fgUTF16toUTF32(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
}

BSS_FORCEINLINE fgElement* fgLayout_GetNext(fgElement* cur)
{
  while((cur = cur->next) != 0 && (cur->flags&FGELEMENT_BACKGROUND) != 0);
  return cur;
}

BSS_FORCEINLINE fgElement* fgLayout_GetPrev(fgElement* cur)
{
  while((cur = cur->prev) != 0 && (cur->flags&FGELEMENT_BACKGROUND) != 0);
  return cur;
}

namespace bss { struct XMLNode; }
struct __VECTOR__UTF8;
struct __VECTOR__UTF16;
struct __VECTOR__UTF32;

extern fgSkin* fgSkinBase_ParseNodeXML(fgSkinBase* self, const bss::XMLNode* root, const char* path);
extern void fgSkinBase_WriteElementAttributesXML(bss::XMLNode* node, fgSkinElement& e, fgSkinBase* root);
extern void fgSkinBase_WriteStyleAttributesXML(bss::XMLNode* node, fgStyle& s, fgSkinBase* root, const char* type);
extern void fgSkinBase_WriteXML(bss::XMLNode* node, fgSkinBase* base, char toplevel);
extern void fgSkin_WriteXML(bss::XMLNode* node, fgSkin* skin);
static kh_fgSkins_s *kh_init_fgSkins();
extern fgSkin* fgStyle_ParseAttributesXML(struct _FG_STYLE* self, const bss::XMLNode* cur, int flags, struct _FG_SKIN_BASE* root, const char* path, const char** id, fgKeyValueArray* userdata);
extern int fgStyle_NodeEvalTransform(const bss::XMLNode* node, fgTransform& t);
extern fgVector* fgText_Conversion(int type, struct __VECTOR__UTF8* text8, struct __VECTOR__UTF16* text16, struct __VECTOR__UTF32* text32);
extern void fgMenu_Show(struct _FG_MENU* self, bool show);
extern void LList_RemoveAll(fgElement* self);
extern void LList_InsertAll(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT next);
template<fgElement*&(*GET)(fgElement*), fgFlag FLAG>
extern fgElement* LList_Find(fgElement* BSS_RESTRICT self);

extern void fgStyle_WriteFloat(bss::Str& s, float abs);
extern void fgStyle_WriteInt(bss::Str& s, int64_t i);
extern void fgStyle_WriteHex(bss::Str& s, uint64_t i);
extern void fgStyle_WriteAbs(bss::Str& s, float abs, short unit);
extern void fgStyle_WriteCoord(bss::Str& s, const Coord& coord, short unit);
extern bss::Str fgStyle_WriteCVec(const CVec& vec, short units);
extern bss::Str fgStyle_WriteAbsRect(const AbsRect& r, short units);
extern bss::Str fgStyle_WriteCRect(const CRect& r, short units);
extern void fgStyle_WriteFlagsIterate(bss::Str& s, const char* type, const char* divider, fgFlag flags, bool remove);
extern void fgBox_SetSpacing(fgElement* self, AbsVec& spacing, const FG_Msg* msg);

struct _FG_BOX_ORDERED_ELEMENTS_;

template<fgFlag FLAGS>
fgElement* fgOrderedGet(struct _FG_BOX_ORDERED_ELEMENTS_* self, const AbsRect* area, const AbsRect* cache);

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

BSS_FORCEINLINE FABS fgLayout_GetElementMinWidth(fgElement* child)
{
  if(!(child->flags&FGELEMENT_EXPANDX))
    return child->mindim.x;
  FABS l = child->layoutdim.x + child->padding.left + child->padding.right + child->margin.left + child->margin.right;
  return (child->mindim.x >= 0.0f && child->mindim.x > l) ? child->mindim.x : l;
}

BSS_FORCEINLINE FABS fgLayout_GetElementMinHeight(fgElement* child)
{
  if(!(child->flags&FGELEMENT_EXPANDY))
    return child->mindim.y;
  FABS l = child->layoutdim.y + child->padding.top + child->padding.bottom + child->margin.top + child->margin.bottom;
  return (child->mindim.y >= 0.0f && child->mindim.y > l) ? child->mindim.y : l;
}

BSS_FORCEINLINE FABS fgLayout_GetElementWidth(fgElement* child)
{
  FABS w = (child->transform.area.left.rel == child->transform.area.right.rel) ? child->transform.area.right.abs - child->transform.area.left.abs : 0.0f;
  FABS m = fgLayout_GetElementMinWidth(child);
  return (m >= 0.0f && m > w) ? m : w;
}

BSS_FORCEINLINE FABS fgLayout_GetElementHeight(fgElement* child)
{
  FABS h = (child->transform.area.top.rel == child->transform.area.bottom.rel) ? child->transform.area.bottom.abs - child->transform.area.top.abs : 0.0f;
  FABS m = fgLayout_GetElementMinHeight(child);
  return (m >= 0.0f && m > h) ? m : h;
}

inline FABS fgLayout_ExpandX(FABS dimx, fgElement* child)
{
  FABS r = child->transform.area.left.abs + fgLayout_GetElementWidth(child);
  return bssmax(dimx, r);
}

inline FABS fgLayout_ExpandY(FABS dimy, fgElement* child)
{
  FABS r = child->transform.area.top.abs + fgLayout_GetElementHeight(child);
  return bssmax(dimy, r);
}

struct _FG_MESSAGEQUEUE {
  ~_FG_MESSAGEQUEUE() { if(mem.load(std::memory_order_acquire)) free(mem.load(std::memory_order_acquire)); }
  bss::RWLock lock;
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

struct kh_fgStyles_s;

struct fgStyleStatic
{
  fgStyleStatic();
  ~fgStyleStatic();
  void Init();
  void Clear();
  struct kh_fgStyles_s* h;
  fgStyleIndex Masks[sizeof(fgStyleIndex) << 3]; // Holds the appropriate mask for each possible bit

  static fgStyleStatic Instance;
};

#endif
