// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FEATHER_CPP_H__
#define __FEATHER_CPP_H__

#include "fgResource.h"
#include "fgText.h"
#include "fgRoot.h"
#include "bss-util\cArraySort.h"
#include "fgLayout.h"
#include "bss-util\cHash.h"
#include "bss-util\cAVLtree.h"

template<void* (FG_FASTCALL *CLONE)(void*), void (FG_FASTCALL *DESTROY)(void*)>
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
  template<void (FG_FASTCALL *DESTROY)(T*), void (FG_FASTCALL *CONSTRUCT)(T*, Args...)>
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
  template<void (FG_FASTCALL *DESTROY)(T*), void (FG_FASTCALL *CONSTRUCT)(T*)>
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

extern BSS_FORCEINLINE void* FG_FASTCALL fgCloneResourceCpp(void* r) { return fgroot_instance->backend.fgCloneResource(r); }
extern BSS_FORCEINLINE void FG_FASTCALL fgDestroyResourceCpp(void* r) { return fgroot_instance->backend.fgDestroyResource(r); }
extern BSS_FORCEINLINE void* FG_FASTCALL fgCloneFontCpp(void* r) { return fgroot_instance->backend.fgCloneFont(r); }
extern BSS_FORCEINLINE void FG_FASTCALL fgDestroyFontCpp(void* r) { return fgroot_instance->backend.fgDestroyFont(r); }

typedef bss_util::cDynArray<fgArbitraryRef<fgCloneResourceCpp, fgDestroyResourceCpp>, FG_UINT, bss_util::CARRAY_CONSTRUCT> fgResourceArray;
typedef bss_util::cDynArray<fgArbitraryRef<fgCloneFontCpp, fgDestroyFontCpp>, FG_UINT, bss_util::CARRAY_CONSTRUCT> fgFontArray;

template<class T>
auto DynGet(const fgVector& r, FG_UINT i) { return ((T&)r)[i]; }
template<class T>
auto DynGetP(const fgVector& r, FG_UINT i) { return ((T&)r).begin()+i; }

template<class T>
char DynArrayRemove(T& a, FG_UINT index)
{
  if(index >= a.Length())
    return 0;
  a.Remove(index);
  return 1;
}

typedef typename fgConstruct<fgStyleLayout, const char*, const char*, fgFlag, const fgTransform*, short, int>::fgConstructor<fgStyleLayout_Destroy, fgStyleLayout_Init> fgStyleLayoutConstruct;
typedef typename fgConstruct<fgClassLayout, const char*, const char*, fgFlag, const fgTransform*, short, int>::fgConstructor<fgClassLayout_Destroy, fgClassLayout_Init> fgClassLayoutConstruct;

extern BSS_FORCEINLINE char fgSortStyleLayout(const fgStyleLayoutConstruct& l, const fgStyleLayoutConstruct& r) { return -SGNCOMPARE(l.order, r.order); }
extern BSS_FORCEINLINE char fgSortClassLayout(const fgClassLayoutConstruct& l, const fgClassLayoutConstruct& r) { return -SGNCOMPARE(l.style.order, r.style.order); }
extern BSS_FORCEINLINE void fgStandardDrawElement(fgElement* self, fgElement* hold, const AbsRect* area, size_t dpi, AbsRect& curarea, bool& clipping);

static const int UNICODE_TERMINATOR = 0;

typedef bss_util::cArraySort<fgStyleLayoutConstruct, fgSortStyleLayout, size_t, bss_util::CARRAY_CONSTRUCT> fgStyleLayoutArray;
typedef bss_util::cDynArray<typename fgConstruct<fgStyle>::fgConstructor<fgStyle_Destroy, fgStyle_Init>, size_t, bss_util::CARRAY_CONSTRUCT> fgStyleArray;
typedef bss_util::cArraySort<fgClassLayoutConstruct, fgSortClassLayout, size_t, bss_util::CARRAY_CONSTRUCT> fgClassLayoutArray;

struct __kh_fgRadioGroup_t;
extern __inline struct __kh_fgRadioGroup_t* fgRadioGroup_init();
extern void fgRadioGroup_destroy(struct __kh_fgRadioGroup_t*);

struct __kh_fgFunctionMap_t;
extern __inline struct __kh_fgFunctionMap_t* fgFunctionMap_init();
extern void fgFunctionMap_destroy(struct __kh_fgFunctionMap_t*);

template<FG_MSGTYPE type, typename... Args>
inline size_t _sendmsg(fgElement* self, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgroot_instance->backend.behaviorhook)(self, &msg);
}

template<FG_MSGTYPE type, typename... Args>
inline size_t _sendsubmsg(fgElement* self, unsigned short sub, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = sub;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return (*fgroot_instance->backend.behaviorhook)(self, &msg);
}

FG_EXTERN bss_util::cHash<std::pair<fgElement*, unsigned short>, void(FG_FASTCALL *)(struct _FG_ELEMENT*, const FG_Msg*)> fgListenerHash;

inline FG_UINT fgStyleGetMask() { return 0; }

template<typename Arg, typename... Args>
inline FG_UINT fgStyleGetMask(Arg arg, Args... args)
{
  return fgStyle_GetName(arg, false) | fgStyleGetMask(args...);
}

BSS_FORCEINLINE size_t fgStandardNeutralSetStyle(fgElement* self, const char* style, unsigned short sub = FGSETSTYLE_NAME)
{
  return _sendsubmsg<FG_SETSTYLE, const void*, size_t>(self, sub, style, fgStyleGetMask("neutral", "hover", "active", "disable"));
}

BSS_FORCEINLINE size_t fgMaskSetStyle(fgElement* self, const char* style, FG_UINT mask)
{
  return _sendsubmsg<FG_SETSTYLE, const void*, size_t>(self, FGSETSTYLE_NAME, style, mask);
}

BSS_FORCEINLINE FABS fgResolveUnit(fgElement* self, FABS x, size_t unit)
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
    return x * ((FABS)self->GetDPI() / (FABS)fgroot_instance->dpi);
  }
}

BSS_FORCEINLINE void fgResolveRectUnit(fgElement* self, AbsRect& r, size_t subtype)
{
  r.left = fgResolveUnit(self, r.left, (subtype & FGUNIT_LEFT_MASK) >> FGUNIT_LEFT);
  r.top = fgResolveUnit(self, r.top, (subtype & FGUNIT_TOP_MASK) >> FGUNIT_TOP);
  r.right = fgResolveUnit(self, r.right, (subtype & FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT);
  r.bottom = fgResolveUnit(self, r.bottom, (subtype & FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM);
}

BSS_FORCEINLINE void fgResolveVecUnit(fgElement* self, AbsVec& v, size_t subtype)
{
  v.x = fgResolveUnit(self, v.x, (subtype & FGUNIT_X_MASK) >> FGUNIT_X);
  v.y = fgResolveUnit(self, v.y, (subtype & FGUNIT_Y_MASK) >> FGUNIT_Y);
}

extern "C" {
  FG_EXTERN size_t FG_FASTCALL fgUTF8toUTF32(const char*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t FG_FASTCALL fgUTF32toUTF8(const int*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t FG_FASTCALL fgUTF32toUTF16(const int*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen);
  FG_EXTERN size_t FG_FASTCALL fgUTF16toUTF32(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, int*BSS_RESTRICT output, size_t buflen);
}

namespace bss_util { struct cXMLNode; }

extern fgSkin* FG_FASTCALL fgSkins_LoadNodeXML(fgSkinBase* self, const bss_util::cXMLNode* root);
extern inline __kh_fgSkins_t *kh_init_fgSkins();
extern void FG_FASTCALL fgStyle_LoadAttributesXML(struct _FG_STYLE* self, const bss_util::cXMLNode* cur, int flags, struct _FG_SKIN_BASE* root, const char* path, char** id);
extern int FG_FASTCALL fgStyle_NodeEvalTransform(const bss_util::cXMLNode* node, fgTransform& t);

#endif
