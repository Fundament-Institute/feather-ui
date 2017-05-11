// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathergui.h"
#include "feathercpp.h"
#include "fgRoot.h"
#include "bss-util/sseVec.h"
#include <limits.h>
#include <math.h>

const fgTransform fgTransform_DEFAULT = { { 0, 0, 0, 0, 0, 1, 0, 1 }, 0, { 0, 0, 0, 0 } };
const fgTransform fgTransform_EMPTY = { { 0, 0, 0, 0, 0, 0, 0, 0 }, 0, { 0, 0, 0, 0 } };
const fgTransform fgTransform_CENTER = { { 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5 }, 0, { 0, 0.5, 0, 0.5 } };
const fgColor fgColor_NONE = { 0 };
const fgColor fgColor_BLACK = { 0xFF000000 };
const fgColor fgColor_WHITE = { 0xFFFFFFFF };
const CRect CRect_EMPTY = { 0,0,0,0,0,0,0,0 };
const AbsVec AbsVec_EMPTY = { 0,0 };
const fgIntVec fgIntVec_EMPTY = { 0,0 };
const fgIntVec fgIntVec_DEFAULTDPI = { 96,96 };

bss::Hash<std::pair<fgElement*, unsigned short>, fgDelegateListener> fgListenerHash;
bss::Hash<char*> fgStringAllocHash;
bss::HashBase<const char*, size_t, true, bss::KH_POINTER_HASHFUNC<const char* const&>, bss::KH_INT_EQUALFUNC<const char*>> fgStringRefHash;

static_assert(sizeof(unsigned int) == sizeof(fgColor), "ERROR: fgColor not size of 32-bit int!");
static_assert(sizeof(FG_Msg) <= sizeof(uint64_t) * 3, "FG_Msg is too big!");
static_assert(sizeof(FG_Msg) == sizeof(void*)*2 + sizeof(uint32_t)*2, "FG_Msg is not 16!");

#ifdef BSS_DEBUG
fgLeakTracker fgLeakTracker::Tracker;
#endif

AbsVec ResolveVec(const CVec* v, const AbsRect* last)
{
  AbsVec r = { v->x.abs, v->y.abs };
  assert(last != 0);
  r.x += fglerp(last->left, last->right, v->x.rel);
  r.y += fglerp(last->top, last->bottom, v->y.rel);
  return r;
}

// This uses a standard inclusive-exclusive rectangle interpretation.
char HitAbsRect(const AbsRect* r, FABS x, FABS y)
{
  assert(r != 0);
  return (x < r->right) && (x >= r->left) && (y < r->bottom) && (y >= r->top);
}

char MsgHitAbsRect(const FG_Msg* msg, const AbsRect* r)
{
  assert(msg != 0 && r != 0);
  return HitAbsRect(r, (FABS)msg->x, (FABS)msg->y);
}

char CompareMargins(const AbsRect* l, const AbsRect* r)
{
  assert(l != 0 && r != 0);
  assert(!std::isnan(l->left) && !std::isnan(r->left));
  assert(!std::isnan(l->top) && !std::isnan(r->top));
  assert(!std::isnan(l->right) && !std::isnan(r->right));
  assert(!std::isnan(l->bottom) && !std::isnan(r->bottom));
  return ((((l->left - r->left) != (l->right + r->right))) << 1)
    | ((((l->top - r->top) != (l->bottom + r->bottom))) << 2)
    | ((((l->left != r->left) || (l->right != r->right))) << 3)
    | ((((l->top != r->top) || (l->bottom != r->bottom))) << 4);
}
char CompareCRects(const CRect* l, const CRect* r)
{
  assert(l != 0 && r != 0);
  assert(!std::isnan(l->left.abs) && !std::isnan(l->left.rel) && !std::isnan(r->left.abs) && !std::isnan(r->left.rel));
  assert(!std::isnan(l->top.abs) && !std::isnan(l->top.rel) && !std::isnan(r->top.abs) && !std::isnan(r->top.rel));
  assert(!std::isnan(l->right.abs) && !std::isnan(l->right.rel) && !std::isnan(r->right.abs) && !std::isnan(r->right.rel));
  assert(!std::isnan(l->bottom.abs) && !std::isnan(l->bottom.rel) && !std::isnan(r->bottom.abs) && !std::isnan(r->bottom.rel));
  return ((((l->left.abs - r->left.abs) != (l->right.abs - r->right.abs))) << 1)
    | ((((l->top.abs - r->top.abs) != (l->bottom.abs - r->bottom.abs))) << 2)
    | ((((l->left.abs != r->left.abs) || (l->right.abs != r->right.abs))) << 3)
    | ((((l->top.abs != r->top.abs) || (l->bottom.abs != r->bottom.abs))) << 4)
    | ((((l->left.rel - r->left.rel) != (l->right.rel - r->right.rel))) << 1)
    | ((((l->top.rel - r->top.rel) != (l->bottom.rel - r->bottom.rel))) << 2)
    | ((((l->left.rel != r->left.rel) || (l->right.rel != r->right.rel))) << 3)
    | ((((l->top.rel != r->top.rel) || (l->bottom.rel != r->bottom.rel))) << 4);
}
char CompareTransforms(const fgTransform* l, const fgTransform* r)
{
  assert(l != 0 && r != 0);
  return CompareCRects(&l->area, &r->area)
    | (((l->center.x.abs != r->center.x.abs) || (l->center.x.rel != r->center.x.rel)) << 5)
    | (((l->center.y.abs != r->center.y.abs) || (l->center.y.rel != r->center.y.rel)) << 6)
    | ((l->rotation != r->rotation) << 7);
}

void MoveCRect(FABS x, FABS y, CRect* r)
{
  AbsVec d = { r->right.abs - r->left.abs, r->bottom.abs - r->top.abs };
  r->left.abs = x;
  r->top.abs = y;
  r->right.abs = x + d.x;
  r->bottom.abs = y + d.y;
}

void ToIntAbsRect(const AbsRect* r, int target[4])
{
  sseVeci(sseVec(BSS_UNALIGNED<const float>(&r->left))) >> BSS_UNALIGNED<int>(target);
}

void ToLongAbsRect(const AbsRect* r, long target[4])
{
#if ULONG_MAX==UINT_MAX
  sseVeci(sseVec(BSS_UNALIGNED<const float>(&r->left))) >> BSS_UNALIGNED<int>((int*)target);
#else
  int hold[4];
  _mm_storeu_si128((__m128i*)hold, _mm_cvtps_epi32(_mm_loadu_ps(&r->left)));
  target[0] = hold[0];
  target[1] = hold[1];
  target[2] = hold[2];
  target[3] = hold[3];
#endif

}

const char* fgCopyText(const char* text, const char* file, size_t line)
{
  if(!text) return 0;
#ifndef FG_NO_TEXT_CACHE
  khiter_t i = fgStringRefHash.Iterator(text);
  if(fgStringRefHash.ExistsIter(i))
  {
    fgStringRefHash.MutableValue(i)++;
    return text;
  }

  i = fgStringAllocHash.Iterator(const_cast<char*>(text));
  if(fgStringAllocHash.ExistsIter(i))
  {
    text = fgStringAllocHash.GetKey(i);
    i = fgStringRefHash.Iterator(text);
    assert(fgStringRefHash.ExistsIter(i));
    fgStringRefHash.MutableValue(i)++;
    return text;
  }

  size_t len = strlen(text) + 1;
  char* ret = (char*)malloc(len);
  memcpy(ret, text, len);
  fgStringAllocHash.Insert(ret);
  fgStringRefHash.Insert(ret, 1);
  return ret;
#else
  size_t len = strlen(text) + 1;
  char* ret = (char*)fgmalloc<char>(len, file, line);
  memcpy(ret, text, len);
  return ret;
#endif
}

bool fgTextFreeCheck(void* p)
{
  return fgStringRefHash.Exists((const char*)p);
}

void fgFreeText(const char* text, const char* file, size_t line)
{
#ifndef FG_NO_TEXT_CACHE
  khiter_t i = fgStringRefHash.Iterator(text);
  if(fgStringRefHash.ExistsIter(i))
  {
    if(--fgStringRefHash.MutableValue(i) <= 0)
    {
      if(!fgStringAllocHash.Remove(const_cast<char*>(text)))
        assert(false);
      if(!fgStringRefHash.Remove(text))
        assert(false);
      free(const_cast<char*>(text));
    }
  }
  else
    assert(false);
#else
  fgfree((char*)text, file, line);
#endif
}

void fgTextLeakDump()
{
  for(auto curiter = fgStringRefHash.begin(); curiter.IsValid(); ++curiter)
  {
    const char* str = fgStringRefHash.GetKey(*curiter);
    size_t refs = fgStringRefHash.GetValue(*curiter);
    fgLog("The string \"%s\" leaked with %zi dangling references\n", str, refs);
    free(const_cast<char*>(str));
  }
}

void fgUpdateMouseState(fgMouseState* state, const FG_Msg* msg)
{
  assert(msg != 0);
  switch(msg->type)
  {
  case FG_MOUSEDOWN:
    state->state |= FGMOUSE_HOVER;
    state->state |= FGMOUSE_INSIDE;
    state->state &= ~FGMOUSE_DRAG;
    break;
  case FG_MOUSEUP:
    state->state |= FGMOUSE_HOVER;
    state->state &= ~FGMOUSE_INSIDE;
    state->state &= ~FGMOUSE_DRAG;
    break;
  case FG_DRAGOVER:
    state->state |= FGMOUSE_DRAG;
    break;
  case FG_MOUSEMOVE:
    state->state |= FGMOUSE_HOVER;
    state->state &= ~FGMOUSE_DRAG;
    break;
  case FG_MOUSEOFF:
    state->state &= ~FGMOUSE_HOVER;
    state->state &= ~FGMOUSE_DRAG;
    break;
  case FG_DROP:
    state->state &= ~FGMOUSE_DRAG;
    state->state &= ~FGMOUSE_INSIDE;
    break;
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEON:
    break;
  case FG_MOUSESCROLL:
    state->x = msg->x;
    state->y = msg->y;
  default:
    return;
  }
  state->x = msg->x;
  state->y = msg->y;
  state->buttons = msg->allbtn;
}

char fgRectIntersect(const AbsRect* l, const AbsRect* r)
{
  return (l->left <= r->right && l->top <= r->bottom && l->right >= r->left && l->bottom >= r->top);
}

void fgRectIntersection(const AbsRect* BSS_RESTRICT l, const AbsRect* BSS_RESTRICT r, AbsRect* out)
{
  out->left = bssmax(l->left, r->left);
  out->top = bssmax(l->top, r->top);
  out->right = bssmin(l->right, r->right);
  out->bottom = bssmin(l->bottom, r->bottom);
}
void fgScaleRectDPI(AbsRect* rect, int dpix, int dpiy)
{
  BSS_ALIGN(16) float scale[4];
  scale[0] = (!dpix) ? 1.0f : ((float)dpix / 96.0f);
  scale[1] = (!dpiy) ? 1.0f : ((float)dpiy / 96.0f);
  scale[2] = scale[0];
  scale[3] = scale[1];
  (sseVec(BSS_UNALIGNED<const float>(&rect->left))*sseVec(scale)) >> BSS_UNALIGNED<float>(&rect->left);
  if(rect->right < rect->left)
    rect->right = rect->left;
  if(rect->bottom < rect->top)
    rect->bottom = rect->top;
}
void fgInvScaleRectDPI(AbsRect* rect, int dpix, int dpiy)
{
  BSS_ALIGN(16) float scale[4];
  scale[0] = (!dpix) ? 1.0f : (96.0f / (float)dpix);
  scale[1] = (!dpiy) ? 1.0f : (96.0f / (float)dpiy);
  scale[2] = scale[0];
  scale[3] = scale[1];
  (sseVec(BSS_UNALIGNED<const float>(&rect->left))*sseVec(scale)) >> BSS_UNALIGNED<float>(&rect->left);
}
void fgScaleVecDPI(AbsVec* v, int dpix, int dpiy)
{
  v->x *= (!dpix) ? 1.0f : ((float)dpix / 96.0f);
  v->y *= (!dpiy) ? 1.0f : ((float)dpiy / 96.0f);
}
void fgInvScaleVecDPI(AbsVec* v, int dpix, int dpiy)
{
  v->x *= (!dpix) ? 1.0f : (96.0f/(float)dpix);
  v->y *= (!dpiy) ? 1.0f : (96.0f/(float)dpiy);
}
void fgResolveDrawRect(const AbsRect* area, AbsRect* outarea, const AbsVec* center, AbsVec* outcenter, fgFlag flags, const fgDrawAuxData* data)
{
  BSS_ALIGN(16) float scale[4];
  scale[0] = (!data->dpi.x) ? 1.0f : ((float)data->dpi.x / 96.0f);
  scale[1] = (!data->dpi.y) ? 1.0f : ((float)data->dpi.y / 96.0f);
  scale[2] = scale[0];
  scale[3] = scale[1];
  (sseVec(BSS_UNALIGNED<const float>(&area->left))*sseVec(scale)) >> BSS_UNALIGNED<float>(&outarea->left);

  fgSnapAbsRect(*outarea, flags);
  if(outarea->right < outarea->left)
    outarea->right = outarea->left;
  if(outarea->bottom < outarea->top)
    outarea->bottom = outarea->top;
  outcenter->x = center->x*scale[0];
  outcenter->y = center->y*scale[1];
}

#ifdef BSS_PLATFORM_WIN32
#include "bss-util/win32_includes.h"

size_t fgUTF8toUTF16(const char*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen)
{
  return (size_t)MultiByteToWideChar(CP_UTF8, 0, input, (int)srclen, output, (int)(!output ? 0 : buflen));
}

size_t fgUTF16toUTF8(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen)
{
  return (size_t)WideCharToMultiByte(CP_UTF8, 0, input, (int)srclen, output, (int)(!output ? 0 : buflen), NULL, NULL);
}

#else
#include <iconv.h>
size_t fgUTF8toUTF16(const char*BSS_RESTRICT input, ptrdiff_t srclen, wchar_t*BSS_RESTRICT output, size_t buflen)
{
static iconv_t iconv_utf8to16 = 0;
  size_t len = srclen < 0 ? strlen(input) : srclen;
  char* out = (char*)output;
  if(!output) return (len * 4) + 1;
  len += 1; // include null terminator
  if(!iconv_utf8to16) iconv_utf8to16 = iconv_open("UTF-8", "UTF-16");
  char* in = (char*)input; // Linux is stupid
  return iconv(iconv_utf8to16, &in, &len, &out, &buflen);
}

extern size_t fgUTF16toUTF8(const wchar_t*BSS_RESTRICT input, ptrdiff_t srclen, char*BSS_RESTRICT output, size_t buflen)
{
  static iconv_t iconv_utf16to8 = 0;
  size_t len = (srclen < 0 ? wcslen(input) : srclen) * 2;
  char* in = (char*)input;
  if(!output) return (len * 2) + 1;
  len += 2; // include null terminator (which is 2 bytes wide here)
  if(!iconv_utf16to8) iconv_utf16to8 = iconv_open("UTF-16", "UTF-8");
  return iconv(iconv_utf16to8, &in, &len, (char**)&output, &buflen);
}
#endif

