// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_BACKEND_H__
#define __FG_BACKEND_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct _FG_ROOT;
struct __VECTOR__KeyValue;
struct _FG_FONT_DESC;

typedef struct _FG_BACKEND {
  size_t(MSC_FASTCALL *GCC_FASTCALL behaviorhook)(struct _FG_ELEMENT* self, const FG_Msg* msg);
  void* (MSC_FASTCALL *GCC_FASTCALL fgCreateFont)(fgFlag flags, const char* font, unsigned int fontsize, const fgIntVec* dpi);
  void* (MSC_FASTCALL *GCC_FASTCALL fgCloneFont)(void* font, const struct _FG_FONT_DESC* desc);
  void (MSC_FASTCALL *GCC_FASTCALL fgDestroyFont)(void* font);
  void (MSC_FASTCALL *GCC_FASTCALL fgDrawFont)(void* font, const int* text, size_t len, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data, void* layout);
  void* (MSC_FASTCALL *GCC_FASTCALL fgFontLayout)(void* font, const int* text, size_t len, float lineheight, float letterspacing, AbsRect* area, fgFlag flags, void* prevlayout);
  void (MSC_FASTCALL *GCC_FASTCALL fgFontGet)(void* font, struct _FG_FONT_DESC* desc);
  void* (MSC_FASTCALL *GCC_FASTCALL fgCreateResource)(fgFlag flags, const char* data, size_t length);
  void* (MSC_FASTCALL *GCC_FASTCALL fgCloneResource)(void* res, fgElement* src);
  void (MSC_FASTCALL *GCC_FASTCALL fgDestroyResource)(void* res);
  void (MSC_FASTCALL *GCC_FASTCALL fgDrawResource)(void* res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data);
  void (MSC_FASTCALL *GCC_FASTCALL fgResourceSize)(void* res, const CRect* uv, AbsVec* dim, fgFlag flags);

  size_t(MSC_FASTCALL *GCC_FASTCALL fgFontIndex)(void* font, const int* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, void* layout);
  AbsVec(MSC_FASTCALL *GCC_FASTCALL fgFontPos)(void* font, const int* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, void* layout);
  void (MSC_FASTCALL *GCC_FASTCALL fgDrawLines)(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center, const fgDrawAuxData* data);
  fgElement* (MSC_FASTCALL *GCC_FASTCALL fgCreate)(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
  short (MSC_FASTCALL *GCC_FASTCALL fgMessageMap)(const char* name);
  void (MSC_FASTCALL *GCC_FASTCALL fgUserDataMap)(fgElement* self, struct __VECTOR__KeyValue* pairs);

  void (MSC_FASTCALL *GCC_FASTCALL fgPushClipRect)(const AbsRect* clip, const fgDrawAuxData* data);
  AbsRect (MSC_FASTCALL *GCC_FASTCALL fgPeekClipRect)(const fgDrawAuxData* data);
  void (MSC_FASTCALL *GCC_FASTCALL fgPopClipRect)(const fgDrawAuxData* data);
  void (MSC_FASTCALL *GCC_FASTCALL fgDragStart)(char type, void* data, fgElement* draw);
  void (MSC_FASTCALL *GCC_FASTCALL fgSetCursor)(unsigned int type, void* custom);
  void (MSC_FASTCALL *GCC_FASTCALL fgClipboardCopy)(unsigned int type, const void* data, size_t length);
  char (MSC_FASTCALL *GCC_FASTCALL fgClipboardExists)(unsigned int type);
  const void* (MSC_FASTCALL *GCC_FASTCALL fgClipboardPaste)(unsigned int type, size_t* length);
  void (MSC_FASTCALL *GCC_FASTCALL fgClipboardFree)(const void* mem);
  void (MSC_FASTCALL *GCC_FASTCALL fgDirtyElement)(fgElement* elem);

  char (MSC_FASTCALL *GCC_FASTCALL fgMessageLoop)(struct _FG_ROOT* root);
  char (MSC_FASTCALL *GCC_FASTCALL fgLoadExtension)(const char* extname, void* fg, size_t sz);
} fgBackend;


FG_EXTERN struct _FG_ROOT* FG_FASTCALL fgInitialize();
FG_EXTERN void FG_FASTCALL fgTerminate(struct _FG_ROOT* root);

FG_EXTERN size_t FG_FASTCALL fgBehaviorHookDefault(fgElement* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgBehaviorHookListener(fgElement* self, const FG_Msg* msg);
FG_EXTERN void* FG_FASTCALL fgCreateFontDefault(fgFlag flags, const char* font, unsigned int fontsize, const fgIntVec* dpi);
FG_EXTERN void* FG_FASTCALL fgCloneFontDefault(void* font, const struct _FG_FONT_DESC* desc);
FG_EXTERN void FG_FASTCALL fgDestroyFontDefault(void* font);
FG_EXTERN void FG_FASTCALL fgDrawFontDefault(void* font, const int* text, size_t len, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data, void* layout);
FG_EXTERN void* FG_FASTCALL fgFontLayoutDefault(void* font, const int* text, size_t len, float lineheight, float letterspacing, AbsRect* area, fgFlag flag, void* prevlayouts);
FG_EXTERN void FG_FASTCALL fgFontGetDefault(void* font, struct _FG_FONT_DESC* desc);
FG_EXTERN void* FG_FASTCALL fgCreateResourceDefault(fgFlag flags, const char* data, size_t length);
FG_EXTERN void* FG_FASTCALL fgCloneResourceDefault(void* res, fgElement* src);
FG_EXTERN void FG_FASTCALL fgDestroyResourceDefault(void* res);
FG_EXTERN void FG_FASTCALL fgDrawResourceDefault(void* res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data);
FG_EXTERN void FG_FASTCALL fgResourceSizeDefault(void* res, const CRect* uv, AbsVec* dim, fgFlag flags);

FG_EXTERN size_t FG_FASTCALL fgFontIndexDefault(void* font, const int* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, void* cache);
FG_EXTERN AbsVec FG_FASTCALL fgFontPosDefault(void* font, const int* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, void* cache);
FG_EXTERN void FG_FASTCALL fgDrawLinesDefault(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center, const fgDrawAuxData* data);
FG_EXTERN fgElement* FG_FASTCALL fgCreateDefault(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN short FG_FASTCALL fgMessageMapDefault(const char* name);
FG_EXTERN void FG_FASTCALL fgUserDataMapDefault(fgElement* self, struct __VECTOR__KeyValue* pairs);
FG_EXTERN void FG_FASTCALL fgUserDataMapDefaultProcess(fgElement* self, struct _FG_KEY_VALUE* pair);
FG_EXTERN void FG_FASTCALL fgUserDataMapCallbacks(fgElement* self, struct __VECTOR__KeyValue* pairs);
FG_EXTERN void FG_FASTCALL fgUserDataMapCallbacksProcess(fgElement* self, struct _FG_KEY_VALUE* pair);

FG_EXTERN void FG_FASTCALL fgPushClipRectDefault(const AbsRect* clip, const fgDrawAuxData* data);
FG_EXTERN AbsRect FG_FASTCALL fgPeekClipRectDefault(const fgDrawAuxData* data);
FG_EXTERN void FG_FASTCALL fgPopClipRectDefault(const fgDrawAuxData* data);
FG_EXTERN void FG_FASTCALL fgDragStartDefault(char type, void* data, fgElement* draw);
FG_EXTERN void FG_FASTCALL fgSetCursorDefault(unsigned int type, void* custom);
FG_EXTERN void FG_FASTCALL fgClipboardCopyDefault(unsigned int type, const void* data, size_t length);
FG_EXTERN char FG_FASTCALL fgClipboardExistsDefault(unsigned int type);
FG_EXTERN const void* FG_FASTCALL fgClipboardPasteDefault(unsigned int type, size_t* length);
FG_EXTERN void FG_FASTCALL fgClipboardFreeDefault(const void* mem);
FG_EXTERN void FG_FASTCALL fgDirtyElementDefault(fgElement* elem);

FG_EXTERN char FG_FASTCALL fgMessageLoopDefault(struct _FG_ROOT* root);
FG_EXTERN char FG_FASTCALL fgLoadExtensionDefault(const char* extname, void* fg, size_t sz);
#ifdef  __cplusplus
}
#endif

#endif
