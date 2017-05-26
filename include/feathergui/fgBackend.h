// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_BACKEND_H__
#define __FG_BACKEND_H__

#include "fgElement.h"
#include "fgLayout.h"
#include <stdarg.h>

#ifdef  __cplusplus
extern "C" {
#endif

struct _FG_ROOT;
struct __VECTOR__KeyValue;
struct _FG_FONT_DESC;

typedef struct _FG_BACKEND {
  enum FGTEXTFMT BackendTextFormat;
  fgFont(*fgCreateFont)(fgFlag flags, const char* family, short weight, char italic, unsigned int size, const AbsVec* dpi);
  fgFont(*fgCloneFont)(fgFont font, const struct _FG_FONT_DESC* desc);
  void (*fgDestroyFont)(fgFont font);
  void (*fgDrawFont)(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data, void* layout);
  void* (*fgFontLayout)(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, AbsRect* area, fgFlag flags, const AbsVec* dpi, void* prevlayout);
  void (*fgFontGet)(fgFont font, struct _FG_FONT_DESC* desc);
  size_t(*fgFontIndex)(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, const AbsVec* dpi, void* layout);
  AbsVec(*fgFontPos)(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, const AbsVec* dpi, void* layout);

  fgAsset(*fgCreateAssetFile)(fgFlag flags, const char* file, const AbsVec* dpi);
  fgAsset(*fgCreateAsset)(fgFlag flags, const char* data, size_t length, const AbsVec* dpi);
  fgAsset(*fgCloneAsset)(fgAsset asset, fgElement* src);
  void (*fgDestroyAsset)(fgAsset asset);
  void (*fgDrawAsset)(fgAsset asset, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data);
  void (*fgAssetSize)(fgAsset asset, const CRect* uv, AbsVec* dim, fgFlag flags, const AbsVec* dpi);

  void (*fgDrawLines)(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center, const fgDrawAuxData* data);
  fgElement* (*fgCreate)(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units);
  const char* (*fgFlagMap)(const char* type, fgFlag flag);
  fgMsgType (*fgMessageMap)(const char* name);
  void (*fgUserDataMap)(fgElement* self, struct __VECTOR__KeyValue* pairs);

  void (*fgPushClipRect)(const AbsRect* clip, const fgDrawAuxData* data);
  AbsRect (*fgPeekClipRect)(const fgDrawAuxData* data);
  void (*fgPopClipRect)(const fgDrawAuxData* data);
  void (*fgDragStart)(char type, void* data, fgElement* draw);
  void (*fgSetCursor)(unsigned int type, void* custom);
  void (*fgClipboardCopy)(unsigned int type, const void* data, size_t length);
  char (*fgClipboardExists)(unsigned int type);
  const void* (*fgClipboardPaste)(unsigned int type, size_t* length);
  void (*fgClipboardFree)(const void* mem);
  void (*fgDirtyElement)(fgElement* elem);

  char (*fgProcessMessages)();
  size_t (*fgLoadExtension)(const char* extname, void* fg, size_t sz);
  int(*fgLogHook)(char level, const char* format, va_list args);
  void (*fgTerminate)();
} fgBackend;


FG_EXTERN struct _FG_ROOT* fgInitialize();

FG_EXTERN void* fgCreateFontDefault(fgFlag flags, const char* family, short weight, char italic, unsigned int size, const AbsVec* dpi);
FG_EXTERN void* fgCloneFontDefault(void* font, const struct _FG_FONT_DESC* desc);
FG_EXTERN void fgDestroyFontDefault(void* font);
FG_EXTERN void fgDrawFontDefault(void* font, const void* text, size_t len, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data, void* layout);
FG_EXTERN void* fgFontLayoutDefault(void* font, const void* text, size_t len, float lineheight, float letterspacing, AbsRect* area, fgFlag flag, const AbsVec* dpi, void* prevlayouts);
FG_EXTERN void fgFontGetDefault(void* font, struct _FG_FONT_DESC* desc);
FG_EXTERN size_t fgFontIndexDefault(void* font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, const AbsVec* dpi, void* cache);
FG_EXTERN AbsVec fgFontPosDefault(void* font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, const AbsVec* dpi, void* cache);

FG_EXTERN fgAsset fgCreateAssetFileDefault(fgFlag flags, const char* file, const AbsVec* dpi);
FG_EXTERN fgAsset fgCreateAssetDefault(fgFlag flags, const char* data, size_t length, const AbsVec* dpi);
FG_EXTERN fgAsset fgCloneAssetDefault(fgAsset asset, fgElement* src);
FG_EXTERN void fgDestroyAssetDefault(fgAsset asset);
FG_EXTERN void fgDrawAssetDefault(fgAsset asset, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data);
FG_EXTERN void fgAssetSizeDefault(fgAsset asset, const CRect* uv, AbsVec* dim, fgFlag flags, const AbsVec* dpi);

FG_EXTERN void fgDrawLinesDefault(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center, const fgDrawAuxData* data);
FG_EXTERN fgElement* fgCreateDefault(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units);
FG_EXTERN const char* fgFlagMapDefault(const char* type, fgFlag flag);
FG_EXTERN fgMsgType fgMessageMapDefault(const char* name);
FG_EXTERN void fgUserDataMapDefault(fgElement* self, struct __VECTOR__KeyValue* pairs);
FG_EXTERN void fgUserDataMapDefaultProcess(fgElement* self, struct _FG_KEY_VALUE* pair);
FG_EXTERN void fgUserDataMapCallbacks(fgElement* self, struct __VECTOR__KeyValue* pairs);
FG_EXTERN void fgUserDataMapCallbacksProcess(fgElement* self, struct _FG_KEY_VALUE* pair);

FG_EXTERN void fgPushClipRectDefault(const AbsRect* clip, const fgDrawAuxData* data);
FG_EXTERN AbsRect fgPeekClipRectDefault(const fgDrawAuxData* data);
FG_EXTERN void fgPopClipRectDefault(const fgDrawAuxData* data);
FG_EXTERN void fgDragStartDefault(char type, void* data, fgElement* draw);
FG_EXTERN void fgSetCursorDefault(unsigned int type, void* custom);
FG_EXTERN void fgClipboardCopyDefault(unsigned int type, const void* data, size_t length);
FG_EXTERN char fgClipboardExistsDefault(unsigned int type);
FG_EXTERN const void* fgClipboardPasteDefault(unsigned int type, size_t* length);
FG_EXTERN void fgClipboardFreeDefault(const void* mem);
FG_EXTERN void fgDirtyElementDefault(fgElement* elem);

FG_EXTERN size_t fgBehaviorHookDefault(fgElement* self, const FG_Msg* msg);
FG_EXTERN size_t fgBehaviorHookSimple(fgElement* self, const FG_Msg* msg);
FG_EXTERN char fgProcessMessagesDefault();
FG_EXTERN size_t fgLoadExtensionDefault(const char* extname, void* fg, size_t sz);
FG_EXTERN void fgTerminateDefault();
FG_EXTERN int fgLogHookDefault(char level, const char* format, va_list args);

FG_EXTERN struct _FG_ROOT* fgLoadBackend(const char* dll);
FG_EXTERN void fgUnloadBackend();

#ifdef  __cplusplus
}
#endif

#endif
