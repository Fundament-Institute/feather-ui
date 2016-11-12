// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_BACKEND_H__
#define __FG_BACKEND_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct _FG_ROOT;

typedef struct _FG_BACKEND {
  size_t(FG_FASTCALL *behaviorhook)(struct _FG_ELEMENT* self, const FG_Msg* msg);
  void* (FG_FASTCALL *fgCreateFont)(fgFlag flags, const char* font, unsigned int fontsize, unsigned int dpi);
  void* (FG_FASTCALL *fgCopyFont)(void* font, unsigned int fontsize, unsigned int dpi);
  void* (FG_FASTCALL *fgCloneFont)(void* font);
  void (FG_FASTCALL *fgDestroyFont)(void* font);
  void* (FG_FASTCALL *fgDrawFont)(void* font, const int* text, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, void* cache);
  void (FG_FASTCALL *fgFontSize)(void* font, const int* text, float lineheight, float letterspacing, AbsRect* area, fgFlag flags);
  void (FG_FASTCALL *fgFontGet)(void* font, float* lineheight, unsigned int* size, unsigned int* dpi);
  void* (FG_FASTCALL *fgCreateResource)(fgFlag flags, const char* data, size_t length);
  void* (FG_FASTCALL *fgCloneResource)(void* res);
  void (FG_FASTCALL *fgDestroyResource)(void* res);
  void (FG_FASTCALL *fgDrawResource)(void* res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags);
  void (FG_FASTCALL *fgResourceSize)(void* res, const CRect* uv, AbsVec* dim, fgFlag flags);

  size_t(FG_FASTCALL *fgFontIndex)(void* font, const int* text, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, void* cache);
  AbsVec(FG_FASTCALL *fgFontPos)(void* font, const int* text, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, void* cache);
  void (FG_FASTCALL *fgDrawLines)(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center);
  fgElement* (FG_FASTCALL *fgCreate)(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
  short (FG_FASTCALL *fgMessageMap)(const char* name);

  void (FG_FASTCALL *fgPushClipRect)(const AbsRect* clip);
  AbsRect (FG_FASTCALL *fgPeekClipRect)();
  void (FG_FASTCALL *fgPopClipRect)();
  void (FG_FASTCALL *fgDragStart)(char type, void* data, fgElement* draw);
  void (FG_FASTCALL *fgSetCursor)(unsigned int type, void* custom);
  void (FG_FASTCALL *fgClipboardCopy)(unsigned int type, const void* data, size_t length);
  char (FG_FASTCALL *fgClipboardExists)(unsigned int type);
  const void* (FG_FASTCALL *fgClipboardPaste)(unsigned int type, size_t* length);
  void (FG_FASTCALL *fgClipboardFree)(const void* mem);
  void (FG_FASTCALL *fgDirtyElement)(fgElement* elem);

  char (FG_FASTCALL *fgMessageLoop)(struct _FG_ROOT* root);
  char (FG_FASTCALL *fgLoadExtension)(const char* extname, void* fg, size_t sz);
} fgBackend;


FG_EXTERN struct _FG_ROOT* FG_FASTCALL fgInitialize();
FG_EXTERN void FG_FASTCALL fgTerminate(struct _FG_ROOT* root);

FG_EXTERN void* FG_FASTCALL fgCreateFontDefault(fgFlag flags, const char* font, unsigned int fontsize, unsigned int dpi);
FG_EXTERN void* FG_FASTCALL fgCopyFontDefault(void* font, unsigned int fontsize, unsigned int dpi);
FG_EXTERN void* FG_FASTCALL fgCloneFontDefault(void* font);
FG_EXTERN void FG_FASTCALL fgDestroyFontDefault(void* font);
FG_EXTERN void* FG_FASTCALL fgDrawFontDefault(void* font, const int* text, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, void* cache);
FG_EXTERN void FG_FASTCALL fgFontSizeDefault(void* font, const int* text, float lineheight, float letterspacing, AbsRect* area, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgFontGetDefault(void* font, float* lineheight, unsigned int* size, unsigned int* dpi);
FG_EXTERN void* FG_FASTCALL fgCreateResourceDefault(fgFlag flags, const char* data, size_t length);
FG_EXTERN void* FG_FASTCALL fgCloneResourceDefault(void* res);
FG_EXTERN void FG_FASTCALL fgDestroyResourceDefault(void* res);
FG_EXTERN void FG_FASTCALL fgDrawResourceDefault(void* res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgResourceSizeDefault(void* res, const CRect* uv, AbsVec* dim, fgFlag flags);

FG_EXTERN size_t FG_FASTCALL fgFontIndexDefault(void* font, const int* text, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, void* cache);
FG_EXTERN AbsVec FG_FASTCALL fgFontPosDefault(void* font, const int* text, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, void* cache);
FG_EXTERN void FG_FASTCALL fgDrawLinesDefault(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center);
FG_EXTERN fgElement* FG_FASTCALL fgCreateDefault(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN short FG_FASTCALL fgMessageMapDefault(const char* name);

FG_EXTERN void FG_FASTCALL fgPushClipRectDefault(const AbsRect* clip);
FG_EXTERN AbsRect FG_FASTCALL fgPeekClipRectDefault();
FG_EXTERN void FG_FASTCALL fgPopClipRectDefault();
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
