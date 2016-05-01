// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_IMPLEMENTATION_H__
#define _FG_IMPLEMENTATION_H__

#include "fgRoot.h"
#include "fgTextbox.h"
#include "fgWindow.h"
#include "fgMenu.h"

#ifdef  __cplusplus
extern "C" {
#endif

// You can pass this to allow a library to work with an arbitrary feather implementation by using the function pointers instead of a static link.
typedef struct _FG_IMPLEMENTATION {
  void* (FG_FASTCALL *fgCreateImageDef)(fgFlag flags, const char* data, size_t length, unsigned int color, const CRect* uv);
  void* (FG_FASTCALL *fgCreateTextDef)(fgFlag flags, const char* text, const char* font, unsigned short fontsize, unsigned short lineheight, unsigned int color);
  //void* (FG_FASTCALL *fgCreateVectorDef(const void* data, size_t length);
  fgElement* (FG_FASTCALL *fgLoadDef)(void* def, const fgTransform* transform, int order);
  void* (FG_FASTCALL *fgCloneDef)(void* def);
  void (FG_FASTCALL *fgDestroyDef)(void* def);

  fgElement* (FG_FASTCALL *fgButton_Create)(fgElement* item, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
  fgElement* (FG_FASTCALL *fgWindow_Create)(const char* caption, fgFlag flags, const fgTransform* transform);
  fgRoot* (FG_FASTCALL *fgInitialize)();
  fgRoot* (FG_FASTCALL *fgSingleton)();
  char (FG_FASTCALL *fgMessageLoop)(fgRoot* root);
  char (FG_FASTCALL *fgLoadExtension)(const char* extname, void* fg, size_t sz);

  void (FG_FASTCALL *fgTerminate)(fgRoot* root);
  void (FG_FASTCALL *fgElement_Init)(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
  void (FG_FASTCALL *fgElement_Destroy)(fgElement* self);
} fgImplementation;

FG_EXTERN void __stdcall fgLoadImplementation(fgImplementation* fg);

#ifdef  __cplusplus
}
#endif

#endif
