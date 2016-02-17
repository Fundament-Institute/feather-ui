// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_IMPLEMENTATION_H__
#define __FG_IMPLEMENTATION_H__

#include "fgRoot.h"
#include "fgTextbox.h"
#include "fgTopWindow.h"
#include "fgMenu.h"

#ifdef  __cplusplus
extern "C" {
#endif

// You can pass this to allow a library to work with an arbitrary feather implementation by using the function pointers instead of a static link.
typedef struct FG_IMPLEMENTATION {
  void* (FG_FASTCALL *fgCreateImageDef)(fgFlag flags, const char* data, size_t length, unsigned int color, const CRect* uv);
  void* (FG_FASTCALL *fgCreateTextDef)(fgFlag flags, const char* text, const char* font, unsigned short fontsize, unsigned short lineheight, unsigned int color);
  //void* (FG_FASTCALL *fgCreateVectorDef(const void* data, size_t length);
  fgChild* (FG_FASTCALL *fgLoadDef)(void* def, const fgElement* element, int order);
  void* (FG_FASTCALL *fgCloneDef)(void* def);
  void (FG_FASTCALL *fgDestroyDef)(void* def);

  fgChild* (FG_FASTCALL *fgButton_Create)(fgChild* item, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
  fgChild* (FG_FASTCALL *fgTopWindow_Create)(const char* caption, fgFlag flags, const fgElement* element);
  fgRoot* (FG_FASTCALL *fgInitialize)();
  fgRoot* (FG_FASTCALL *fgSingleton)();
  char (FG_FASTCALL *fgMessageLoop)(fgRoot* root);
  char (FG_FASTCALL *fgLoadExtension)(void* fg, const char* extname);

  void (FG_FASTCALL *fgTerminate)(fgRoot* root);
  void (FG_FASTCALL *fgChild_Init)(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
  void (FG_FASTCALL *fgChild_Destroy)(fgChild* self);
} fgImplementation;

FG_EXTERN void __stdcall fgLoadImplementation(fgImplementation* fg);

#ifdef  __cplusplus
}
#endif

#endif
