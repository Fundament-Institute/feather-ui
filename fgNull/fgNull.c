// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgNull.h"
#include "fgText.h"
#include "fgResource.h"
#include "fgRoot.h"
#include "fgButton.h"
#include "fgMenu.h"
#include "fgTopWindow.h"
#include "bss_defines.h"

#if defined(BSS_DEBUG) && defined(BSS_CPU_x86_64)
#pragma comment(lib, "../bin/feathergui64_d.lib")
#elif defined(BSS_CPU_x86_64)
#pragma comment(lib, "../bin/feathergui64.lib")
#elif defined(BSS_DEBUG)
#pragma comment(lib, "../bin/feathergui_d.lib")
#else
#pragma comment(lib, "../bin/feathergui.lib")
#endif

fgRoot* _fgroot=0; // fgRoot singleton variable


void* FG_FASTCALL fgCreateFont(fgFlag flags, const char* font, unsigned int fontsize, float lineheight, float letterspacing) { return (void*)0xFFFFFFFF; }
void* FG_FASTCALL fgCloneFont(void* font) { return (void*)0xFFFFFFFF; }
void FG_FASTCALL fgDestroyFont(void* font) { }
void FG_FASTCALL fgDrawFont(void* font, const char* text, unsigned int color, const AbsRect* area, FABS rotation, AbsVec* center, fgFlag flags) { }
void FG_FASTCALL fgFontSize(void* font, const char* text, AbsRect* area, fgFlag flags) { }

void* FG_FASTCALL fgCreateResource(fgFlag flags, const char* data, size_t length) { return (void*)0xFFFFFFFF; }
void* FG_FASTCALL fgCloneResource(void* res) { return (void*)0xFFFFFFFF; }
void FG_FASTCALL fgDestroyResource(void* res) { }
void FG_FASTCALL fgDrawResource(void* res, const CRect* uv, unsigned int color, const AbsRect* area, FABS rotation, AbsVec* center, fgFlag flags) { }
void FG_FASTCALL fgResourceSize(void* res, const CRect* uv, AbsVec* dim, fgFlag flags) { }

#define DEFAULT_CREATE(type, init, ...) \
  type* r = (type*)malloc(sizeof(type)); \
  init(r, __VA_ARGS__); \
  return (fgChild*)r

fgChild* FG_FASTCALL fgResource_Create(void* res, const CRect* uv, unsigned int color, fgFlag flags, fgChild* parent, const fgElement* element)
{
  DEFAULT_CREATE(fgResource, fgResource_Init, res, uv, color, flags, parent, element);
}
fgChild* FG_FASTCALL fgText_Create(char* text, void* font, unsigned int color, fgFlag flags, fgChild* parent, const fgElement* element)
{
  DEFAULT_CREATE(fgText, fgText_Init, text, font, color, flags, parent, element);
}
fgChild* FG_FASTCALL fgButton_Create(fgChild* item, fgFlag flags, fgChild* parent, const fgElement* element)
{
  DEFAULT_CREATE(fgButton, fgButton_Init, flags, parent, element);
}
fgChild* FG_FASTCALL fgTopWindow_Create(const char* caption, fgFlag flags, const fgElement* element)
{
  fgTopWindow* r = (fgTopWindow*)malloc(sizeof(fgTopWindow));
  fgTopWindow_Init(r,flags, element);
  fgChild_VoidMessage((fgChild*)r, FG_SETPARENT, fgSingleton());
  fgChild_VoidMessage((fgChild*)r, FG_SETTEXT, caption);
  return (fgChild*)r;
}

fgRoot* FG_FASTCALL fgInitialize()
{
  fgRoot* r = (fgRoot*)malloc(sizeof(fgRoot));
  assert(!_fgroot);
  fgRoot_Init(r);
  return _fgroot=r;
}
char FG_FASTCALL fgLoadExtension(void* fg, const char* extname) { return -1; }
void fgPushClipRect(AbsRect* clip) {}
void fgPopClipRect() {}