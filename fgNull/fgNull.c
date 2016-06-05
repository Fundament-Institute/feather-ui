// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgNull.h"
#include "fgText.h"
#include "fgResource.h"
#include "fgRoot.h"
#include "fgButton.h"
#include "fgMenu.h"
#include "fgWindow.h"
#include "fgRadioButton.h"
#include "fgSlider.h"
#include "fgProgressbar.h"
#include "fgList.h"
#include "fgDebug.h"

#if defined(BSS_DEBUG) && defined(BSS_CPU_x86_64)
#pragma comment(lib, "../bin/feathergui_d.lib")
#elif defined(BSS_CPU_x86_64)
#pragma comment(lib, "../bin/feathergui.lib")
#elif defined(BSS_DEBUG)
#pragma comment(lib, "../bin32/feathergui32_d.lib")
#else
#pragma comment(lib, "../bin32/feathergui32.lib")
#endif

fgRoot* _fgroot=0; // fgRoot singleton variable


void* FG_FASTCALL fgCreateFont(fgFlag flags, const char* font, unsigned int fontsize, unsigned int dpi) { return (void*)~0; }
void* FG_FASTCALL fgCopyFont(void* font, unsigned int fontsize, unsigned int dpi) { return (void*)~0; }
void* FG_FASTCALL fgCloneFont(void* font) { return (void*)~0; }
void FG_FASTCALL fgDestroyFont(void* font) { }
void* FG_FASTCALL fgDrawFont(void* font, const int* text, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, AbsVec* center, fgFlag flags, void* cache) { return 0; }
void FG_FASTCALL fgFontSize(void* font, const int* text, float lineheight, float letterspacing, AbsRect* area, fgFlag flags) { }
void FG_FASTCALL fgFontGet(void* font, float* lineheight, unsigned int* size, unsigned int* dpi)
{ 
  if(lineheight) *lineheight = 0.0f;
  if(size) *size = 0;
  if(dpi) *dpi = 0;
}

void* FG_FASTCALL fgCreateResource(fgFlag flags, const char* data, size_t length) { return (void*)~0; }
void* FG_FASTCALL fgCloneResource(void* res) { return (void*)~0; }
void FG_FASTCALL fgDestroyResource(void* res) { }
void FG_FASTCALL fgDrawResource(void* res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, AbsVec* center, fgFlag flags) {}
void FG_FASTCALL fgResourceSize(void* res, const CRect* uv, AbsVec* dim, fgFlag flags) { }

FG_EXTERN fgElement* FG_FASTCALL fgCreate(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  return fgCreateDefault(type, parent, next, name, flags, transform);
}

fgRoot* FG_FASTCALL fgInitialize()
{
  fgRoot* r = (fgRoot*)malloc(sizeof(fgRoot));
  assert(!_fgroot);
  AbsRect area = { 0 };
  fgRoot_Init(r, &area, 0);
  return _fgroot=r;
}

AbsRect* clipstack = 0;
size_t clipcapacity = 0;
size_t clipnum = 0;

void fgPushClipRect(AbsRect* clip)
{
  if(clipcapacity >= clipnum)
  {
    clipcapacity = clipcapacity * 2;
    clipstack = realloc(clipstack, sizeof(AbsRect)*clipcapacity);
  }
  clipstack[clipnum++] = *clip;
}
AbsRect fgPeekClipRect()
{
  static const AbsRect BLANK = { 0,0,0,0 };
  return !clipnum ? BLANK : clipstack[clipnum - 1];
}
void fgPopClipRect()
{
  --clipnum;
}

char FG_FASTCALL fgLoadExtension(const char* extname, void* fg, size_t sz) { return -1; }
void FG_FASTCALL fgDrawLine(AbsVec p1, AbsVec p2, unsigned int color) {}
void fgSetCursor(unsigned int type, void* custom) {}
void fgClipboardCopy(unsigned int type, const void* data, size_t length) {}
char fgClipboardExists(unsigned int type) { return 0; }
const void* fgClipboardPaste(unsigned int type, size_t* length) { return 0; }
void fgClipboardFree(const void* mem) {}
void fgDirtyElement(fgTransform* elem) {}