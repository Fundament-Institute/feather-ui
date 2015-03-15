// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgNull.h"
#include "fgStatic.h"
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
void (FG_FASTCALL *debugmsghook)(fgWindow* self, const FG_Msg* msg)=0;

#define BUILDDEBUGMSG(TYPE) char FG_FASTCALL fgdebug_Message_##TYPE##(TYPE* self, const FG_Msg* msg) \
{ \
  if(debugmsghook!=0) (*debugmsghook)((fgWindow*)self,msg); \
  return TYPE##_Message(self,msg); \
}

BUILDDEBUGMSG(fgButton)
//BUILDDEBUGMSG(fgMenu)
BUILDDEBUGMSG(fgTopWindow)
BUILDDEBUGMSG(fgWindow)

fgStatic* FG_FASTCALL fgLoadImageData(const void* data, size_t length, fgFlag flags)
{
  fgStatic* r = (fgStatic*)malloc(sizeof(fgStatic));
  fgStatic_Init(r);
  return r;
}
fgStatic* FG_FASTCALL fgLoadText(const char* text, fgFlag flags, const char* font, unsigned short fontsize, unsigned short lineheight)
{
  fgStatic* r = (fgStatic*)malloc(sizeof(fgStatic));
  fgStatic_Init(r);
  return r;
}

// These are reference implementations of definitions. Copy-paste them into your implementation, then define fgLoadImageData and fgLoadText.
// Alternatively, you may replace them with your own definitions, which is usually what non-game implementations do.
typedef struct __IMAGE_DEF {
  fgFlag flags;
  unsigned int color;
  const char* data;
  size_t length;
  CRect uv;
} fgImageDef;

typedef struct __TEXT_DEF {
  fgFlag flags;
  unsigned int color;
  const char* text;
  const char* font;
  unsigned short fontsize;
  unsigned short lineheight;
} fgTextDef;

void* FG_FASTCALL fgLoadImageDef(fgFlag flags, const char* data, size_t length, unsigned int color, const CRect* uv)
{
  fgImageDef* def = malloc(sizeof(fgImageDef));
  def->flags=flags|FGSTATIC_MARKER; // Lets us know this is an image definition
  def->color=color;
  def->data=data;
  def->length=length;
  def->uv=*uv;
  return def;
}
void* FG_FASTCALL fgLoadTextDef(fgFlag flags, const char* text, const char* font, unsigned short fontsize, unsigned short lineheight, unsigned int color)
{
  fgTextDef* def = malloc(sizeof(fgTextDef));
  def->flags=flags;
  def->color=color;
  def->text=text;
  def->font=font;
  def->fontsize=fontsize;
  def->lineheight=lineheight;
  return def;
}
fgStatic* FG_FASTCALL fgEmptyStatic(fgFlag flags)
{
  fgStatic* r = (fgStatic*)malloc(sizeof(fgStatic));
  fgStatic_Init(r);
  r->element.flags=flags;
  return r;
}
fgStatic* FG_FASTCALL fgLoadDefaultText(const char* text)
{
  return fgLoadText(text,0,0,0,0);
}
fgStatic* FG_FASTCALL fgLoadDef(void* def, const fgElement* element, int order)
{
  fgStatic* r;
  fgFlag flags=*(fgFlag*)def;
  if(flags&FGSTATIC_MARKER) {
    fgImageDef* idef=(fgImageDef*)def;
    r=fgLoadImageData(idef->data,idef->length,idef->flags);
    (r->message)(r,FG_RSETCOLOR,(void*)idef->color,0);
    (r->message)(r,FG_RSETUV,&idef->uv,0);
  } else {
    fgTextDef* tdef=(fgTextDef*)def;
    r=fgLoadText(tdef->text,tdef->flags,tdef->font,tdef->fontsize,tdef->lineheight);
    (r->message)(r,FG_RSETCOLOR,(void*)tdef->color,0);
  }
  return r;
}

void FG_FASTCALL fgDestroyDef(void* def)
{
  free(def);
}

fgWindow* FG_FASTCALL fgButton_Create(fgStatic* item, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags)
{
  fgButton* r = (fgButton*)malloc(sizeof(fgButton));
  r->window.message=&fgdebug_Message_fgButton;
  fgButton_Init(r,parent,element,id,flags);
  return r;
}
fgWindow* FG_FASTCALL fgMenu_Create(fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags)
{
  fgMenu* r = (fgMenu*)malloc(sizeof(fgMenu));
  //r->grid.window.message=&fgdebug_Message_fgMenu;
  //fgMenu_Init(r,parent,element,id,flags);
  fgGrid_Init(0,0,0,0,0);
  return r;
}
fgWindow* FG_FASTCALL fgTopWindow_Create(const char* caption, const fgElement* element, FG_UINT id, fgFlag flags)
{
  fgTopWindow* r = (fgTopWindow*)malloc(sizeof(fgTopWindow));
  fgTopWindow_Init(r,element,id,flags);
  r->window.message=&fgdebug_Message_fgTopWindow;
  r->region.message=&fgdebug_Message_fgWindow;
  fgWindow_VoidMessage((fgWindow*)r,FG_SETPARENT,fgSingleton());
  return r;
}

fgRoot* FG_FASTCALL fgInitialize()
{
  fgRoot* r = (fgRoot*)malloc(sizeof(fgRoot));
  assert(!_fgroot);
  fgRoot_Init(r);
  return _fgroot=r;
}
fgRoot* FG_FASTCALL fgSingleton()
{
  return _fgroot;
}
char FG_FASTCALL fgLoadExtension(void* fg, const char* extname) { return -1; }
