// Copyright ©2015 Black Sphere Studios
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

#define BUILDDEBUGMSG(TYPE) size_t FG_FASTCALL fgdebug_Message_##TYPE##(TYPE* self, const FG_Msg* msg) \
{ \
  if(debugmsghook!=0) (*debugmsghook)((fgWindow*)self,msg); \
  return TYPE##_Message(self,msg); \
}

BUILDDEBUGMSG(fgButton)
//BUILDDEBUGMSG(fgMenu)
BUILDDEBUGMSG(fgTopWindow)
BUILDDEBUGMSG(fgWindow)

static const fgFlag FGIMAGE_MARKER = (1 << ((sizeof(fgFlag) << 3) - 1));

// These are reference implementations of definitions.
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

void* FG_FASTCALL fgCreateImageDef(fgFlag flags, const char* data, size_t length, unsigned int color, const CRect* uv)
{
  fgImageDef* def = malloc(sizeof(fgImageDef));
  def->flags=flags | FGIMAGE_MARKER; // Lets us know this is an image definition
  def->color=color;
  def->data=data;
  def->length=length;
  def->uv=*uv;
  return def;
}
void* FG_FASTCALL fgCreateTextDef(fgFlag flags, const char* text, const char* font, unsigned short fontsize, unsigned short lineheight, unsigned int color)
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
void* FG_FASTCALL fgDefaultTextDef(fgFlag flags, const char* text)
{
  return fgCreateTextDef(flags, text, 0, 0, 0, 0);
}
fgChild* FG_FASTCALL fgLoadDef(void* def, const fgElement* element, int order)
{
  fgChild* r = malloc(sizeof(fgChild));
  fgChild_Init(r, 0, 0, element);
  fgFlag flags=*(fgFlag*)def;
  if(flags & FGIMAGE_MARKER) {
    fgImageDef* idef=(fgImageDef*)def;
    fgChild_IntMessage(r, FG_SETCOLOR, idef->color, 0);
    fgChild_VoidMessage(r, FG_SETCOLOR, &idef->uv);
  } else {
    fgTextDef* tdef=(fgTextDef*)def;
    fgChild_IntMessage(r, FG_SETFONTCOLOR, tdef->color, 0);
  }
  return r;
}

void FG_FASTCALL fgDestroyDef(void* def)
{
  free(def);
}

fgChild* FG_FASTCALL fgButton_Create(fgChild* item, fgFlag flags, fgChild* parent, const fgElement* element)
{
  fgButton* r = (fgButton*)malloc(sizeof(fgButton));
  r->window.element.message=&fgdebug_Message_fgButton;
  fgButton_Init(r,flags,parent,element);
  return (fgChild*)r;
}
fgChild* FG_FASTCALL fgTopWindow_Create(const char* caption, fgFlag flags, const fgElement* element)
{
  fgTopWindow* r = (fgTopWindow*)malloc(sizeof(fgTopWindow));
  fgTopWindow_Init(r,flags, element);
  r->window.element.message=&fgdebug_Message_fgTopWindow;
  r->region.message=&fgdebug_Message_fgWindow;
  fgChild_VoidMessage((fgChild*)r, FG_SETPARENT, fgSingleton());
  fgChild_VoidMessage((fgChild*)r, FG_SETCAPTION, caption);
  return (fgChild*)r;
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
