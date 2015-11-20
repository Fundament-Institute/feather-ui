// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "fgButton.h"
#include "fgTopWindow.h"
#include "fgText.h"
#include "fgResource.h"

fgChild* fgLayoutLoadMapping(const char* name, fgFlag flags, fgChild* parent, fgElement* element)
{
  if(!strcmp(name, "fgChild")) {
    fgChild* r = (fgChild*)malloc(sizeof(fgChild));
    fgChild_Init(r, flags, parent, element);
    return r;
  }
  else if(!strcmp(name, "fgWindow")) {
    fgWindow* r = (fgWindow*)malloc(sizeof(fgWindow));
    fgWindow_Init(r, flags, parent, element);
    return (fgChild*)r;
  }
  else if(!strcmp(name, "fgTopWindow"))
  {
    fgChild* r = fgTopWindow_Create(0, flags, element);
    fgChild_VoidMessage(r, FG_SETPARENT, parent);
    return r;
  }
  else if(!strcmp(name, "fgButton"))
    return fgButton_Create(0, flags, parent, element);
  else if(!strcmp(name, "fgText"))
    return fgText_Create(0, 0, 0xFFFFFFFF, flags, parent, element);
  else if(!strcmp(name, "fgResource"))
    return fgResource_Create(0, 0, 0xFFFFFFFF, flags, parent, element);

  return 0;
}

void FG_FASTCALL fgLayout_Init(fgLayout* self)
{
  memset(self, 0, sizeof(fgLayout));
}
void FG_FASTCALL fgLayout_Destroy(fgLayout* self)
{
  for(FG_UINT i = 0; i < self->resources.l; ++i)
    fgDestroyResource(fgVector_Get(self->resources, i, void*));
  fgVector_Destroy(&self->resources);

  for(FG_UINT i = 0; i < self->fonts.l; ++i)
    fgDestroyFont(fgVector_Get(self->fonts, i, void*));
  fgVector_Destroy(&self->fonts);

  for(FG_UINT i = 0; i < self->layout.l; ++i)
    fgClassLayout_Destroy(fgVector_GetP(self->layout, i, fgClassLayout));
  fgVector_Destroy(&self->layout);
}
FG_UINT FG_FASTCALL fgLayout_AddResource(fgLayout* self, void* resource)
{
  fgVector_Add(self->resources, resource, void*);
  return self->resources.l - 1;
}
char FG_FASTCALL fgLayout_RemoveResource(fgLayout* self, FG_UINT resource)
{
  if(resource >= self->resources.l)
    return 0;
  fgDestroyResource(fgLayout_GetResource(self, resource));
  fgVector_Remove(&self->resources, resource, sizeof(void*));
  return 1;
}
void* FG_FASTCALL fgLayout_GetResource(fgLayout* self, FG_UINT resource)
{
  return fgVector_Get(self->resources, resource, void*);
}
FG_UINT FG_FASTCALL fgLayout_AddFont(fgLayout* self, void* font)
{
  fgVector_Add(self->fonts, font, void*);
  return self->fonts.l - 1;
}
char FG_FASTCALL fgLayout_RemoveFont(fgLayout* self, FG_UINT font)
{
  if(font >= self->fonts.l)
    return 0;
  fgDestroyFont(fgLayout_GetFont(self, font));
  fgVector_Remove(&self->fonts, font, sizeof(void*));
  return 1;
}
void* FG_FASTCALL fgLayout_GetFont(fgLayout* self, FG_UINT font)
{
  return fgVector_Get(self->fonts, font, void*);
}
FG_UINT FG_FASTCALL fgLayout_AddLayout(fgLayout* self, char* name, fgElement* element, fgFlag flags)
{
  fgVector_CheckSize(&self->layout, sizeof(fgClassLayout));
  FG_UINT r = self->layout.l++;
  fgClassLayout_Init(fgLayout_GetLayout(self, r), name, element, flags);
  return r;
}
char FG_FASTCALL fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout)
{
  if(layout >= self->layout.l)
    return 0;
  fgClassLayout_Destroy(fgLayout_GetLayout(self, layout));
  fgVector_Remove(&self->layout, layout, sizeof(fgClassLayout));
  return 1;
}
fgClassLayout* FG_FASTCALL fgLayout_GetLayout(fgLayout* self, FG_UINT layout)
{
  return fgVector_Get(self->layout, layout, void*);
}

void FG_FASTCALL fgClassLayout_Init(fgClassLayout* self, char* name, fgElement* element, fgFlag flags)
{
  fgStyleLayout_Init(&self->style, name, element, flags);
  fgVector_Init(&self->children);
}
void FG_FASTCALL fgClassLayout_Destroy(fgClassLayout* self)
{
  fgStyleLayout_Destroy(&self->style);
  fgVector_Destroy(&self->children);
}

FG_UINT FG_FASTCALL fgClassLayout_AddChild(fgClassLayout* self, char* name, fgElement* element, fgFlag flags)
{
  fgVector_CheckSize(&self->children, sizeof(fgClassLayout));
  FG_UINT r = self->children.l++;
  fgClassLayout_Init(fgClassLayout_GetChild(self, r), name, element, flags);
  return r;
}
char FG_FASTCALL fgClassLayout_RemoveChild(fgClassLayout* self, FG_UINT child)
{
  if(child >= self->children.l)
    return 0;
  fgClassLayout_Destroy(fgClassLayout_GetChild(self, child));
  fgVector_Remove(&self->children, child, sizeof(fgClassLayout));
  return 1;
}
fgClassLayout* FG_FASTCALL fgClassLayout_GetChild(fgClassLayout* self, FG_UINT child)
{
  return fgVector_GetP(self->children, child, fgClassLayout);
}