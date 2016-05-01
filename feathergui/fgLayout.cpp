// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "fgButton.h"
#include "fgWindow.h"
#include "fgText.h"
#include "fgResource.h"
#include "feathercpp.h"

fgElement* fgLayoutLoadMapping(const char* type, fgElement* parent, fgElement* next, const char* name, fgFlag flags, const fgTransform* transform)
{
  if(!strcmp(type, "fgElement")) { // This could be a hash, but this is low priority as this is never called in a performance critical loop.
    fgElement* r = (fgElement*)malloc(sizeof(fgElement));
    fgElement_Init(r, parent, next, name, flags, transform);
    return r;
  }
  else if(!strcmp(type, "fgControl")) {
    fgControl* r = (fgControl*)malloc(sizeof(fgControl));
    fgControl_Init(r, parent, next, name, flags, transform);
    return (fgElement*)r;
  }
  else if(!strcmp(type, "fgWindow"))
  {
    fgElement* r = fgWindow_Create(0, flags, transform);
    _sendmsg<FG_SETPARENT, void*, void*>(r, parent, next);
    return r;
  }
  else if(!strcmp(type, "fgButton"))
    return fgButton_Create(0, parent, next, name, flags, transform);
  else if(!strcmp(type, "fgText"))
    return fgText_Create(0, 0, 0xFFFFFFFF, parent, next, name, flags, transform);
  else if(!strcmp(type, "fgResource"))
    return fgResource_Create(0, 0, 0xFFFFFFFF, parent, next, name, flags, transform);

  return 0;
}

void FG_FASTCALL fgLayout_Init(fgLayout* self)
{
  memset(self, 0, sizeof(fgLayout));
}
void FG_FASTCALL fgLayout_Destroy(fgLayout* self)
{
  reinterpret_cast<fgResourceArray&>(self->resources).~cDynArray();
  reinterpret_cast<fgFontArray&>(self->fonts).~cDynArray();
  reinterpret_cast<fgClassLayoutArray&>(self->layout).~cDynArray();
}
FG_UINT FG_FASTCALL fgLayout_AddResource(fgLayout* self, void* resource)
{
  return ((fgResourceArray&)self->resources).Add(resource);
}
char FG_FASTCALL fgLayout_RemoveResource(fgLayout* self, FG_UINT resource)
{
  return DynArrayRemove((fgResourceArray&)self->resources, resource);
}
void* FG_FASTCALL fgLayout_GetResource(fgLayout* self, FG_UINT resource)
{
  return DynGet<fgResourceArray>(self->resources, resource).ref;
}
FG_UINT FG_FASTCALL fgLayout_AddFont(fgLayout* self, void* font)
{
  return ((fgFontArray&)self->fonts).Add(font);
}
char FG_FASTCALL fgLayout_RemoveFont(fgLayout* self, FG_UINT font)
{
  return DynArrayRemove((fgFontArray&)self->fonts, font);
}
void* FG_FASTCALL fgLayout_GetFont(fgLayout* self, FG_UINT font)
{
  return ((fgFontArray&)self->fonts)[font].ref;
}
FG_UINT FG_FASTCALL fgLayout_AddLayout(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform)
{
  return ((fgClassLayoutArray&)self->layout).AddConstruct(type, name, flags, transform);
}
char FG_FASTCALL fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout)
{
  return DynArrayRemove((fgClassLayoutArray&)self->layout, layout);
}
fgClassLayout* FG_FASTCALL fgLayout_GetLayout(fgLayout* self, FG_UINT layout)
{
  return self->layout.p + layout;
}

void FG_FASTCALL fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgStyleLayout_Init(&self->style, type, name, flags, transform);
  memset(&self->children, 0, sizeof(fgVector));
}
void FG_FASTCALL fgClassLayout_Destroy(fgClassLayout* self)
{
  fgStyleLayout_Destroy(&self->style);
  reinterpret_cast<fgClassLayoutArray&>(self->children).~cDynArray();
}

FG_UINT FG_FASTCALL fgClassLayout_AddChild(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform)
{
  return ((fgClassLayoutArray&)self->children).AddConstruct(type, name, flags, transform);
}
char FG_FASTCALL fgClassLayout_RemoveChild(fgClassLayout* self, FG_UINT child)
{
  return DynArrayRemove((fgFontArray&)self->children, child);
}
fgClassLayout* FG_FASTCALL fgClassLayout_GetChild(fgClassLayout* self, FG_UINT child)
{
  return self->children.p + child;
}