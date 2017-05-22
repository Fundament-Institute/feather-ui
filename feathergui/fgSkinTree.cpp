// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgSkinTree.h"
#include "feathercpp.h"

char CompStylePair(const fgSkinTree::fgStylePair& l, const fgSkinTree::fgStylePair& r)
{
  uint8_t lb = bss::BitCount(l.map);
  uint8_t rb = bss::BitCount(r.map);
  char ret = SGNCOMPARE(rb, lb);
  return !ret ? SGNCOMPARE(r.map, l.map) : ret;
}
typedef bss::ArraySort<fgSkinTree::fgStylePair, &CompStylePair> StyleArraySort;

void fgSkinElement_Init(fgSkinElement* self, const char* type, fgFlag flags, const fgTransform* transform, short units, int order)
{
  self->type = fgCopyText(type, __FILE__, __LINE__);
  self->transform = *transform;
  self->units = units;
  self->flags = flags;
  self->order = order;
  self->skin = 0;
  fgStyle_Init(&self->style);
}
void fgSkinElement_Destroy(fgSkinElement* self)
{
  if(self->type) fgFreeText(self->type, __FILE__, __LINE__);
  fgStyle_Destroy(&self->style);
}

void fgSkinTree_Init(fgSkinTree* self)
{
  bss::bssFill(*self, 0);
}
void fgSkinTree_Destroy(fgSkinTree* self)
{
  reinterpret_cast<fgSkinLayoutArray&>(self->children).~ArraySort();

  for(size_t i = 0; i < self->styles.l; ++i)
    fgStyle_Destroy(&self->styles.p[i].style);
}

size_t fgSkinTree_AddChild(fgSkinTree* self, const char* type, fgFlag flags, const fgTransform* transform, short units, int order)
{
  size_t r = ((fgSkinLayoutArray&)self->children).Insert(fgSkinLayoutConstruct(type, flags, transform, units, order));
  self->children.p[r].instance = fgroot_instance->backend.fgCreate(type, 0, 0, 0, flags, (units == -1) ? 0 : transform, units);
  return r;
}
char fgSkinTree_RemoveChild(fgSkinTree* self, FG_UINT child)
{
  return DynArrayRemove((fgSkinLayoutArray&)self->children, child);
}
fgSkinLayout* fgSkinTree_GetChild(const fgSkinTree* self, FG_UINT child)
{
  return self->children.p + child;
}

FG_UINT fgSkinTree_AddStyle(fgSkinTree* self, const char* names)
{
  FG_UINT style = fgStyle_GetAllNames(names);
  self->stylemask |= style;
  fgSkinTree::fgStylePair pair = { style, 0 };
  if(((StyleArraySort&)self->styles).Find(pair) == (size_t)~0)
    ((StyleArraySort&)self->styles).Insert(pair);
  return style;
}

char fgSkinTree_RemoveStyle(fgSkinTree* self, FG_UINT style)
{
  fgSkinTree::fgStylePair pair = { style, 0 };
  size_t i = ((StyleArraySort&)self->styles).Find(pair);
  if(i == (size_t)~0)
    return 0;
  fgStyle_Destroy(&self->styles.p[i].style);
  return ((StyleArraySort&)self->styles).Remove(i);
}
fgStyle* fgSkinTree_GetStyle(const fgSkinTree* self, FG_UINT style)
{
  fgSkinTree::fgStylePair pair = { style, 0 };
  size_t i = ((StyleArraySort&)self->styles).Find(pair);
  return (i == (size_t)~0) ? 0 : (&self->styles.p[i].style);
}

void fgSkinLayout_Init(fgSkinLayout* self, const char* type, fgFlag flags, const fgTransform* transform, short units, int order)
{
  bss::bssFill(*self, 0);
  fgSkinElement_Init(&self->element, type, flags, transform, units, order);
  self->instance = 0;
}
void fgSkinLayout_Destroy(fgSkinLayout* self)
{
  fgSkinElement_Destroy(&self->element);
  if(self->instance) VirtualFreeChild(self->instance);
  fgSkinTree_Destroy(&self->tree);
}
