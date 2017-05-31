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

void fgSkinElement_Init(fgSkinElement* self, const char* type, fgFlag flags, const fgTransform* transform, fgMsgType units, int order)
{
  self->type = fgCopyText(type, __FILE__, __LINE__);
  self->transform = *transform;
  self->units = units;
  self->flags = flags;
  self->order = order;
  self->skin = 0;
  fgStyle_Init(&self->style);
}

void fgSkinElement_InitCopy(fgSkinElement* self, const fgSkinElement* from)
{
  self->transform = from->transform;
  self->units = from->units;
  self->flags = from->flags;
  self->type = fgCopyText(from->type, __FILE__, __LINE__);
  fgStyle_InitCopy(&self->style, &from->style);
  self->order = from->order;
  self->skin = from->skin;
}

void fgSkinElement_ResolveCopy(fgSkinElement* self, struct _FG_SKIN_BASE* parent)
{
  if(self->skin)
    self->skin = parent->GetAnySkin(self->skin->base.name);
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
void fgSkinTree_InitCopy(fgSkinTree* self, const fgSkinTree* from)
{
  new(&self->children) fgSkinLayoutArray(reinterpret_cast<const fgSkinLayoutArray&>(from->children));
  self->stylemask = from->stylemask;
  self->styles.s = from->styles.s;
  self->styles.l = from->styles.l;
  self->styles.p = bss::bssMalloc<fgSkinTree::fgStylePair>(self->styles.s); // do NOT use fgmalloc, this is not freed via fgfree
  for(size_t i = 0; i < from->styles.l; ++i)
  {
    self->styles.p[i].map = from->styles.p[i].map;
    fgStyle_InitCopy(&self->styles.p[i].style, &from->styles.p[i].style);
  }
}
void fgSkinTree_ResolveCopy(fgSkinTree* self, struct _FG_SKIN_BASE* parent)
{
  for(size_t i = 0; i < self->children.l; ++i)
  {
    fgSkinElement_ResolveCopy(&self->children.p[i].element, parent);
    fgSkinTree_ResolveCopy(&self->children.p[i].tree, parent);
  }
}

void fgSkinTree_Destroy(fgSkinTree* self)
{
  reinterpret_cast<fgSkinLayoutArray&>(self->children).~ArraySort();

  for(size_t i = 0; i < self->styles.l; ++i)
    fgStyle_Destroy(&self->styles.p[i].style);
  ((StyleArraySort&)self->styles).~ArraySort();
}

size_t fgSkinTree_AddChild(fgSkinTree* self, const char* type, fgFlag flags, const fgTransform* transform, fgMsgType units, int order)
{
  return ((fgSkinLayoutArray&)self->children).Insert(fgSkinLayoutConstruct(type, flags, transform, units, order));
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
  {
    fgLog(FGLOG_INFO, "Tried to remove nonexistant style %u", style);
    return 0;
  }
  fgStyle_Destroy(&self->styles.p[i].style);
  return ((StyleArraySort&)self->styles).Remove(i);
}
fgStyle* fgSkinTree_GetStyle(const fgSkinTree* self, FG_UINT style)
{
  fgSkinTree::fgStylePair pair = { style, 0 };
  size_t i = ((StyleArraySort&)self->styles).Find(pair);
  return (i == (size_t)~0) ? 0 : (&self->styles.p[i].style);
}

void fgSkinLayout_Init(fgSkinLayout* self, const char* type, fgFlag flags, const fgTransform* transform, fgMsgType units, int order)
{
  bss::bssFill(*self, 0);
  fgSkinElement_Init(&self->element, type, flags, transform, units, order);
  self->instance = fgroot_instance->backend.fgCreate(type, 0, 0, 0, flags, (units == (fgMsgType)~0) ? 0 : transform, units);
}
void fgSkinLayout_InitCopy(fgSkinLayout* self, const fgSkinLayout* from)
{
  fgSkinElement_InitCopy(&self->element, &from->element);
  fgSkinTree_InitCopy(&self->tree, &from->tree);
  self->instance = fgroot_instance->backend.fgCreate(self->element.type, 0, 0, 0, self->element.flags, (self->element.units == (fgMsgType)~0) ? nullptr : &self->element.transform, self->element.units);
  _sendsubmsg<FG_SETSTYLE, void*, size_t>(self->instance, FGSETSTYLE_POINTER, (void*)&self->element.style, ~0);
}

void fgSkinLayout_Destroy(fgSkinLayout* self)
{
  fgSkinElement_Destroy(&self->element);
  if(self->instance) VirtualFreeChild(self->instance);
  fgSkinTree_Destroy(&self->tree);
}
