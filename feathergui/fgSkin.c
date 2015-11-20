// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgSkin.h"
#include "khash.h"
#include "fgResource.h"
#include "fgText.h"

KHASH_INIT(fgSkins, const char*, fgSkin, 1, kh_str_hash_funcins, kh_str_hash_insequal);

fgVector resources; // type: void*
fgVector children; // type: fgStyleLayout
fgVector styles; // type: fgStyle
fgVector subskins; // type: fgSkin (should be used for prechildren ONLY)
struct __kh_fgSkins_t* skinmap;

void FG_FASTCALL fgSkin_Init(fgSkin* self)
{
  memset(self, 0, sizeof(fgSkin));
}

char FG_FASTCALL fgSkin_DestroySkinElement(fgSkin* self, khiter_t iter)
{
  if(kh_exist(self->skinmap, iter))
  {
    fgSkin_Destroy(&kh_val(self->skinmap, iter));
    free((char*)kh_key(self->skinmap, iter));
    kh_del(fgSkins, self->skinmap, iter);
    return 1;
  }
  return 0;
}

void FG_FASTCALL fgSkin_Destroy(fgSkin* self)
{
  for(FG_UINT i = 0; i < self->resources.l; ++i)
    fgDestroyResource(fgVector_Get(self->resources, i, void*));
  fgVector_Destroy(&self->resources);

  for(FG_UINT i = 0; i < self->fonts.l; ++i)
    fgDestroyFont(fgVector_Get(self->fonts, i, void*));
  fgVector_Destroy(&self->fonts);

  for(FG_UINT i = 0; i < self->children.l; ++i)
    fgStyleLayout_Destroy(fgVector_GetP(self->children, i, fgStyleLayout));
  fgVector_Destroy(&self->children);

  for(FG_UINT i = 0; i < self->styles.l; ++i)
    fgStyle_Destroy(fgVector_GetP(self->styles, i, fgStyle));
  fgVector_Destroy(&self->styles);

  for(FG_UINT i = 0; i < self->subskins.l; ++i)
    fgSkin_Destroy(fgVector_GetP(self->subskins, i, fgSkin));
  fgVector_Destroy(&self->subskins);

  if(self->skinmap)
  {
    khiter_t cur = kh_begin(self->skinmap);
    while(cur != kh_end(self->skinmap)) fgSkin_DestroySkinElement(self, cur++);
    kh_destroy_fgSkins(self->skinmap);
  }
}
FG_UINT FG_FASTCALL fgSkin_AddResource(fgSkin* self, void* resource)
{
  fgVector_Add(self->resources, resource, void*);
  return self->resources.l - 1;
}
char FG_FASTCALL fgSkin_RemoveResource(fgSkin* self, FG_UINT resource)
{
  if(resource >= self->resources.l)
    return 0;
  fgDestroyResource(fgSkin_GetResource(self, resource));
  fgVector_Remove(&self->resources, resource, sizeof(void*));
  return 1;
}
void* FG_FASTCALL fgSkin_GetResource(fgSkin* self, FG_UINT resource)
{
  return fgVector_Get(self->resources, resource, void*);
}
FG_UINT FG_FASTCALL fgSkin_AddFont(fgSkin* self, void* font)
{
  fgVector_Add(self->fonts, font, void*);
  return self->fonts.l - 1;
}
char FG_FASTCALL fgSkin_RemoveFont(fgSkin* self, FG_UINT font)
{
  if(font >= self->fonts.l)
    return 0;
  fgDestroyFont(fgSkin_GetFont(self, font));
  fgVector_Remove(&self->fonts, font, sizeof(void*));
  return 1;
}
void* FG_FASTCALL fgSkin_GetFont(fgSkin* self, FG_UINT font)
{
  return fgVector_Get(self->fonts, font, void*);
}
FG_UINT FG_FASTCALL fgSkin_AddChild(fgSkin* self, char* name, fgElement* element, fgFlag flags)
{
  fgVector_CheckSize(&self->children, sizeof(fgStyleLayout));
  FG_UINT r = self->children.l++;
  fgStyleLayout_Init(fgSkin_GetChild(self, r), name, element, flags);
  return r;
}
char FG_FASTCALL fgSkin_RemoveChild(fgSkin* self, FG_UINT child)
{
  if(child >= self->children.l)
    return 0;
  fgStyleLayout_Destroy(fgSkin_GetChild(self, child));
  fgVector_Remove(&self->children, child, sizeof(fgStyleLayout));
  return 1;
}
fgStyleLayout* FG_FASTCALL fgSkin_GetChild(fgSkin* self, FG_UINT child)
{
  return fgVector_GetP(self->children, child, fgStyleLayout);
}

FG_UINT fgVector_AddStyle(fgVector* self, ptrdiff_t index)
{
  fgVector_CheckSize(self, sizeof(fgStyle));
  FG_UINT r = self->l++;
  fgStyle_Init(((fgStyle*)self->p) + (self->l));
  return r;
}
FG_UINT fgVector_RemoveStyle(fgVector* self, FG_UINT style)
{
  if(style >= self->l)
    return 0;
  fgStyle_Destroy(((fgStyle*)self->p) + style);
  fgVector_Remove(self, style, sizeof(fgStyle));
  return 1;
}

FG_UINT FG_FASTCALL fgSkin_AddStyle(fgSkin* self)
{
  return fgVector_AddStyle(&self->styles, 0);
}
char FG_FASTCALL fgSkin_RemoveStyle(fgSkin* self, FG_UINT style)
{
  return fgVector_RemoveStyle(&self->styles, style);
}
fgStyle* FG_FASTCALL fgSkin_GetStyle(fgSkin* self, FG_UINT style)
{
  return fgVector_GetP(self->styles, style, fgStyle);
}

FG_UINT FG_FASTCALL fgSkin_AddSubSkin(fgSkin* self, int index)
{
  fgVector_CheckSize(&self->subskins, sizeof(fgSkin));
  FG_UINT r = self->subskins.l++;
  fgSkin_Init(fgSkin_GetSubSkin(self, r));
  return r;
}
char FG_FASTCALL fgSkin_RemoveSubSkin(fgSkin* self, FG_UINT subskin)
{
  if(subskin >= self->subskins.l)
    return 0;
  fgSkin_Destroy(fgSkin_GetSubSkin(self, subskin));
  fgVector_Remove(&self->subskins, subskin, sizeof(fgSkin));
  return 1;
}
fgSkin* FG_FASTCALL fgSkin_GetSubSkin(fgSkin* self, FG_UINT subskin)
{
  return fgVector_GetP(self->subskins, subskin, fgSkin);
}

fgSkin* FG_FASTCALL fgSkin_AddSkin(fgSkin* self, char* name)
{
  if(!self->skinmap)
    self->skinmap = kh_init_fgSkins();

  if(!name) return 0;
  int r = 0;
  khiter_t iter = kh_put(fgSkins, self->skinmap, name, &r);
  if(r != 0) // If r is 0 the element already exists so we don't want to re-initialize it
  {
    fgSkin_Init(&kh_val(self->skinmap, iter));
    kh_key(self->skinmap, iter) = fgCopyText(name);
  }

  return &kh_val(self->skinmap, iter);
}
char FG_FASTCALL fgSkin_RemoveSkin(fgSkin* self, char* name)
{
  if(!self->skinmap || !name)
    return 0;

  return fgSkin_DestroySkinElement(self, kh_get(fgSkins, self->skinmap, name));
}
fgSkin* FG_FASTCALL fgSkin_GetSkin(fgSkin* self, char* name)
{
  return &kh_val(self->skinmap, kh_get(fgSkins, self->skinmap, name));
}

void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, char* name, fgElement* element, fgFlag flags)
{
  self->name = fgCopyText(name);
  self->element = *element;
  self->flags = flags;
  fgStyle_Init(&self->style);
}
void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self)
{
  free(self->name);
  fgStyle_Destroy(&self->style);
}

void FG_FASTCALL fgStyle_Init(fgStyle* self)
{
  memset(&self, 0, sizeof(fgStyle));
}

void FG_FASTCALL fgStyle_Destroy(fgStyle* self)
{
  for(FG_UINT i = 0; i < self->substyles.l; ++i)
    fgStyle_Destroy(fgVector_GetP(self->substyles, i, fgStyle));
  fgVector_Destroy(&self->substyles);
  while(self->styles)
    fgStyle_RemoveStyleMsg(self, self->styles);
}

FG_UINT FG_FASTCALL fgStyle_AddSubstyle(fgStyle* self, ptrdiff_t index)
{
  return fgVector_AddStyle(&self->substyles, index);
}

char FG_FASTCALL fgStyle_RemoveSubstyle(fgStyle* self, FG_UINT substyle)
{
  return fgVector_RemoveStyle(&self->substyles, substyle);
}

fgStyle* FG_FASTCALL fgStyle_GetSubstyle(fgStyle* self, FG_UINT substyle)
{
  return fgVector_GetP(self->substyles, substyle, fgStyle);
}

fgStyleMsg* FG_FASTCALL fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, const void* arg1, size_t arglen1, const void* arg2, size_t arglen2)
{
  fgStyleMsg* r = malloc(sizeof(fgStyleMsg) + arglen1 + arglen2);
  memcpy(&r->msg, msg, sizeof(FG_Msg));

  if(arg1)
  {
    r->msg.other1 = r + 1;
    memcpy(r->msg.other1, arg1, arglen1);
  }

  if(arg2)
  {
    r->msg.other2 = ((char*)(r + 1)) + arglen1;
    memcpy(r->msg.other2, arg2, arglen2);
  }

  r->next = self->styles;
  self->styles = r;
  return r;
}

void FG_FASTCALL fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg)
{
  if(self->styles == msg)
    self->styles = msg->next;
  else
  {
    fgStyleMsg* cur = self->styles;
    while(cur && cur->next != msg) cur = cur->next;
    if(cur) cur->next = msg->next;
  }
  free(msg);
}