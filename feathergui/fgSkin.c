// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgSkin.h"
#include "khash.h"

KHASH_INIT(fgSkins, const char*, fgSkin, 1, kh_str_hash_funcins, kh_str_hash_insequal);

FG_EXTERN void FG_FASTCALL fgSkin_Init(fgSkin* self)
{
  fgVector_Init(&self->defs);
  fgVector_Init(&self->styles);
  fgVector_Init(&self->subskins);
}

FG_EXTERN void FG_FASTCALL fgSkin_Destroy(fgSkin* self)
{
  for(FG_UINT i = 0; i < self->defs.l; ++i)
    fgDestroyDef(fgVector_Get(self->defs, i, fgFullDef).def);
  fgVector_Destroy(&self->defs);

  for(FG_UINT i = self->styles.l; i > 0;)
    fgStyle_RemoveStyle(&self->styles, --i);
  fgVector_Destroy(&self->styles);

  for(FG_UINT i = 0; i < self->subskins.l; ++i)
    fgSkin_Destroy(fgVector_GetP(self->subskins, i, fgSkin));
  fgVector_Destroy(&self->subskins);
}
FG_EXTERN FG_UINT FG_FASTCALL fgStyle_AddStyle(fgVector* self)
{
  fgVector_CheckSize(self, sizeof(fgStyle));
  fgStyle* style = ((fgStyle*)self->p) + (self->l++);
  style->styles = 0;
  fgVector_Init(&style->substyles);
  return self->l - 1;
}
FG_EXTERN void FG_FASTCALL fgStyle_RemoveStyle(fgVector* self, FG_UINT index)
{
  fgStyle* style = ((fgStyle*)self->p) + index;
  while(style->styles) fgStyle_RemoveStyleMsg(style, style->styles);

  for(FG_UINT i = style->substyles.l; i > 0;)
    fgStyle_RemoveStyle(&style->substyles, --i);
  fgVector_Destroy(&style->substyles);
}
FG_EXTERN fgStyleMsg* FG_FASTCALL fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, const void* arg1, size_t arglen1, const void* arg2, size_t arglen2)
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
  return r;
}
FG_EXTERN void FG_FASTCALL fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg)
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
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddStatic(fgSkin* self, void* def, const fgElement* elem, int order)
{
  fgVector_CheckSize(&self->defs, sizeof(fgFullDef));
  fgFullDef* item = ((fgFullDef*)self->defs.p) + (self->defs.l++);
  item->def = fgCloneDef(def);
  item->element = *elem;
  item->order = order;
  return self->defs.l-1;
}
FG_EXTERN void FG_FASTCALL fgSkin_RemoveStatic(fgSkin* self, size_t index)
{
  fgDestroyDef(fgVector_Get(self->defs, index, fgFullDef).def);
  fgVector_Remove(&self->defs, index, sizeof(fgFullDef));
}

FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddSkin(fgSkin* self)
{
  fgVector_CheckSize(&self->subskins, sizeof(fgSkin));
  fgSkin* item = ((fgSkin*)self->subskins.p) + (self->subskins.l++);
  fgSkin_Init(item);
  return self->subskins.l - 1;
}
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_GetSkin(fgSkin* self, FG_UINT index)
{
  return fgVector_GetP(self->subskins, index, fgSkin);
}
FG_EXTERN void FG_FASTCALL fgSkin_RemoveSkin(fgSkin* self, FG_UINT index)
{
  fgSkin_Destroy(fgVector_GetP(self->subskins, index, fgSkin));
  fgVector_Remove(&self->subskins, index, sizeof(fgSkin));
}
FG_EXTERN struct __kh_fgSkins_t* FG_FASTCALL fgSkins_Init()
{
  return kh_init_fgSkins();
}
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_Add(struct __kh_fgSkins_t* self, const char* name)
{
  if(!name) return 0;
  int r = 0;
  khiter_t iter = kh_put(fgSkins, self, name, &r);
  if(r != 0) // If r is 0 the element already exists so we don't want to re-initialize it
  {
    fgSkin_Init(&kh_val(self, iter));
    size_t len = strlen(name)+1;
    kh_key(self, iter) = malloc(len);
    memcpy((char*)kh_key(self, iter), name, len);
  }

  return &kh_val(self, iter);
}
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_Get(struct __kh_fgSkins_t* self, const char* name)
{
  if(!name) return 0;
  khiter_t iter = kh_get(fgSkins, self, name);
  return kh_exist(self, iter)?(&kh_val(self, iter)):0;
}
void FG_FASTCALL fgSkins_RemoveElement(struct __kh_fgSkins_t* self, khiter_t iter)
{
  if(kh_exist(self, iter))
  {
    fgSkin_Destroy(&kh_val(self, iter));
    free((char*)kh_key(self, iter));
  }
}
FG_EXTERN void FG_FASTCALL fgSkins_Remove(struct __kh_fgSkins_t* self, const char* name)
{
  fgSkins_RemoveElement(self, kh_get(fgSkins, self, name));
}
FG_EXTERN void FG_FASTCALL fgSkins_Destroy(struct __kh_fgSkins_t* self)
{
  // Properly destroy all data in the hash table
  khiter_t cur = kh_begin(self);
  while(cur != kh_end(self)) fgSkins_RemoveElement(self, cur++);
  kh_destroy_fgSkins(self);
}
FG_EXTERN void FG_FASTCALL fgSkins_Apply(struct __kh_fgSkins_t* self, fgChild* child)
{
  char* name = 0;
  if(!child) return;

  fgChild* cur = child->root; // Set all the children skins first, because a skin can override potential child skins.
  while(cur)
  {
    fgSkins_Apply(self, cur);
    cur = cur->next;
  }

  fgChild_VoidMessage(child, FG_GETNAME, &name);
  fgSkin* skin = !name?0:fgSkins_Get(self, name);

  if(!skin)
  {
    fgChild_VoidMessage(child, FG_GETCLASSNAME, &name);
    skin = fgSkins_Get(self, name);
  }
  fgChild_VoidMessage(child, FG_SETSKIN, skin); // IF we can't find a skin we deliberately set the skin to NULL.
}