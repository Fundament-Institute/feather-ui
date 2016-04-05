// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "bss-util/khash.h"
#include "fgRadioButton.h"
#include "fgSkin.h"
#include "fgRoot.h"

#ifdef BSS_32BIT
#define kh_ptr_hash_func(key) kh_int_hash_func((size_t)key)
#define kh_ptr_hash_equal(a, b) kh_int_hash_equal(a, b)
#else
#define kh_ptr_hash_func(key) kh_int64_hash_func((size_t)key)
#define kh_ptr_hash_equal(a, b) kh_int64_hash_equal(a, b)
#endif

KHASH_INIT(fgRadioGroup, fgChild*, fgRadiobutton*, 1, kh_ptr_hash_func, kh_ptr_hash_equal);

__inline struct __kh_fgRadioGroup_t* fgRadioGroup_init() { return kh_init_fgRadioGroup(); }
__inline void fgRadioGroup_destroy(struct __kh_fgRadioGroup_t* p) { kh_destroy_fgRadioGroup(p); }

void FG_FASTCALL fgRadiobutton_Init(fgRadiobutton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgRadiobutton_Destroy, (FN_MESSAGE)&fgRadiobutton_Message);
}
void FG_FASTCALL fgRadiobutton_Destroy(fgRadiobutton* self)
{
  fgSendMsg<FG_SETPARENT>(*self); // Ensure we remove ourselves from the hash
  fgWindow_Destroy(&self->window.window);
}

fgRadiobutton** fgRadiobutton_GetHash(fgChild* parent)
{
  khiter_t iter = kh_get(fgRadioGroup, fgroot_instance->radiohash, parent);
  if(iter != kh_end(fgroot_instance->radiohash) && kh_exist(fgroot_instance->radiohash, iter))
    return &kh_val(fgroot_instance->radiohash, iter);
  return 0;
}
fgRadiobutton* fgRadiobutton_GetHashRef(fgChild* parent)
{
  fgRadiobutton** p = fgRadiobutton_GetHash(parent);
  return !p ? 0 : *p;
}

size_t FG_FASTCALL fgRadiobutton_Message(fgRadiobutton* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  fgChild*& parent = self->window.window.element.parent;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgCheckbox_Message(&self->window, msg);
    self->radionext = 0;
    self->radioprev = 0;
    return 1;
  case FG_SETSTATE:
    if(msg->otherint && parent != 0) // if you about to check this radio button, uncheck all of them in it's fgChild group
    {
      fgRadiobutton* cur = fgRadiobutton_GetHashRef(parent);
      while(cur)
      {
        fgChild_IntMessage((fgChild*)cur, FG_SETSTATE, 0, 0);
        cur = cur->radionext;
      }
    }
    self->window.checked = msg->otherint;
    break;
  case FG_SETPARENT:
    if(msg->other != parent)
    {
      if(parent)
      {
        fgRadiobutton** root = fgRadiobutton_GetHash(parent);
        assert(root != 0);
        if(self->radioprev != 0) self->radioprev->radionext = self->radionext;
        else if(root) *root = self->radionext;
        if(self->radionext != 0) self->radionext->radioprev = self->radioprev;
        if(root && !*root)
          kh_del(fgRadioGroup, fgroot_instance->radiohash, kh_get(fgRadioGroup, fgroot_instance->radiohash, parent));
      }
      fgCheckbox_Message(&self->window, msg);
      if(parent)
      {
        int r = 0;
        fgRadiobutton*& root = kh_val(fgroot_instance->radiohash, kh_put(fgRadioGroup, fgroot_instance->radiohash, parent, &r));
        if(r) root = 0; // if r is nonzero we inserted it and thus must initialize root
        self->radionext = root;
        self->radioprev = 0;
        if(root) root->radioprev = self;
        root = self;
      }
    }
    return 1;
  case FG_ACTION:
    if(!self->window.checked)
      fgChild_IntMessage(*self, FG_SETSTATE, 1, 0);
    return 1;
  case FG_GETCLASSNAME:
    return (size_t)"fgRadioButton";
  }
  return fgCheckbox_Message(&self->window, msg);

}