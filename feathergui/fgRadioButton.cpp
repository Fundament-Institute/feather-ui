// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "bss-util/khash.h"
#include "fgRadiobutton.h"
#include "fgSkin.h"
#include "fgRoot.h"

KHASH_INIT(fgRadioGroup, fgElement*, fgRadiobutton*, 1, kh_ptr_hash_func, kh_int_hash_equal);

__inline struct __kh_fgRadioGroup_t* fgRadioGroup_init() { return kh_init_fgRadioGroup(); }
__inline void fgRadioGroup_destroy(struct __kh_fgRadioGroup_t* p) { kh_destroy_fgRadioGroup(p); }

void FG_FASTCALL fgRadiobutton_Init(fgRadiobutton* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgRadiobutton_Destroy, (fgMessage)&fgRadiobutton_Message);
}
void FG_FASTCALL fgRadiobutton_Destroy(fgRadiobutton* self)
{
  _sendmsg<FG_SETPARENT>(*self); // Ensure we remove ourselves from the hash
  fgControl_Destroy(&self->window.control);
}

fgRadiobutton** fgRadiobutton_GetHash(fgElement* parent)
{
  khiter_t iter = kh_get(fgRadioGroup, fgroot_instance->radiohash, parent);
  if(iter != kh_end(fgroot_instance->radiohash) && kh_exist(fgroot_instance->radiohash, iter))
    return &kh_val(fgroot_instance->radiohash, iter);
  return 0;
}
fgRadiobutton* fgRadiobutton_GetHashRef(fgElement* parent)
{
  fgRadiobutton** p = fgRadiobutton_GetHash(parent);
  return !p ? 0 : *p;
}

size_t FG_FASTCALL fgRadiobutton_Message(fgRadiobutton* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  fgElement*& parent = self->window.control.element.parent;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgCheckbox_Message(&self->window, msg);
    self->radionext = 0;
    self->radioprev = 0;
    return FG_ACCEPT;
  case FG_SETVALUE:
    if(msg->subtype > FGVALUE_INT64)
      return 0;
    if(msg->otherint && parent != 0) // if you about to check this radio button, uncheck all others in it's fgElement group
    {
      fgRadiobutton* cur = fgRadiobutton_GetHashRef(parent);
      while(cur)
      {
        if(cur != self)
          fgIntMessage((fgElement*)cur, FG_SETVALUE, 0, 0);
        cur = cur->radionext;
      }
    }
    self->window.checked = (char)msg->otherint;
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
    return FG_ACCEPT;
  case FG_ACTION:
    if(!self->window.checked)
      fgIntMessage(*self, FG_SETVALUE, 1, 0);
    return FG_ACCEPT;
  case FG_GETCLASSNAME:
    return (size_t)"RadioButton";
  }
  return fgCheckbox_Message(&self->window, msg);

}