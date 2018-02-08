// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "bss-util/khash.h"
#include "fgRadiobutton.h"
#include "fgSkin.h"
#include "fgRoot.h"

KHASH_INIT(fgRadioGroup, fgElement*, fgRadiobutton*, 1, kh_ptr_hash_func, kh_int_hash_equal);

struct kh_fgRadioGroup_s* fgRadioGroup_init() { return kh_init_fgRadioGroup(); }
void fgRadioGroup_destroy(struct kh_fgRadioGroup_s* p) { kh_destroy_fgRadioGroup(p); }

void fgRadiobutton_Init(fgRadiobutton* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgRadiobutton_Destroy, (fgMessage)&fgRadiobutton_Message);
}
void fgRadiobutton_Destroy(fgRadiobutton* self)
{
  _sendmsg<FG_SETPARENT>(*self); // Ensure we remove ourselves from the hash
  self->window->message = (fgMessage)fgCheckbox_Message;
  fgCheckbox_Destroy(&self->window);
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

size_t fgRadiobutton_Message(fgRadiobutton* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgCheckbox_Message(&self->window, msg);
    self->radionext = 0;
    self->radioprev = 0;
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgRadiobutton* hold = reinterpret_cast<fgRadiobutton*>(msg->e);
      hold->radionext = 0;
      hold->radioprev = 0;
      fgCheckbox_Message(&self->window, msg);
    }
    return sizeof(fgRadiobutton);
  case FG_SETVALUE:
    if(msg->subtype > FGVALUE_INT64)
    {
      fgLog(FGLOG_INFO, "%s set invalid value type: %hu", fgGetFullName(*self).c_str(), msg->subtype);
      return 0;
    }
    if(msg->i && self->window->parent != 0) // if you about to check this radio button, uncheck all others in it's fgElement group
    {
      fgRadiobutton* cur = fgRadiobutton_GetHashRef(self->window->parent);
      while(cur)
      {
        if(cur != self)
          _sendmsg<FG_SETVALUE, size_t>((fgElement*)cur, 0);
        cur = cur->radionext;
      }
    }
    self->window.checked = (char)msg->i;
    break;
  case FG_PARENTCHANGE:
    if(msg->e2) // remove ourselves from the old parent, if there was one
    {
      fgRadiobutton** root = fgRadiobutton_GetHash(msg->e2);
      assert(root != 0);
      if(self->radioprev != 0) self->radioprev->radionext = self->radionext;
      else if(root) *root = self->radionext;
      if(self->radionext != 0) self->radionext->radioprev = self->radioprev;
      if(root && !*root)
        kh_del(fgRadioGroup, fgroot_instance->radiohash, kh_get(fgRadioGroup, fgroot_instance->radiohash, msg->e2));
    }
    if(msg->e) // add ourselves to the new parent, if there is one.
    {
      int r = 0;
      fgRadiobutton*& root = kh_val(fgroot_instance->radiohash, kh_put(fgRadioGroup, fgroot_instance->radiohash, msg->e, &r));
      if(r) root = 0; // if r is nonzero we inserted it and thus must initialize root
      self->radionext = root;
      self->radioprev = 0;
      if(root) root->radioprev = self;
      root = self;
    }
    return FG_ACCEPT;
  case FG_ACTION:
    if(!self->window.checked)
      _sendmsg<FG_SETVALUE, size_t>(*self, 1);
    return FG_ACCEPT;
  case FG_GETCLASSNAME:
    return (size_t)"RadioButton";
  }
  return fgCheckbox_Message(&self->window, msg);

}