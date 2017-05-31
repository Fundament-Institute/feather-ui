// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgDebug.h"
#include "fgRoot.h"
#include "feathercpp.h"
#include "bss-util/DynArray.h"

fgDebug* fgdebug_instance = nullptr;

const char* fgDebug_GetMessageString(uint16_t msg);

void fgDebug_Init(fgDebug* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, uint16_t units)
{
  fgElement_InternalSetup(self->tabs, parent, next, name, flags, transform, units, (fgDestroy)&fgDebug_Destroy, (fgMessage)&fgDebug_Message);
}

void fgDebug_ClearLog(fgDebug* self)
{
  for(size_t i = 0; i < self->messagestrings.l; ++i)
    fgfree(self->messagestrings.p[i], __FILE__, __LINE__);
  self->messagestrings.l = 0;
  self->messagelog.l = 0;
}

fgDebug* fgDebug_Get()
{
  return fgdebug_instance;
}

fgElement* fgDebug_GetTreeItem(fgElement* root, fgElement* target)
{
  if(target->parent != 0 && target->parent != &fgroot_instance->gui.element)
    root = fgDebug_GetTreeItem(root, target->parent);
  if(!root)
    return 0;
  fgElement* cur = root->root;
  while(cur)
  {
    if(!(cur->flags&FGELEMENT_BACKGROUND) && cur->userdata == target)
      return cur;
    cur = cur->next;
  }
  return 0;
}

void fgDebug_Destroy(fgDebug* self)
{
  fgDebug_Hide();
  fgDebug_ClearLog(self);
  ((bss::DynArray<char*>&)self->messagestrings).~DynArray();
  ((bss::DynArray<fgDebugMessage>&)self->messagelog).~DynArray();
  self->tabs->message = (fgMessage)fgTabcontrol_Message;
  fgTabcontrol_Destroy(&self->tabs);
  VirtualFreeChild(&self->overlay);
  fgdebug_instance = 0; // we can't null this until after we're finished destroying ourselves 
}

void fgDebug_DrawMessages(fgDebug* self, AbsRect* rect, fgDrawAuxData* aux)
{
  _FG_FONT_DESC desc;
  desc.lineheight = self->lineheight;
  if(desc.lineheight == 0)
    fgroot_instance->backend.fgFontGet(self->font, &desc);
  FABS bottom = rect->bottom;
  size_t i = self->messagelog.l;
  while(bottom > rect->top && (i-- > 0))
  {
    ((bss::DynArray<int>*)&self->text32)->Clear();
    ((bss::DynArray<wchar_t>*)&self->text16)->Clear();
    ((bss::DynArray<char>*)&self->text8)->Clear();
    AbsRect txtarea = { rect->left, bottom - desc.lineheight, rect->right, bottom };
    ((bss::DynArray<char>*)&self->text8)->Reserve(fgDebug_WriteMessage(self->messagelog.p + i, 0, 0) + 1);
    self->text8.l = fgDebug_WriteMessage(self->messagelog.p + i, self->text8.p, self->text8.s) + 1;
    fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
    if(v)
      fgroot_instance->backend.fgDrawFont(self->font, v->p, v->l, desc.lineheight, self->letterspacing, self->color.color, &txtarea, 0, &AbsVec_EMPTY, 0, aux, 0);

    bottom = txtarea.top;
  }
}
void fgDebug_SetHover(fgDebug* self, fgElement* hover)
{
  if(self->hover != hover)
  {
    self->hover = hover;
    if(self->hover)
    {
      AbsRect out;
      ResolveOuterRect(hover, &out);
      self->overlay.SetArea(CRect{ out.left, 0, out.top, 0, out.right, 0, out.bottom, 0 });
      fgSetTopmost(&self->overlay);
    }
    else
      fgClearTopmost(&self->overlay);
    self->overlay.SetFlag(FGELEMENT_HIDDEN, !self->hover);
  }
}
size_t fgDebug_OverlayMessage(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
    case FG_DRAW:
      if(fgdebug_instance->hover != 0)
      {
        AbsRect outer;
        ResolveOuterRect(fgdebug_instance->hover, &outer);
        AbsRect clip = outer;
        clip.left += fgdebug_instance->hover->margin.left;
        clip.top += fgdebug_instance->hover->margin.top;
        clip.right -= fgdebug_instance->hover->margin.right;
        clip.bottom -= fgdebug_instance->hover->margin.bottom;
        AbsRect inner;
        GetInnerRect(fgdebug_instance->hover, &inner, &clip);
        fgDrawAuxData* data = (fgDrawAuxData*)msg->p2;

        const CRect ZeroCRect = { 0,0,0,0,0,0,0,0 };
        const AbsVec ZeroAbsVec = { 0,0 };
        fgroot_instance->backend.fgDrawAsset(0, &ZeroCRect, 0x666666FF, 0, 0.0f, &outer, 0, &ZeroAbsVec, FGRESOURCE_RECT, data);
        fgroot_instance->backend.fgDrawAsset(0, &ZeroCRect, 0x6666FFFF, 0, 0.0f, &clip, 0, &ZeroAbsVec, FGRESOURCE_RECT, data);
        fgroot_instance->backend.fgDrawAsset(0, &ZeroCRect, 0x6666FF66, 0, 0.0f, &inner, 0, &ZeroAbsVec, FGRESOURCE_RECT, data);

        /*if(self->font)
        {
        unsigned int pt, dpi;
        float lh = self->lineheight;
        if(lh == 0.0f)
        fgroot_instance->backend.fgFontGet(self->font, &lh, &pt, &dpi);
        char utf8buf[50] = { 0 };
        int utf32buf[50] = { 0 };
        AbsRect txtarea = totalarea;

        int len = snprintf(utf8buf, 50, "%.0f, %.0f", totalarea.left, totalarea.top);
        fgUTF8toUTF32(utf8buf, len, utf32buf, 50);
        txtarea.bottom = totalarea.top;
        txtarea.top -= lh;
        fgroot_instance->backend.fgDrawFont(self->font, utf32buf, self->lineheight, self->letterspacing, self->color.color, &txtarea, 0, &AbsVec { 0,0 }, 0, 0);

        len = snprintf(utf8buf, 50, "%.0f, %.0f", totalarea.right, totalarea.bottom);
        fgUTF8toUTF32(utf8buf, len, utf32buf, 50);
        txtarea.bottom = totalarea.bottom - lh;
        txtarea.top = totalarea.bottom;
        txtarea.left = txtarea.right;
        fgroot_instance->backend.fgDrawFont(self->font, utf32buf, self->lineheight, self->letterspacing, self->color.color, &txtarea, 0, &AbsVec { 0,0 }, 0, 0);
        }*/
      }
      break;
  }

  return fgElement_Message(self, msg);
}

static const char* PROPERTY_LIST[] = {
  "Class",
  "Name",
  "Area",
  "Center",
  "Rotation",
  "Margin",
  "Padding",
  "Min Size",
  "Max Size",
  "Layout Size",
  "Scaling",
  "Parent",
  "Flags",
  "Skin",
  "Style",
  "User ID",
  "Userdata",
};

size_t fgDebug_PropertyMessage(fgText* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEUP:
    if(self->element.parent && self->element.parent->parent)
    {
      AbsRect rect;
      fgElement_RelativeTo(&self->element, self->element.parent->parent, &rect);
      CRect area = { rect.left, 0, rect.top, 0, rect.right, 0, rect.bottom, 0 };
      fgElement* textbox = fgdebug_instance->editbox;
      if(textbox->userdata)
        ((fgElement*)textbox->userdata)->SetFlag(FGELEMENT_HIDDEN, false);
      fgdebug_instance->lastscroll = self->element.parent->parent->padding.topleft;
      textbox->SetArea(area);
      textbox->SetText(self->element.GetText());
      textbox->SetFlag(FGELEMENT_HIDDEN, false);
      textbox->userdata = self;
      textbox->userid = self->element.userid;
      self->element.SetFlag(FGELEMENT_HIDDEN, true);
    }
    break;
  }

  return fgText_Message(self, msg);
}
void fgDebug_ApplyProperty(fgElement* target, int id, const char* prop)
{
  switch(id)
  {
  case 0:
    return;
  case 1:
    target->SetName(prop);
    break;
  case 2:
  {
    CRect area;
    fgMsgType units = fgStyle_ParseCRect(prop, &area);
    target->SetArea(area, units);
  }
  break;
  case 3: // center
  {
    fgTransform t = target->transform;
    fgMsgType units = fgStyle_ParseCVec(prop, &t.center);
    target->SetTransform(t, units);
  }
  break;
  case 4: // rotation
  {
    char* p;
    fgTransform t = target->transform;
    t.rotation = strtof(prop, &p);
    target->SetTransform(t, 0);
  }
    break;
  case 5:
  {
    AbsRect margin;
    fgMsgType units = fgStyle_ParseAbsRect(prop, &margin);
    target->SetMargin(margin, units);
  }
    break;
  case 6:
  {
    AbsRect padding;
    fgMsgType units = fgStyle_ParseAbsRect(prop, &padding);
    target->SetPadding(padding, units);
  }
    break;
  case 7:
  {
    AbsVec dim;
    fgMsgType units = fgStyle_ParseAbsVec(prop, &dim);
    target->SetDim(dim.x, dim.y, FGDIM_MIN, units);
  }
    break;
  case 8:
  {
    AbsVec dim;
    fgMsgType units = fgStyle_ParseAbsVec(prop, &dim);
    target->SetDim(dim.x, dim.y, FGDIM_MAX, units);
  }
  break;
  case 9:
    fgStyle_ParseAbsVec(prop, &target->layoutdim);
    break;
  case 10:
  {
    AbsVec scale;
    fgStyle_ParseAbsVec(prop, &scale);
    target->SetScaling(scale.x, scale.y);
  }
  break;
  case 11: // setparent
    break;
  case 12:
  {
    fgFlag rmflags;
    fgFlag add = fgSkinBase_ParseFlagsFromString(prop, &rmflags, '\n', 0);
    fgFlag def = fgGetTypeFlags(target->GetClassName());
    target->SetFlags((def | add)&(~rmflags));
  }
    break;
  case 13: // skin
    break;
  case 14:
    target->SetStyle(atoi(prop), ~0);
    break;
  case 15:
    target->userid = atoi(prop);
    break;
  }
}

void fgDebug_DisplayProperties(fgElement* e)
{
  fgGrid& g = fgdebug_instance->properties;
  g.GetRow(0)->GetItem(1)->SetText(e->GetClassName());
  g.GetRow(1)->GetItem(1)->SetText(e->GetName());
  g.GetRow(2)->GetItem(1)->SetText(fgStyle_WriteCRect(e->transform.area, 0).c_str());
  g.GetRow(3)->GetItem(1)->SetText(fgStyle_WriteCVec(e->transform.center, 0).c_str());
  g.GetRow(4)->GetItem(1)->SetText(bss::StrF("%g", e->transform.rotation).c_str());
  g.GetRow(5)->GetItem(1)->SetText(fgStyle_WriteAbsRect(e->margin, 0).c_str());
  g.GetRow(6)->GetItem(1)->SetText(fgStyle_WriteAbsRect(e->padding, 0).c_str());
  g.GetRow(7)->GetItem(1)->SetText(bss::StrF("%g %g", e->mindim.x, e->mindim.y).c_str());
  g.GetRow(8)->GetItem(1)->SetText(bss::StrF("%g %g", e->maxdim.x, e->maxdim.y).c_str());
  g.GetRow(9)->GetItem(1)->SetText(bss::StrF("%g %g", e->layoutdim.x, e->layoutdim.y).c_str());
  g.GetRow(10)->GetItem(1)->SetText(bss::StrF("%g %g", e->scaling.x, e->scaling.y).c_str());
  if(e->parent && e->parent->GetName())
    g.GetRow(11)->GetItem(1)->SetText(bss::StrF("%s (%p)", e->parent->GetName(), e->parent).c_str());
  else if(e->parent)
    g.GetRow(11)->GetItem(1)->SetText(bss::StrF("%p", e->parent).c_str());
  else
    g.GetRow(11)->GetItem(1)->SetText("");

  fgFlag def = fgGetTypeFlags(e->GetClassName());
  fgFlag rm = def & (~e->flags);
  fgFlag add = (~def) & e->flags;

  bss::Str flags;
  fgStyle_WriteFlagsIterate(flags, e->GetClassName(), "\n", add, false);
  fgStyle_WriteFlagsIterate(flags, e->GetClassName(), "\n", rm, true);
  g.GetRow(12)->GetItem(1)->SetText(flags.c_str()[0] ? (flags.c_str() + 1) : "");
  g.GetRow(13)->GetItem(1)->SetText(!e->skin ? "[none]" : e->skin->base.name);
  g.GetRow(14)->GetItem(1)->SetText(bss::StrF("%X", e->style).c_str());
  g.GetRow(15)->GetItem(1)->SetText(bss::StrF("%u", e->userid).c_str());
  g.GetRow(16)->GetItem(1)->SetText(bss::StrF("%p", e->userdata).c_str());
}

size_t fgDebug_EditBoxMessage(fgTextbox* self, const FG_Msg* msg)
{
  fgFlag otherint = (fgFlag)msg->u;
  switch(msg->type)
  {
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = bss::bssSetBit<fgFlag>(self->scroll->flags, otherint, msg->u2 != 0);
  case FG_SETFLAGS:
    if((self->scroll->flags ^ otherint) & FGELEMENT_HIDDEN)
    {
      if(self->scroll->userdata && (otherint & FGELEMENT_HIDDEN))
      {
        ((fgElement*)self->scroll->userdata)->SetFlag(FGELEMENT_HIDDEN, false);
        self->scroll->userdata = 0;
      }
    }
    break;
  case FG_ACTION:
    if(!msg->subtype)
    {
      fgElement* e = (fgElement*)((fgElement*)fgdebug_instance->elements->userdata)->userdata;
      fgDebug_ApplyProperty(e, self->scroll->userid, self->scroll->GetText());
      self->scroll->SetFlag(FGELEMENT_HIDDEN, true);
      fgDebug_DisplayProperties(e);
    }
    break;
  }

  return fgTextbox_Message(self, msg);
}

void fgDebug_ContextMenu(struct _FG_ELEMENT* e, const FG_Msg* m)
{
  if(m->e)
  {
    switch(m->e->userid)
    {
    case 0: // delete
      if(fgdebug_instance->elements->userdata)
      {
        fgElement* e = (fgElement*)((fgElement*)fgdebug_instance->elements->userdata)->userdata;
        if(e)
          VirtualFreeChild(e);
      }
      break;
    }
  }
}
void fgDebug_PropertiesAction(fgElement* e, const FG_Msg* m)
{
  switch(m->subtype)
  {
  case FGSCROLLBAR_BAR:
  case FGSCROLLBAR_PAGE:
  case FGSCROLLBAR_BUTTON:
  case FGSCROLLBAR_CHANGE:
  case FGSCROLLBAR_SCROLLTO:
  case FGSCROLLBAR_SCROLLTOABS:
    if(!(fgdebug_instance->editbox->flags&FGELEMENT_HIDDEN))
    {
      CRect area = fgdebug_instance->editbox->transform.area;
      area.top.abs += e->padding.top - fgdebug_instance->lastscroll.y;
      area.bottom.abs += e->padding.top - fgdebug_instance->lastscroll.y;
      fgdebug_instance->editbox->SetArea(area);
      fgdebug_instance->lastscroll = e->padding.topleft;
    }
  }
}
void fgDebug_ContextMenuInsert(struct _FG_ELEMENT* e, const FG_Msg* m)
{
  if(m->e && m->e->userdata && fgdebug_instance->elements->userdata)
  {
    fgElement* target = (fgElement*)reinterpret_cast<fgElement*>(fgdebug_instance->elements->userdata)->userdata;
    if(target)
      fgCreate((const char*)m->e->userdata, target->parent, target, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText((const char*)m->e->userdata);
  }
}
size_t fgDebug_Message(fgDebug* self, const FG_Msg* msg)
{
  assert(fgroot_instance != 0);
  assert(self != 0);
  fgFlag otherint = (fgFlag)msg->u;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
  {
    if(fgdebug_instance != nullptr)
      VirtualFreeChild(*fgdebug_instance);

    fgdebug_instance = self;
    memsubset<fgDebug, fgElement>(self, 0); // lineheight must be zero'd before a potential transform unit resolution.
    fgTabcontrol_Message(&self->tabs, msg);
    self->tablayout = self->tabs->AddItemText("Layout");
    self->tabmessages = self->tabs->AddItemText("Messages");
    self->tablayout->GetSelectedItem()->Action();

    const fgTransform tf_elements = { { 0,0,0,0,0,1,0,0.7 }, 0,{ 0,0,0,0 } };
    const fgTransform tf_properties = { { 0,0,0,0.7,0,1,0,1 }, 0,{ 0,0,0,0 } };
    const fgTransform tf_overlay = { { 0,0,0,0,0,0,0,0 }, 0,{ fgroot_instance->gui.element.transform.area.left.abs,0,fgroot_instance->gui.element.transform.area.top.abs,0 } };
    fgTreeview_Init(&self->elements, self->tablayout, 0, "Debug$elements", FGFLAGS_INTERNAL, &tf_elements, 0);
    fgGrid_Init(&self->properties, self->tablayout, 0, "Debug$properties", FGBOX_TILEY | FGFLAGS_INTERNAL, &tf_properties, 0);
    fgText_Init(&self->contents, self->tabmessages, 0, "Debug$contents", FGELEMENT_HIDDEN | FGFLAGS_INTERNAL, &fgTransform_EMPTY, 0);
    fgElement_Init(&self->overlay, fgroot_instance->gui, 0, "Debug$overlay", FGELEMENT_HIDDEN | FGELEMENT_IGNORE | FGELEMENT_BACKGROUND | FGELEMENT_NOCLIP | FGFLAGS_INTERNAL, &tf_overlay, 0);
    self->overlay.message = (fgMessage)fgDebug_OverlayMessage;
    fgTextbox_Init(&self->editbox, self->properties, 0, "Debug$editbox", FGELEMENT_HIDDEN | FGTEXTBOX_SINGLELINE | FGTEXTBOX_ACTION | FGELEMENT_BACKGROUND, &fgTransform_EMPTY, 0);
    self->editbox->message = (fgMessage)fgDebug_EditBoxMessage;
    
    fgElement* column0 = self->properties.InsertColumn("Name");
    fgElement* column1 = self->properties.InsertColumn("Value");
    column0->SetFlag(FGELEMENT_EXPANDX, false);
    column1->SetFlag(FGELEMENT_EXPANDX, false);
    column0->transform.area.right.abs = column0->transform.area.left.abs + 100;
    column1->transform.area.right.abs = column1->transform.area.left.abs + 300;
    column0->SetArea(column0->transform.area);
    column1->SetArea(column1->transform.area);
    self->properties.header->SetValueF(2.0f);
    const fgTransform tf_prop = { { 0,0,0,0,0,1,0,0 }, 0,{ 0,0,0,0 } };
    for(size_t i = 0; i < sizeof(PROPERTY_LIST) / sizeof(const char*); ++i)
    {
      fgGridRow* r = self->properties.InsertRow();
      fgCreate("text", *r, 0, 0, FGELEMENT_EXPANDY, &fgTransform_EMPTY, 0)->SetText(PROPERTY_LIST[i]);
      fgElement* prop = fgCreate("text", *r, 0, 0, FGELEMENT_EXPANDY, &fgTransform_EMPTY, 0);
      prop->message = (fgMessage)fgDebug_PropertyMessage;
      prop->userid = i;
    }
    fgSubmenu_Init(&self->context, *self, 0, "Debug$context", fgGetTypeFlags("Submenu") | FGFLAGS_INTERNAL, &fgTransform_EMPTY, 0);
    self->context->AddItemText("Delete");
    //self->context->AddItemText("Move");
    fgElement* insertmenu = fgCreate("Submenu", self->context->AddItemText("Insert"), 0, 0, FGFLAGS_DEFAULTS, 0, 0);
    fgIterateControls(insertmenu, [](void* p, const char* s) { fgElement* e = (fgElement*)p; e->AddItemText(s)->userdata = (void*)s; });
    self->context->AddListener(FG_ACTION, fgDebug_ContextMenu);
    insertmenu->AddListener(FG_ACTION, fgDebug_ContextMenuInsert);

    self->elements->SetContextMenu(self->context);
    self->properties->ReorderChild(self->editbox, 0);
    self->properties->AddListener(FG_ACTION, fgDebug_PropertiesAction);
    self->behaviorhook = &fgBehaviorHookDefault;
  }
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgDebug* hold = reinterpret_cast<fgDebug*>(msg->e);
      fgTabcontrol_Message(&self->tabs, msg);
      self->elements->Clone(hold->elements);
      self->properties->Clone(hold->properties);
      self->contents->Clone(hold->contents);
      self->context->Clone(hold->context);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->elements);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->properties);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->contents);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->context);
      reinterpret_cast<bss::DynArray<fgDebugMessage>&>(hold->messagelog) = reinterpret_cast<bss::DynArray<fgDebugMessage>&>(self->messagelog);
      reinterpret_cast<bss::DynArray<char*>&>(hold->messagestrings) = reinterpret_cast<bss::DynArray<char*>&>(self->messagestrings);
      hold->behaviorhook = self->behaviorhook;
      hold->depth = self->depth;
      hold->hover = self->hover;
      hold->ignore = self->ignore;
      hold->font = fgroot_instance->backend.fgCloneFont(self->font, 0);
      hold->color = self->color;
      hold->lineheight = self->lineheight;
      hold->letterspacing = self->letterspacing;
      hold->oldpadding = self->oldpadding;
      reinterpret_cast<bss::DynArray<int>&>(hold->text32) = reinterpret_cast<bss::DynArray<int>&>(self->text32);
      reinterpret_cast<bss::DynArray<wchar_t>&>(hold->text16) = reinterpret_cast<bss::DynArray<wchar_t>&>(self->text16);
      reinterpret_cast<bss::DynArray<char>&>(hold->text8) = reinterpret_cast<bss::DynArray<char>&>(self->text8);
    }
    return sizeof(fgDebug);
  case FG_DRAW:
    {
      AbsRect clip;
      ResolveRect(self->elements, &clip);
      //fgDebug_DrawMessages(self, &clip, (fgDrawAuxData*)msg->p2);
    }
    break;
  case FG_SETFLAG: // We need to perform extra logic on show/hide
    otherint = bss::bssSetBit<fgFlag>(self->tabs->flags, otherint, msg->u2 != 0);
  case FG_SETFLAGS:
    if((otherint^self->tabs->flags) & FGELEMENT_HIDDEN)
    { // handle a layout flag change
      size_t r = fgTabcontrol_Message(&self->tabs, msg);
      self->overlay.SetFlag(FGELEMENT_HIDDEN, otherint&FGELEMENT_HIDDEN);
      if(otherint&FGELEMENT_HIDDEN)
      {
        fgroot_instance->fgBehaviorHook = self->behaviorhook;
        if(!(self->tabs->flags&FGDEBUG_OVERLAY))
          fgroot_instance->gui->SetPadding(self->oldpadding);
        if(self->tabs->flags&FGDEBUG_CLEARONHIDE)
          fgDebug_ClearLog(self);
      }
      else
      {
        self->behaviorhook = fgroot_instance->fgBehaviorHook;
        fgDebug_BuildTree(self->elements);
        self->oldpadding = fgroot_instance->gui->padding;
        if(!(self->tabs->flags&FGDEBUG_OVERLAY))
          fgroot_instance->gui->SetPadding(AbsRect{ self->oldpadding.left, self->oldpadding.top, self->oldpadding.right + 300, self->oldpadding.bottom });

        fgroot_instance->fgBehaviorHook = &fgRoot_BehaviorDebug;
      }
      return r;
    }
    break;
  case FG_SETFONT:
    if(self->font) fgroot_instance->backend.fgDestroyFont(self->font);
    self->font = 0;
    if(msg->p)
    {
      fgFontDesc desc;
      fgroot_instance->backend.fgFontGet(msg->p, &desc);
      AbsVec dpi = self->tabs->GetDPI();
      bool identical = (dpi.x == desc.dpi.x && dpi.y == desc.dpi.y);
      desc.dpi = dpi;
      self->font = fgroot_instance->backend.fgCloneFont(msg->p, identical ? 0 : &desc);
    }
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->f;
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETLETTERSPACING:
    self->letterspacing = msg->f;
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETCOLOR:
    self->color.color = (unsigned int)msg->i;
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_GETFONT:
    return reinterpret_cast<size_t>(self->font);
  case FG_GETLINEHEIGHT:
    return *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_GETLETTERSPACING:
    return *reinterpret_cast<size_t*>(&self->letterspacing);
  case FG_GETCOLOR:
    return self->color.color;
  case FG_GETCLASSNAME:
    return (size_t)"Debug";
  }
  return fgTabcontrol_Message(&self->tabs, msg);
}

char* fgDebug_CopyText(fgDebug* self, const char* s)
{
  if(!s) return 0;
  size_t len = strlen(s) + 1;
  char* ret = fgmalloc<char>(len, __FILE__, __LINE__);
  memcpy(ret, s, len);
  ((bss::DynArray<char*>&)self->messagestrings).AddConstruct(ret);
  return ret;
}
size_t fgTreeItem_DebugMessage(fgTreeItem* self, const FG_Msg* msg)
{
  assert(fgdebug_instance != 0);
  assert(self != 0);

  switch(msg->type)
  {
  case FG_DESTROY:
    if(fgdebug_instance->elements->userdata == self)
      fgdebug_instance->elements->userdata = 0;
    break;
  case FG_MOUSEON:
    fgDebug_SetHover(fgdebug_instance, (fgElement*)self->control.element.userdata);
    return FG_ACCEPT;
  case FG_MOUSEOFF:
    fgDebug_SetHover(fgdebug_instance, 0);
    return FG_ACCEPT;
  case FG_GOTFOCUS:
    if(self->control.element.userdata)
    {
      fgdebug_instance->editbox->SetFlag(FGELEMENT_HIDDEN, true);
      fgDebug_DisplayProperties((fgElement*)self->control.element.userdata);
      fgdebug_instance->elements->userdata = self;
    }
  }

  return fgTreeItem_Message(self, msg);
}

const char* fgDebug_GetElementName(fgDebug* self, fgElement* e)
{
  if(!e)
    return 0;
  const char* r = e->GetName();
  return !r ? e->GetClassName() : fgDebug_CopyText(self, r);
}

fgElement* fgDebug_GetElementUnderMouse()
{
  AbsRect cache;
  fgElement* cur = fgroot_instance->gui;
  fgElement* hover = 0;
  while(cur)
  {
    hover = cur;
    cur = fgElement_GetChildUnderMouse(cur, fgroot_instance->mouse.x, fgroot_instance->mouse.y, &cache);
  }
  return hover;
}

void fgDebug_TreeInsert(fgElement* parent, fgElement* element, fgElement* treeview, fgElement* contextmenu)
{
  assert(fgdebug_instance != 0);
  assert(fgroot_instance != 0);
  assert(element != 0);
  fgElement* root = parent;

  if(treeview != 0)
  {
    fgElement* next = 0;
    if(element->next)
      next = fgDebug_GetTreeItem(parent, element->next);
    root = fgroot_instance->backend.fgCreate("TreeItem", parent, next, 0, FGELEMENT_EXPAND, 0, 0);
    root->SetContextMenu(contextmenu);
    root->message = (fgMessage)&fgTreeItem_DebugMessage;
    root->userdata = element;
  }
  else
    treeview = !fgdebug_instance ? parent : *fgdebug_instance;

  fgElement* text = fgroot_instance->backend.fgCreate("Text", root, 0, 0, FGELEMENT_EXPAND, 0, 0);

  if(element->GetName())
    text->SetText(element->GetName());
  else
    text->SetText(element->GetClassName());

  if(element == treeview)
    return;

  fgElement* cur = element->root;
  while(cur)
  {
    fgDebug_TreeInsert(root, cur, treeview, contextmenu);
    cur = cur->next;
  }
}

size_t fgRoot_BehaviorDebug(fgElement* self, const FG_Msg* msg)
{
  assert(fgdebug_instance != 0);
  assert(self != 0);
  static const FG_Msg* msgbuffer = 0;
  static size_t depthbuffer = 0;
  static size_t* indexbuffer = 0;

  if(msg->type == FG_DRAW) // Don't log FG_DRAW calls or the whole log will be flooded with them.
  {
    fgdebug_instance->ignore += 1;
    size_t r = (*fgdebug_instance->behaviorhook)(self, msg);
    fgdebug_instance->ignore -= 1;
    return r;
  }
  assert(fgdebug_instance->behaviorhook != &fgRoot_BehaviorDebug);
  if(fgdebug_instance->ignore > 0) // If the ignore flag is set, don't log any messages
    return (*fgdebug_instance->behaviorhook)(self, msg);

  if(self == &fgdebug_instance->overlay)
    return (*fgdebug_instance->behaviorhook)(self, msg);

  fgElement* parent = self->parent;
  while(parent != 0) // Make sure we do not log any messages associated with the debug object itself.
  {
    if(parent == fgdebug_instance->tabs)
      return (*fgdebug_instance->behaviorhook)(self, msg);
    parent = parent->parent;
  }

  switch(msg->type)
  {
  case FG_MOUSEMOVE:
  case FG_KEYDOWN:
  case FG_KEYUP:
    if(fgroot_instance->GetKey(FG_KEY_MENU))
    {
      fgdebug_instance->ignore += 1;
      fgDebug_SetHover(fgdebug_instance, fgDebug_GetElementUnderMouse());
      fgdebug_instance->ignore -= 1;
      return FG_ACCEPT;
    }
    else
      fgDebug_SetHover(fgdebug_instance, 0);
    break;
  case FG_MOUSEUP:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSESCROLL:
    if(fgroot_instance->GetKey(FG_KEY_MENU))
      return FG_ACCEPT;
  case FG_MOUSEDOWN:
    if(fgroot_instance->GetKey(FG_KEY_MENU))
    {
      fgdebug_instance->ignore += 1;
      fgDebug_SetHover(fgdebug_instance, fgDebug_GetElementUnderMouse());
      fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, fgdebug_instance->hover);
      if(treeitem)
        treeitem->MouseDown(msg->x, msg->y, msg->button, msg->allbtn);
      fgdebug_instance->ignore -= 1;
      return FG_ACCEPT;
    }
    break;
  case FG_REMOVECHILD:
    if(msg->e && msg->e->parent == self)
    {
      fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, msg->e);
      if(treeitem)
        VirtualFreeChild(treeitem);
    }
    break;
  }

  if(msgbuffer)
    *indexbuffer = fgDebug_LogMessage(fgdebug_instance, msgbuffer, 0, depthbuffer);

  msgbuffer = msg;
  depthbuffer = fgdebug_instance->depth;
  size_t index = (size_t)-1;
  indexbuffer = &index;
  ++fgdebug_instance->depth;
  size_t r = (*fgdebug_instance->behaviorhook)(self, msg);
  --fgdebug_instance->depth;

  if(msgbuffer)
  {
    switch(msgbuffer->type)
    {
    case FG_SETFLAGS:
    case FG_SETMARGIN:
    case FG_SETPADDING:
    case FG_SETPARENT:
    case FG_ADDITEM:
    case FG_ADDCHILD:
    case FG_REMOVECHILD:
    case FG_REORDERCHILD:
    case FG_LAYOUTCHANGE:
    case FG_LAYOUTLOAD:
    case FG_SETSKIN:
    case FG_SETSTYLE:
    case FG_DRAGOVER:
    case FG_DROP:
    case FG_KEYUP:
    case FG_KEYDOWN:
    case FG_KEYCHAR:
    case FG_MOUSEMOVE:
    case FG_MOUSEDOWN:
    case FG_MOUSEUP:
      if(!r)
        break;
    default:
      *indexbuffer = fgDebug_LogMessage(fgdebug_instance, msgbuffer, 0, depthbuffer);
    }
    msgbuffer = 0;
  }

  if(index < fgdebug_instance->messagelog.l)
    fgdebug_instance->messagelog.p[index].value = r;

  switch(msg->type)
  {
  case FG_ADDCHILD:
    if(msg->e)
    {
      fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, self);
      if(treeitem)
        fgDebug_TreeInsert(treeitem, msg->e, treeitem, fgdebug_instance->context);
      else
        break;
    }
  case FG_REORDERCHILD:
    if(msg->e && msg->e->parent == self)
    {
      fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, msg->e);
      fgElement* nextitem = !msg->e->next ? 0 : fgDebug_GetTreeItem(fgdebug_instance->elements, msg->e->next);
      fgElement* selfitem = fgDebug_GetTreeItem(fgdebug_instance->elements, self);
      if(selfitem && treeitem)
        selfitem->ReorderChild(treeitem, nextitem);
    }
    break;
  case FG_SETNAME:
  {
    fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, self);
    if(treeitem)
    {
      fgElement* text = fgElement_GetChildByType(treeitem, "text");
      if(text)
        text->SetText(!self->GetName() ? self->GetClassName() : self->GetName());
    }
  }
  case FG_SETAREA:
  case FG_SETTRANSFORM:
  case FG_SETMARGIN:
  case FG_SETPADDING:
  case FG_SETDIM:
  case FG_SETSCALING:
  case FG_SETPARENT:
  case FG_SETFLAG:
  case FG_SETFLAGS:
  case FG_SETSKIN:
  case FG_SETSTYLE:
  case FG_SETUSERDATA:
  case FG_LAYOUTCHANGE:
    if(fgdebug_instance->elements->userdata && self == reinterpret_cast<fgElement*>(fgdebug_instance->elements->userdata)->userdata)
      fgDebug_DisplayProperties(self);
    break;
  }
  return r;
}

void fgDebug_BuildTree(fgElement* treeview)
{
  assert(fgroot_instance != 0);
  // Clean out the tree
  fgElement* cur;
  fgElement* hold = treeview->root;
  while((cur = hold) != 0)
  {
    hold = hold->next;
    if(!(cur->flags&FGELEMENT_BACKGROUND))
      VirtualFreeChild(cur);
  }

  fgDebug_TreeInsert(treeview, &fgroot_instance->gui.element, 0, fgdebug_instance->context);
}

void fgDebug_Show(const fgTransform* tf, char overlay)
{
  static const fgTransform TF_DEFAULT = { {-300,1,0,0,0,1,0,1}, 0, {0,0,0,0} };
  if(!tf)
    tf = &TF_DEFAULT;

  assert(fgroot_instance != 0);
  if(!fgdebug_instance)
    fgdebug_instance = reinterpret_cast<fgDebug*>(fgCreate("debug", *fgroot_instance, 0, 0, FGELEMENT_HIDDEN | (overlay ? FGDEBUG_OVERLAY : 0) | FGELEMENT_BACKGROUND, tf, 0));
  assert(fgdebug_instance != 0);
  if(fgroot_instance->fgBehaviorHook == &fgRoot_BehaviorDebug)
    return; // Prevent an infinite loop

  fgdebug_instance->tabs->SetFlag(FGELEMENT_HIDDEN, false);
}
void fgDebug_Hide()
{
  if(fgdebug_instance)
    fgdebug_instance->tabs->SetFlag(FGELEMENT_HIDDEN, true);
}

size_t fgDebug_LogMessage(fgDebug* self, const FG_Msg* msg, unsigned long long time, size_t depth)
{
  fgDebugMessage m = { 0 };
  m.type = msg->type;
  m.subtype = msg->subtype;
  m.time = time;
  m.depth = depth;
  self->ignore += 1;

  switch(msg->type)
  {
  default:
    m.arg1.p = msg->p;
    m.arg2.p = msg->p2;
    break;
  case FG_MOVE:
    m.arg1.element = msg->e;
    m.arg1.name = fgDebug_GetElementName(self, m.arg1.element);
    m.arg2.p = msg->p2;
    break;
  case FG_SETPARENT:
  case FG_ADDITEM:
  case FG_ADDCHILD:
  case FG_REORDERCHILD:
    m.arg2.element = msg->e2;
    m.arg2.name = fgDebug_GetElementName(self, m.arg2.element);
  case FG_REMOVEITEM:
  case FG_REMOVECHILD:
  case FG_DROP:
  case FG_GETSKIN:
    m.arg1.name = fgDebug_GetElementName(self, m.arg1.element);
    m.arg1.element = msg->e;
    break;
  case FG_SETUV:
  case FG_SETAREA:
    if(msg->p != nullptr)
      m.arg1.crect = *(CRect*)msg->p;
    break;
  case FG_SETTRANSFORM:
    if(msg->p != nullptr)
      m.arg1.transform = *(fgTransform*)msg->p;
    break;
  case FG_SETMARGIN:
    if(msg->p != nullptr)
      m.arg1.rect = *(AbsRect*)msg->p;
    break;
  case FG_SETPADDING:
    if(msg->p != nullptr)
      m.arg1.rect = *(AbsRect*)msg->p;
    break;
  case FG_LAYOUTFUNCTION:
    m.arg1.message = (FG_Msg*)msg->p;
    if(msg->p2 != nullptr)
      m.arg2.crect = *(CRect*)msg->p2;
    break;
  case FG_SETDIM:
    m.arg1.f = msg->f;
    m.arg2.f = msg->f2;
    break;
  case FG_SETSCALING:
    m.arg1.f = msg->f;
    m.arg2.f = msg->f2;
    break;
  case FG_MOUSEDOWN:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEUP:
    m.mouse.button = msg->button;
    m.mouse.allbtn = msg->allbtn;
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSEMOVE:
  case FG_DRAGOVER:
    m.mouse.x = msg->x;
    m.mouse.y = msg->y;
    break;
  case FG_DRAW:
    if(msg->p != nullptr)
      m.arg1.rect = *(AbsRect*)msg->p;
    m.arg2.i = msg->u2;
    break;
  case FG_INJECT:
    m.arg1.message = (FG_Msg*)msg->p;
    if(msg->p2 != nullptr)
      m.arg2.rect = *(AbsRect*)msg->p2;
    break;
  case FG_SETSTYLE:
    if(!msg->subtype)
      m.arg1.s = fgDebug_CopyText(self, (const char*)msg->p);
    else
      m.arg1.p = msg->p;
    m.arg2.u = msg->u2;
    break;
  case FG_SETUSERDATA:
    m.arg1.p = msg->p;
    m.arg2.s = fgDebug_CopyText(self, (const char*)msg->p2);
    break;
  case FG_GETUSERDATA:
  case FG_SETTEXT:
  case FG_SETNAME:
    m.arg1.s = fgDebug_CopyText(self, (const char*)msg->p);
    break;
  case FG_MOUSESCROLL:
    m.mouse.scrolldelta = msg->scrolldelta;
    m.mouse.scrollhdelta = msg->scrollhdelta;
    m.mouse.x = msg->x;
    m.mouse.y = msg->y;
    break;
  case FG_TOUCHBEGIN:
  case FG_TOUCHEND:
  case FG_TOUCHMOVE:
    m.mouse.touchindex = msg->touchindex;
    m.mouse.x = msg->x;
    m.mouse.y = msg->y;
    break;
  case FG_KEYUP:
  case FG_KEYDOWN:
    m.keys.keycode = msg->keycode;
    m.keys.sigkeys = msg->sigkeys;
    break;
  case FG_KEYCHAR:
    m.keys.keychar = msg->keychar;
    m.keys.sigkeys = msg->sigkeys;
    break;
  case FG_CLEAR:
    break;
  }
  self->ignore -= 1;

  return ((bss::DynArray<fgDebugMessage>&)self->messagelog).Add(m);
}

const char* _dbg_getstr(const char* s)
{
  return !s ? "[null]" : s;
}

#define OUTPUT_CRECT(r) r.left.abs,r.left.rel,r.top.abs,r.top.rel,r.right.abs,r.right.rel,r.bottom.abs,r.bottom.rel
#define OUTPUT_CVEC(r) r.x.abs,r.x.rel,r.y.abs,r.y.rel
#define OUTPUT_RECT(r) r.left,r.top,r.right,r.bottom

const char* fgDebug_GetMessageString(fgMsgType msg)
{
  switch(msg)
  {
  case FG_CONSTRUCT: return "FG_CONSTRUCT";
  case FG_DESTROY: return "FG_DESTROY";
  case FG_CLONE: return "FG_CLONE";
  case FG_MOVE: return "FG_MOVE";
  case FG_SETALPHA: return "FG_SETALPHA";
  case FG_SETAREA: return "FG_SETAREA";
  case FG_SETTRANSFORM: return "FG_SETTRANSFORM";
  case FG_SETFLAG: return "FG_SETFLAG";
  case FG_SETFLAGS: return "FG_SETFLAGS";
  case FG_SETMARGIN: return "FG_SETMARGIN";
  case FG_SETPADDING: return "FG_SETPADDING";
  case FG_SETPARENT: return "FG_SETPARENT";
  case FG_ADDCHILD: return "FG_ADDCHILD";
  case FG_REMOVECHILD: return "FG_REMOVECHILD";
  case FG_REORDERCHILD: return "FG_REORDERCHILD";
  case FG_PARENTCHANGE: return "FG_PARENTCHANGE";
  case FG_LAYOUTCHANGE: return "FG_LAYOUTCHANGE";
  case FG_LAYOUTFUNCTION: return "FG_LAYOUTFUNCTION";
  case FG_LAYOUTLOAD: return "FG_LAYOUTLOAD";
  case FG_DRAGOVER: return "FG_DRAGOVER";
  case FG_DROP: return "FG_DROP";
  case FG_DRAW: return "FG_DRAW";
  case FG_INJECT: return "FG_INJECT";
  case FG_SETSKIN: return "FG_SETSKIN";
  case FG_GETSKIN: return "FG_GETSKIN";
  case FG_SETSTYLE: return "FG_SETSTYLE";
  case FG_GETSTYLE: return "FG_GETSTYLE";
  case FG_GETCLASSNAME: return "FG_GETCLASSNAME";
  case FG_GETDPI: return "FG_GETDPI";
  case FG_SETDPI: return "FG_SETDPI";
  case FG_SETUSERDATA: return "FG_SETUSERDATA";
  case FG_GETUSERDATA: return "FG_GETUSERDATA";
  case FG_SETDIM: return "FG_SETDIM";
  case FG_GETDIM: return "FG_GETDIM";
  case FG_SETSCALING: return "FG_SETSCALING";
  case FG_GETSCALING: return "FG_GETSCALING";
  case FG_MOUSEDOWN: return "FG_MOUSEDOWN";
  case FG_MOUSEDBLCLICK: return "FG_MOUSEDBLCLICK";
  case FG_MOUSEUP: return "FG_MOUSEUP";
  case FG_MOUSEON: return "FG_MOUSEON";
  case FG_MOUSEOFF: return "FG_MOUSEOFF";
  case FG_MOUSEMOVE: return "FG_MOUSEMOVE";
  case FG_MOUSESCROLL: return "FG_MOUSESCROLL";
  case FG_TOUCHBEGIN: return "FG_TOUCHBEGIN";
  case FG_TOUCHEND: return "FG_TOUCHEND";
  case FG_TOUCHMOVE: return "FG_TOUCHMOVE";
  case FG_KEYUP: return "FG_KEYUP";
  case FG_KEYDOWN: return "FG_KEYDOWN";
  case FG_KEYCHAR: return "FG_KEYCHAR";
  case FG_JOYBUTTONDOWN: return "FG_JOYBUTTONDOWN";
  case FG_JOYBUTTONUP: return "FG_JOYBUTTONUP";
  case FG_JOYAXIS: return "FG_JOYAXIS";
  case FG_GOTFOCUS: return "FG_GOTFOCUS";
  case FG_LOSTFOCUS: return "FG_LOSTFOCUS";
  case FG_SETNAME: return "FG_SETNAME";
  case FG_GETNAME: return "FG_GETNAME";
  case FG_SETCONTEXTMENU: return "FG_SETCONTEXTMENU";
  case FG_GETCONTEXTMENU: return "FG_GETCONTEXTMENU";
  case FG_NEUTRAL: return "FG_NEUTRAL";
  case FG_HOVER: return "FG_HOVER";
  case FG_ACTIVE: return "FG_ACTIVE";
  case FG_ACTION: return "FG_ACTION";
  case FG_GETITEM: return "FG_GETITEM";
  case FG_ADDITEM: return "FG_ADDITEM";
  case FG_REMOVEITEM: return "FG_REMOVEITEM";
  case FG_SETITEM: return "FG_SETITEM";
  case FG_CLEAR: return "FG_CLEAR";
  case FG_SELECTION: return "FG_SELECTION";
  case FG_GETSELECTEDITEM: return "FG_GETSELECTEDITEM";
  case FG_GETVALUE: return "FG_GETVALUE";
  case FG_SETVALUE: return "FG_SETVALUE";
  case FG_GETRANGE: return "FG_GETRANGE";
  case FG_SETRANGE: return "FG_SETRANGE";
  case FG_SETASSET: return "FG_SETASSET";
  case FG_SETUV: return "FG_SETUV";
  case FG_SETCOLOR: return "FG_SETCOLOR";
  case FG_SETOUTLINE: return "FG_SETOUTLINE";
  case FG_SETFONT: return "FG_SETFONT";
  case FG_SETLINEHEIGHT: return "FG_SETLINEHEIGHT";
  case FG_SETLETTERSPACING: return "FG_SETLETTERSPACING";
  case FG_SETTEXT: return "FG_SETTEXT";
  case FG_GETASSET: return "FG_GETASSET";
  case FG_GETUV: return "FG_GETUV";
  case FG_GETCOLOR: return "FG_GETCOLOR";
  case FG_GETOUTLINE: return "FG_GETOUTLINE";
  case FG_GETFONT: return "FG_GETFONT";
  case FG_GETLINEHEIGHT: return "FG_GETLINEHEIGHT";
  case FG_GETLETTERSPACING: return "FG_GETLETTERSPACING";
  case FG_GETTEXT: return "FG_GETTEXT";
  case FG_DEBUGMESSAGE: return "FG_DEBUGMESSAGE";
  }
  return "UNKNOWN MESSAGE";
}

template<typename F, typename... Args>
ptrdiff_t fgDebug_WriteMessageFn(fgDebugMessage* msg, F fn, Args... args)
{
  int spaces = (int)(msg->depth * 2);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    return (*fn)(args..., "%*sFG_CONSTRUCT()", spaces, "");
  case FG_DESTROY:
    return (*fn)(args..., "%*sFG_DESTROY()", spaces, "");
  case FG_CLONE:
    return (*fn)(args..., "%*sFG_CLONE()", spaces, "");
  case FG_GETSTYLE:
    return (*fn)(args..., "%*sFG_GETSTYLE() - %p", spaces, "", msg->valuep);
  case FG_GETDPI:
    return (*fn)(args..., "%*sFG_GETDPI() - %ti", spaces, "", msg->value);
  case FG_GETCLASSNAME:
    return (*fn)(args..., "%*sFG_GETCLASSNAME() - %s", spaces, "", (char*)msg->valuep);
  case FG_GETNAME:
    return (*fn)(args..., "%*sFG_GETNAME() - %s", spaces, "", (char*)msg->valuep);
  case FG_NEUTRAL:
    return (*fn)(args..., "%*sFG_NEUTRAL()", spaces, "");
  case FG_HOVER:
    return (*fn)(args..., "%*sFG_HOVER()", spaces, "");
  case FG_ACTIVE:
    return (*fn)(args..., "%*sFG_ACTIVE()", spaces, "");
  case FG_ACTION:
    return (*fn)(args..., "%*sFG_ACTION()", spaces, "");
  case FG_GOTFOCUS:
    return (*fn)(args..., "%*sFG_GOTFOCUS()", spaces, "");
  case FG_LOSTFOCUS:
    return (*fn)(args..., "%*sFG_LOSTFOCUS()", spaces, "");
  case FG_GETDIM:
    return (*fn)(args..., "%*sFG_GETDIM:%hhu()", spaces, "", msg->subtype);
  case FG_GETITEM:
    return (*fn)(args..., "%*sFG_GETITEM() - 0x%p", spaces, "", msg->valuep);
  case FG_GETASSET:
    return (*fn)(args..., "%*sFG_GETASSET() - 0x%p", spaces, "", msg->valuep);
  case FG_GETUV:
    //return (*fn)(args..., "%*sFG_GETUV() - CRect{%f,%f,%f,%f,%f,%f,%f,%f}", spaces, "");
    return (*fn)(args..., "%*sFG_GETUV() - CRect", spaces, "");
  case FG_GETCOLOR:
    return (*fn)(args..., "%*sFG_GETCOLOR() - %#zX", spaces, "", msg->value);
  case FG_GETOUTLINE:
    return (*fn)(args..., "%*sFG_GETOUTLINE() - %f", spaces, "", msg->valuef);
  case FG_GETFONT:
    return (*fn)(args..., "%*sFG_GETFONT() - 0x%p", spaces, "", msg->valuep);
  case FG_GETLINEHEIGHT:
    return (*fn)(args..., "%*sFG_GETLINEHEIGHT() - %f", spaces, "", msg->valuef);
  case FG_GETLETTERSPACING:
    return (*fn)(args..., "%*sFG_GETLETTERSPACING() - %f", spaces, "", msg->valuef);
  case FG_MOVE:
    return (*fn)(args..., "%*sFG_MOVE:%hhu(%s [0x%p], %zu)", spaces, "", msg->subtype, _dbg_getstr(msg->arg1.name), msg->arg1.element, msg->arg2.u);
  case FG_SETALPHA:
    return (*fn)(args..., "%*sFG_SETALPHA(%f)", spaces, "", msg->arg1.f);
  case FG_SETAREA:
    return (*fn)(args..., "%*sFG_SETAREA(CRect{%f,%f,%f,%f,%f,%f,%f,%f})", spaces, "", OUTPUT_CRECT(msg->arg1.crect));
  case FG_SETTRANSFORM:
    return (*fn)(args..., "%*sFG_SETTRANSFORM(fgTransform{{%f,%f,%f,%f,%f,%f,%f,%f}, %f, {%f,%f,%f,%f})", spaces, "",
      OUTPUT_CRECT(msg->arg1.transform.area),
      msg->arg1.transform.rotation,
      OUTPUT_CVEC(msg->arg1.transform.center));
  case FG_SETFLAG:
    return (*fn)(args..., "%*sFG_SETFLAG(%#zX, %s)", spaces, "", msg->arg1.u, msg->arg2.u ? "true" : "false");
  case FG_SETFLAGS:
    return (*fn)(args..., "%*sFG_SETFLAGS(%#zX)", spaces, "", msg->arg1.u);
  case FG_SETMARGIN:
    return (*fn)(args..., "%*sFG_SETMARGIN(AbsRect{%f, %f, %f, %f})", spaces, "", OUTPUT_RECT(msg->arg1.rect));
  case FG_SETPADDING:
    return (*fn)(args..., "%*sFG_SETPADDING(AbsRect{%f, %f, %f, %f})", spaces, "", OUTPUT_RECT(msg->arg1.rect));
  case FG_LAYOUTCHANGE:
    return (*fn)(args..., "%*sFG_LAYOUTCHANGE:%hhu(0x%p, 0x%p)", spaces, "", msg->subtype, msg->arg1.p, msg->arg2.p);
  case FG_SETPARENT:
    return (*fn)(args..., "%*sFG_SETPARENT(%s [0x%p], %s [0x%p])", spaces, "", _dbg_getstr(msg->arg1.name), msg->arg1.element, _dbg_getstr(msg->arg2.name), msg->arg2.element);
  case FG_ADDCHILD:
    return (*fn)(args..., "%*sFG_ADDCHILD(%s [0x%p], %s [0x%p])", spaces, "", _dbg_getstr(msg->arg1.name), msg->arg1.element, _dbg_getstr(msg->arg2.name), msg->arg2.element);
  case FG_REMOVECHILD:
    return (*fn)(args..., "%*sFG_REMOVECHILD(%s [0x%p])", spaces, "", _dbg_getstr(msg->arg1.name), msg->arg1.element);
  case FG_REORDERCHILD:
    return (*fn)(args..., "%*sFG_REORDERCHILD(%s [0x%p], %s [0x%p])", spaces, "", _dbg_getstr(msg->arg1.name), msg->arg1.element, _dbg_getstr(msg->arg2.name), msg->arg2.element);
  case FG_GETSKIN:
    return (*fn)(args..., "%*sFG_GETSKIN(%s [0x%p]) - 0x%p", spaces, "", _dbg_getstr(msg->arg1.name), msg->arg1.element, msg->valuep);
  case FG_LAYOUTFUNCTION:
    return (*fn)(args..., "%*sFG_LAYOUTFUNCTION(0x%p, CRect{%f,%f,%f,%f,%f,%f,%f,%f})", spaces, "", msg->arg1.p, OUTPUT_CRECT(msg->arg2.crect));
  case FG_LAYOUTLOAD:
    return (*fn)(args..., "%*sFG_LAYOUTLOAD(0x%p)", spaces, "", msg->arg1.p);
  case FG_SETASSET:
    return (*fn)(args..., "%*sFG_SETASSET(0x%p)", spaces, "", msg->arg1.p);
  case FG_SETFONT:
    return (*fn)(args..., "%*sFG_SETFONT(0x%p)", spaces, "", msg->arg1.p);
  case FG_MOUSEDOWN:
    return (*fn)(args..., "%*sFG_MOUSEDOWN(x:%i, y:%i, %#hhX, %#hhX)", spaces, "", msg->mouse.x, msg->mouse.y, msg->mouse.button, msg->mouse.allbtn);
  case FG_MOUSEDBLCLICK:
    return (*fn)(args..., "%*sFG_MOUSEDBLCLICK(x:%i, y:%i, %#hhX, %#hhX)", spaces, "", msg->mouse.x, msg->mouse.y, msg->mouse.button, msg->mouse.allbtn);
  case FG_MOUSEUP:
    return (*fn)(args..., "%*sFG_MOUSEUP(x:%i, y:%i, %#hhX, %#hhX)", spaces, "", msg->mouse.x, msg->mouse.y, msg->mouse.button, msg->mouse.allbtn);
  case FG_MOUSEON:
    return (*fn)(args..., "%*sFG_MOUSEON(x:%i, y:%i)", spaces, "", msg->mouse.x, msg->mouse.y);
  case FG_MOUSEOFF:
    return (*fn)(args..., "%*sFG_MOUSEOFF(x:%i, y:%i)", spaces, "", msg->mouse.x, msg->mouse.y);
  case FG_MOUSEMOVE:
    return (*fn)(args..., "%*sFG_MOUSEMOVE(x:%i, y:%i)", spaces, "", msg->mouse.x, msg->mouse.y);
  case FG_DRAGOVER:
    return (*fn)(args..., "%*sFG_DRAGOVER(x:%i, y:%i)", spaces, "", msg->mouse.x, msg->mouse.y);
  case FG_DROP:
    return (*fn)(args..., "%*sFG_SETFONT(%s [0x%p])", spaces, "", _dbg_getstr(msg->arg1.name), msg->arg1.element);
  case FG_DRAW:
    return (*fn)(args..., "%*sFG_DRAW:%hhu(AbsRect{%f, %f, %f, %f}, 0x%p)", spaces, "", msg->subtype, OUTPUT_RECT(msg->arg1.rect), msg->arg2.p);
  case FG_INJECT:
    return (*fn)(args..., "%*sFG_INJECT(0x%p, AbsRect{%f, %f, %f, %f})", spaces, "", msg->arg1.p, OUTPUT_RECT(msg->arg2.rect));
  case FG_SETSKIN:
    return (*fn)(args..., "%*sFG_SETSKIN(0x%p)", spaces, "", msg->arg1.p);
  case FG_SETSTYLE:
    if(!msg->subtype)
      return (*fn)(args..., "%*sFG_SETSTYLE(%s, %zu)", spaces, "", _dbg_getstr(msg->arg1.s), msg->arg2.u);
    else if(msg->subtype == 1)
      return (*fn)(args..., "%*sFG_SETSTYLE:1(%zu, %zu)", spaces, "", msg->arg1.u, msg->arg2.u);
    return (*fn)(args..., "%*sFG_SETSTYLE:%hhu(0x%p, %zu)", spaces, "", msg->subtype, msg->arg1.p, msg->arg2.u);
  case FG_GETVALUE:
    return (*fn)(args..., "%*sFG_GETVALUE() - %ti", spaces, "", msg->value);
  case FG_GETRANGE:
    return (*fn)(args..., "%*sFG_GETRANGE() - %ti", spaces, "", msg->value);
  case FG_SELECTION:
    return (*fn)(args..., "%*sFG_SELECTION(0x%p, 0x%p)", spaces, "", msg->arg1.p, msg->arg2.p);
  case FG_GETSELECTEDITEM:
    return (*fn)(args..., "%*sFG_GETSELECTEDITEM(%zu) - 0x%p", spaces, "", msg->arg1.u, msg->valuep);
  case FG_SETDPI:
    return (*fn)(args..., "%*sFG_SETDPI(%ti)", spaces, "", msg->arg1.i);
  case FG_SETCOLOR:
    return (*fn)(args..., "%*sFG_SETCOLOR(%#zX)", spaces, "", msg->arg1.u);
  case FG_SETUSERDATA:
    return (*fn)(args..., "%*sFG_SETUSERDATA(0x%p, %s)", spaces, "", msg->arg1.p, _dbg_getstr(msg->arg2.s));
  case FG_GETUSERDATA:
    return (*fn)(args..., "%*sFG_GETUSERDATA(%s) - 0x%p", spaces, "", _dbg_getstr(msg->arg1.s), msg->valuep);
  case FG_SETTEXT:
    return (*fn)(args..., "%*sFG_SETTEXT(%s)", spaces, "", _dbg_getstr(msg->arg1.s));
  case FG_SETNAME:
    return (*fn)(args..., "%*sFG_SETNAME(%s)", spaces, "", _dbg_getstr(msg->arg1.s));
  case FG_SETCONTEXTMENU:
    return (*fn)(args..., "%*sFG_SETCONTEXTMENU(0x%p)", spaces, "", msg->arg1.p);
  case FG_GETCONTEXTMENU:
    return (*fn)(args..., "%*sFG_GETCONTEXTMENU(0x%p)", spaces, "", msg->arg1.p);
  case FG_MOUSESCROLL:
    return (*fn)(args..., "%*sFG_MOUSESCROLL(x:%i, y:%i, v:%#hi, h:%#hi)", spaces, "", msg->mouse.x, msg->mouse.y, msg->mouse.scrolldelta, msg->mouse.scrollhdelta);
  case FG_TOUCHBEGIN:
    return (*fn)(args..., "%*sFG_TOUCHBEGIN(x:%i, y:%i, %hi)", spaces, "", msg->mouse.x, msg->mouse.y, msg->mouse.touchindex);
  case FG_TOUCHEND:
    return (*fn)(args..., "%*sFG_TOUCHEND(x:%i, y:%i, %hi)", spaces, "", msg->mouse.x, msg->mouse.y, msg->mouse.touchindex);
  case FG_TOUCHMOVE:
    return (*fn)(args..., "%*sFG_TOUCHMOVE(x:%i, y:%i, %hi)", spaces, "", msg->mouse.x, msg->mouse.y, msg->mouse.touchindex);
  case FG_KEYUP:
    return (*fn)(args..., "%*sFG_KEYUP(%#hhX, %#hhX)", spaces, "", msg->keys.keycode, msg->keys.sigkeys);
  case FG_KEYDOWN:
    return (*fn)(args..., "%*sFG_KEYDOWN(%#hhX, %#hhX)", spaces, "", msg->keys.keycode, msg->keys.sigkeys);
  case FG_KEYCHAR:
    return (*fn)(args..., "%*sFG_KEYCHAR(%c (%u), %#hhX)", spaces, "", (char)msg->keys.keychar, msg->keys.keychar, msg->keys.sigkeys);
  case FG_JOYBUTTONDOWN:
    return (*fn)(args..., "%*sFG_JOYBUTTONDOWN(%hi, %s)", spaces, "", msg->joybutton, msg->joydown ? "true" : "false");
  case FG_JOYBUTTONUP:
    return (*fn)(args..., "%*sFG_JOYBUTTONUP(%hi, %s)", spaces, "", msg->joybutton, msg->joydown ? "true" : "false");
  case FG_JOYAXIS:
    return (*fn)(args..., "%*sFG_JOYAXIS(%hi, %f)", spaces, "", msg->joyaxis, msg->joyvalue);
  case FG_SETDIM:
    return (*fn)(args..., "%*sFG_SETDIM:%hhu(%f, %f)", spaces, "", msg->subtype, msg->arg1.f, msg->arg2.f);
  case FG_SETSCALING:
    return (*fn)(args..., "%*sFG_SETSCALING:%hhu(%f, %f)", spaces, "", msg->subtype, msg->arg1.f, msg->arg2.f);
  case FG_SETVALUE:
    return (*fn)(args..., "%*sFG_SETVALUE(%ti, %zu)", spaces, "", msg->arg1.i, msg->arg2.u);
  case FG_SETRANGE:
    return (*fn)(args..., "%*sFG_SETRANGE(%ti, %zu)", spaces, "", msg->arg1.i);
  case FG_SETUV:
    return (*fn)(args..., "%*sFG_SETUV(CRect{%f,%f,%f,%f,%f,%f,%f,%f})", spaces, "", OUTPUT_CRECT(msg->arg2.crect));
  case FG_SETLETTERSPACING:
    return (*fn)(args..., "%*sFG_SETLETTERSPACING(%g)", spaces, "", msg->arg1.f);
  case FG_SETLINEHEIGHT:
    return (*fn)(args..., "%*sFG_SETLINEHEIGHT(%g)", spaces, "", msg->arg1.f);
  case FG_SETOUTLINE:
    return (*fn)(args..., "%*sFG_SETOUTLINE(%g)", spaces, "", msg->arg1.f);
  }

  return 0;
}

ptrdiff_t fgDebug_WriteMessage(fgDebugMessage* msg, char* buf, size_t bufsize)
{
  return fgDebug_WriteMessageFn<int(*)(char* const, size_t, char const* const, ...), char* const, size_t>(msg, &snprintf, buf, bufsize);
}

void fgDebug_DumpMessages(const char* file)
{
  FILE* f;
  FOPEN(f, file, "wb");
  for(size_t i = 0; i < fgdebug_instance->messagelog.l; ++i)
    fgDebug_WriteMessageFn<int(*)(FILE*, char const* const, ...), FILE*>(fgdebug_instance->messagelog.p + i, &fprintf, f);
  fclose(f);
}
