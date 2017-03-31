// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgDebug.h"
#include "fgRoot.h"
#include "feathercpp.h"
#include "bss-util/cDynArray.h"

fgDebug* fgdebug_instance = nullptr;

const char* fgDebug_GetMessageString(unsigned short msg);

void fgDebug_Init(fgDebug* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(&self->element, parent, next, name, flags, transform, units, (fgDestroy)&fgDebug_Destroy, (fgMessage)&fgDebug_Message);
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
  ((bss_util::cDynArray<char*>&)self->messagestrings).~cDynArray();
  ((bss_util::cDynArray<fgDebugMessage>&)self->messagelog).~cDynArray();
  fgElement_Destroy(&self->element);
  fgdebug_instance = 0; // we can't null this until after we're finished destroying ourselves 
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
  "User Hash",
};

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
    ((bss_util::cDynArray<int>*)&self->text32)->Clear();
    ((bss_util::cDynArray<wchar_t>*)&self->text16)->Clear();
    ((bss_util::cDynArray<char>*)&self->text8)->Clear();
    AbsRect txtarea = { rect->left, bottom - desc.lineheight, rect->right, bottom };
    ((bss_util::cDynArray<char>*)&self->text8)->Reserve(fgDebug_WriteMessage(self->messagelog.p + i, 0, 0) + 1);
    self->text8.l = fgDebug_WriteMessage(self->messagelog.p + i, self->text8.p, self->text8.s) + 1;
    fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
    fgroot_instance->backend.fgDrawFont(self->font, v->p, v->l, desc.lineheight, self->letterspacing, self->color.color, &txtarea, 0, &AbsVec_EMPTY, 0, aux, 0);

    bottom = txtarea.top;
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
    fgElement_Message(&self->element, msg);
    const fgTransform tf_elements = { { -300,1,0,0,0,1,0,1 }, 0,{ 0,0,0,0 } };
    const fgTransform tf_properties = { { -300,1,-200,1,0,1,0,1 }, 0,{ 0,0,0,0 } };
    const fgTransform tf_contents = { { 0,0,-200,1,200,0,0,1 }, 0,{ 0,0,0,0 } };
    fgTreeview_Init(&self->elements, *self, 0, "Debug$elements", 0, &tf_elements, 0);
    fgGrid_Init(&self->properties, *self, 0, "Debug$properties", FGELEMENT_HIDDEN, &tf_properties, 0);
    fgText_Init(&self->contents, *self, 0, "Debug$contents", FGELEMENT_HIDDEN, &tf_contents, 0);
    self->properties.InsertColumn("Name");
    self->properties.InsertColumn("Value");
    for(size_t i = 0; i < sizeof(PROPERTY_LIST) / sizeof(const char*); ++i)
    {
      fgGridRow* r = self->properties.InsertRow();
      r->InsertItem(PROPERTY_LIST[i]);
      r->InsertItem("");
    }
    fgSubmenu_Init(&self->context, *self, 0, "Debug$context", fgGetTypeFlags("Submenu"), &fgTransform_EMPTY, 0);
    self->context->AddItemText("Delete");
    self->context->AddItemText("Move");
    fgIterateControls(fgCreate("Submenu", self->context->AddItemText("Insert"), 0, 0, FGELEMENT_USEDEFAULTS, 0, 0), [](void* p, const char* s) { fgElement* e = (fgElement*)p; e->AddItemText(s); });
    self->elements->SetContextMenu(self->context);
    self->behaviorhook = &fgBehaviorHookDefault;
  }
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgElement_Message(&self->element, msg);
    }
    return sizeof(fgText);
  case FG_DRAW:
    if(self->hover != 0)
    {
      AbsRect outer;
      ResolveOuterRect(self->hover, &outer);
      AbsRect clip = outer;
      clip.left += self->hover->margin.left;
      clip.top += self->hover->margin.top;
      clip.right -= self->hover->margin.right;
      clip.bottom -= self->hover->margin.bottom;
      AbsRect inner;
      GetInnerRect(self->hover, &inner, &clip);
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
    
    {
      AbsRect clip;
      ResolveRect(self->elements, &clip);
      //fgDebug_DrawMessages(self, &clip, (fgDrawAuxData*)msg->p2);
    }
    break;
  case FG_SETFLAG: // We need to perform extra logic on show/hide
    otherint = bss_util::bssSetBit<fgFlag>(self->element.flags, otherint, msg->u2 != 0);
  case FG_SETFLAGS:
    if((otherint^self->element.flags) & FGELEMENT_HIDDEN)
    { // handle a layout flag change
      size_t r = fgElement_Message(&self->element, msg);
      if(otherint&FGELEMENT_HIDDEN)
      {
        fgroot_instance->backend.fgBehaviorHook = self->behaviorhook;
        fgroot_instance->gui->SetPadding(self->oldpadding);
        if(self->element.flags&FGDEBUG_CLEARONHIDE)
          fgDebug_ClearLog(self);
      }
      else
      {
        self->elements->SetTransform(fgTransform{ { -300,1,0,0,0,1,0,1 }, 0,{ 0,0,0,0 } });
        self->element.SetTransform(fgTransform_DEFAULT);
        self->behaviorhook = fgroot_instance->backend.fgBehaviorHook;
        fgDebug_BuildTree(self->elements);
        self->oldpadding = fgroot_instance->gui->padding;
        if(!(self->element.flags&FGDEBUG_OVERLAY))
          fgroot_instance->gui->SetPadding(AbsRect{ self->oldpadding.left, self->oldpadding.top, self->oldpadding.right + 300, self->oldpadding.bottom });

        fgroot_instance->backend.fgBehaviorHook = &fgRoot_BehaviorDebug;
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
      fgIntVec dpi = self->element.GetDPI();
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
  return fgElement_Message(&self->element, msg);
}

char* fgDebug_CopyText(fgDebug* self, const char* s)
{
  if(!s) return 0;
  size_t len = strlen(s) + 1;
  char* ret = fgmalloc<char>(len, __FILE__, __LINE__);
  memcpy(ret, s, len);
  ((bss_util::cDynArray<char*>&)self->messagestrings).AddConstruct(ret);
  return ret;
}

size_t fgTreeItem_DebugMessage(fgTreeItem* self, const FG_Msg* msg)
{
  assert(fgdebug_instance != 0);
  assert(self != 0);

  switch(msg->type)
  {
  case FG_MOUSEON:
    fgdebug_instance->hover = (fgElement*)self->control.element.userdata;
    return FG_ACCEPT;
  case FG_MOUSEOFF:
    fgdebug_instance->hover = 0;
    return FG_ACCEPT;
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

  fgElement* parent = self->parent;
  while(parent != 0) // Make sure we do not log any messages associated with the debug object itself.
  {
    if(parent == &fgdebug_instance->element)
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
      fgdebug_instance->hover = fgDebug_GetElementUnderMouse();
      fgdebug_instance->ignore -= 1;
      return FG_ACCEPT;
    }
    else
      fgdebug_instance->hover = 0;
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
      fgdebug_instance->hover = fgDebug_GetElementUnderMouse();
      fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, fgdebug_instance->hover);
      if(treeitem)
        treeitem->MouseDown(msg->x, msg->y, msg->button, msg->allbtn);
      fgdebug_instance->ignore -= 1;
      return FG_ACCEPT;
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

  if(msg->type == FG_REMOVECHILD && msg->e != 0 && !msg->e->parent)
  {
    fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, msg->e);
    if(treeitem)
      VirtualFreeChild(treeitem);
  }
  if(msg->type == FG_ADDCHILD && msg->e != 0 && msg->e->parent == self)
  {
    fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, self);
    if(treeitem)
      fgDebug_TreeInsert(fgdebug_instance->elements, msg->e, 0, fgdebug_instance->context);
  }
  if(msg->type == FG_REORDERCHILD && msg->e != 0 && msg->e->parent == self)
  {
    fgElement* treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, msg->e);
    if(treeitem)
      VirtualFreeChild(treeitem);
    treeitem = fgDebug_GetTreeItem(fgdebug_instance->elements, self);
    if(treeitem)
      fgDebug_TreeInsert(fgdebug_instance->elements, msg->e, 0, fgdebug_instance->context);
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

void fgDebug_Show(float left, float right, char overlay)
{
  assert(fgroot_instance != 0);
  if(!fgdebug_instance)
    fgdebug_instance = reinterpret_cast<fgDebug*>(fgCreate("debug", *fgroot_instance, 0, 0, FGELEMENT_HIDDEN | (overlay ? FGDEBUG_OVERLAY : 0), &fgTransform_DEFAULT, 0));
  assert(fgdebug_instance != 0);
  if(fgroot_instance->backend.fgBehaviorHook == &fgRoot_BehaviorDebug)
    return; // Prevent an infinite loop

  fgdebug_instance->element.SetFlag(FGELEMENT_HIDDEN, false);
}
void fgDebug_Hide()
{
  assert(fgdebug_instance != 0);
  assert(fgdebug_instance != 0);
  fgdebug_instance->element.SetFlag(FGELEMENT_HIDDEN, true);
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
  }
  self->ignore -= 1;

  return ((bss_util::cDynArray<fgDebugMessage>&)self->messagelog).Add(m);
}

const char* _dbg_getstr(const char* s)
{
  return !s ? "[null]" : s;
}

#define OUTPUT_CRECT(r) r.left.abs,r.left.rel,r.top.abs,r.top.rel,r.right.abs,r.right.rel,r.bottom.abs,r.bottom.rel
#define OUTPUT_CVEC(r) r.x.abs,r.x.rel,r.y.abs,r.y.rel
#define OUTPUT_RECT(r) r.left,r.top,r.right,r.bottom

const char* fgDebug_GetMessageString(unsigned short msg)
{
  switch(msg)
  {
  case FG_CONSTRUCT: return "FG_CONSTRUCT";
  case FG_DESTROY: return "FG_DESTROY"; 
  case FG_CLONE: return "FG_CLONE";
  case FG_GETSTYLE: return "FG_GETSTYLE";
  case FG_GETDPI: return "FG_GETDPI";
  case FG_GETCLASSNAME: return "FG_GETCLASSNAME";
  case FG_GETNAME: return "FG_GETNAME";
  case FG_NEUTRAL: return "FG_NEUTRAL";
  case FG_HOVER: return "FG_HOVER";
  case FG_ACTIVE: return "FG_ACTIVE";
  case FG_ACTION: return "FG_ACTION";
  case FG_GOTFOCUS: return "FG_GOTFOCUS";
  case FG_LOSTFOCUS: return "FG_LOSTFOCUS";
  case FG_GETDIM: return "FG_GETDIM";
  case FG_GETITEM: return "FG_GETITEM";
  case FG_GETASSET: return "FG_GETASSET";
  case FG_GETUV: return "FG_GETUV";
  case FG_GETCOLOR: return "FG_GETCOLOR";
  case FG_GETOUTLINE: return "FG_GETOUTLINE";
  case FG_GETFONT: return "FG_GETFONT";
  case FG_GETLINEHEIGHT: return "FG_GETLINEHEIGHT";
  case FG_GETLETTERSPACING: return "FG_GETLETTERSPACING";
  case FG_MOVE: return "FG_MOVE";
  case FG_SETALPHA: return "FG_SETALPHA";
  case FG_SETAREA: return "FG_SETAREA";
  case FG_SETTRANSFORM: return "FG_SETTRANSFORM";
  case FG_SETFLAG: return "FG_SETFLAG";
  case FG_SETFLAGS: return "FG_SETFLAGS";
  case FG_SETMARGIN: return "FG_SETMARGIN";
  case FG_SETPADDING: return "FG_SETPADDING";
  case FG_LAYOUTCHANGE: return "FG_LAYOUTCHANGE";
  case FG_PARENTCHANGE: return "FG_PARENTCHANGE";
  case FG_SETPARENT: return "FG_SETPARENT";
  case FG_ADDCHILD: return "FG_ADDCHILD";
  case FG_REMOVECHILD: return "FG_REMOVECHILD";
  case FG_REORDERCHILD: return "FG_REORDERCHILD";
  case FG_GETSKIN: return "FG_GETSKIN";
  case FG_LAYOUTFUNCTION: return "FG_LAYOUTFUNCTION";
  case FG_LAYOUTLOAD: return "FG_LAYOUTLOAD";
  case FG_SETASSET: return "FG_SETASSET";
  case FG_SETFONT: return "FG_SETFONT";
  case FG_MOUSEDOWN: return "FG_MOUSEDOWN";
  case FG_MOUSEDBLCLICK: return "FG_MOUSEDBLCLICK";
  case FG_MOUSEUP: return "FG_MOUSEUP";
  case FG_MOUSEON: return "FG_MOUSEON";
  case FG_MOUSEOFF: return "FG_MOUSEOFF";
  case FG_MOUSEMOVE: return "FG_MOUSEMOVE";
  case FG_DRAGOVER: return "FG_DRAGOVER";
  case FG_DROP: return "FG_SETFONT";
  case FG_DRAW: return "FG_DRAW";
  case FG_INJECT: return "FG_INJECT";
  case FG_SETSKIN: return "FG_SETSKIN";
  case FG_SETSTYLE: return "FG_SETSTYLE";
  case FG_GETVALUE: return "FG_GETVALUE";
  case FG_GETRANGE: return "FG_GETRANGE";
  case FG_GETSELECTEDITEM: return "FG_GETSELECTEDITEM";
  case FG_SETDPI: return "FG_SETDPI";
  case FG_SETCOLOR: return "FG_SETCOLOR";
  case FG_SETUSERDATA: return "FG_SETUSERDATA";
  case FG_GETUSERDATA: return "FG_GETUSERDATA";
  case FG_SETTEXT: return "FG_SETTEXT";
  case FG_SETNAME: return "FG_SETNAME";
  case FG_SETCONTEXTMENU: return "FG_SETCONTEXTMENU";
  case FG_GETCONTEXTMENU: return "FG_GETCONTEXTMENU";
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
  case FG_SETDIM: return "FG_SETDIM";
  case FG_SETVALUE: return "FG_SETVALUE";
  case FG_SETRANGE: return "FG_SETRANGE";
  case FG_SETUV: return "FG_SETUV";
  case FG_SETLETTERSPACING: return "FG_SETLETTERSPACING";
  case FG_SETLINEHEIGHT: return "FG_SETLINEHEIGHT";
  case FG_SETOUTLINE: return "FG_SETOUTLINE";
  case FG_SETSCALING: return "FG_SETSCALING";
  case FG_GETSCALING: return "FG_GETSCALING";
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
    return (*fn)(args..., "%*sFG_SETLETTERSPACING(%.2g)", spaces, "", msg->arg1.f);
  case FG_SETLINEHEIGHT:
    return (*fn)(args..., "%*sFG_SETLINEHEIGHT(%.2g)", spaces, "", msg->arg1.f);
  case FG_SETOUTLINE:
    return (*fn)(args..., "%*sFG_SETOUTLINE(%.2g)", spaces, "", msg->arg1.f);
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
