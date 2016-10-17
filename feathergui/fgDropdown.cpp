// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgDropdown.h"
#include "feathercpp.h"

void FG_FASTCALL fgDropdown_Init(fgDropdown* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgDropdown_Destroy, (fgMessage)&fgDropdown_Message);
}

void FG_FASTCALL fgDropdown_Destroy(fgDropdown* self)
{
  fgBox_Destroy(&self->box);
  fgControl_Destroy(&self->control);
}

size_t FG_FASTCALL fgDropdownBox_Message(fgBox* self, const FG_Msg* msg)
{
  fgDropdown* parent = (fgDropdown*)self->window->parent;
  switch(msg->type)
  {
  case FG_MOUSEDOWN:
    assert(parent != 0);
    if(parent->dropflag && !MsgHitCRect(msg, *self))
    {
      self->window->SetFlag(FGELEMENT_HIDDEN, true);
      if(fgroot_instance->topmost == *self)
        fgroot_instance->topmost = 0;
    }
    break;
  case FG_MOUSEUP:
    assert(parent != 0);
    {
      AbsRect cache;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(target)
      {
        if(parent->selected)
          fgStandardNuetralSetStyle(parent->selected, "selected", FGSETSTYLE_REMOVEFLAG);
        //parent->selected = target;
      }
      if(parent->dropflag)
      {
        self->window->SetFlag(FGELEMENT_HIDDEN, true);
        if(fgroot_instance->topmost == *self)
          fgroot_instance->topmost = 0;
      }
    }
    if(!parent->dropflag)
      parent->dropflag = 1;
    break;
  }
  return fgBox_Message(self, msg);
}

size_t FG_FASTCALL fgDropdown_Message(fgDropdown* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgBox_Init(&self->box, *self, 0, "fgDropdown$box", FGELEMENT_BACKGROUND | FGELEMENT_IGNORE | FGELEMENT_NOCLIP | FGELEMENT_EXPANDY | FGBOX_TILEY, &fgTransform { {0, 0, 0, 1, 0, 1, 0, 1 }, 0, {0, 0} });
    self->box->message = (fgMessage)&fgDropdownBox_Message;
    self->selected = 0;
    self->dropflag = 0;
    self->hover.color = 0x99999999;
    break;
  case FG_MOUSEDOWN:
    self->dropflag = 0;
    self->box->SetFlag(FGELEMENT_HIDDEN, false);
    fgroot_instance->topmost = self->box;
    return (*self->box->message)(self->box, msg);
  case FG_MOUSEUP:
    //if(!self->dropflag)
    //  return fgControl_Message(&self->box.window.control, msg);
    //{
    //  AbsRect cache;
    //  fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    //  if(target)
    //  {
    //    if(self->selected)
    //      fgStandardNuetralSetStyle(self->selected, "selected", FGSETSTYLE_REMOVEFLAG);
    //    self->selected = target;
    //    fgStandardNuetralSetStyle(target, "selected", FGSETSTYLE_SETFLAG);
    //  }
    //}
    //self->dropflag = 0;
    //if(fgroot_instance->topmost == *self)
    //  fgroot_instance->topmost = 0;
    break;
  case FG_MOUSEMOVE:
    //if(fgCaptureWindow == *self)
      break;
    //return FG_ACCEPT; // If the dropdown hasn't been dropped down, eat the mousemove events.
  case FG_DRAW:
    if(fgCaptureWindow != *self)
    {
      fgControl_Message(&self->control, msg); // Render things normally first

      if(self->selected) // Then, yank the selected item out of our box child and render it manually, if it exists
      {
        AbsRect* area = (AbsRect*)msg->other;
        AbsRect out;
        bool clipping = false;
        ResolveRect(self->selected, &out);
        fgStandardApplyClipping(self->selected, area, clipping);

        FABS dx = out.right - out.left;
        FABS dy = out.bottom - out.top;
        out.left = (((area->right - area->left) - dx) * 0.5f) + area->left;
        out.top = (((area->bottom - area->top) - dy) * 0.5f) + area->top;
        out.right = out.left + dx;
        out.bottom = out.top + dy;
        //_sendsubmsg<FG_DRAW, void*, size_t>(self->selected, 0, &out, msg->otheraux);
      }
      return FG_ACCEPT;
    }
    break;
  case FG_REMOVECHILD:
  case FG_ADDCHILD:
    if(((fgElement*)msg->other)->flags & FGELEMENT_BACKGROUND)
      break;
    return (*self->box->message)(self->box, msg);
  case FG_GETCOLOR:
    return self->hover.color;
  case FG_SETCOLOR:
    self->hover.color = (size_t)msg->otherint;
    return FG_ACCEPT;
  case FG_GETSELECTEDITEM:
    return (size_t)self->selected;
  case FG_GETCLASSNAME:
    return (size_t)"Dropdown";
  }
  fgControl_Message(&self->control, msg);
}

/*

if(fgCaptureWindow != *self)
{ // In this case, we only render background elements and the single selected item (if there is one). Because of this, there's no need to use the ordered rendering because by definition we aren't rendering anything that would be ordered.
char culled = (msg->subtype & 1);
fgElement* hold = culled ? (*self)->rootnoclip : (*self)->root;
AbsRect curarea;
bool clipping = false;
fgFlag flags = self->box->flags;
self->box->flags |= FGSCROLLBAR_HIDEV;
AbsRect* area = (AbsRect*)msg->other;

while(hold)
{
if((hold->flags & FGELEMENT_BACKGROUND))
fgStandardDrawElement(*self, hold, area, msg->otheraux, curarea, clipping);
if(hold == self->selected)
{
AbsRect out;
ResolveRectCache(hold, &out, area, (hold->flags & FGELEMENT_BACKGROUND) ? 0 : &self->box->padding);
fgStandardApplyClipping(hold, area, clipping);

FABS dx = out.right - out.left;
FABS dy = out.bottom - out.top;
out.left = (((area->right - area->left) - dx) * 0.5f) + area->left;
out.top = (((area->bottom - area->top) - dy) * 0.5f) + area->top;
out.right = out.left + dx;
out.bottom = out.top + dy;
_sendsubmsg<FG_DRAW, void*, size_t>(hold, 0, &out, msg->otheraux);
}
hold = culled ? hold->nextnoclip : hold->next;
}

if(clipping)
fgPopClipRect();

self->box->flags = flags;
return FG_ACCEPT;
}
else
{
FG_Msg m = *msg;
AbsRect r = *(AbsRect*)msg->other;
if(r.right - r.left < self->box.window.realsize.x)
r.right = r.left + self->box.window.realsize.x;
if(r.bottom - r.top < self->box.window.realsize.y)
r.bottom = r.top + self->box.window.realsize.y;
m.other = &r;
fgBox_Message(&self->box, &m);
return FG_ACCEPT;
}
case FG_INJECT:
if(fgCaptureWindow == *self)
{
AbsRect r;
ResolveRect(*self, &r);
if(r.right - r.left < self->box.window.realsize.x)
r.right = r.left + self->box.window.realsize.x;
if(r.bottom - r.top < self->box.window.realsize.y)
r.bottom = r.top + self->box.window.realsize.y;
fgElement* cur = self->box->lastinject;
while(cur) // Try to inject to any children we have
{
if(!(cur->flags&FGELEMENT_IGNORE) && _sendmsg<FG_INJECT, const void*, const void*>(cur, msg->other, &r)) // We have to check FGELEMENT_IGNORE because the noclip list may have render-only elements in it.
return FG_ACCEPT; // If the message is NOT rejected, return 1 immediately to indicate we accepted the message.
cur = cur->previnject; // Otherwise the child rejected the message.
}

return (*fgroot_instance->behaviorhook)(*self, (const FG_Msg*)msg->other);
}
*/