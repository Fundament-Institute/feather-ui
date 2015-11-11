// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWindow.h"
#include "fgRoot.h"

fgWindow* fgFocusedWindow = 0;
fgWindow* fgLastHover = 0;

void FG_FASTCALL fgWindow_Init(fgWindow* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{ 
  assert(self!=0);
  memset(self,0,sizeof(fgWindow)); 
  self->element.destroy=&fgWindow_Destroy; 
  self->element.message=&fgWindow_Message;
  self->element.flags=flags;
  if(element) self->element.element=*element;
}
void FG_FASTCALL fgWindow_Destroy(fgWindow* self)
{
  assert(self!=0);
  fgChild_VoidMessage((fgChild*)self,FG_SETPARENT,0);

  if(self->name) free(self->name);
  fgChild_Destroy(&self->element);
}

size_t FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg)
{
  int otherint = msg->otherint;
  assert(self!=0);
  assert(msg!=0);
  
  switch(msg->type)
  {
  case FG_MOUSEUP: // This is done to take care of the "click button, hold mouse, move over another button, release" problem.
  case FG_MOUSEMOVE:
    if(MsgHitCRect(msg,&self->element))
    {
      if(fgLastHover!=self)
      {
        if(fgLastHover!=0)
          fgChild_VoidMessage((fgChild*)fgLastHover,FG_MOUSEOFF,0);
        fgLastHover=self;
        fgChild_VoidMessage((fgChild*)self,FG_MOUSEON,0);
      }
      return 0;
    }
    return 1;
  case FG_MOUSEDOWN:
    if(fgFocusedWindow != self)
      fgChild_VoidMessage((fgChild*)self,FG_GOTFOCUS,0);
    if(msg->button == 2 && self->contextmenu!=0)
      fgChild_VoidMessage((fgChild*)self->contextmenu,FG_GOTFOCUS,0);
    return !MsgHitCRect(msg,&self->element);
  case FG_GOTFOCUS:
    if(fgFocusedWindow) // We do this here so you can disable getting focus by blocking this message without messing things up
      fgChild_VoidMessage((fgChild*)fgFocusedWindow,FG_LOSTFOCUS,0);
    fgFocusedWindow=self;
    break;
  case FG_LOSTFOCUS:
    assert(fgFocusedWindow==self);
    fgFocusedWindow=0;
    break;
  case FG_CLONE:
  {
    fgWindow* hold = msg->other;
    if(!hold)
      hold = malloc(sizeof(fgWindow));
    hold->id = self->id;
    hold->contextmenu = hold->contextmenu;
    fgChild_VoidMessage((fgChild*)hold, FG_SETNAME, self->name);
    //fgChild_VoidMessage(hold, FG_SETSKIN, self->skin);

    FG_Msg m = *msg;
    m.other = hold;
    return fgChild_Message((fgChild*)self, msg);
  }
    break;
  case FG_GETCLASSNAME:
    (*(const char**)msg->other) = "fgWindow";
    return 0;
  case FG_SETNAME:
    if(self->name) free(self->name);
    self->name = 0;
    if(msg->other)
    {
      size_t len = strlen(msg->other)+1;
      self->name = malloc(len);
      memcpy(self->name, msg->other, len);
    }
    return 0;
  case FG_GETNAME:
    (*(const char**)msg->other) = self->name;
    return 0;
  }
  return fgChild_Message((fgChild*)self, msg);
}

size_t FG_FASTCALL fgWindow_HoverProcess(fgWindow* self, const FG_Msg* msg)
{
  char t;
  assert(self!=0 && msg!=0);
  switch(msg->type)
  {
  case FG_MOUSEMOVE:
    if(msg->allbtn&FG_MOUSELBUTTON) //If the button is active, eat the mousemove
      return 0;
    break;
  case FG_MOUSEON:
    fgChild_VoidMessage((fgChild*)self,FG_HOVER,0);
    break;
  case FG_MOUSEUP:
    t=MsgHitCRect(msg,&self->element);
    if(fgFocusedWindow==self && t) // Does this happen while we have focus AND the event is inside our control?
      fgChild_VoidMessage((fgChild*)self, FG_ACTION, 0); // Fire off a message
    fgChild_VoidMessage((fgChild*)self,t?FG_HOVER:FG_NUETRAL, 0);
    break;
  case FG_MOUSEOFF: // FG_MOUSEUP falls through
    fgChild_VoidMessage((fgChild*)self,FG_NUETRAL, 0);
    break;
  case FG_MOUSEDOWN:
    if(msg->button==FG_MOUSELBUTTON && MsgHitCRect(msg,&self->element)) {
      fgChild_VoidMessage((fgChild*)self,FG_ACTIVE, 0);
    }
    break;
  }
  return fgWindow_Message(self,msg);
}

//void FG_FASTCALL DoSkinCheck(fgWindow* self, fgStatic** skins, const FG_Msg* msg)
//{
//  if(skins[msg->otheraux]!=0)
//    fgChild_VoidMessage(self,FG_REMOVESTATIC,skins[msg->otheraux]);
//  skins[msg->otheraux]=(fgStatic*)msg->other;
//}


      /*if(active && flip!=0) // Enable clipping, remove from nonclipping array
      {
        for(i=0; i<fgNonClipping.l; ++i)
          if(fgNonClipping.p[i]==self)
          {
            Remove_VectWindow(&fgNonClipping,i);
            break;
          }
      }
      else if(!active && flip!=0) // Disable clipping, add to nonclipping array
        Add_VectWindow(&fgNonClipping,self); //DON'T sort this; if any parent window changed its zorder everything would explode
      break;*/
    /*case FGWIN_HIDDEN:
      if(active && flip!=0) // Hide window
      {
        hold=self->element.parent;
        fgChild_VoidMessage(self,FG_SETPARENT,0);
        self->element.next=hold; // Store parent in next variable so as not to confuse SetParent or the destructor
      }
      else if(!active && flip!=0) // Show window
      {
        assert(self->element.next!=0); // next will either have the stored secret parent in it, or be NULL. You can't "Show" something that has no parent!
        fgChild_VoidMessage(self,FG_SETPARENT,self->element.next);
      }
      break;*/