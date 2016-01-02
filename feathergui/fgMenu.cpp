// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

/*#include "fgMenu.h"
#include <time.h>

static const double DROPDOWN_TIME=0.4; // 400 milliseconds for dropdown


void FG_FASTCALL fgMenu_Init(fgMenu* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags)
{
  FG_Msg msg = {0};
  assert(self!=0);
  fgWindow_Init(&self->window,parent,element,id,flags);
  self->window.window.message=&fgMenu_Message;
  self->window.window.element.destroy=&fgMenu_Destroy;
  self->dropdown=fgRoot_AllocAction(&fgMenu_DoDropdown,0,0);
  fgGrid_Init(&self->grid,(fgWindow*)self,element,id,flags);
  //fgWindow_Init(&self->overlay,(fgWindow*)self,0,0,0);
  self->expanded=0;
  msg.type=FG_SETFLAG;
  msg.otherint=FGWIN_NOCLIP;
  msg.otherintaux=1;
  fgMenu_Message(self,&msg); // All menus are nonclipping by necessity unless it morphs into a menubar
}
void FG_FASTCALL fgMenu_Destroy(fgMenu* self)
{
  fgRoot_DeallocAction(fgSingleton(),self->dropdown);
  //fgWindow_Destroy(&self->overlay);
  fgWindow_Destroy((fgWindow*)&self->grid);
  fgWindow_Destroy(&self->window);
}

char FG_FASTCALL fgMenu_Message(fgMenu* self, const FG_Msg* msg)
{
  long long curtime;
  struct FG_MENUITEM* item;
  AbsVec loc={0};
  assert(self!=0 && msg!=0);

  switch(msg->type)
  {
  case FG_MOUSEMOVE:
    if(self->expanded!=0) break;
    curtime = clock()/(CLOCKS_PER_SEC/1000);
    item=(struct FG_MENUITEM*)fgGrid_HitElement(&self->grid,msg->x,msg->y);
    if(!item || item->submenu==(void*)-1) // if it didn't hit anything, or it hit a divider, hide highlight
      (*self->highlight->message)(self->highlight,FG_RSHOW,0,0);
    else if(self->dropdown->arg!=item) // Otherwise reposition highlight properly, if position actually changed
    {
      self->dropdown->arg=item;
      self->dropdown->time=fgSingleton()->time+DROPDOWN_TIME;
      fgRoot_ModifyAction(fgSingleton(),self->dropdown);
      loc.y=item->render.element.element.area.top.abs; // This works because the parent's the same, so both locations are relative to the same thing.
      MoveCRect(loc,&self->highlight->element.element.area);
      (*self->highlight->message)(self->highlight,FG_MOVE,0,0);
      (*self->highlight->message)(self->highlight,FG_RSHOW,(void*)1,0);
    }
    break;
  case FG_MOUSEOFF:
    if(self->expanded!=0) break;
    (*self->highlight->message)(self->highlight,FG_RSHOW,0,0);
    break;
  case FG_ADDITEM:
    return 0;
  case FG_GOTFOCUS:
    fgWindow_IntMessage((fgWindow*)self,FG_SHOW,1);
    break;
  case FG_LOSTFOCUS:
    fgWindow_IntMessage((fgWindow*)self,FG_SHOW,0);
    break;
  case FG_GETCLASSNAME:
    return "fgMenu";
  }
  return fgWindow_Message((fgWindow*)self,msg);
}

FG_EXTERN char FG_FASTCALL fgMenu_DoDropdown(fgMenu* self)
{
  fgWindow_BasicMessage((fgWindow*)self,FG_GOTFOCUS); // By transfering focus to our child menu it will show itself.
  return 0;
}
//*/