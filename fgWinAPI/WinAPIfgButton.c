// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWinAPI.h"
#include "win32_includes.h"
#include <CommCtrl.h>

extern WinAPIfgRoot* _fgroot;
extern void FG_FASTCALL WinAPIfgImage_Destroy(WinAPIfgImage*);
extern void FG_FASTCALL WinAPIfgText_Destroy(WinAPIfgText*);

char FG_FASTCALL WinAPIfgButton_Message(WinAPIfgButton* self, const FG_Msg* msg)
{
  char r =  fgWindow_Message(self,msg);
  WinAPIMessage* wmsg = (WinAPIMessage*)msg->other;
  
  switch(msg->type)
  {
  case FG_MOVE:
    WinAPI_FG_MOVE(self);
    break;
  case WINAPIFGTOP_MSGFILTER:
    switch(wmsg->message)
    {
    case WM_COMMAND:
      switch(HIWORD(wmsg->wParam))
      {
      case BN_CLICKED:
        fgWindow_BasicMessage((fgWindow*)self,FG_CLICKED);
        break;
      }
      break;
    }
    break;
  }

  return r;
}

fgWindow* FG_FASTCALL fgButton_Create(fgStatic* item, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags)
{
  AbsRect absp;
  AbsRect absr;
  RECT rsize = { CW_USEDEFAULT,CW_USEDEFAULT,CW_USEDEFAULT,CW_USEDEFAULT };
  DWORD style=WS_CHILD|WS_VISIBLE;
  wchar_t* txt=0;
  WinAPIfgButton* r = bssmalloc<WinAPIfgButton>(1);
  //r->window.message=&fgdebug_Message_fgButton;
  fgWindow_Init((fgWindow*)r,parent,element,id,flags);
  r->wn.window.message=&WinAPIfgButton_Message;
  r->wn.window.element.destroy=&WinAPIfgWindow_Destroy;
  if(item->element.destroy==&WinAPIfgText_Destroy) { // If true its text-based
    WinAPIutf8to16(&txt,((WinAPIfgText*)item)->text);
    style|=BS_TEXT|BS_VCENTER|BS_CENTER;
  } else if(item->element.destroy==&WinAPIfgImage_Destroy) {
    style|=BS_BITMAP;
  }
  assert(parent!=0);

  ResolveRect((fgChild*)parent,&absp); //Get size
  absp.right-=absp.left;
  absp.bottom-=absp.top;
  absp.left=0;
  absp.top=0;
  ResolveRectCache(&absr,(fgElement*)r,&absp);
  ToLongAbsRect(&absr,&rsize.left);
  rsize.right-=rsize.left;
  rsize.bottom-=rsize.top;
  if(!rsize.right) rsize.right=CW_USEDEFAULT;
  if(!rsize.bottom) rsize.bottom=CW_USEDEFAULT;

  r->wn.handle=CreateWindowExW(0,WC_BUTTON,txt,style,rsize.left,rsize.top,rsize.right,rsize.bottom,((WinAPIfgWindow*)parent)->handle,0,_fgroot->instance,0);
  SetWindowLongPtr(r->wn.handle,GWLP_USERDATA,(LONG_PTR)r); // Set userdata
  r->wn.DefWndProc=(WNDPROC)GetWindowLongPtr(r->wn.handle,GWLP_WNDPROC); // Grab old WndProc
  SetWindowLongPtr(r->wn.handle,GWLP_WNDPROC,(WNDPROC)&fgWindowWndProc); // Subclass control
  
  if(txt) free(txt);
  if(item) VirtualFreeChild((fgChild*)item); // We eat this
  return (fgWindow*)r;
}