// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWinAPI.h"
#include "win32_includes.h"
#include <dwmapi.h>

//typedef HRESULT (STDAPICALLTYPE *DWMCOMPENABLE)(BOOL*);
extern WinAPIfgRoot* _fgroot;
extern HBRUSH hbrBlack;
extern HRESULT (__stdcall *dwmextend)(HWND, const MARGINS*);   

HWND TopWndCreate(HINSTANCE instance, fgWindow* wn, const wchar_t* name, long style, INT x, INT y, INT cx, INT cy)
{
	HWND hWnd;
  HINSTANCE hInstance = !instance?GetModuleHandleW(0):instance;

  //if(icon)
	//  wcex.hIcon = (HICON)LoadImageW(hInstance, icon, IMAGE_ICON, 0,0, LR_LOADFROMFILE); 
	
  //if(composite!=0) 
  //  hWnd = CreateWindowExW((composite==2||composite==4)?WS_EX_TRANSPARENT|WS_EX_COMPOSITED|WS_EX_LAYERED:WS_EX_COMPOSITED, 
  //                          L"FeatherWindow", L"", style, INT(wleft), INT(wtop), INT(rwidth), INT(rheight), NULL, NULL, hInstance, NULL);
  //else 
  hWnd = CreateWindowExW(0,L"FeatherWindow", name, style, x, y, cx ,cy, NULL, NULL, hInstance, NULL);
  SetWindowLongPtr(hWnd,GWLP_USERDATA,(LONG_PTR)wn); // Cast to LONG_PTR so it works in x64, not just x86
  
	UpdateWindow(hWnd);
  //SetWindowPos(hWnd,composite==2||composite==3?HWND_TOPMOST:HWND_TOP, INT(wleft), INT(wtop), INT(rwidth), INT(rheight),
  //              SWP_NOCOPYBITS|SWP_NOMOVE|SWP_NOACTIVATE);
  
 // if(composite==2||composite==4) SetLayeredWindowAttributes(hWnd, 0, 0xFF, LWA_ALPHA);
	//else 
 // {
 //   SetActiveWindow(hWnd);
	//  SetForegroundWindow(hWnd);
 // }
  SetCursor(LoadCursor(NULL, IDC_ARROW));

  return hWnd;
}

char WinAPIfgTop_Message(WinAPIfgTop* self, const FG_Msg* msg)
{
  CRect narea;
  MARGINS m;
  RECT rc;
  WinAPIMessage* wmsg = (WinAPIMessage*)msg->p;
	HDC         hDC;

  switch(msg->type)
  {
  case FG_REMOVESTATIC:
  case FG_REMOVECHILD:
    if(((fgChild*)msg->p)->parent==(fgChild*)self)
      return fgWindow_Message(&self->wn.window,msg);
    return fgWindow_Message(&self->region,msg);
  case FG_ADDSTATIC:
  case FG_ADDCHILD:
    if(msg->u2==1)
      return fgWindow_Message(&self->wn.window,msg);
    return fgWindow_Message(&self->region,msg);
  case WINAPIFGTOP_WINDOWMOVE:
    narea.left.abs=((WINDOWPOS*)msg->p)->x;
    narea.left.rel=0;
    narea.top.abs=((WINDOWPOS*)msg->p)->y;
    narea.top.rel=0;
    narea.right.abs=((WINDOWPOS*)msg->p)->x+((WINDOWPOS*)msg->p)->cx;
    narea.right.rel=0;
    narea.bottom.abs=((WINDOWPOS*)msg->p)->y+((WINDOWPOS*)msg->p)->cy;
    narea.bottom.rel=0;
    fgWindow_SetArea(&self->wn.window,&narea);
    return 0;
  case FGTOPWINDOW_SETMARGIN:
    ToIntAbsRect((AbsRect*)msg->p,self->margin);
    m.cxLeftWidth=self->margin[0];
    m.cyTopHeight=self->margin[1];
    m.cxRightWidth=self->margin[2];
    m.cyBottomHeight=self->margin[3];
    (*dwmextend)(self->wn.handle,&m);

    if(self->margin[0]!=0 || self->margin[1]!=0 || self->margin[2]!=0 || self->margin[3]!=0)
      SetWindowLongPtr(self->wn.handle,GWL_EXSTYLE,GetWindowLongPtr(self->wn.handle,GWL_EXSTYLE)|WS_EX_COMPOSITED);
    else
      SetWindowLongPtr(self->wn.handle,GWL_EXSTYLE,GetWindowLongPtr(self->wn.handle,GWL_EXSTYLE)&(~WS_EX_COMPOSITED));

    InvalidateRect(self->wn.handle,0,TRUE);
	  UpdateWindow(self->wn.handle);
    return 0;
  case WINAPIFGTOP_MSGFILTER: // A top level window automatically paints a black background underneath the extended frame margins.
    switch(wmsg->message)
    {
    case WM_ERASEBKGND: 
      //if(GetWindowLongPtr(self->wn.handle,GWL_EXSTYLE)&WS_EX_COMPOSITED)
      if(self->margin[0]!=0 || self->margin[1]!=0 || self->margin[2]!=0 || self->margin[3]!=0)
      {
		    hDC = (HDC)wmsg->wParam; 
        GetClientRect(self->wn.handle,&rc);
        rc.right-=rc.left;
        rc.bottom-=rc.top;
        rc.left=0;
        rc.top=0;
        FillRect(hDC, &rc, hbrBlack); 
        rc.left=self->margin[0];
        rc.top=self->margin[1];
        rc.right-=self->margin[2];
        rc.bottom-=self->margin[3];
        FillRect(hDC, &rc, (HBRUSH)GetClassLong(self->wn.handle,GCL_HBRBACKGROUND)); 
        return -1;
      }
      break;
    }
    break;
  }

  return fgWindow_Message((fgWindow*)self,msg);
}
void WinAPIfgTop_Destroy(WinAPIfgTop* self)
{
  fgWindow_Destroy(&self->region); // Removes the region from the window child list so we don't try to free something on the stack.
  WinAPIfgWindow_Destroy(self);
}

fgWindow* fgTopWindow_Create(const char* caption, const fgElement* element, FG_UINT id, fgFlag flags)
{
  AbsRect absr;
  RECT rsize = { CW_USEDEFAULT,CW_USEDEFAULT,CW_USEDEFAULT,CW_USEDEFAULT };
  wchar_t* buf=0;
  long style=WS_OVERLAPPED | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE;
  WinAPIfgTop* r = bssmalloc<WinAPIfgTop>(1);
  fgWindow_Init(&r->wn.window,(fgWindow*)_fgroot,element,id,flags);
  fgWindow_Init(&r->region,(fgWindow*)r,element,0,0);
  if(flags&FGTOPWINDOW_MINIMIZE) style|=WS_MINIMIZEBOX;
  if(flags&FGTOPWINDOW_RESTORE) style|=WS_MAXIMIZEBOX;
  if(!(flags&FGTOPWINDOW_NOTITLEBAR)) style|=WS_CAPTION|WS_SYSMENU;
  if(flags&FGTOPWINDOW_RESIZABLE) style|=WS_THICKFRAME;
  else
  {
    if(element!=0) style|=WS_POPUP;
    if(!(flags&FGTOPWINDOW_NOTITLEBAR)) style|=WS_BORDER;
  }

  if(element!=0) 
  {
    ResolveRect((fgChild*)r,&absr); //Get size
    ToLongAbsRect(&absr,&rsize.left);

	  AdjustWindowRect(&rsize, style, FALSE);
	  rsize.right = (rsize.right - rsize.left);
	  rsize.bottom = (rsize.bottom - rsize.top);
  }

  r->wn.window.message=&WinAPIfgTop_Message;
  r->wn.window.element.destroy=&WinAPIfgTop_Destroy;
  fgWindow_VoidMessage((fgWindow*)r,FG_SETPARENT,_fgroot);
  
  if(caption!=0)
    WinAPIutf8to16(&buf,caption);

  r->wn.DefWndProc=&DefWindowProcW;
  r->wn.handle=TopWndCreate(_fgroot->instance,&r->wn.window, buf, style, rsize.left, rsize.top, rsize.right, rsize.bottom);
  if(buf) free(buf);
  return (fgWindow*)r;
}