// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgWindowD2D.h"
#include "fgRoot.h"
#include "bss-util/bss_win32_includes.h"
#include <d2d1_1.h>
#include "fgRoundRect.h"
#include "fgCircle.h"
#include "fgTriangle.h"

fgWindowD2D* fgWindowD2D::windowlist = 0;

void fgWindowD2D_Init(fgWindowD2D* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  self->list.next = self->list.prev = 0;
  bss_util::AltLLAdd<fgWindowD2D, fgWindowD2D::GETNODE>(self, fgWindowD2D::windowlist);
  fgElement_InternalSetup(self->window, parent, next, name, flags, transform, units, (fgDestroy)&fgWindowD2D_Destroy, (fgMessage)&fgWindowD2D_Message);
}
void fgWindowD2D_Destroy(fgWindowD2D* self)
{
  fgContext_Destroy(&self->context);

  bss_util::AltLLRemove<fgWindowD2D, fgWindowD2D::GETNODE>(self, fgWindowD2D::windowlist);
  if(!fgWindowD2D::windowlist)
    PostQuitMessage(0);

  if(self->handle)
  {
    SetWindowLongPtrW(self->handle, GWLP_USERDATA, 0); // Prevent the WM_DESTROY message from propogating
    DestroyWindow(self->handle);
  }

  self->window->message = (fgMessage)fgWindow_Message;
  fgWindow_Destroy(&self->window);
}
size_t fgWindowD2D_Message(fgWindowD2D* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    self->handle = 0;
    self->dpi.x = 0;
    self->dpi.y = 0;
    fgContext_Construct(&self->context);
    return fgWindow_Message(&self->window, msg);
  case FG_PARENTCHANGE:
    if(msg->e != 0 && self->window->parent == &fgSingleton()->gui.element)
      return fgSendMsg<FG_SETPARENT, const void*>(self->window, fgSingleton()->monitors);
    if(!self->window->parent && self->handle != 0)
    {
      self->context.DiscardResources();
      SetWindowLongPtrW(self->handle, GWLP_USERDATA, 0); // Prevent the WM_DESTROY message from propogating
      DestroyWindow(self->handle);
    }
    else if(self->window->parent != 0 && !self->handle)
    {
      AbsRect out;
      ResolveRect(self->window, &out);
      fgScaleRectDPI(&out, 96, 96);
      self->handle = self->context.WndCreate(out, 0, self, L"fgWindowD2D", self->dpi);
      ShowWindow(self->handle, SW_SHOW);
      UpdateWindow(self->handle);
      //SetWindowPos(self->handle, HWND_TOP, INT(wleft), INT(wtop), INT(rwidth), INT(rheight), SWP_NOCOPYBITS | SWP_NOMOVE | SWP_NOACTIVATE);
      SetCursor(LoadCursor(NULL, IDC_ARROW));
      self->context.CreateResources(self->handle);
    }
    break;
  case FG_DRAW:
  {
    fgDrawAuxDataEx exdata;
    self->context.BeginDraw(self->handle, self->window, (AbsRect*)msg->p, exdata);
    FG_Msg m = *msg;
    m.p2 = &exdata;
    fgWindow_Message(&self->window, &m);
    /*fgElement* topmost = fgSingleton()->topmost;
    if(topmost && GetElementWindow(topmost) == self) // Draw topmost before the drag object
    {
    AbsRect out;
    ResolveRect(topmost, &out);
    topmost->Draw(&out, &exdata.data);
    }*/
    self->context.EndDraw();
  }
    return FG_ACCEPT;
  case FG_MOVE:
    if(self->handle && msg->subtype != (uint16_t)~0)
    {
      AbsRect out;
      ResolveRect(self->window, &out);
      fgScaleRectDPI(&out, 96, 96);
      SetWindowPos(self->handle, HWND_TOP, out.left, out.top, out.right - out.left, out.bottom - out.top, SWP_NOSENDCHANGING);
    }
    break;
  case FG_SETDPI:
    self->dpi.x = msg->i;
    self->dpi.y = msg->i2;
    break;
  case FG_ACTION:
    switch(msg->i)
    {
    case FGWINDOW_MINIMIZE:
      ShowWindow(self->handle, SW_MINIMIZE);
      return FG_ACCEPT;
    case FGWINDOW_UNMINIMIZE:
      ShowWindow(self->handle, SW_RESTORE);
      return FG_ACCEPT;
    case FGWINDOW_RESTORE:
      ShowWindow(self->handle, SW_RESTORE);
      self->window.maximized = 0;
    case FGWINDOW_MAXIMIZE:
      if(msg->i == FGWINDOW_MAXIMIZE)
      {
        ShowWindow(self->handle, SW_MAXIMIZE);
        self->window.maximized = 1;
      }
      RECT r;
      if(SUCCEEDED(GetClientRect(self->handle, &r)))
      {
        CRect area = { r.left, 0, r.top, 0, r.right, 0, r.bottom };
        fgSendSubMsg<FG_SETAREA, void*>(self->window, (uint16_t)~0, &area);
      }
      return FG_ACCEPT;
    case FGWINDOW_CLOSE:
      VirtualFreeChild(self->window);
      return FG_ACCEPT;
    }
  }
  return fgWindow_Message(&self->window, msg);
}

longptr_t __stdcall fgWindowD2D::WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  fgWindowD2D* self = reinterpret_cast<fgWindowD2D*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  if(self)
  {
    switch(message)
    {
    case WM_PAINT:
    {
      AbsRect area;
      ResolveRect(self->window, &area);
      self->window->Draw(&area, 0);
      return 0;
    }
    case WM_DESTROY:
      VirtualFreeChild(self->window);
      return 1;
    case 0x02E0: //WM_DPICHANGED
      //self->window->SetDPI(LOWORD(wParam), HIWORD(wParam));
      return 0;
    //case WM_WINDOWPOSCHANGING:
    //{
    //  WINDOWPOS* pos = (WINDOWPOS*)lParam;
    //  CRect area = { pos->x, 0, pos->y, 0, pos->x + pos->cx, 0, pos->y + pos->cy, 0 };
    //  fgSendSubMsg<FG_SETAREA, void*>(self->window, 1, &area);
    //  break;
    //}
    }
    return self->context.WndProc(hWnd, message, wParam, lParam, self->window);
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}