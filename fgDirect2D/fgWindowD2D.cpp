// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgWindowD2D.h"
#include "fgDirect2D.h"
#include "fgRoot.h"
#include "win32_includes.h"
#include <d2d1_1.h>
#include <Windowsx.h>
#include "fgRoundRect.h"
#include "fgCircle.h"
#include "fgTriangle.h"
#include "fgDebug.h"

fgWindowD2D* fgWindowD2D::windowlist = 0;

void fgWindowD2D_Init(fgWindowD2D* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  if(parent != &fgSingleton()->gui.element)
    fgElement_InternalSetup(self->window, parent, next, name, flags, transform, units, (fgDestroy)&fgWindow_Destroy, (fgMessage)&fgWindow_Message);
  else
  {
    self->list.next = self->list.prev = 0;
    bss::AltLLAdd<fgWindowD2D, fgWindowD2D::GETNODE>(self, fgWindowD2D::windowlist);
    fgElement_InternalSetup(self->window, parent, next, name, flags, transform, units, (fgDestroy)&fgWindowD2D_Destroy, (fgMessage)&fgWindowD2D_Message);
  }
}
void fgWindowD2D_Destroy(fgWindowD2D* self)
{
  fgContext_Destroy(&self->context);

  bss::AltLLRemove<fgWindowD2D, fgWindowD2D::GETNODE>(self, fgWindowD2D::windowlist);
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
      ResolveOuterRect(self->window, &out);
      fgScaleRectDPI(&out, 96, 96);
      unsigned long style = 0;
      if(self->window->flags&FGWINDOW_RESIZABLE)
        style |= WS_THICKFRAME;
      if(self->window->flags&FGWINDOW_MINIMIZABLE)
        style |= WS_MINIMIZEBOX;
      if(self->window->flags&FGWINDOW_MAXIMIZABLE)
        style |= WS_MAXIMIZEBOX;

      self->handle = self->context.WndCreate(out, style, style ? WS_EX_APPWINDOW : 0, self, L"fgWindowD2D", self->dpi);
      fgContext::ApplyWin32Size(self->window, self->handle, self->dpi);
#ifdef TEST_DPI
      self->dpi.x = self->dpi.y = TEST_DPI;
#endif
      self->window->SetDPI(self->dpi.x, self->dpi.y); // Send a message with the DPI change so fonts get resized correctly
      RECT rect; // Force the NCCALCSIZE message to fire via SetWindowPos
      GetWindowRect(self->handle, &rect); 
      SetWindowPos(self->handle, HWND_TOP, INT(rect.left), INT(rect.top), INT(rect.right - rect.left), INT(rect.bottom - rect.top), SWP_FRAMECHANGED); 
      ShowWindow(self->handle, SW_SHOW);
      UpdateWindow(self->handle);
      self->context.CreateResources(self->handle);
    }
    break;
  case FG_DRAW:
  {
    fgDrawAuxDataEx exdata;
    self->context.BeginDraw(self->handle, self->window, (AbsRect*)msg->p, exdata, &self->window->margin);
    FG_Msg m = *msg;
    m.p2 = &exdata;
    fgWindow_Message(&self->window, &m);
    self->context.EndDraw();
  }
  return FG_ACCEPT;
  case FG_MOVE:
    if(self->handle && msg->subtype != (uint16_t)~0)
    {
      AbsRect out;
      ResolveOuterRect(self->window, &out);
      fgScaleRectDPI(&out, self->dpi.x, self->dpi.y);
      RECT r = { out.left, out.top, out.right, out.bottom };
      //AdjustWindowRectEx(&r, GetWindowLong(self->handle, GWL_STYLE), FALSE, GetWindowLong(self->handle, GWL_EXSTYLE));
      SetWindowPos(self->handle, HWND_TOP, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOSENDCHANGING);
    }
    break;
  case FG_SETDPI:
    self->dpi.x = msg->f;
    self->dpi.y = msg->f2;
    break;
  case FG_GETDPI:
    if(!self->dpi.x)
      break;
    return (size_t)&self->dpi;
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
      return FG_ACCEPT;
    case FGWINDOW_MAXIMIZE:
      ShowWindow(self->handle, SW_MAXIMIZE);
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
  static const int BORDERWIDTH = 5;
  fgWindowD2D* self = reinterpret_cast<fgWindowD2D*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  if(self)
  {
    switch(message)
    {
    case WM_CREATE:
    {
      CREATESTRUCT* info = reinterpret_cast<CREATESTRUCT*>(lParam);
      SetWindowPos(hWnd,HWND_TOP,info->x, info->y, info->cx, info->cy, SWP_FRAMECHANGED);
    }
    break; 
    case WM_NCCALCSIZE:
      if(wParam == TRUE)
        return 0;
      break;
    case WM_NCHITTEST:
      if(self->window->flags&(FGWINDOW_RESIZABLE|FGWINDOW_MAXIMIZABLE))
      {
        RECT WindowRect;
        GetWindowRect(hWnd, &WindowRect);
        int x = GET_X_LPARAM(lParam) - WindowRect.left;
        int y = GET_Y_LPARAM(lParam) - WindowRect.top;
        FG_Msg msg = { FG_DEBUGMESSAGE };
        msg.x = GET_X_LPARAM(lParam);
        msg.y = GET_Y_LPARAM(lParam);

        if((self->window->flags&FGWINDOW_RESIZABLE) && !self->window.maximized)
        {
          if(x < BORDERWIDTH && y < BORDERWIDTH)
            return HTTOPLEFT;
          if(x > WindowRect.right - WindowRect.left - BORDERWIDTH && y < BORDERWIDTH)
            return HTTOPRIGHT;
          if(x > WindowRect.right - WindowRect.left - BORDERWIDTH && y > WindowRect.bottom - WindowRect.top - BORDERWIDTH)
            return HTBOTTOMRIGHT;
          if(x < BORDERWIDTH && y > WindowRect.bottom - WindowRect.top - BORDERWIDTH)
            return HTBOTTOMLEFT;
          if(x < BORDERWIDTH)
            return HTLEFT;
          if(y < BORDERWIDTH)
            return HTTOP;
          if(x > WindowRect.right - WindowRect.left - BORDERWIDTH)
            return HTRIGHT;
          if(y > WindowRect.bottom - WindowRect.top - BORDERWIDTH)
            return HTBOTTOM;
        }
        if(y < self->window->padding.top + self->window->margin.top && reinterpret_cast<fgElement*>(fgStandardInject(self->window, &msg, 0)) == self->window)
          return HTCAPTION;
        return HTCLIENT;
      }
      break;
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
      self->window->SetDPI(LOWORD(wParam), HIWORD(wParam));
      return 0;
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
      SetCapture(hWnd);
      break;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
      ReleaseCapture();
      break;
    case WM_MOVE:
      if(fgDirect2D::instance->debugtarget != 0 && hWnd == fgDirect2D::instance->debugtarget->handle)
      {
        AbsRect area;
        ResolveRect(fgDirect2D::instance->debugtarget->window, &area);
        SetWindowPos(fgDirect2D::instance->debughwnd, hWnd, area.right, area.top, 300, area.bottom - area.top, SWP_NOSENDCHANGING | SWP_NOACTIVATE);
      }
    case WM_SIZE:
      switch(wParam)
      {
      case SIZE_MAXIMIZED:
        self->window.maximized = 1;
        break;
      case SIZE_MINIMIZED:
      case SIZE_RESTORED:
        self->window.maximized = 0;
        break;
      }
      fgContext::ApplyWin32Size(self->window, self->handle, self->dpi);
      break;
    case WM_ACTIVATE:
      if(fgDirect2D::instance->debugtarget != 0 && hWnd == fgDirect2D::instance->debugtarget->handle)
        SetWindowPos(fgDirect2D::instance->debughwnd, hWnd, 0, 0, 0, 0, SWP_NOSENDCHANGING | SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE);
      break;
    }
    return self->context.WndProc(hWnd, message, wParam, lParam, self->window);
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}