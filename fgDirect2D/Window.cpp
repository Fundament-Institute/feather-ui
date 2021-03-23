// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "Window.h"
#include "BackendD2D.h"
#include <dwmapi.h>
#include <Windowsx.h>
#include <wincodec.h>
#include <dwrite_1.h>
#include "RoundRect.h"
#include "Arc.h"
#include "Circle.h"
#include "Triangle.h"
#include "Modulation.h"
#include "util.h"

namespace D2D {
  __KHASH_IMPL(wic, , const Asset*, ID2D1Bitmap*, 1, kh_ptr_hash_func, kh_int_hash_equal);
}

using namespace D2D;
static float PI = 3.14159265359f;

EXTERN_C IMAGE_DOS_HEADER __ImageBase; // Accesses the HINSTANCE for this module

FG_FORCEINLINE FG_Vec GetPoints(longptr_t lParam)
{
  return FG_Vec{ static_cast<float> GET_X_LPARAM(lParam), static_cast<float> GET_Y_LPARAM(lParam) };
}
FG_FORCEINLINE FG_Vec AdjustPoints(longptr_t lParam, HWND__* hWnd)
{
  RECT rect;
  GetWindowRect(hWnd, &rect);
  auto v = GetPoints(lParam);
  v.x += rect.left;

  return FG_Vec{ v.x + static_cast<float>(rect.left), v.y + static_cast<float>(rect.top) };
}

Window::Window(Backend* _backend, FG_MsgReceiver* _element, FG_Vec* pos, FG_Vec* dim, uint64_t flags, const char* caption,
               void* context)
{
  target                = 0;
  context               = 0;
  color                 = 0;
  edgecolor             = 0;
  roundrect             = 0;
  triangle              = 0;
  arc                   = 0;
  circle                = 0;
  scaling               = 0;
  element               = _element;
  backend               = _backend;
  margin                = { 0 };
  assets                = kh_init_wic();
  inside                = false;
  unsigned long style   = (flags & FG_Window_NOBORDER) ? WS_POPUP : 0;
  unsigned long exstyle = (flags & FG_Window_NOCAPTION) ? WS_EX_TOOLWINDOW : WS_EX_APPWINDOW;
  if(flags & FG_Window_MINIMIZABLE)
    style |= WS_MINIMIZEBOX | WS_SYSMENU;
  if(flags & FG_Window_RESIZABLE)
    style |= WS_THICKFRAME;
  if(flags & FG_Window_MAXIMIZABLE)
    style |= WS_MAXIMIZEBOX | WS_SYSMENU;

  hWnd   = Window::WndCreate(pos, dim, style, exstyle, this, Backend::WindowClass, caption, dpi);
  target = reinterpret_cast<ID2D1HwndRenderTarget*>(context);

  RECT rect;
  GetWindowRect(hWnd, &rect);
  ShowWindow(hWnd, SW_SHOW);
  UpdateWindow(hWnd);
}
Window::~Window() { DiscardResources(); }

void Window::WndRegister(WNDPROC f, const wchar_t* name)
{
  // Register window class
  WNDCLASSEX wcex    = { sizeof(WNDCLASSEX) };
  wcex.style         = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
  wcex.lpfnWndProc   = f;
  wcex.cbClsExtra    = 0;
  wcex.cbWndExtra    = sizeof(LONG_PTR);
  wcex.hInstance     = ((HINSTANCE)&__ImageBase);
  wcex.hbrBackground = NULL;
  wcex.lpszMenuName  = NULL;
  wcex.hCursor       = NULL;
  wcex.lpszClassName = name;

  RegisterClassExW(&wcex);
}

longptr_t __stdcall Window::WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  static const int BORDERWIDTH              = 5;
  static tagTRACKMOUSEEVENT _trackingstruct = { sizeof(tagTRACKMOUSEEVENT), TME_LEAVE, 0, 0 };
  Window* self                              = reinterpret_cast<Window*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));

  switch(message)
  {
  case WM_CREATE:
  {
    CREATESTRUCT* info = reinterpret_cast<CREATESTRUCT*>(lParam);
    SetWindowPos(hWnd, HWND_TOP, info->x, info->y, info->cx, info->cy, SWP_FRAMECHANGED);
    SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(info->lpCreateParams));
  }
  break;
  case WM_NCCALCSIZE:
    if(wParam == TRUE && (GetWindowLongPtrW(hWnd, GWL_STYLE) & WS_POPUP) != 0)
      return 0;
    break;
  case WM_LBUTTONDOWN:
  case WM_RBUTTONDOWN:
  case WM_MBUTTONDOWN: SetCapture(hWnd); break;
  case WM_LBUTTONUP:
  case WM_RBUTTONUP:
  case WM_MBUTTONUP: ReleaseCapture(); break;
  }

  if(self)
  {
    switch(message)
    {
    case WM_SIZE:
      if(self->target)
        self->target->Resize(D2D1::SizeU(LOWORD(lParam), HIWORD(lParam)));
    case WM_MOVE:
    {
      FG_Msg msg               = { FG_Kind_SETWINDOWFLAGS };
      msg.setWindowFlags.flags = self->backend->Behavior(self, FG_Msg{ FG_Kind_GETWINDOWFLAGS }).getWindowFlags;

      switch(wParam)
      {
      case SIZE_MAXIMIZED:
        msg.setWindowFlags.flags |= FG_Window_MAXIMIZED;
        self->backend->Behavior(self, msg);
        break;
      case SIZE_MINIMIZED: msg.setWindowFlags.flags &= ~FG_Window_MINIMIZED; self->backend->Behavior(self, msg);
      case SIZE_RESTORED:
        msg.setWindowFlags.flags &= ~(FG_Window_MAXIMIZED & FG_Window_MINIMIZED);
        self->backend->Behavior(self, msg);
        break;
      }
      self->ApplyWin32Size();
      break;
    }
    case WM_NCHITTEST: // TODO: Make this a custom message passed to the window behavior
    {
      auto flags = self->backend->Behavior(self, FG_Msg{ FG_Kind_GETWINDOWFLAGS }).getWindowFlags;
      if(flags & (FG_Window_RESIZABLE | FG_Window_MAXIMIZABLE))
      {
        RECT WindowRect;
        GetWindowRect(hWnd, &WindowRect);
        int x = GET_X_LPARAM(lParam) - WindowRect.left;
        int y = GET_Y_LPARAM(lParam) - WindowRect.top;

        if((flags & FG_Window_RESIZABLE) && !(flags & FG_Window_MAXIMIZED))
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
        // if(y < (self->window->padding.top * (fgSingleton()->dpi.y / Backend::BASE_DPI)) + self->window->margin.top)
        //  return HTCAPTION;
        return HTCLIENT;
      }
      break;
    }
    case WM_DESTROY:
    {
      FG_Msg msg               = { FG_Kind_SETWINDOWFLAGS };
      msg.setWindowFlags.flags = self->backend->Behavior(self, FG_Msg{ FG_Kind_GETWINDOWFLAGS }).getWindowFlags |
                                 FG_Window_CLOSED;
      self->backend->Behavior(self, msg);
      FG_Msg action         = FG_Msg{ FG_Kind_ACTION };
      action.action.subkind = 1; // FG_WINDOW_ONCLOSE
      self->backend->Behavior(self, action);
      delete self;
      return 1;
    }
    case WM_MOUSEWHEEL:
    {
      DWORD pos = GetMessagePos();
      self->SetMouseScroll(GetPoints(pos), 0, GET_WHEEL_DELTA_WPARAM(wParam), GetMessageTime());
      break;
    }
    case WM_MOUSEHWHEEL:
    {
      DWORD pos = GetMessagePos();
      self->SetMouseScroll(GetPoints(pos), GET_WHEEL_DELTA_WPARAM(wParam), 0, GetMessageTime());
      break;
    }
    case WM_DPICHANGED:
      self->dpi = FG_Vec{ static_cast<float>(LOWORD(wParam)), static_cast<float>(HIWORD(wParam)) };
      self->backend->RefreshMonitors();
      return 0;
    case WM_MOUSEMOVE:
    {
      FG_Vec pt = AdjustPoints(lParam, hWnd); // Call this up here so we don't do it twice
      if(!self->inside)
      {
        _trackingstruct.hwndTrack = hWnd;
        BOOL result               = TrackMouseEvent(&_trackingstruct);
        self->inside              = true;
        self->SetMouse(pt, FG_Kind_MOUSEON, 0, wParam, GetMessageTime());
      }
      self->SetMouse(pt, FG_Kind_MOUSEMOVE, 0, wParam, GetMessageTime());
      break;
    }
    case WM_LBUTTONDOWN:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEDOWN, FG_MouseButton_L, wParam, GetMessageTime());
      break;
    case WM_LBUTTONUP:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEUP, FG_MouseButton_L, wParam, GetMessageTime());
      break;
    case WM_LBUTTONDBLCLK:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEDBLCLICK, FG_MouseButton_L, wParam, GetMessageTime());
      break;
    case WM_RBUTTONDOWN:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEDOWN, FG_MouseButton_R, wParam, GetMessageTime());
      break;
    case WM_RBUTTONUP:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEUP, FG_MouseButton_R, wParam, GetMessageTime());
      break;
    case WM_RBUTTONDBLCLK:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEDBLCLICK, FG_MouseButton_R, wParam, GetMessageTime());
      break;
    case WM_MBUTTONDOWN:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEDOWN, FG_MouseButton_M, wParam, GetMessageTime());
      break;
    case WM_MBUTTONUP:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEUP, FG_MouseButton_M, wParam, GetMessageTime());
      break;
    case WM_MBUTTONDBLCLK:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEDBLCLICK, FG_MouseButton_M, wParam, GetMessageTime());
      break;
    case WM_XBUTTONDOWN:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEDOWN,
                     (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MouseButton_X1 : FG_MouseButton_X2), wParam,
                     GetMessageTime());
      break;
    case WM_XBUTTONUP:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEUP,
                     (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MouseButton_X1 : FG_MouseButton_X2), wParam,
                     GetMessageTime());
      break;
    case WM_XBUTTONDBLCLK:
      self->SetMouse(AdjustPoints(lParam, hWnd), FG_Kind_MOUSEDBLCLICK,
                     (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MouseButton_X1 : FG_MouseButton_X2), wParam,
                     GetMessageTime());
      break;
    case WM_SYSKEYUP:
    case WM_SYSKEYDOWN:
      self->SetKey(static_cast<uint8_t>(wParam), (HIWORD(lParam) & (KF_EXTENDED | 0xff)), message == WM_SYSKEYDOWN,
                   (lParam & 0x40000000) != 0, GetMessageTime());
      break;
    case WM_KEYUP:
    case WM_KEYDOWN: // Windows return codes are the opposite of feathergui's - returning 0 means we accept, anything else
                     // rejects, so we invert the return code here.
      return !self->SetKey(static_cast<uint8_t>(wParam), (HIWORD(lParam) & (KF_EXTENDED | 0xff)), message == WM_KEYDOWN,
                           (lParam & 0x40000000) != 0, GetMessageTime());
    case WM_UNICHAR:
      if(wParam == UNICODE_NOCHAR)
        return TRUE;
    case WM_CHAR: self->SetChar(static_cast<int>(wParam), GetMessageTime()); return 0;
    case WM_MOUSELEAVE:
    {
      DWORD pos = GetMessagePos();
      self->SetMouse(GetPoints(pos), FG_Kind_MOUSEOFF, 0, (size_t)~0, GetMessageTime());
      self->inside = false;
      break;
    }
    case WM_TOUCH:
    {
      auto count  = LOWORD(wParam);
      auto inputs = reinterpret_cast<TOUCHINPUT*>(_alloca(sizeof(TOUCHINPUT) * count));
      GetTouchInputInfo((HTOUCHINPUT)lParam, count, inputs, sizeof(TOUCHINPUT));
      bool handled = false;

      for(uint16_t i = 0; i < count; ++i)
        handled = handled || (self->SetTouch(inputs[i]) != 0);

      if(!handled)
        break; // If we don't handle it, break to DefWindowProc, which will close the touch handle

      CloseTouchInputHandle((HTOUCHINPUT)lParam);
      return 0;
    }
    case WM_DROPFILES: break; // TODO: http://www.catch22.net/tuts/win32/drag-and-drop-introduction
    case WM_PAINT:
    {
      FG_Msg msg = { FG_Kind_DRAW };
      RECT WindowRect;
      GetWindowRect(hWnd, &WindowRect);
      msg.draw.area.left   = 0;
      msg.draw.area.top    = 0;
      msg.draw.area.right  = static_cast<float>(WindowRect.right - WindowRect.left);
      msg.draw.area.bottom = static_cast<float>(WindowRect.bottom - WindowRect.top);
      self->backend->BeginDraw(self->backend, hWnd, &msg.draw.area);
      self->backend->Behavior(self, msg);
      self->backend->EndDraw(self->backend, hWnd);
      ValidateRect(self->hWnd, NULL);
      return 0;
    }
    case WM_DISPLAYCHANGE:
      self->backend->RefreshMonitors();
      InvalidateRect(hWnd, NULL, FALSE);
      return 0;
    }
  }

  return DefWindowProc(hWnd, message, wParam, lParam);
}

HWND__* Window::WndCreate(FG_Vec* pos, FG_Vec* dim, unsigned long style, uint32_t exflags, void* self, const wchar_t* cls,
                          const char* caption, FG_Vec& dpi)
{
  exflags |= WS_EX_COMPOSITED | WS_EX_LAYERED;

  const wchar_t* wcaption = L"";
  if(caption)
  {
    size_t len = UTF8toUTF16(caption, -1, 0, 0);
    if(len > 4096)
      return 0;
    wcaption = (wchar_t*)ALLOCA(sizeof(wchar_t) * len);
    UTF8toUTF16(caption, -1, const_cast<wchar_t*>(wcaption), len);
  }

  RECT rsize = {
    !pos ? CW_USEDEFAULT : static_cast<int>(floor(pos->x)),
    !pos ? CW_USEDEFAULT : static_cast<int>(floor(pos->y)),
  };
  rsize.right  = rsize.left + (!dim ? CW_USEDEFAULT : static_cast<int>(ceil(dim->x)));
  rsize.bottom = rsize.top + (!dim ? CW_USEDEFAULT : static_cast<int>(ceil(dim->y)));

  AdjustWindowRectEx(&rsize, style, (style & WS_POPUP) ? FALSE : TRUE, exflags);
  HWND handle = CreateWindowExW(exflags, cls, wcaption, style, (style & WS_POPUP) ? rsize.left : CW_USEDEFAULT,
                                (style & WS_POPUP) ? rsize.top : CW_USEDEFAULT, rsize.right - rsize.left,
                                rsize.bottom - rsize.top, NULL, NULL, (HINSTANCE)&__ImageBase, self);
  HDC hdc     = GetDC(handle);
  dpi         = { static_cast<float>(GetDeviceCaps(hdc, LOGPIXELSX)), static_cast<float>(GetDeviceCaps(hdc, LOGPIXELSY)) };
  ReleaseDC(handle, hdc);

  if(backend->dwmblurbehind != 0)
  {
    // MARGINS margins = { -1,-1,-1,-1 };
    //(*dwmextend)(handle, &margins); //extends glass effect
    HRGN region               = CreateRectRgn(-1, -1, 0, 0);
    DWM_BLURBEHIND blurbehind = { DWM_BB_ENABLE | DWM_BB_BLURREGION | DWM_BB_TRANSITIONONMAXIMIZED, TRUE, region, FALSE };
    (*backend->dwmblurbehind)(handle, &blurbehind);
    DeleteObject(region);
  }

  return handle;
}

FG_Err Window::PushClip(const FG_Rect& area)
{
  target->PushAxisAlignedClip(D2D1::RectF(area.left, area.top, area.right, area.bottom), D2D1_ANTIALIAS_MODE_ALIASED);
  return 0;
}
FG_Err Window::PopClip()
{
  target->PopAxisAlignedClip();
  return 0;
}

void Window::BeginDraw(const FG_Rect& area)
{
  CreateResources();
  target->BeginDraw();
  PushClip(area);
}

void Window::Clear(FG_Color color) { target->Clear(ToD2Color(color.v)); }

void Window::EndDraw()
{
  PopClip();
  if(target->EndDraw() == D2DERR_RECREATE_TARGET)
    DiscardResources();
}

void Window::CreateResources()
{
  if(!target)
  {
    RECT rc;
    GetClientRect(hWnd, &rc);
    HRESULT hr = backend->CreateHWNDTarget(
      D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT,
                                   D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_PREMULTIPLIED)),
      D2D1::HwndRenderTargetProperties(hWnd, D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top)), &target);

    if(SUCCEEDED(hr))
    {
      hr = target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &color);
      hr = target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &edgecolor);
      if(FAILED(hr))
        (*backend->_log)(backend->_root, FG_Level_ERROR, "CreateSolidColorBrush failed with error code %li", hr);
      hr = target->QueryInterface<ID2D1DeviceContext>(&context);
      if(SUCCEEDED(hr))
      {
        context->SetUnitMode(
          D2D1_UNIT_MODE_PIXELS); // Force direct2D to render in pixels because feathergui does the DPI conversion for us.
        hr = context->CreateEffect(CLSID_Arc, &arc);
        hr = context->CreateEffect(CLSID_Circle, &circle);
        hr = context->CreateEffect(CLSID_RoundRect, &roundrect);
        hr = context->CreateEffect(CLSID_Triangle, &triangle);
        hr = context->CreateEffect(CLSID_D2D1Scale, &scaling);
        if(FAILED(hr))
          (*backend->_log)(backend->_root, FG_Level_ERROR, "CreateEffect failed with error code %li", hr);
      }
      else
        (*backend->_log)(
          backend->_root, FG_Level_ERROR,
          "QueryInterface<ID2D1DeviceContext> failed with error code %li, custom effects won't be available. Make sure your device supports Direct2D 1.1",
          hr);
    }
    else
      (*backend->_log)(backend->_root, FG_Level_ERROR, "CreateHwndRenderTarget failed with error code %li", hr);
  }
}
void Window::DiscardResources()
{
  for(khint_t iter = kh_begin(assets); iter != kh_end(assets); ++iter)
  {
    if(kh_exist(assets, iter))
    {
      auto& a = kh_key(assets, iter)->instances;
      for(size_t i = 0; i < a.Length(); ++i)
        if(a[i] == this)
        {
          a.Remove(i);
          break;
        }

      kh_val(assets, iter)->Release();
    }
  }

  if(color)
    color->Release();
  if(edgecolor)
    edgecolor->Release();
  if(roundrect)
    roundrect->Release();
  if(triangle)
    triangle->Release();
  if(arc)
    arc->Release();
  if(circle)
    circle->Release();
  if(scaling)
    scaling->Release();
  if(context)
    context->Release();
  if(target)
    target->Release();
  color     = 0;
  edgecolor = 0;
  context   = 0;
  roundrect = 0;
  triangle  = 0;
  arc       = 0;
  circle    = 0;
  target    = 0;
}
void Window::SetCaption(const char* caption)
{
  if(caption)
  {
    size_t len = UTF8toUTF16(caption, -1, 0, 0);
    if(len < 4096)
    {
      auto wcaption = reinterpret_cast<wchar_t*>(ALLOCA(sizeof(wchar_t) * len));
      UTF8toUTF16(caption, -1, wcaption, len);
      SetWindowTextW(hWnd, wcaption);
    }
  }
}
void Window::SetArea(FG_Vec* pos, FG_Vec* dim)
{
  FG_Vec zero = {};
  UINT flags  = SWP_NOSENDCHANGING;
  if(!pos)
  {
    flags |= SWP_NOMOVE;
    pos = &zero;
  }
  if(!dim)
  {
    flags |= SWP_NOSIZE;
    dim = &zero;
  }

  SetWindowPos(hWnd, HWND_TOP, static_cast<int>(floorf(pos->x)), static_cast<int>(floorf(pos->y)),
               static_cast<int>(ceilf(dim->x)), static_cast<int>(ceilf(dim->y)), SWP_NOSENDCHANGING);
}
void Window::SetFlags(uint64_t flags)
{
  if((flags & FG_Window_MINIMIZED) && !IsIconic(hWnd))
    ShowWindow(hWnd, SW_MINIMIZE);
  if((flags & FG_Window_MAXIMIZED) && !IsZoomed(hWnd))
    ShowWindow(hWnd, SW_MAXIMIZE);
  if(!(flags & (FG_Window_MINIMIZED | FG_Window_MAXIMIZED)) && (IsIconic(hWnd) || IsZoomed(hWnd)))
    ShowWindow(hWnd, SW_RESTORE);

  if((flags & FG_Window_CLOSED) && IsWindowVisible(hWnd))
    ShowWindow(hWnd, SW_HIDE);
  if(!(flags & FG_Window_CLOSED) && !IsWindowVisible(hWnd))
    ShowWindow(hWnd, SW_SHOW);

  // LONG curstyle = GetWindowLongA(hWnd, GWL_STYLE);
  // LONG check = curstyle &
}

uint8_t Window::GetModKey()
{
  // GetKeyState only reflects keyboard state up to this current keyboard event, which is what we want
  uint8_t m = 0;
  if(GetKeyState(VK_SHIFT) & 0x8000)
    m |= FG_ModKey_SHIFT;
  if(GetKeyState(VK_CONTROL) & 0x8000)
    m |= FG_ModKey_CONTROL;
  if(GetKeyState(VK_MENU) & 0x8000)
    m |= FG_ModKey_ALT;
  if((GetKeyState(VK_LWIN) | GetKeyState(VK_RWIN)) & 0x8000)
    m |= FG_ModKey_SUPER;
  if(GetKeyState(VK_CAPITAL) & 1)
    m |= FG_ModKey_CAPSLOCK;
  if(GetKeyState(VK_NUMLOCK) & 1)
    m |= FG_ModKey_NUMLOCK;
  return m;
}

size_t Window::SetKey(uint8_t keycode, uint16_t scancode, bool down, bool held, unsigned long time)
{
  if(!scancode)
    scancode = (uint16_t)MapVirtualKeyW((UINT)keycode, MAPVK_VK_TO_VSC);

  FG_Msg evt         = { down ? FG_Kind_KEYDOWN : FG_Kind_KEYUP };
  evt.keyUp.key      = keycode;
  evt.keyUp.scancode = scancode;
  evt.keyUp.modkeys  = GetModKey();

  if(held)
    evt.keyUp.modkeys |= FG_ModKey_HELD;

  return backend->Behavior(this, evt).keyUp;
}

size_t Window::SetChar(int key, unsigned long time)
{
  FG_Msg evt          = { FG_Kind_KEYCHAR };
  evt.keyChar.unicode = key;
  evt.keyChar.modkeys = GetModKey();

  return backend->Behavior(this, evt).keyChar;
}

void Window::ApplyWin32Size()
{
  RECT r;

  if(SUCCEEDED(GetWindowRect(hWnd, &r)))
  {
    auto flags = backend->Behavior(this, FG_Msg{ FG_Kind_GETWINDOWFLAGS }).getWindowFlags;
    if(flags & FG_Window_MAXIMIZED)
    {
      RECT rsize = r;
      AdjustWindowRectEx(&rsize, GetWindowLong(hWnd, GWL_STYLE), FALSE, GetWindowLong(hWnd, GWL_EXSTYLE));
      margin = { static_cast<float>(rsize.right - r.right), static_cast<float>(rsize.bottom - r.bottom),
                 static_cast<float>(rsize.right - r.right), static_cast<float>(rsize.bottom - r.bottom) };
    }
    else
      margin = { 0, 0, 0, 0 };

    FG_Msg msg             = { FG_Kind_SETWINDOWRECT };
    msg.setWindowRect.rect = { static_cast<float>(r.left), static_cast<float>(r.top), static_cast<float>(r.right),
                               static_cast<float>(r.bottom) };
    backend->Behavior(this, msg);
  }
}

size_t Window::SetMouse(const FG_Vec& points, FG_Kind type, unsigned char button, size_t wparam, unsigned long time)
{
  FG_Msg evt      = { type };
  evt.mouseMove.x = points.x;
  evt.mouseMove.y = points.y;

  if(wparam != (size_t)-1)
  {
    evt.mouseMove.all = 0; // we must keep these bools accurate at all times
    evt.mouseMove.all |= FG_MouseButton_L & (-((wparam & MK_LBUTTON) != 0));
    evt.mouseMove.all |= FG_MouseButton_R & (-((wparam & MK_RBUTTON) != 0));
    evt.mouseMove.all |= FG_MouseButton_M & (-((wparam & MK_MBUTTON) != 0));
    evt.mouseMove.all |= FG_MouseButton_X1 & (-((wparam & MK_XBUTTON1) != 0));
    evt.mouseMove.all |= FG_MouseButton_X2 & (-((wparam & MK_XBUTTON2) != 0));
  }
  else // if wparam is -1 it signals that it is invalid, so we simply leave our assignments at their last known value.
  {
    evt.mouseMove.all |= ((GetKeyState(VK_LBUTTON) != 0) << 0);
    evt.mouseMove.all |= ((GetKeyState(VK_RBUTTON) != 0) << 1);
    evt.mouseMove.all |= ((GetKeyState(VK_MBUTTON) != 0) << 2);
    evt.mouseMove.all |= ((GetKeyState(VK_XBUTTON1) != 0) << 3);
    evt.mouseMove.all |= ((GetKeyState(VK_XBUTTON2) != 0) << 4);
  }

  if(!button)
    return backend->Behavior(this, evt).mouseMove;

  evt.mouseDown.button = button;

  switch(type)
  {
  case FG_Kind_MOUSEDBLCLICK:                  // L down
  case FG_Kind_MOUSEDOWN:                      // L down
    evt.mouseDown.all |= evt.mouseDown.button; // Ensure the correct button position is reflected in the allbtn parameter
                                               // regardless of the current state of the mouse
    break;
  case FG_Kind_MOUSEUP: // L up
    evt.mouseDown.all &= ~evt.mouseDown.button;
    break;
  }

  return backend->Behavior(this, evt).mouseDown;
}

size_t Window::SetTouch(TOUCHINPUT& input)
{
  FG_Msg evt = { FG_Kind_TOUCHMOVE };

  if(input.dwFlags & TOUCHEVENTF_MOVE)
    evt.touchMove.flags |= FG_Touch_MOVE;
  if(input.dwFlags & TOUCHEVENTF_PALM)
    evt.touchMove.flags |= FG_Touch_PALM;
  if(input.dwFlags & TOUCHEVENTF_INRANGE)
    evt.touchMove.flags |= FG_Touch_HOVER;
  if(input.dwFlags & TOUCHEVENTF_DOWN)
    evt.kind = FG_Kind_TOUCHBEGIN;
  if(input.dwFlags & TOUCHEVENTF_UP)
    evt.kind = FG_Kind_TOUCHEND;

  evt.touchMove.x = static_cast<float>(input.x);
  evt.touchMove.y = static_cast<float>(input.y);
  evt.touchMove.z = NAN;
  if(input.dwMask & TOUCHINPUTMASKF_CONTACTAREA)
    evt.touchMove.r = (input.cxContact > input.cyContact) ? input.cxContact : input.cyContact;

  evt.touchMove.pressure = NAN;
  evt.touchMove.index    = backend->GetTouchIndex(input.dwID, evt.kind);

  return backend->Behavior(this, evt).touchMove;
}

size_t Window::SetMouseScroll(const FG_Vec& points, uint16_t x, uint16_t y, unsigned long time)
{
  FG_Msg evt             = { FG_Kind_MOUSESCROLL };
  evt.mouseScroll.x      = points.x;
  evt.mouseScroll.y      = points.y;
  evt.mouseScroll.delta  = y / (float)WHEEL_DELTA;
  evt.mouseScroll.hdelta = -x / (float)WHEEL_DELTA; // inverted for consistency with X11 and MacOSX

  return backend->Behavior(this, evt).mouseScroll;
}

void Window::InvalidateHWND(const FG_Rect* area)
{
  if(!area)
    InvalidateRect(hWnd, NULL, FALSE);
  else
  {
    RECT rect = { static_cast<LONG>(floorf(area->left)), static_cast<LONG>(floorf(area->top)),
                  static_cast<LONG>(ceilf(area->right)), static_cast<LONG>(ceilf(area->bottom)) };
    InvalidateRect(hWnd, &rect, FALSE);
  }

  InvalidateRect(hWnd, NULL, FALSE);
}
ID2D1Bitmap* Window::GetBitmapFromSource(const Asset* p)
{
  ID2D1Bitmap* b = NULL;
  int r;
  khiter_t i = kh_put_wic(assets, p, &r);
  if(!r)
    b = kh_val(assets, i);
  else if(r == -1)
    return 0;
  else
  {
    IWICBitmapSource* tex;
    reinterpret_cast<IUnknown*>(p->data.data)->QueryInterface<IWICBitmapSource>(&tex);

    if(!tex)
      return 0;

    if(FAILED(target->CreateBitmapFromWicBitmap(tex, nullptr, &b)))
    {
      tex->Release();
      return 0;
    }

    p->instances.Add(this);
    tex->Release();
    kh_val(assets, i) = b;
  }
  b->AddRef();
  return b;
}
void Window::DiscardAsset(const Asset* p)
{
  auto iter = kh_get_wic(assets, p);
  if(iter < kh_end(assets) && kh_exist(assets, iter))
  {
    kh_val(assets, iter)->Release();
    kh_del_wic(assets, iter);
  }
}

void Window::PushTransform(const D2D_MATRIX_3X2_F& m)
{
  D2D1::Matrix3x2F old;
  context->GetTransform(&old);
  transforms.push(old * m);
  context->SetTransform(transforms.top());
}
void Window::PopTransform()
{
  transforms.pop();
  context->SetTransform(transforms.empty() ? D2D1::IdentityMatrix() : transforms.top());
}

void Window::PushRotate(float rotate, const FG_Rect& area)
{
  if(rotate != 0.0f)
    PushTransform(D2D1::Matrix3x2F::Rotation(rotate * (180.0f / PI),
                                             D2D1::Point2F((area.left + area.right) / 2, (area.top + area.bottom) / 2)));
}

void Window::PopRotate(float rotate)
{
  if(rotate != 0.0f)
    PopTransform();
}

FG_Err Window::DrawTextD2D(FG_Font* font, void* fontlayout, FG_Rect* area, FG_Color fill, float blur, float rotate, float z)
{
  if(!fontlayout)
    return -1;

  PushRotate(rotate, *area);
  IDWriteTextLayout* layout = (IDWriteTextLayout*)fontlayout;
  color->SetColor(ToD2Color(fill.v));
  layout->SetMaxWidth(area->right - area->left);
  layout->SetMaxHeight(area->bottom - area->top);

  D2D1_TEXT_ANTIALIAS_MODE aa = D2D1_TEXT_ANTIALIAS_MODE_DEFAULT;
  switch(font->aa)
  {
  case FG_AntiAliasing_NO_AA: aa = D2D1_TEXT_ANTIALIAS_MODE_ALIASED; break;
  case FG_AntiAliasing_AA: aa = D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE; break;
  case FG_AntiAliasing_LCD:
  case FG_AntiAliasing_LCD_V: aa = D2D1_TEXT_ANTIALIAS_MODE_CLEARTYPE; break;
  }

  if(aa != target->GetTextAntialiasMode())
    target->SetTextAntialiasMode(aa);

  target->DrawTextLayout(D2D1::Point2F(area->left, area->top), layout, color, D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT);

  PopRotate(rotate);

  return 0;
}

template<int N, typename Arg, typename... Args>
inline FG_Err Window::DrawEffect(ID2D1Effect* effect, const FG_Rect& area, float rotate, const Arg arg, const Args&... args)
{
  effect->SetValue<Arg, int>(N, arg);
  if constexpr(sizeof...(args) > 0)
    return DrawEffect<N + 1, Args...>(effect, area, rotate, args...);

  PushRotate(rotate, area);
  D2D1_RECT_F rect = D2D1::RectF(area.left, area.top, area.right, area.bottom);
  context->DrawImage(effect, D2D1::Point2F(area.left, area.top), rect, D2D1_INTERPOLATION_MODE_LINEAR,
                     D2D1_COMPOSITE_MODE_SOURCE_OVER);
  PopRotate(rotate);

  return 0;
}

FG_Err Window::DrawAsset(FG_Asset* asset, const FG_Rect& area, FG_Rect* source, FG_Color color, float time, float rotate,
                         float z)
{
  ID2D1Bitmap* bitmap = GetBitmapFromSource(static_cast<const Asset*>(asset));
  fgassert(bitmap);

  D2D1_RECT_F rect      = D2D1::RectF(area.left, area.top, area.right, area.bottom);
  auto size             = bitmap->GetPixelSize();
  D2D1_RECT_F uvresolve = D2D1::RectF(0, 0, size.width, size.height);
  if(source)
    uvresolve = D2D1::RectF(source->left, source->top, source->right, source->bottom);

  auto scale = D2D1::Vector2F((rect.right - rect.left) / (uvresolve.right - uvresolve.left),
                              (rect.bottom - rect.top) / (uvresolve.bottom - uvresolve.top));
  PushRotate(rotate, area);
  if(scale.x == 1.0f && scale.y == 1.0f)
    target->DrawBitmap(bitmap, rect, color.a / 255.0f, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, &uvresolve);
  else
  {
    auto e = scaling;
    e->SetValue(D2D1_SCALE_PROP_SCALE, scale);
    e->SetValue(D2D1_SCALE_PROP_INTERPOLATION_MODE, D2D1_SCALE_INTERPOLATION_MODE_ANISOTROPIC);
    e->SetInput(0, bitmap);
    context->DrawImage(e, D2D1::Point2F(rect.left, rect.top),
                       D2D1::RectF(floorf(uvresolve.left * scale.x), floorf(uvresolve.top * scale.y),
                                   ceilf(uvresolve.right * scale.x), ceilf(uvresolve.bottom * scale.y + 1.0f)));
  }

  PopRotate(rotate);
  bitmap->Release();
  return 0;
}

FG_Err Window::DrawRect(const FG_Rect& area, const FG_Rect& corners, FG_Color fillColor, float border, FG_Color borderColor,
                        float blur, FG_Asset* asset, float rotate, float z)
{
  FG_Rect expand = { area.left - blur, area.top - blur, area.right + blur, area.bottom + blur };

  DrawEffect<0>(roundrect, expand, rotate, D2D1::Vector4F(expand.left, expand.top, expand.right, expand.bottom),
                D2D1::Vector4F(corners.left, corners.top, corners.right, corners.bottom), fillColor, borderColor, border,
                blur);
  return 0;
}

FG_Err Window::DrawArc(const FG_Rect& area, FG_Vec angles, FG_Color fillColor, float border, FG_Color borderColor,
                       float blur, float innerRadius, FG_Asset* asset, float z)
{
  DrawEffect<0>(arc, area, 0.0f, D2D1::Vector4F(area.left, area.top, area.right, area.bottom),
                D2D1::Vector4F(angles.x + (angles.y / 2.0f) - (PI / 2.0f), angles.y / 2.0f, innerRadius, 0.0f), fillColor,
                borderColor, border, blur);
  return 0;
}

FG_Err Window::DrawCircle(const FG_Rect& area, FG_Color fillColor, float border, FG_Color borderColor, float blur,
                          float innerRadius, float innerBorder, FG_Asset* asset, float z)
{
  DrawEffect<0>(circle, area, 0.0f, D2D1::Vector4F(area.left, area.top, area.right, area.bottom),
                D2D1::Vector4F(innerRadius, innerBorder, 0.0f, 0.0f), fillColor, borderColor, border, blur);
  return 0;
}

FG_Err Window::DrawTriangle(const FG_Rect& area, const FG_Rect& corners, FG_Color fillColor, float border,
                            FG_Color borderColor, float blur, FG_Asset* asset, float rotate, float z)
{
  DrawEffect<0>(triangle, area, rotate, D2D1::Vector4F(area.left, area.top, area.right, area.bottom),
                D2D1::Vector4F(corners.left, corners.top, corners.right, corners.bottom), fillColor, borderColor, border,
                blur);
  return 0;
}

FG_Err Window::DrawLines(FG_Vec* points, uint32_t count, FG_Color fill)
{
  color->SetColor(ToD2Color(fill.v));
  for(size_t i = 1; i < count; ++i)
    target->DrawLine(D2D1_POINT_2F{ points[i - 1].x, points[i - 1].y }, D2D1_POINT_2F{ points[i].x, points[i].y }, color,
                     1.0F, 0);
  return 0;
}

FG_Err Window::DrawCurve(FG_Vec* anchors, uint32_t count, FG_Color fillColor, float stroke, FG_Color strokeColor)
{
  return -1;
}

FG_Err Window::DrawShader(FG_Shader* shader, FG_Asset* vertices, FG_Asset* indices, ...) { return -1; }
