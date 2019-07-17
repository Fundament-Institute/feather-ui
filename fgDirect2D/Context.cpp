// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "Context.h"
#include "Direct2D.h"
#include "feather/keys.h"
#include "feather/rtree.h"
#include "feather/component/window.h"
#include "win32_includes.h"
#include <d2d1_1.h>
#include <dwmapi.h>
#include <Windowsx.h>
#include <wincodec.h>
#include "RoundRect.h"
#include "Circle.h"
#include "Triangle.h"
#include "Modulation.h"
#include "util.h"

namespace D2D {
  __KHASH_IMPL(wic, , fgAssetD2D*, ID2D1Bitmap*, 1, kh_ptr_hash_func, kh_int_hash_equal);
}

using namespace D2D;

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

FG_FORCEINLINE fgVec GetPoints(longptr_t lParam)
{
  union { POINTS FAR* p; longptr_t l; } u;
  u.l = lParam;
  return fgVec{ (float)u.p->x, (float)u.p->y };
}
FG_FORCEINLINE fgVec AdjustPoints(longptr_t lParam, HWND__* hWnd)
{
  RECT rect;
  GetWindowRect(hWnd, &rect);
  auto v = GetPoints(lParam);
  v.x += rect.left;

  return fgVec{ v.x + (float)rect.left, v.y + (float)rect.top };
}

Context::Context(const fgRoot* root, fgDocumentNode* _node, struct FG__OUTLINE_NODE* _display)
{
  target = 0;
  context = 0;
  color = 0;
  edgecolor = 0;
  roundrect = 0;
  triangle = 0;
  circle = 0;
  scale = 0;
  node = _node;
  backend = static_cast<Direct2D*>(root->backend);
  margin = { 0 };
  assets = kh_init_wic();
  display = _display;
  inside = false;
  invalid = false;

  dpi = reinterpret_cast<DisplayData*>(display->auxdata)->dpi;
  //hWnd = Context::WndCreate(fgRect{ 0 }, WS_POPUP, WS_EX_TOOLWINDOW, backend, L"WindowD2D", dpi);
  hWnd = Context::WndCreate(fgRect{ 0, 0, 800, 600 }, WS_THICKFRAME, WS_EX_APPWINDOW, this, L"WindowD2D", dpi);
  
  ShowWindow(hWnd, SW_SHOW);
  UpdateWindow(hWnd);
}
Context::~Context()
{
  DiscardResources();
}

void Context::WndRegister(WNDPROC f, const wchar_t* name)
{
  // Register window class
  WNDCLASSEX wcex = { sizeof(WNDCLASSEX) };
  wcex.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
  wcex.lpfnWndProc = f;
  wcex.cbClsExtra = 0;
  wcex.cbWndExtra = sizeof(LONG_PTR);
  wcex.hInstance = ((HINSTANCE)& __ImageBase);
  wcex.hbrBackground = NULL;
  wcex.lpszMenuName = NULL;
  wcex.hCursor = NULL;
  wcex.lpszClassName = name;

  RegisterClassExW(&wcex);
}

longptr_t __stdcall Context::TopWndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  static const int BORDERWIDTH = 5;
  Context* self = reinterpret_cast<Context*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  if(self)
  {
    switch(message)
    {
    case WM_CREATE:
    {
      CREATESTRUCT* info = reinterpret_cast<CREATESTRUCT*>(lParam);
      SetWindowPos(hWnd, HWND_TOP, info->x, info->y, info->cx, info->cy, SWP_FRAMECHANGED);
    }
    break;
    case WM_NCCALCSIZE:
      if(wParam == TRUE)
        return 0;
      break;
    case WM_NCHITTEST: // TODO: Make this a custom message passed to the window behavior
    {
      fgMessage msg = { FG_MSG_STATE_GET, FG_WINDOW_STATE };
      fgCalcNode flags;
      msg.getState = { 0, &flags };
      fgSendMessage(self->backend->root, self->node, &msg);
      if(flags.value.i & (FG_WINDOW_RESIZABLE | FG_WINDOW_MAXIMIZABLE))
      {
        RECT WindowRect;
        GetWindowRect(hWnd, &WindowRect);
        int x = GET_X_LPARAM(lParam) - WindowRect.left;
        int y = GET_Y_LPARAM(lParam) - WindowRect.top;

        if((flags.value.i & FG_WINDOW_RESIZABLE) && !(flags.value.i & FG_WINDOW_MAXIMIZED))
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
        //if(y < (self->window->padding.top * (fgSingleton()->dpi.y / 96.0f)) + self->window->margin.top)
        //  return HTCAPTION;
        return HTCLIENT;
      }
      break;
    }
    case WM_DESTROY:
    {
      fgMessage msg = { FG_MSG_STATE_SET, FG_WINDOW_STATE };
      fgMessage get = { FG_MSG_STATE_GET, FG_WINDOW_STATE };
      get.getState = { 0, &msg.setState.value };
      fgSendMessage(self->backend->root, self->node, &get);
      msg.setState.value.value.i |= FG_WINDOW_CLOSED;
      fgSendMessage(self->backend->root, self->node, &msg);
      msg = { FG_MSG_ACTION, FG_WINDOW_ONCLOSE };
      fgSendMessage(self->backend->root, self->node, &msg);
      delete self;
      return 1;
    }
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
    case WM_SIZE:
    {
      fgMessage msg = { FG_MSG_STATE_SET, FG_WINDOW_STATE };
      fgMessage get = { FG_MSG_STATE_GET, FG_WINDOW_STATE };
      get.getState.value = &msg.setState.value;
      fgSendMessage(self->backend->root, self->node, &get);

      switch(wParam)
      {
      case SIZE_MAXIMIZED:
        msg.setState.value.value.i |= FG_WINDOW_MAXIMIZED;
        fgSendMessage(self->backend->root, self->node, &msg);
        break;
      case SIZE_MINIMIZED:
        msg.setState.value.value.i &= ~FG_WINDOW_MINIMIZED;
        fgSendMessage(self->backend->root, self->node, &msg);
      case SIZE_RESTORED:
        msg.setState.value.value.i &= ~(FG_WINDOW_MAXIMIZED & FG_WINDOW_MINIMIZED);
        fgSendMessage(self->backend->root, self->node, &msg);
        break;
      }
      self->ApplyWin32Size(hWnd);
    }
    break;
    }
    return Context::WndProc(hWnd, message, wParam, lParam);
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

longptr_t __stdcall Context::WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  static tagTRACKMOUSEEVENT _trackingstruct = { sizeof(tagTRACKMOUSEEVENT), TME_LEAVE, 0, 0 };
  Context* self = reinterpret_cast<Context*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  switch(message)
  {
  case WM_SIZE:
    if(self->target)
      self->target->Resize(D2D1::SizeU(LOWORD(lParam), HIWORD(lParam)));
    return 0;
  case WM_MOUSEWHEEL:
  {
    POINTS pointstemp = { 0, GET_WHEEL_DELTA_WPARAM(wParam) };
    DWORD pos = GetMessagePos();
    self->SetMouse(GetPoints(pos), FG_MSG_MOUSE_SCROLL, 0, *(size_t*)& pointstemp, GetMessageTime());
  }
  break;
  case WM_DPICHANGED:
    self->dpi = fgVec{ (float)LOWORD(wParam), (float)HIWORD(wParam) };
    self->backend->RefreshMonitors();
    return 0;
  case WM_MOUSEHWHEEL:
  {
    POINTS pointstemp = { GET_WHEEL_DELTA_WPARAM(wParam), 0 };
    DWORD pos = GetMessagePos();
    self->SetMouse(GetPoints(pos), FG_MSG_MOUSE_SCROLL, 0, *(size_t*)& pointstemp, GetMessageTime());
  }
  break;
  case WM_MOUSEMOVE:
  {
    fgVec pt = AdjustPoints(lParam, hWnd); // Call this up here so we don't do it twice
    if(!self->inside)
    {
      _trackingstruct.hwndTrack = hWnd;
      BOOL result = TrackMouseEvent(&_trackingstruct);
      self->inside = true;
      self->SetMouse(pt, FG_MSG_MOUSE_ON, 0, wParam, GetMessageTime());
    }
    self->SetMouse(pt, FG_MSG_MOUSE_MOVE, 0, wParam, GetMessageTime());
    break;
  }
  case WM_LBUTTONDOWN:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_DOWN, FG_MOUSE_LBUTTON, wParam, GetMessageTime());
    break;
  case WM_LBUTTONUP:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_UP, FG_MOUSE_LBUTTON, wParam, GetMessageTime());
    break;
  case WM_LBUTTONDBLCLK:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_DBLCLICK, FG_MOUSE_LBUTTON, wParam, GetMessageTime());
    break;
  case WM_RBUTTONDOWN:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_DOWN, FG_MOUSE_RBUTTON, wParam, GetMessageTime());
    break;
  case WM_RBUTTONUP:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_UP, FG_MOUSE_RBUTTON, wParam, GetMessageTime());
    break;
  case WM_RBUTTONDBLCLK:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_DBLCLICK, FG_MOUSE_RBUTTON, wParam, GetMessageTime());
    break;
  case WM_MBUTTONDOWN:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_DOWN, FG_MOUSE_MBUTTON, wParam, GetMessageTime());
    break;
  case WM_MBUTTONUP:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_UP, FG_MOUSE_MBUTTON, wParam, GetMessageTime());
    break;
  case WM_MBUTTONDBLCLK:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_DBLCLICK, FG_MOUSE_MBUTTON, wParam, GetMessageTime());
    break;
  case WM_XBUTTONDOWN:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_DOWN, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSE_XBUTTON1 : FG_MOUSE_XBUTTON2), wParam, GetMessageTime());
    break;
  case WM_XBUTTONUP:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_UP, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSE_XBUTTON1 : FG_MOUSE_XBUTTON2), wParam, GetMessageTime());
    break;
  case WM_XBUTTONDBLCLK:
    self->SetMouse(AdjustPoints(lParam, hWnd), FG_MSG_MOUSE_DBLCLICK, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSE_XBUTTON1 : FG_MOUSE_XBUTTON2), wParam, GetMessageTime());
    break;
  case WM_SYSKEYUP:
  case WM_SYSKEYDOWN:
    self->SetKey((uint8_t)wParam, message == WM_SYSKEYDOWN, (lParam & 0x40000000) != 0, GetMessageTime());
    break;
  case WM_KEYUP:
  case WM_KEYDOWN: // Windows return codes are the opposite of feathergui's - returning 0 means we accept, anything else rejects, so we invert the return code here.
    return !self->SetKey((uint8_t)wParam, message == WM_KEYDOWN, (lParam & 0x40000000) != 0, GetMessageTime());
  case WM_UNICHAR:
    if(wParam == UNICODE_NOCHAR) return TRUE;
  case WM_CHAR:
    self->SetChar((int)wParam, GetMessageTime());
    return 0;
  case WM_MOUSELEAVE:
  {
    DWORD pos = GetMessagePos();
    self->SetMouse(GetPoints(pos), FG_MSG_MOUSE_OFF, 0, (size_t)~0, GetMessageTime());
    self->inside = false;
    break;
  }
  case WM_PAINT:
  {
    fgMessage msg = { FG_MSG_DRAW };
    msg.draw.data = self;
    msg.draw.area = fgResolveNodeArea(self->node->rtnode);
    self->BeginDraw(hWnd, msg.draw.area);
    fgSendMessage(self->backend->root, self->node, &msg);
    self->EndDraw();
    ValidateRect(self->hWnd, NULL);
    return 0;
  }
  case WM_DISPLAYCHANGE:
    self->backend->RefreshMonitors();
    self->display = self->backend->GetDisplay(MonitorFromWindow(self->hWnd, MONITOR_DEFAULTTONEAREST));
    InvalidateRect(hWnd, NULL, FALSE);
    return 0;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

HWND__* Context::WndCreate(const fgRect& out, unsigned long style, uint32_t exflags, void* self, const wchar_t* cls, fgVec& dpi)
{
  exflags |= WS_EX_COMPOSITED | WS_EX_LAYERED;

  RECT rsize = { (LONG)floor(out.left), (LONG)floor(out.top), (LONG)ceil(out.right), (LONG)ceil(out.bottom) };

  //AdjustWindowRectEx(&rsize, style, FALSE, exflags); // So long as we are drawing all over the nonclient area, we don't actually want to correct this
  int rwidth = rsize.right - rsize.left;
  int rheight = rsize.bottom - rsize.top;

  HWND handle = CreateWindowExW(exflags, cls, L"", style, (style & WS_POPUP) ? rsize.left : CW_USEDEFAULT, (style & WS_POPUP) ? rsize.top : CW_USEDEFAULT, INT(rwidth), INT(rheight), NULL, NULL, (HINSTANCE)& __ImageBase, NULL);
  HDC hdc = GetDC(handle);
  dpi = { (float)GetDeviceCaps(hdc, LOGPIXELSX), (float)GetDeviceCaps(hdc, LOGPIXELSY) };
  ReleaseDC(handle, hdc);

  SetWindowLongPtrW(handle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));

  if(backend->dwmblurbehind != 0)
  {
    //MARGINS margins = { -1,-1,-1,-1 };
    //(*dwmextend)(handle, &margins); //extends glass effect
    HRGN region = CreateRectRgn(-1, -1, 0, 0);
    DWM_BLURBEHIND blurbehind = { DWM_BB_ENABLE | DWM_BB_BLURREGION | DWM_BB_TRANSITIONONMAXIMIZED, TRUE, region, FALSE };
    (*backend->dwmblurbehind)(handle, &blurbehind);
    DeleteObject(region);
  }

  return handle;
}

fgError Context::PushClip(fgRect area)
{
  target->PushAxisAlignedClip(D2D1::RectF(area.left, area.top, area.right, area.bottom), D2D1_ANTIALIAS_MODE_ALIASED);
  return 0;
}
fgError Context::PopClip()
{
  target->PopAxisAlignedClip();
  return 0;
}

void Context::BeginDraw(HWND handle, const fgRect& area)
{
  invalid = true;
  CreateResources(handle);
  target->BeginDraw();
  target->Clear(D2D1::ColorF(0, 0));
  target->SetTransform(D2D1::Matrix3x2F::Translation(-area.left, -area.top));
  PushClip(area);
}

void Context::EndDraw()
{
  PopClip();
  if(target->EndDraw() == 0x8899000C) // D2DERR_RECREATE_TARGET
    DiscardResources();
  invalid = false;
}

void Context::CreateResources(HWND handle)
{
  if(!target)
  {
    RECT rc;
    GetClientRect(handle, &rc);
    HRESULT hr = backend->factory->CreateHwndRenderTarget(
      D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT, D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_PREMULTIPLIED)),
      D2D1::HwndRenderTargetProperties(handle, D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top)),
      &target);

    if(SUCCEEDED(hr))
    {
      hr = target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &color);
      hr = target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &edgecolor);
      if(FAILED(hr))
        fgLog(backend->root, FGLOG_ERROR, "CreateSolidColorBrush failed with error code %li", hr);
      hr = target->QueryInterface<ID2D1DeviceContext>(&context);
      if(SUCCEEDED(hr))
      {
        context->SetUnitMode(D2D1_UNIT_MODE_PIXELS); // Force direct2D to render in pixels because feathergui does the DPI conversion for us.
        hr = context->CreateEffect(CLSID_RoundRect, &roundrect);
        hr = context->CreateEffect(CLSID_Circle, &circle);
        hr = context->CreateEffect(CLSID_Triangle, &triangle);
        hr = context->CreateEffect(CLSID_D2D1Scale, &scale);
        if(FAILED(hr))
          fgLog(backend->root, FGLOG_ERROR, "CreateEffect failed with error code %li", hr);
      }
      else
        fgLog(backend->root, FGLOG_ERROR, "QueryInterface<ID2D1DeviceContext> failed with error code %li, custom effects won't be available. Make sure your device supports Direct2D 1.1", hr);
    }
    else
      fgLog(backend->root, FGLOG_ERROR, "CreateHwndRenderTarget failed with error code %li", hr);
  }
}
void Context::DiscardResources()
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
  if(circle)
    circle->Release();
  if(scale)
    scale->Release();
  if(context)
    context->Release();
  if(target)
    target->Release();
  color = 0;
  edgecolor = 0;
  context = 0;
  roundrect = 0;
  triangle = 0;
  circle = 0;
  target = 0;
}

size_t Context::SetKey(uint8_t keycode, bool down, bool held, unsigned long time)
{
  fgMessage evt = { down ? FG_MSG_KEY_DOWN : FG_MSG_KEY_UP };
  evt.key.code = keycode;
  evt.key.sigkeys = 0;
  if(fgGetKey(backend->root, FG_KEY_SHIFT)) evt.key.sigkeys = evt.key.sigkeys | 1; //VK_SHIFT
  if(fgGetKey(backend->root, FG_KEY_CONTROL)) evt.key.sigkeys = evt.key.sigkeys | 2; //VK_CONTROL
  if(fgGetKey(backend->root, FG_KEY_MENU)) evt.key.sigkeys = evt.key.sigkeys | 4; //VK_MENU
  if(held) evt.key.sigkeys = evt.key.sigkeys | 8;

  return fgInject(backend->root, 0, &evt).error;
}

size_t Context::SetChar(int key, unsigned long time)
{
  fgMessage evt = { FG_MSG_KEY_CHAR };
  evt.keyChar.unicode = key;
  evt.keyChar.sigkeys = 0;

  if(fgGetKey(backend->root, FG_KEY_SHIFT)) evt.keyChar.sigkeys = evt.keyChar.sigkeys | 1; //VK_SHIFT
  if(fgGetKey(backend->root, FG_KEY_CONTROL)) evt.keyChar.sigkeys = evt.keyChar.sigkeys | 2; //VK_CONTROL
  if(fgGetKey(backend->root, FG_KEY_MENU)) evt.keyChar.sigkeys = evt.keyChar.sigkeys | 4; //VK_MENU
  // if(held) evt.keyChar.sigkeys = evt.keyChar.sigkeys | 8;

  return fgInject(backend->root, 0, &evt).error;
}

void Context::ApplyWin32Size(HWND__* handle)
{
  RECT r;

  if(SUCCEEDED(GetWindowRect(handle, &r)))
  {
    fgMessage msg = { FG_MSG_STATE_GET, FG_WINDOW_STATE };
    fgCalcNode flags;
    msg.getState.value = &flags;
    fgSendMessage(backend->root, node, &msg);
    if(flags.value.i & FG_WINDOW_MAXIMIZED)
    {
      RECT rsize = r;
      AdjustWindowRectEx(&rsize, GetWindowLong(handle, GWL_STYLE), FALSE, GetWindowLong(handle, GWL_EXSTYLE));
      margin = { (float)(rsize.right - r.right), (float)(rsize.bottom - r.bottom), (float)(rsize.right - r.right), (float)(rsize.bottom - r.bottom) };
    }
    else
      margin = { 0,0,0,0 };

    fgRect rect = { (float)r.left, (float)r.top, (float)r.right, (float)r.bottom };
    msg = { FG_MSG_STATE_SET, FG_WINDOW_LEFT };
    msg.setState.value = { FG_CALC_FLOAT, r.left };
    fgSendMessage(backend->root, node, &msg);
    msg.type = FG_WINDOW_TOP;
    msg.setState.value = { FG_CALC_FLOAT, r.top };
    fgSendMessage(backend->root, node, &msg);
    msg.type = FG_WINDOW_RIGHT;
    msg.setState.value = { FG_CALC_FLOAT, r.right };
    fgSendMessage(backend->root, node, &msg);
    msg.type = FG_WINDOW_BOTTOM;
    msg.setState.value = { FG_CALC_FLOAT, r.bottom };
    fgSendMessage(backend->root, node, &msg);
  }
}

size_t Context::SetMouse(fgVec& points, fgMsgType type, unsigned char button, size_t wparam, unsigned long time)
{
  fgMessage evt = { type };
  evt.mouse.x = points.x;
  evt.mouse.y = points.y;

  if(type == FG_MSG_MOUSE_SCROLL)
  {
    POINTS delta = MAKEPOINTS(wparam);
    evt.mouseScroll.delta = delta.y;
    evt.mouseScroll.hdelta = delta.x;
  }
  else
  {
    if(wparam != (size_t)-1) //if wparam is -1 it signals that it is invalid, so we simply leave our assignments at their last known value.
    {
      evt.mouse.all = 0; //we must keep these bools accurate at all times
      evt.mouse.all |= FG_MOUSE_LBUTTON & (-((wparam & MK_LBUTTON) != 0));
      evt.mouse.all |= FG_MOUSE_RBUTTON & (-((wparam & MK_RBUTTON) != 0));
      evt.mouse.all |= FG_MOUSE_MBUTTON & (-((wparam & MK_MBUTTON) != 0));
      evt.mouse.all |= FG_MOUSE_XBUTTON1 & (-((wparam & MK_XBUTTON1) != 0));
      evt.mouse.all |= FG_MOUSE_XBUTTON2 & (-((wparam & MK_XBUTTON2) != 0));
    }
    else if(backend->root->n_pointers > 0)
      evt.mouse.all = backend->root->pointers[0].buttons;
    evt.mouse.button = button;
  }

  switch(type)
  {
  case FG_MSG_MOUSE_DBLCLICK: //L down
  case FG_MSG_MOUSE_DOWN: //L down
    evt.mouse.all |= evt.mouse.button; // Ensure the correct button position is reflected in the allbtn parameter regardless of the current state of the mouse
    break;
  case FG_MSG_MOUSE_UP: //L up
    evt.mouse.all &= ~evt.mouse.button;
    break;
  }

  return fgInject(backend->root, 0, &evt).error;
}

void Context::InvalidateHWND(HWND__* hWnd)
{
  if(!invalid)
  {
    InvalidateRect(hWnd, NULL, FALSE);
    invalid = true;
  }
}
ID2D1Bitmap* Context::GetBitmapFromSource(fgAssetD2D* p)
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
    reinterpret_cast<IUnknown*>(p->asset.data)->QueryInterface<IWICBitmapSource>(&tex);

    if(!tex)
      return 0;

    if(FAILED(target->CreateBitmapFromWicBitmap(tex, nullptr, &b)))
    {
      tex->Release();
      return 0;
    }

    tex->Release();
    kh_val(assets, i) = b;
  }
  b->AddRef();
  return b;
}
