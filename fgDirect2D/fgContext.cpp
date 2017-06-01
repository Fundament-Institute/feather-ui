// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgContext.h"
#include "fgDirect2D.h"
#include "fgRoot.h"
#include "fgWindow.h"
#include "win32_includes.h"
#include "bss-util/sseVec.h"
#include <d2d1_1.h>
#include <dwmapi.h>
#include "fgRoundRect.h"
#include "fgCircle.h"
#include "fgTriangle.h"
#include "fgModulation.h"
#include "util.h"

using namespace bss;

EXTERN_C IMAGE_DOS_HEADER __ImageBase;
typedef HRESULT(STDAPICALLTYPE *DWMCOMPENABLE)(BOOL*);
typedef HRESULT(STDAPICALLTYPE *DWMBLURBEHIND)(HWND, const DWM_BLURBEHIND*);
DWMBLURBEHIND dwmblurbehind = 0;

void fgContext_Construct(fgContext* self)
{
  self->color = 0;
  self->edgecolor = 0;
  self->target = 0;
  self->roundrect = 0;
  self->triangle = 0;
  self->circle = 0;
  self->scale = 0;
  self->context = 0;
  self->inside = false;
  self->invalid = false;
  new (&self->cliprect) std::stack<AbsRect>();
  new (&self->wichash) Hash<IWICBitmapSource*, ID2D1Bitmap*>();
}
void fgContext_Destroy(fgContext* self)
{
  self->DiscardResources();
  self->cliprect.~stack();
  for(auto i : self->wichash)
    self->wichash.GetValue(i)->Release();
  self->wichash.~Hash();
}

void fgContext::SetDWMCallbacks()
{
  // Check for desktop composition
  HMODULE dwm = LoadLibraryW(L"dwmapi.dll");
  if(dwm)
  {
    DWMCOMPENABLE dwmcomp = (DWMCOMPENABLE)GetProcAddress(dwm, "DwmIsCompositionEnabled");
    if(!dwmcomp) { FreeLibrary(dwm); dwm = 0; }
    else
    {
      BOOL res;
      (*dwmcomp)(&res);
      if(res == FALSE) { FreeLibrary(dwm); dwm = 0; } //fail
    }
    dwmblurbehind = (DWMBLURBEHIND)GetProcAddress(dwm, "DwmEnableBlurBehindWindow");

    if(!dwmblurbehind)
    {
      FreeLibrary(dwm);
      dwm = 0;
      dwmblurbehind = 0;
    }
  }
}
void fgContext::WndRegister(WNDPROC f, const wchar_t* name)
{
  // Register window class
  WNDCLASSEX wcex = { sizeof(WNDCLASSEX) };
  wcex.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
  wcex.lpfnWndProc = f;
  wcex.cbClsExtra = 0;
  wcex.cbWndExtra = sizeof(LONG_PTR);
  wcex.hInstance = ((HINSTANCE)&__ImageBase);
  wcex.hbrBackground = NULL;
  wcex.lpszMenuName = NULL;
  wcex.hCursor = NULL;
  wcex.lpszClassName = name;

  RegisterClassExW(&wcex);
}

longptr_t __stdcall fgContext::WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam, fgElement* src)
{
  static tagTRACKMOUSEEVENT _trackingstruct = { sizeof(tagTRACKMOUSEEVENT), TME_LEAVE, 0, 0 };
  switch(message)
  {
  case WM_SIZE:
    if(target)
      target->Resize(D2D1::SizeU(LOWORD(lParam), HIWORD(lParam)));
    return 0;
  case WM_DISPLAYCHANGE:
    InvalidateRect(hWnd, NULL, FALSE);
    return 0;
  case WM_MOUSEWHEEL:
  {
    POINTS pointstemp = { 0, GET_WHEEL_DELTA_WPARAM(wParam) };
    DWORD pos = GetMessagePos();
    SetMouse(AdjustPointsDPI(MAKELPPOINTS(pos)), FG_MOUSESCROLL, 0, *(size_t*)&pointstemp, GetMessageTime());
  }
  break;
  case 0x020E: // WM_MOUSEHWHEEL - this is only sent on vista machines, but our minimum version is XP, so we manually look for the message anyway and process it if it happens to get sent to us.
  {
    POINTS pointstemp = { GET_WHEEL_DELTA_WPARAM(wParam), 0 };
    DWORD pos = GetMessagePos();
    SetMouse(AdjustPointsDPI(MAKELPPOINTS(pos)), FG_MOUSESCROLL, 0, *(size_t*)&pointstemp, GetMessageTime());
  }
  break;
  case WM_MOUSEMOVE:
  {
    AbsVec pt = AdjustPoints(MAKELPPOINTS(lParam), hWnd); // Call this up here so we don't do it twice
    if(!inside)
    {
      _trackingstruct.hwndTrack = hWnd;
      BOOL result = TrackMouseEvent(&_trackingstruct);
      inside = true;
      SetMouse(pt, FG_MOUSEON, 0, wParam, GetMessageTime());
    }
    SetMouse(pt, FG_MOUSEMOVE, 0, wParam, GetMessageTime());
    break;
  }
  case WM_LBUTTONDOWN:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEDOWN, FG_MOUSELBUTTON, wParam, GetMessageTime());
    break;
  case WM_LBUTTONUP:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEUP, FG_MOUSELBUTTON, wParam, GetMessageTime());
    break;
  case WM_LBUTTONDBLCLK:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEDBLCLICK, FG_MOUSELBUTTON, wParam, GetMessageTime());
    break;
  case WM_RBUTTONDOWN:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEDOWN, FG_MOUSERBUTTON, wParam, GetMessageTime());
    break;
  case WM_RBUTTONUP:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEUP, FG_MOUSERBUTTON, wParam, GetMessageTime());
    break;
  case WM_RBUTTONDBLCLK:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEDBLCLICK, FG_MOUSERBUTTON, wParam, GetMessageTime());
    break;
  case WM_MBUTTONDOWN:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEDOWN, FG_MOUSEMBUTTON, wParam, GetMessageTime());
    break;
  case WM_MBUTTONUP:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEUP, FG_MOUSEMBUTTON, wParam, GetMessageTime());
    break;
  case WM_MBUTTONDBLCLK:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEDBLCLICK, FG_MOUSEMBUTTON, wParam, GetMessageTime());
    break;
  case WM_XBUTTONDOWN:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEDOWN, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime());
    break;
  case WM_XBUTTONUP:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEUP, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime());
    break;
  case WM_XBUTTONDBLCLK:
    SetMouse(AdjustPoints(MAKELPPOINTS(lParam), hWnd), FG_MOUSEDBLCLICK, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime());
    break;
  case WM_SYSKEYUP:
  case WM_SYSKEYDOWN:
  case WM_KEYUP:
  case WM_KEYDOWN: // Windows return codes are the opposite of feathergui's - returning 0 means we accept, anything else rejects, so we invert the return code here.
    return !SetKey((uint8_t)wParam, message == WM_KEYDOWN || message == WM_SYSKEYDOWN, (lParam & 0x40000000) != 0, GetMessageTime());
  case WM_UNICHAR:
    if(wParam == UNICODE_NOCHAR) return TRUE;
  case WM_CHAR:
    SetChar((int)wParam, GetMessageTime());
    return 0;
  case WM_MOUSELEAVE:
  {
    DWORD pos = GetMessagePos();
    SetMouse(AdjustPointsDPI(MAKELPPOINTS(pos)), FG_MOUSEOFF, 0, (size_t)~0, GetMessageTime());
    inside = false;
  }
  break;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

HWND__* fgContext::WndCreate(const AbsRect& out, unsigned long style, uint32_t exflags, void* self, const wchar_t* cls, AbsVec& dpi)
{
  exflags |= WS_EX_COMPOSITED | WS_EX_LAYERED;

  RECT rsize = { out.left, out.top, out.right, out.bottom };

  //AdjustWindowRectEx(&rsize, style, FALSE, exflags); // So long as we are drawing all over the nonclient area, we don't actually want to correct this
  int rwidth = rsize.right - rsize.left;
  int rheight = rsize.bottom - rsize.top;

  HWND handle = CreateWindowExW(exflags, cls, L"", style, (style&WS_POPUP) ? rsize.left : CW_USEDEFAULT, (style&WS_POPUP) ? rsize.top : CW_USEDEFAULT, INT(rwidth), INT(rheight), NULL, NULL, (HINSTANCE)&__ImageBase, NULL);
  HDC hdc = GetDC(handle);
  dpi = { (FABS)GetDeviceCaps(hdc, LOGPIXELSX), (FABS)GetDeviceCaps(hdc, LOGPIXELSY) };
  ReleaseDC(handle, hdc);

  SetWindowLongPtrW(handle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
  
  if(dwmblurbehind != 0)
  {
    //MARGINS margins = { -1,-1,-1,-1 };
    //(*dwmextend)(handle, &margins); //extends glass effect
    HRGN region = CreateRectRgn(-1, -1, 0, 0);
    DWM_BLURBEHIND blurbehind = { DWM_BB_ENABLE | DWM_BB_BLURREGION | DWM_BB_TRANSITIONONMAXIMIZED, TRUE, region, FALSE };
    (*dwmblurbehind)(handle, &blurbehind);
    DeleteObject(region);
  }
  
  return handle;
}

void fgContext::BeginDraw(HWND handle, fgElement* element, const AbsRect* area, fgDrawAuxDataEx& exdata, AbsRect* margin)
{
  invalid = true;
  CreateResources(handle);
  target->BeginDraw();
  target->Clear(D2D1::ColorF(0, 0));
  fgElement* hold = element->root;
  fgassert(!cliprect.size());
  AbsRect truearea = *area;
  if(margin)
  {
    const sseVecT<FABS> m(-1.0f, -1.0f, 1.0f, 1.0f);
    (sseVecT<FABS>(BSS_UNALIGNED<const float>(&area->left)) + (sseVecT<FABS>(BSS_UNALIGNED<const float>(&margin->left))*m)) >> BSS_UNALIGNED<float>(&truearea.left);
  }
  const AbsVec& dpi = element->GetDPI();
  fgModulationRectDPI(&truearea, dpi.x, dpi.y);
  target->SetTransform(D2D1::Matrix3x2F::Translation(-truearea.left, -truearea.top));
  cliprect.push(*area); // Dont' use the DPI scale here because the cliprect is scaled later in the pipeline.

  exdata = {
    {
      sizeof(fgDrawAuxDataEx),
      dpi,
      { 1,1 },
      { 0,0 }
    },
    this,
  };
}

void fgContext::EndDraw()
{
  cliprect.pop();
  fgassert(!cliprect.size());
  if(target->EndDraw() == 0x8899000C) // D2DERR_RECREATE_TARGET
    DiscardResources();
  invalid = false;
}

void fgContext::CreateResources(HWND handle)
{
  if(!target)
  {
    RECT rc;
    GetClientRect(handle, &rc);
    HRESULT hr = fgDirect2D::instance->factory->CreateHwndRenderTarget(
      D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT, D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_PREMULTIPLIED)),
      D2D1::HwndRenderTargetProperties(handle, D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top)),
      &target);

    if(SUCCEEDED(hr))
    {
      hr = target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &color);
      hr = target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &edgecolor);
      if(FAILED(hr))
        fgLog(FGLOG_ERROR, "CreateSolidColorBrush failed with error code %li", hr);
      hr = target->QueryInterface<ID2D1DeviceContext>(&context);
      if(SUCCEEDED(hr))
      {
        context->SetUnitMode(D2D1_UNIT_MODE_PIXELS); // Force direct2D to render in pixels because feathergui does the DPI conversion for us.
        hr = context->CreateEffect(CLSID_fgRoundRect, &roundrect);
        hr = context->CreateEffect(CLSID_fgCircle, &circle);
        hr = context->CreateEffect(CLSID_fgTriangle, &triangle);
        hr = context->CreateEffect(CLSID_D2D1Scale, &scale);
        if(FAILED(hr))
          fgLog(FGLOG_ERROR, "CreateEffect failed with error code %li", hr);
      }
      else
        fgLog(FGLOG_ERROR, "QueryInterface<ID2D1DeviceContext> failed with error code %li, custom effects won't be available. Make sure your device supports Direct2D 1.1", hr);
    }
    else
      fgLog(FGLOG_ERROR, "CreateHwndRenderTarget failed with error code %li", hr);
  }
}
void fgContext::DiscardResources()
{
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
  target = 0;
  context = 0;
  roundrect = 0;
  triangle = 0;
  circle = 0;
}

size_t fgContext::SetKey(uint8_t keycode, bool down, bool held, unsigned long time)
{
  fgRoot* root = fgSingleton();
  FG_Msg evt = { 0 };
  evt.type = down ? FG_KEYDOWN : FG_KEYUP;
  evt.keycode = keycode;
  evt.sigkeys = 0;
  if(root->GetKey(FG_KEY_SHIFT)) evt.sigkeys = evt.sigkeys | 1; //VK_SHIFT
  if(root->GetKey(FG_KEY_CONTROL)) evt.sigkeys = evt.sigkeys | 2; //VK_CONTROL
  if(root->GetKey(FG_KEY_MENU)) evt.sigkeys = evt.sigkeys | 4; //VK_MENU
  if(held) evt.sigkeys = evt.sigkeys | 8;

  return root->inject(root, &evt);
}

void fgContext::SetChar(int key, unsigned long time)
{
  fgRoot* root = fgSingleton();
  FG_Msg evt = { 0 };
  evt.type = FG_KEYCHAR;
  evt.keychar = key;
  evt.sigkeys = 0;

  if(root->GetKey(FG_KEY_SHIFT)) evt.sigkeys = evt.sigkeys | 1; //VK_SHIFT
  if(root->GetKey(FG_KEY_CONTROL)) evt.sigkeys = evt.sigkeys | 2; //VK_CONTROL
  if(root->GetKey(FG_KEY_MENU)) evt.sigkeys = evt.sigkeys | 4; //VK_MENU
                                                               //if(held) evt.sigkeys = evt.sigkeys|8;
  root->inject(root, &evt);
}

AbsVec fgContext::AdjustPoints(tagPOINTS* points, HWND hWnd)
{
  RECT rect;
  GetWindowRect(hWnd, &rect);
  AbsVec pt = { points->x + rect.left, points->y + rect.top };
  AdjustDPI(pt, fgSingleton()->dpi);
  return pt;
}
void fgContext::AdjustDPI(AbsVec& pt, AbsVec& dpi)
{
  pt.x *= (96.0f / dpi.x);
  pt.y *= (96.0f / dpi.y);
}
AbsVec fgContext::AdjustPointsDPI(tagPOINTS* points)
{
  AbsVec pt = { points->x, points->y };
  AdjustDPI(pt, fgSingleton()->dpi);
  return pt;
}

void fgContext::ApplyWin32Size(fgWindow& self, HWND__* handle, const AbsVec& dpi)
{
  RECT r;

  if(SUCCEEDED(GetWindowRect(handle, &r)))
  {
    if(self.maximized)
    {
      RECT rsize = r;
      AdjustWindowRectEx(&rsize, GetWindowLong(handle, GWL_STYLE), FALSE, GetWindowLong(handle, GWL_EXSTYLE));
      self->SetMargin(AbsRect{ (FABS)(rsize.right-r.right), (FABS)(rsize.bottom - r.bottom), (FABS)(rsize.right - r.right), (FABS)(rsize.bottom - r.bottom) }, (uint16_t)~0);
    }
    else
      self->SetMargin(AbsRect{ 0,0,0,0 }, (uint16_t)~0);

    AbsRect rect = { r.left, r.top, r.right, r.bottom };
    fgInvScaleRectDPI(&rect, dpi.x, dpi.y);
    CRect area = { rect.left, 0, rect.top, 0, rect.right, 0, rect.bottom };
    fgSendSubMsg<FG_SETAREA, void*>(self, (uint16_t)~0, &area);
  }
}

void fgContext::SetMouse(AbsVec& points, unsigned short type, unsigned char button, size_t wparam, unsigned long time)
{
  fgRoot* root = fgSingleton();
  FG_Msg evt = { 0 };
  evt.type = type;
  evt.x = points.x;
  evt.y = points.y;

  if(type == FG_MOUSESCROLL)
  {
    POINTS delta = MAKEPOINTS(wparam);
    evt.scrolldelta = delta.y;
    evt.scrollhdelta = delta.x;
  }
  else
  {
    if(wparam != (size_t)-1) //if wparam is -1 it signals that it is invalid, so we simply leave our assignments at their last known value.
    {
      evt.allbtn = 0; //we must keep these bools accurate at all times
      evt.allbtn |= FG_MOUSELBUTTON&(-((wparam&MK_LBUTTON) != 0));
      evt.allbtn |= FG_MOUSERBUTTON&(-((wparam&MK_RBUTTON) != 0));
      evt.allbtn |= FG_MOUSEMBUTTON&(-((wparam&MK_MBUTTON) != 0));
      evt.allbtn |= FG_MOUSEXBUTTON1&(-((wparam&MK_XBUTTON1) != 0));
      evt.allbtn |= FG_MOUSEXBUTTON2&(-((wparam&MK_XBUTTON2) != 0));
    }
    else
      evt.allbtn = root->mouse.buttons;
    evt.button = button;
  }

  switch(type)
  {
  case FG_MOUSEDBLCLICK: //L down
  case FG_MOUSEDOWN: //L down
    evt.allbtn |= evt.button; // Ensure the correct button position is reflected in the allbtn parameter regardless of the current state of the mouse
    break;
  case FG_MOUSEUP: //L up
    evt.allbtn &= ~evt.button;
    break;
  }

  root->inject(root, &evt);
}

void fgContext::InvalidateHWND(HWND__* hWnd)
{
  if(!invalid)
  {
    InvalidateRect(hWnd, NULL, FALSE);
    invalid = true;
  }
}
ID2D1Bitmap* fgContext::GetBitmapFromSource(IWICBitmapSource* p)
{
  ID2D1Bitmap* b = NULL;
  khiter_t i = wichash.Iterator(p);
  if(wichash.ExistsIter(i))
  {
    ID2D1Bitmap* b = wichash.GetValue(i);
    b->AddRef();
    return b;
  }

  if(FAILED(target->CreateBitmapFromWicBitmap(p, nullptr, &b)))
    return 0;
  
  wichash.Insert(p, b);
  b->AddRef();
  return b;
}
