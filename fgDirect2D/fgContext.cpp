// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgContext.h"
#include "fgDirect2D.h"
#include "fgRoot.h"
#include "bss-util/bss_win32_includes.h"
#include <d2d1_1.h>
#include <dwmapi.h>
#include "fgRoundRect.h"
#include "fgCircle.h"
#include "fgTriangle.h"

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
  self->inside = false;
  new (&self->cliprect) std::stack<AbsRect>();
}
void fgContext_Destroy(fgContext* self)
{
  self->DiscardResources();
  self->cliprect.~stack();
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
  wcex.hCursor = LoadCursor(NULL, IDI_APPLICATION);
  wcex.lpszClassName = name;

  RegisterClassExW(&wcex);
}

#define MAKELPPOINTS(l)       ((POINTS FAR *)&(l))

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
      POINTS pointstemp = { 0 };
      pointstemp.y = GET_WHEEL_DELTA_WPARAM(wParam);
      SetMouse(&pointstemp, FG_MOUSESCROLL, 0, wParam, GetMessageTime(), src);
    }
    break;
    case 0x020E: // WM_MOUSEHWHEEL - this is only sent on vista machines, but our minimum version is XP, so we manually look for the message anyway and process it if it happens to get sent to us.
    {
      POINTS pointstemp = { 0 };
      pointstemp.x = GET_WHEEL_DELTA_WPARAM(wParam);
      SetMouse(&pointstemp, FG_MOUSESCROLL, 0, wParam, GetMessageTime(), src);
    }
    break;
    case WM_MOUSEMOVE:
      if(!inside)
      {
        _trackingstruct.hwndTrack = hWnd;
        BOOL result = TrackMouseEvent(&_trackingstruct);
        inside = true;
        SetMouse(MAKELPPOINTS(lParam), FG_MOUSEON, 0, wParam, GetMessageTime(), src);
      }
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEMOVE, 0, wParam, GetMessageTime(), src);
      break;
    case WM_LBUTTONDOWN:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDOWN, FG_MOUSELBUTTON, wParam, GetMessageTime(), src);
      SetCapture(hWnd);
      break;
    case WM_LBUTTONUP:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEUP, FG_MOUSELBUTTON, wParam, GetMessageTime(), src);
      ReleaseCapture();
      break;
    case WM_LBUTTONDBLCLK:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDBLCLICK, FG_MOUSELBUTTON, wParam, GetMessageTime(), src);
      break;
    case WM_RBUTTONDOWN:
      SetCapture(hWnd);
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDOWN, FG_MOUSERBUTTON, wParam, GetMessageTime(), src);
      break;
    case WM_RBUTTONUP:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEUP, FG_MOUSERBUTTON, wParam, GetMessageTime(), src);
      ReleaseCapture();
      break;
    case WM_RBUTTONDBLCLK:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDBLCLICK, FG_MOUSERBUTTON, wParam, GetMessageTime(), src);
      break;
    case WM_MBUTTONDOWN:
      SetCapture(hWnd);
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDOWN, FG_MOUSEMBUTTON, wParam, GetMessageTime(), src);
      break;
    case WM_MBUTTONUP:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEUP, FG_MOUSEMBUTTON, wParam, GetMessageTime(), src);
      ReleaseCapture();
      break;
    case WM_MBUTTONDBLCLK:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDBLCLICK, FG_MOUSEMBUTTON, wParam, GetMessageTime(), src);
      break;
    case WM_XBUTTONDOWN:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDOWN, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime(), src);
      break;
    case WM_XBUTTONUP:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEUP, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime(), src);
      break;
    case WM_XBUTTONDBLCLK:
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDBLCLICK, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime(), src);
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
      SetMouse(MAKELPPOINTS(lParam), FG_MOUSEOFF, 0, 0, GetMessageTime(), src);
      inside = false;
      break;
    }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

HWND__* fgContext::WndCreate(const AbsRect& out, uint32_t exflags, void* self, const wchar_t* cls, fgIntVec& dpi)
{
  unsigned long style = WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;

  RECT rsize = { out.left, out.top, out.right, out.bottom };

  AdjustWindowRect(&rsize, style, FALSE);
  int rwidth = rsize.right - rsize.left;
  int rheight = rsize.bottom - rsize.top;

  HWND handle = CreateWindowExW(WS_EX_COMPOSITED | exflags, cls, L"", style, rsize.left, rsize.top, INT(rwidth), INT(rheight), NULL, NULL, (HINSTANCE)&__ImageBase, NULL);
  HDC hdc = GetDC(handle);
  dpi = { (int)GetDeviceCaps(hdc, LOGPIXELSX), (int)GetDeviceCaps(hdc, LOGPIXELSY) };
  
  SetWindowLongPtrW(handle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));

  if(dwmblurbehind != 0)
  {
    //MARGINS margins = { -1,-1,-1,-1 };
    //(*dwmextend)(handle, &margins); //extends glass effect
    HRGN region = CreateRectRgn(-1, -1, 0, 0);
    DWM_BLURBEHIND blurbehind = { DWM_BB_ENABLE | DWM_BB_BLURREGION | DWM_BB_TRANSITIONONMAXIMIZED, TRUE, region, FALSE };
    (*dwmblurbehind)(handle, &blurbehind);
  }

  return handle;
}

void fgContext::Draw(HWND handle, fgElement* element, const AbsRect* area)
{
  CreateResources(handle);
  target->BeginDraw();
  target->Clear(D2D1::ColorF(0, 0));
  fgElement* hold = element->root;
  assert(!cliprect.size());

  target->SetTransform(D2D1::Matrix3x2F::Translation(-area->left, -area->top));
  cliprect.push(*area);

  AbsRect curarea;
  fgDrawAuxDataEx exdata = {
    {
      sizeof(fgDrawAuxDataEx),
      element->GetDPI(),
      { 1,1 },
      { 0,0 }
    },
    this,
  };

  fgStandardDraw(element, area, &exdata.data, 0);

  /*fgElement* topmost = fgSingleton()->topmost;
  if(topmost && GetElementWindow(topmost) == self) // Draw topmost before the drag object
  {
    AbsRect out;
    ResolveRect(topmost, &out);
    topmost->Draw(&out, &exdata.data);
  }*/

  cliprect.pop();
  assert(!cliprect.size());
  if(target->EndDraw() == 0x8899000C) // D2DERR_RECREATE_TARGET
    DiscardResources();
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
      hr = target->QueryInterface<ID2D1DeviceContext>(&context);
      if(SUCCEEDED(hr))
      {
        hr = context->CreateEffect(CLSID_fgRoundRect, &roundrect);
        hr = context->CreateEffect(CLSID_fgCircle, &circle);
        hr = context->CreateEffect(CLSID_fgTriangle, &triangle);
      }
      hr = hr;
    }
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

  return fgRoot_Inject(root, &evt);
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
  fgRoot_Inject(root, &evt);
}

void fgContext::SetMouse(tagPOINTS* points, unsigned short type, unsigned char button, size_t wparam, unsigned long time, fgElement* src)
{
  fgRoot* root = fgSingleton();
  FG_Msg evt = { 0 };
  evt.type = type;
  evt.button = button;
  evt.x = root->mouse.x; // Set these up here for any mouse events that don't contain coordinates.
  evt.y = root->mouse.y;

  if(type == FG_MOUSEOFF || type == FG_MOUSEON)
  {
    fgRoot_Inject(root, &evt);
    return;
  }

  if(wparam != (size_t)-1) //if wparam is -1 it signals that it is invalid, so we simply leave our assignments at their last known value.
  {
    uint8_t bt = 0; //we must keep these bools accurate at all times
    bt |= FG_MOUSELBUTTON&(-((wparam&MK_LBUTTON) != 0));
    bt |= FG_MOUSERBUTTON&(-((wparam&MK_RBUTTON) != 0));
    bt |= FG_MOUSEMBUTTON&(-((wparam&MK_MBUTTON) != 0));
    bt |= FG_MOUSEXBUTTON1&(-((wparam&MK_XBUTTON1) != 0));
    bt |= FG_MOUSEXBUTTON2&(-((wparam&MK_XBUTTON2) != 0));
    root->mouse.buttons = bt;
  }

  if(type != FG_MOUSESCROLL) //The WM_MOUSEWHEEL event does not send mousecoord data
  {
    AbsRect out;
    ResolveRect(src, &out);
    fgIntVec dpi = src->GetDPI();
    AbsVec scale = { (!dpi.x) ? 1.0f : (96.0f / (float)dpi.x), (!dpi.y) ? 1.0f : (96.0f / (float)dpi.y) };
    evt.x = points->x * scale.x + out.left;
    evt.y = points->y * scale.y + out.top;
    evt.allbtn = root->mouse.buttons;
    root->mouse.x = evt.x;
    root->mouse.y = evt.y;
  }

  switch(type)
  {
  case FG_MOUSESCROLL:
    evt.scrolldelta = points->y;
    evt.scrollhdelta = points->x;
    break;
  case FG_MOUSEDBLCLICK: //L down
  case FG_MOUSEDOWN: //L down
    evt.allbtn |= evt.button; // Ensure the correct button position is reflected in the allbtn parameter regardless of the current state of the mouse
    break;
  case FG_MOUSEUP: //L up
    evt.allbtn &= ~evt.button;
    break;
  }

  fgRoot_Inject(root, &evt);
}