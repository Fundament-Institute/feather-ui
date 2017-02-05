// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgDirect2D.h"
#include "fgRoundRect.h"
#include "bss-util/bss_win32_includes.h"
#include <dwmapi.h>
#include <d2d1_1.h>

EXTERN_C IMAGE_DOS_HEADER __ImageBase;
typedef HRESULT(STDAPICALLTYPE *DWMCOMPENABLE)(BOOL*);
typedef HRESULT(STDAPICALLTYPE *DWMBLURBEHIND)(HWND, const DWM_BLURBEHIND*);
DWMBLURBEHIND dwmblurbehind = 0;
uint32_t fgWindowD2D::wincount = 0;

void fgWindowD2D_Init(fgWindowD2D* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  ++fgWindowD2D::wincount;
  fgElement_InternalSetup(self->window, parent, next, name, flags, transform, units, (fgDestroy)&fgWindowD2D_Destroy, (fgMessage)&fgWindowD2D_Message);
}
void fgWindowD2D_Destroy(fgWindowD2D* self)
{
  self->cliprect.~stack<AbsRect>();
  self->DiscardResources();

  if(!--fgWindowD2D::wincount)
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
    self->color = 0;
    self->edgecolor = 0;
    self->target = 0;
    self->handle = 0;
    self->dpi.x = 0;
    self->dpi.y = 0;
    self->roundrect = 0;
    self->triangle = 0;
    self->circle = 0;
    self->inside = false;
    new (&self->cliprect) std::stack<AbsRect>();
    return fgWindow_Message(&self->window, msg);
  case FG_PARENTCHANGE:
    if(msg->e != 0 && self->window->parent == &fgSingleton()->gui.element)
      return fgSendMsg<FG_SETPARENT, const void*>(self->window, fgSingleton()->monitors);
    if(!self->window->parent && self->handle != 0)
    {
      self->DiscardResources();
      SetWindowLongPtrW(self->handle, GWLP_USERDATA, 0); // Prevent the WM_DESTROY message from propogating
      DestroyWindow(self->handle);
    }
    else if(self->window->parent != 0 && !self->handle)
    {
      self->WndCreate();
      self->CreateResources();
    }
    break;
  case FG_DRAW:
    self->CreateResources();
    self->target->BeginDraw();
    self->target->Clear(D2D1::ColorF(0, 0));
    {
      fgElement* hold = self->window->root;
      assert(!self->cliprect.size());

      AbsRect area = *(AbsRect*)msg->p;
      self->target->SetTransform(D2D1::Matrix3x2F::Translation(-area.left, -area.top));
      self->cliprect.push(area);

      AbsRect curarea;
      fgDrawAuxDataEx exdata = {
        {
          sizeof(fgDrawAuxDataEx),
          self->dpi,
          { 1,1 },
          { 0,0 }
        },
        self,
      };

      fgStandardDraw(self->window, &area, &exdata.data, 0);

      fgElement* topmost = fgSingleton()->topmost;
      if(topmost && GetElementWindow(topmost) == self) // Draw topmost before the drag object
      {
        AbsRect out;
        ResolveRect(topmost, &out);
        topmost->Draw(&out, &exdata.data);
      }

      self->cliprect.pop();
      assert(!self->cliprect.size());
    }
    if(self->target->EndDraw() == 0x8899000C) // D2DERR_RECREATE_TARGET
      self->DiscardResources();
    return FG_ACCEPT;
  case FG_MOVE:
    if(self->handle && msg->subtype != (uint16_t)-1)
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
      break;
    case FGWINDOW_RESTORE:
    case FGWINDOW_UNMINIMIZE:
      ShowWindow(self->handle, SW_RESTORE);
      break;
    case FGWINDOW_MAXIMIZE:
      ShowWindow(self->handle, SW_MAXIMIZE);
      RECT r;
      if(SUCCEEDED(GetClientRect(self->handle, &r)))
      {
        CRect area = { r.left, 0, r.top, 0, r.right, 0, r.bottom };
        fgSendSubMsg<FG_SETAREA, void*>(self->window, 1, &area);
      }
      break;
    case FGWINDOW_CLOSE:
      VirtualFreeChild(self->window);
      return FG_ACCEPT;
    }
  }
  return fgWindow_Message(&self->window, msg);
}

void fgWindowD2D::WndRegister()
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
   
  // Register window class
  WNDCLASSEX wcex = { sizeof(WNDCLASSEX) };
  wcex.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
  wcex.lpfnWndProc = fgWindowD2D::WndProc;
  wcex.cbClsExtra = 0;
  wcex.cbWndExtra = sizeof(LONG_PTR);
  wcex.hInstance = ((HINSTANCE)&__ImageBase);
  wcex.hbrBackground = NULL;
  wcex.lpszMenuName = NULL;
  wcex.hCursor = LoadCursor(NULL, IDI_APPLICATION);
  wcex.lpszClassName = L"fgDirect2D";

  RegisterClassExW(&wcex);
}

#define MAKELPPOINTS(l)       ((POINTS FAR *)&(l))

longptr_t __stdcall fgWindowD2D::WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  static tagTRACKMOUSEEVENT _trackingstruct = { sizeof(tagTRACKMOUSEEVENT), TME_LEAVE, 0, 0 };
  fgWindowD2D* self = reinterpret_cast<fgWindowD2D*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  if(self)
  {
    switch(message)
    {
    case WM_SIZE:
      if(self->target)
        self->target->Resize(D2D1::SizeU(LOWORD(lParam), HIWORD(lParam)));
      return 0;
    case WM_DISPLAYCHANGE:
      InvalidateRect(self->handle, NULL, FALSE);
      return 0;
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
    case WM_MOUSEWHEEL:
    {
      POINTS pointstemp = { 0 };
      pointstemp.y = GET_WHEEL_DELTA_WPARAM(wParam);
      self->SetMouse(&pointstemp, FG_MOUSESCROLL, 0, wParam, GetMessageTime());
    }
    break;
    case 0x020E: // WM_MOUSEHWHEEL - this is only sent on vista machines, but our minimum version is XP, so we manually look for the message anyway and process it if it happens to get sent to us.
    {
      POINTS pointstemp = { 0 };
      pointstemp.x = GET_WHEEL_DELTA_WPARAM(wParam);
      self->SetMouse(&pointstemp, FG_MOUSESCROLL, 0, wParam, GetMessageTime());
    }
    break;
    case WM_MOUSEMOVE:
      if(!self->inside)
      {
        _trackingstruct.hwndTrack = hWnd;
        BOOL result = TrackMouseEvent(&_trackingstruct);
        self->inside = true;
        self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEON, 0, wParam, GetMessageTime());
      }
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEMOVE, 0, wParam, GetMessageTime());
      break;
    case WM_LBUTTONDOWN:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDOWN, FG_MOUSELBUTTON, wParam, GetMessageTime());
      SetCapture(hWnd);
      break;
    case WM_LBUTTONUP:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEUP, FG_MOUSELBUTTON, wParam, GetMessageTime());
      ReleaseCapture();
      break;
    case WM_LBUTTONDBLCLK:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDBLCLICK, FG_MOUSELBUTTON, wParam, GetMessageTime());
      break;
    case WM_RBUTTONDOWN:
      SetCapture(hWnd);
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDOWN, FG_MOUSERBUTTON, wParam, GetMessageTime());
      break;
    case WM_RBUTTONUP:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEUP, FG_MOUSERBUTTON, wParam, GetMessageTime());
      ReleaseCapture();
      break;
    case WM_RBUTTONDBLCLK:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDBLCLICK, FG_MOUSERBUTTON, wParam, GetMessageTime());
      break;
    case WM_MBUTTONDOWN:
      SetCapture(hWnd);
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDOWN, FG_MOUSEMBUTTON, wParam, GetMessageTime());
      break;
    case WM_MBUTTONUP:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEUP, FG_MOUSEMBUTTON, wParam, GetMessageTime());
      ReleaseCapture();
      break;
    case WM_MBUTTONDBLCLK:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDBLCLICK, FG_MOUSEMBUTTON, wParam, GetMessageTime());
      break;
    case WM_XBUTTONDOWN:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDOWN, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime());
      break;
    case WM_XBUTTONUP:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEUP, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime());
      break;
    case WM_XBUTTONDBLCLK:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEDBLCLICK, (GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? FG_MOUSEXBUTTON1 : FG_MOUSEXBUTTON2), wParam, GetMessageTime());
      break;
    case WM_SYSKEYUP:
    case WM_SYSKEYDOWN:
    case WM_KEYUP:
    case WM_KEYDOWN: // Windows return codes are the opposite of feathergui's - returning 0 means we accept, anything else rejects, so we invert the return code here.
      return !self->SetKey((uint8_t)wParam, message == WM_KEYDOWN || message == WM_SYSKEYDOWN, (lParam & 0x40000000) != 0, GetMessageTime());
    case WM_UNICHAR:
      if(wParam == UNICODE_NOCHAR) return TRUE;
    case WM_CHAR:
      self->SetChar((int)wParam, GetMessageTime());
      return 0;
    case WM_SYSCOMMAND:
      if((wParam & 0xFFF0) == SC_SCREENSAVE || (wParam & 0xFFF0) == SC_MONITORPOWER)
        return 0; //No screensavers!
      break;
    case WM_MOUSELEAVE:
      self->SetMouse(MAKELPPOINTS(lParam), FG_MOUSEOFF, 0, 0, GetMessageTime());
      self->inside = false;
      break;
    //case WM_WINDOWPOSCHANGING:
    //{
    //  WINDOWPOS* pos = (WINDOWPOS*)lParam;
    //  CRect area = { pos->x, 0, pos->y, 0, pos->x + pos->cx, 0, pos->y + pos->cy, 0 };
    //  fgSendSubMsg<FG_SETAREA, void*>(self->window, 1, &area);
    //  break;
    //}
    }
  }

  return DefWindowProc(hWnd, message, wParam, lParam);
}

void fgWindowD2D::WndCreate()
{
  unsigned long style = WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;

  AbsRect out;
  ResolveRect(window, &out);
  fgScaleRectDPI(&out, 96, 96);
  RECT rsize = { out.left, out.top, out.right, out.bottom };

  AdjustWindowRect(&rsize, style, FALSE);
  int rwidth = rsize.right - rsize.left;
  int rheight = rsize.bottom - rsize.top;
  
  handle = CreateWindowExW(WS_EX_COMPOSITED, L"fgDirect2D", L"", style, rsize.left, rsize.top, INT(rwidth), INT(rheight), NULL, NULL, (HINSTANCE)&__ImageBase, NULL);
  HDC hdc = GetDC(handle);
  UINT xdpi = (UINT)GetDeviceCaps(hdc, LOGPIXELSX);
  UINT ydpi = (UINT)GetDeviceCaps(hdc, LOGPIXELSY);
  dpi.x = xdpi;
  dpi.y = ydpi;
  
  SetWindowLongPtrW(handle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

  //MARGINS margins = { -1,-1,-1,-1 };
  //(*dwmextend)(handle, &margins); //extends glass effect
  HRGN region = CreateRectRgn(-1, -1, 0, 0);
  DWM_BLURBEHIND blurbehind = { DWM_BB_ENABLE | DWM_BB_BLURREGION | DWM_BB_TRANSITIONONMAXIMIZED, TRUE, region, FALSE };
  (*dwmblurbehind)(handle, &blurbehind);

  ShowWindow(handle, SW_SHOW);
  UpdateWindow(handle);
  //SetWindowPos(handle, HWND_TOP, INT(wleft), INT(wtop), INT(rwidth), INT(rheight), SWP_NOCOPYBITS | SWP_NOMOVE | SWP_NOACTIVATE);

  SetCursor(LoadCursor(NULL, IDC_ARROW));
}

void fgWindowD2D::CreateResources()
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
        hr = context->CreateEffect(CLSID_fgRoundRect, &roundrect);
      hr = hr;
    }
  }
}
void fgWindowD2D::DiscardResources()
{
  if(color)
    color->Release();
  if(edgecolor)
    edgecolor->Release();
  if(target)
    target->Release();
  if(roundrect)
    roundrect->Release();
  if(triangle)
    triangle->Release();
  if(circle)
    circle->Release();
  color = 0;
  edgecolor = 0;
  target = 0;
  context = 0;
  roundrect = 0;
  triangle = 0;
  circle = 0;
}

size_t fgWindowD2D::SetKey(uint8_t keycode, bool down, bool held, unsigned long time)
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

void fgWindowD2D::SetChar(int key, unsigned long time)
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

void fgWindowD2D::SetMouse(tagPOINTS* points, unsigned short type, unsigned char button, size_t wparam, unsigned long time)
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
    ResolveRect(window, &out);
    fgIntVec dpi = window->GetDPI();
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