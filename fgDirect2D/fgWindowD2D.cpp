// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgDirect2D.h"
#include "bss-util/bss_win32_includes.h"
#include <dwmapi.h>
#include <d2d1.h>

EXTERN_C IMAGE_DOS_HEADER __ImageBase;
typedef HRESULT(STDAPICALLTYPE *DWMCOMPENABLE)(BOOL*);
typedef HRESULT(STDAPICALLTYPE *DWMBLURBEHIND)(HWND, const DWM_BLURBEHIND*);
DWMBLURBEHIND dwmblurbehind = 0;

void FG_FASTCALL fgWindowD2D_Init(fgWindowD2D* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(self->window, parent, next, name, flags, transform, units, (fgDestroy)&fgWindowD2D_Destroy, (fgMessage)&fgWindowD2D_Message);
}
void FG_FASTCALL fgWindowD2D_Destroy(fgWindowD2D* self)
{
  if(self->target)
    self->target->Release();
  if(self->handle)
    DestroyWindow(self->handle);
  if(self->color)
    self->color->Release();
  if(self->edgecolor)
    self->edgecolor->Release();
  fgWindow_Destroy(&self->window);
}
size_t FG_FASTCALL fgWindowD2D_Message(fgWindowD2D* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_DRAW:
    self->target->BeginDraw();
    self->target->EndDraw();
    return FG_ACCEPT;
  case FG_CONSTRUCT:
    self->color = 0;
    self->edgecolor = 0;
    self->target = 0;
    self->handle = 0;
    if(fgWindow_Message(&self->window, msg))
    {
      AbsRect out;
      ResolveRect(self->window, &out);
      RECT rc = { out.left, out.top, out.right, out.bottom };
      self->handle = self->WndCreate(rc, self->window->flags);

      GetClientRect(self->handle, &rc);
      HRESULT hr = fgDirect2D::instance->factory->CreateHwndRenderTarget(
        D2D1::RenderTargetProperties(),
        D2D1::HwndRenderTargetProperties(self->handle, D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top)),
        &self->target);
      if(SUCCEEDED(hr))
        self->target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &self->color);
      if(SUCCEEDED(hr))
        self->target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &self->edgecolor);
    }
    return FG_ACCEPT;
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
longptr_t __stdcall fgWindowD2D::WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  return DefWindowProc(hWnd, message, wParam, lParam);
}

HWND fgWindowD2D::WndCreate(RECT& rsize, fgFlag flags)
{
  unsigned long style = WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_POPUP;
  if(flags & FGWINDOW_MINIMIZABLE) style |= WS_MINIMIZEBOX;
  if(flags & FGWINDOW_MAXIMIZABLE) style |= WS_MAXIMIZEBOX;
  if(flags & FGWINDOW_RESIZABLE) style |= WS_SIZEBOX;

  AdjustWindowRect(&rsize, style, FALSE);
  int rwidth = rsize.right - rsize.left;
  int rheight = rsize.bottom - rsize.top;

  HWND hWnd = CreateWindowExW(WS_EX_COMPOSITED, L"fgDirect2D", L"", style, CW_USEDEFAULT, CW_USEDEFAULT, INT(rwidth), INT(rheight), NULL, NULL, (HINSTANCE)&__ImageBase, NULL);

  //MARGINS margins = { -1,-1,-1,-1 };
  //(*dwmextend)(hWnd, &margins); //extends glass effect
  HRGN region = CreateRectRgn(-1, -1, 0, 0);
  DWM_BLURBEHIND blurbehind = { DWM_BB_ENABLE | DWM_BB_BLURREGION | DWM_BB_TRANSITIONONMAXIMIZED, TRUE, region, FALSE };
  (*dwmblurbehind)(hWnd, &blurbehind);

  ShowWindow(hWnd, SW_SHOW);
  UpdateWindow(hWnd);
  //SetWindowPos(hWnd, HWND_TOP, INT(wleft), INT(wtop), INT(rwidth), INT(rheight), SWP_NOCOPYBITS | SWP_NOMOVE | SWP_NOACTIVATE);

  SetCursor(LoadCursor(NULL, IDC_ARROW));
  SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
  return hWnd;
}