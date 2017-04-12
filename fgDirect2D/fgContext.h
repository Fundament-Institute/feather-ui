// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_CONTEXT_D2D_H__
#define __FG_CONTEXT_D2D_H__

#include "fgElement.h"
#include <stack>

//#define TEST_DPI 120

struct HWND__;
struct tagRECT;
struct tagPOINTS;
struct ID2D1HwndRenderTarget;
struct ID2D1SolidColorBrush;
struct ID2D1Effect;
struct ID2D1DeviceContext;
struct fgDrawAuxDataEx;

#if defined(_WIN64)
typedef long long longptr_t;
#else
typedef __w64 long longptr_t;
#endif

struct fgContext {
  ID2D1HwndRenderTarget* target;
  ID2D1DeviceContext* context;
  ID2D1SolidColorBrush* color;
  ID2D1SolidColorBrush* edgecolor;
  ID2D1Effect* roundrect;
  ID2D1Effect* triangle;
  ID2D1Effect* circle;
  std::stack<AbsRect> cliprect;
  bool inside;
  bool invalid;

  void CreateResources(HWND__* handle);
  void DiscardResources();
  void BeginDraw(HWND__* handle, fgElement* element, const AbsRect* area, fgDrawAuxDataEx& exdata);
  void EndDraw();
  size_t SetKey(uint8_t keycode, bool down, bool held, unsigned long time);
  void SetChar(int key, unsigned long time);
  void SetMouse(tagPOINTS* points, unsigned short type, unsigned char button, size_t wparam, unsigned long time);
  longptr_t __stdcall WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam, fgElement* src);
  void InvalidateHWND(HWND__* hWnd);

  static HWND__* WndCreate(const AbsRect& out, uint32_t exflags, void* self, const wchar_t* cls, fgIntVec& dpi);
  static void SetDWMCallbacks();
  static void WndRegister(longptr_t(__stdcall* f)(HWND__*, unsigned int, size_t, longptr_t), const wchar_t* name);
  static tagPOINTS* AdjustPoints(tagPOINTS* points, fgElement* src);
  static tagPOINTS* AdjustDPI(tagPOINTS* points, fgElement* src);

  static uint32_t wincount;
};

struct fgDrawAuxDataEx {
  fgDrawAuxData data;
  fgContext* context;
};

void fgContext_Construct(fgContext* self);
void fgContext_Destroy(fgContext* self);

#endif