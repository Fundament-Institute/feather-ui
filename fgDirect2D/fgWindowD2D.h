// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_WINDOW_D2D_H__
#define __FG_WINDOW_D2D_H__

#include "fgWindow.h"
#include <stack>

struct HWND__;
struct tagRECT;
struct tagPOINTS;
struct ID2D1HwndRenderTarget;
struct ID2D1SolidColorBrush;

#if defined(_WIN64)
typedef long long longptr_t;
#else
typedef __w64 long longptr_t;
#endif

struct fgWindowD2D {
  fgWindow window;
  HWND__* handle;
  ID2D1HwndRenderTarget* target;
  ID2D1SolidColorBrush* color;
  ID2D1SolidColorBrush* edgecolor;
  fgIntVec dpi;
  std::stack<AbsRect> cliprect;
  bool inside;

  void WndCreate();
  void CreateResources();
  void DiscardResources();
  size_t SetKey(uint8_t keycode, bool down, bool held, unsigned long time);
  void SetChar(int key, unsigned long time);
  void SetMouse(tagPOINTS* points, unsigned short type, unsigned char button, size_t wparam, unsigned long time);

  static void WndRegister();
  static longptr_t __stdcall WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam);

  static uint32_t wincount;
};

struct fgDrawAuxDataEx {
  fgDrawAuxData data;
  fgWindowD2D* window;
};


#ifdef  __cplusplus
extern "C" {
#endif

FG_EXTERN void fgWindowD2D_Init(fgWindowD2D* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgWindowD2D_Destroy(fgWindowD2D* self);
FG_EXTERN size_t fgWindowD2D_Message(fgWindowD2D* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif