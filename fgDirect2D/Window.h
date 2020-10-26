// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__CONTEXT_H
#define D2D__CONTEXT_H

#include "CompactArray.h"
#include "khash.h"
#include "backend.h"
#include "win32_includes.h"
#include <d2d1_1.h>
#include <stack>
#include <vector>

//#define TEST_DPI 120

#if defined(_WIN64)
typedef long long longptr_t;
#else
typedef __w64 long longptr_t;
#endif

namespace D2D {
  class Backend;
  struct Window;

  struct Asset : FG_Asset
  {
    mutable CompactArray<Window*> instances;
  };

  KHASH_DECLARE(wic, const Asset*, ID2D1Bitmap*);

  // A window wraps a drawing context and processes messages for it
  struct Window
  {
    HWND__* hWnd;
    ID2D1HwndRenderTarget* target;
    ID2D1DeviceContext* context;
    ID2D1SolidColorBrush* color;
    ID2D1SolidColorBrush* edgecolor;
    ID2D1Effect* roundrect;
    ID2D1Effect* triangle;
    ID2D1Effect* circle;
    ID2D1Effect* scale;
    FG_MsgReceiver* element;
    Backend* backend;
    FG_Rect margin; // When maximized, the actual window area is larger than what we need to draw on, so we must compensate
    std::stack<FG_Asset*> layers;
    std::stack<D2D_MATRIX_3X2_F> transforms;
    struct kh_wic_s* assets;
    FG_Vec dpi; // The DPI should always be per-display, but windows is insane so we must be prepared to override this.

    bool inside;

    Window(Backend* backend, FG_MsgReceiver* element, FG_Vec* pos, FG_Vec* dim, uint64_t flags, const char* caption,
           void* context);
    ~Window();
    void CreateResources();
    void DiscardResources();
    void DiscardAsset(const Asset* p);
    void BeginDraw(const FG_Rect& area);
    void EndDraw();
    void SetCaption(const char* caption);
    void SetArea(FG_Vec* pos, FG_Vec* dim);
    void SetFlags(uint64_t flags);
    size_t SetKey(uint8_t keycode, uint16_t scancode, bool down, bool held, unsigned long time);
    size_t SetChar(int key, unsigned long time);
    size_t SetMouse(const FG_Vec& points, FG_Kind type, unsigned char button, size_t wparam, unsigned long time);
    size_t SetMouseScroll(const FG_Vec& points, uint16_t x, uint16_t y, unsigned long time);
    size_t SetTouch(tagTOUCHINPUT& input);
    void InvalidateHWND(const FG_Rect* area);
    ID2D1Bitmap* GetBitmapFromSource(const Asset* p);
    HWND__* WndCreate(FG_Vec* pos, FG_Vec* dim, unsigned long style, uint32_t exflags, void* self, const wchar_t* cls,
                      const char* caption, FG_Vec& dpi);
    void ApplyWin32Size();
    void Clear(FG_Color color);
    int PushClip(const FG_Rect& area);
    int PopClip();
    void PushTransform(const D2D_MATRIX_3X2_F& m);
    void PopTransform();

    static void WndRegister(longptr_t(__stdcall* f)(HWND__*, unsigned int, size_t, longptr_t), const wchar_t* name);
    static longptr_t __stdcall WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam);
    static uint8_t GetModKey();

    static uint32_t wincount;
  };
}

#endif
