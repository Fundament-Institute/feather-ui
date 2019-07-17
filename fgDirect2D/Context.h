// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__CONTEXT_H
#define D2D__CONTEXT_H

#include "feather/document.h"
#include "feather/root.h"
#include "CompactArray.h"
#include "Display.h"
#include <stack>
#include <vector>

//#define TEST_DPI 120

struct HWND__;
struct tagRECT;
struct tagPOINTS;
struct ID2D1HwndRenderTarget;
struct ID2D1SolidColorBrush;
struct ID2D1Effect;
struct ID2D1DeviceContext;
struct ID2D1Bitmap;
struct ID2D1Layer;
struct FG__OUTLINE_NODE;

#define LOGEMPTY
#define LOGFAILURE(x, f, ...) { HRESULT hr = (x); if(FAILED(hr)) { fgLog(instance->root, FGLOG_ERROR, f, __VA_ARGS__); } }
#define LOGFAILURERET(x, r, f, ...) { HRESULT hr = (x); if(FAILED(hr)) { fgLog(instance->root, FGLOG_ERROR, f, __VA_ARGS__); return r; } }
#define LOGFAILURERETNULL(x, f, ...) LOGFAILURERET(x,LOGEMPTY,f,__VA_ARGS__)

#if defined(_WIN64)
typedef long long longptr_t;
#else
typedef __w64 long longptr_t;
#endif

namespace D2D {
  struct Direct2D;
  struct Context;

  struct fgAssetD2D
  {
    fgAsset asset;
    CompactArray<Context*> instances;
  };

  KHASH_DECLARE(wic, fgAssetD2D*, ID2D1Bitmap*);

  struct Context
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
    fgDocumentNode* node;
    Direct2D* backend;
    fgRect margin; // When maximized, the actual window area is larger than the area we intend to draw on, so we must compensate
    std::stack<ID2D1Layer*> layers;
    struct kh_wic_s* assets;
    fgVec dpi; // The DPI should always be per-display, but windows is insane so we must be prepared to override this.
    struct FG__OUTLINE_NODE* display;

    bool inside;
    bool invalid;

    Context(const fgRoot* root, fgDocumentNode* node, struct FG__OUTLINE_NODE* display);
    ~Context();
    void CreateResources(HWND__* handle);
    void DiscardResources();
    void BeginDraw(HWND__* handle, const fgRect& area);
    void EndDraw();
    size_t SetKey(uint8_t keycode, bool down, bool held, unsigned long time);
    size_t SetChar(int key, unsigned long time);
    size_t SetMouse(fgVec& points, fgMsgType type, unsigned char button, size_t wparam, unsigned long time);
    void InvalidateHWND(HWND__* hWnd);
    ID2D1Bitmap* GetBitmapFromSource(fgAssetD2D* p);
    HWND__* WndCreate(const fgRect& out, unsigned long style, uint32_t exflags, void* self, const wchar_t* cls, fgVec& dpi);
    void ApplyWin32Size(HWND__* handle);
    fgError PushClip(fgRect area);
    fgError PopClip();

    static void WndRegister(longptr_t(__stdcall* f)(HWND__*, unsigned int, size_t, longptr_t), const wchar_t* name);
    static longptr_t __stdcall WndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam);
    static longptr_t __stdcall TopWndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam);
    
    static uint32_t wincount;
  };
}

#endif
