// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgDirect2D.h"
#include "fgMonitor.h"
#include "fgResource.h"
#include "fgText.h"
#include "bss-util/bss_defines.h"
#include "bss-util/bss_win32_includes.h"
#include <d2d1.h>
#include <stdint.h>
#include <wincodec.h>
#include <dwrite.h>
#include <malloc.h>

#define GETEXDATA(data) assert(data->fgSZ == sizeof(fgDrawAuxDataEx)); if(data->fgSZ != sizeof(fgDrawAuxDataEx)) return; fgDrawAuxDataEx* exdata = (fgDrawAuxDataEx*)data;

fgDirect2D* fgDirect2D::instance = 0;

BOOL __stdcall SpawnMonitorsProc(HMONITOR monitor, HDC hdc, LPRECT, LPARAM lparam)
{
  static fgMonitor* prev = 0;
  fgDirect2D* root = reinterpret_cast<fgDirect2D*>(lparam);
  MONITORINFO info = { sizeof(MONITORINFO), 0 };
  GetMonitorInfo(monitor, &info);
  UINT xdpi = (UINT)GetDeviceCaps(hdc, LOGPIXELSX);
  UINT ydpi = (UINT)GetDeviceCaps(hdc, LOGPIXELSY);
  //GetDpiForMonitor(0, MDT_Effective_DPI, &xdpi, &ydpi);

  AbsRect area = { info.rcMonitor.left, info.rcMonitor.top, info.rcMonitor.right, info.rcMonitor.bottom };

  fgMonitor* cur = reinterpret_cast<fgMonitor*>(calloc(1, sizeof(fgMonitor)));
  fgIntVec dpi = { xdpi, ydpi };
  fgMonitor_Init(cur, 0, &root->root, prev, &area, &dpi);

  return TRUE;
}

void FG_FASTCALL fgTerminateD2D()
{
  PostQuitMessage(0);
  fgDirect2D* d2d = fgDirect2D::instance;
  assert(d2d);
  VirtualFreeChild(d2d->root.gui);
  if(d2d->factory)
    d2d->factory->Release();
  if(d2d->wicfactory)
    d2d->wicfactory->Release();
  if(d2d->writefactory)
    d2d->writefactory->Release();
  CoUninitialize();

  fgDirect2D::instance = 0;
}

fgWindowD2D* GetElementWindow(fgElement* cur)
{
  while(cur && cur->destroy != (fgDestroy)fgWindowD2D_Destroy) cur = cur->parent;
  return (fgWindowD2D*)cur;
}

inline D2D1_COLOR_F ToD2Color(unsigned int color)
{
  fgColor c = { color };
  return D2D1::ColorF((c.r << 16) | (c.g << 8) | c.b, c.a / 255.0f);
}

template<class T, void (MSC_FASTCALL *GCC_FASTCALL INIT)(T* BSS_RESTRICT, fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, const char*, fgFlag, const fgTransform*, unsigned short)>
BSS_FORCEINLINE fgElement* d2d_create_default(fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units, const char* file, size_t line)
{
  T* r = (T*)malloc(sizeof(T));
  INIT(r, parent, next, name, flags, transform, units);
  ((fgElement*)r)->free = &free;
  return (fgElement*)r;
}

fgElement* FG_FASTCALL fgCreateD2D(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  if(!_stricmp(type, "window"))
    return d2d_create_default<fgWindowD2D, fgWindowD2D_Init>(parent, next, name, flags, transform, units, __FILE__, __LINE__);
  return fgCreateDefault(type, parent, next, name, flags, transform, units);
}

void* FG_FASTCALL fgCreateFontD2D(fgFlag flags, const char* font, uint32_t fontsize, const fgIntVec* dpi)
{
  size_t len = fgUTF8toUTF16(font, -1, 0, 0);
  DYNARRAY(wchar_t, wtext, len);
  fgUTF8toUTF16(font, -1, wtext, len);
  IDWriteTextFormat* format = 0;
  fgDirect2D::instance->writefactory->CreateTextFormat(wtext, 0, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, fontsize, L"en-us", &format);
  return format;
}

void* FG_FASTCALL fgCloneFontD2D(void* font, const struct _FG_FONT_DESC* desc)
{
  IDWriteTextFormat* f = (IDWriteTextFormat*)font;
  if(!desc || desc->pt == f->GetFontSize())
    f->AddRef();
  else
  {
    DYNARRAY(wchar_t, wtext, f->GetFontFamilyNameLength()+1);
    f->GetFontFamilyName(wtext, f->GetFontFamilyNameLength() + 1);
    fgDirect2D::instance->writefactory->CreateTextFormat(wtext, 0, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, desc->pt, L"en-us", &f);
  }
  return f;
}
void FG_FASTCALL fgDestroyFontD2D(void* font) { ((IDWriteTextFormat*)font)->Release(); }
void FG_FASTCALL fgDrawFontD2D(void* font, const int* text, size_t srclen, float lineheight, float letterspacing, unsigned int color, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data, void* cache)
{
  GETEXDATA(data);
  IDWriteTextFormat* format = (IDWriteTextFormat*)font;
  IDWriteTextLayout* layout = (IDWriteTextLayout*)cache;
  
  exdata->window->color->SetColor(ToD2Color(color));
  size_t len = fgUTF32toUTF16(text, srclen, 0, 0);
  DYNARRAY(wchar_t, wtext, len);
  fgUTF32toUTF16(text, srclen, wtext, len);
  
  if(!layout)
    exdata->window->target->DrawTextW(wtext, len, format, D2D1::RectF(area->left, area->top, area->right, area->bottom), exdata->window->color, D2D1_DRAW_TEXT_OPTIONS_CLIP);
  else
    exdata->window->target->DrawTextLayout(D2D1::Point2F(area->left, area->top), layout, exdata->window->color, D2D1_DRAW_TEXT_OPTIONS_CLIP);
}
IDWriteTextLayout* CreateD2DLayout(IDWriteTextFormat* format, const int* text, size_t srclen, const AbsRect* area)
{
  size_t len = fgUTF32toUTF16(text, srclen, 0, 0);
  DYNARRAY(wchar_t, wtext, len);
  fgUTF32toUTF16(text, srclen, wtext, len);
  IDWriteTextLayout* layout = 0;
  FABS x = area->right - area->left;
  FABS y = area->bottom - area->top;
  fgDirect2D::instance->writefactory->CreateTextLayout(wtext, len, format, x <= 0.0f ? INFINITY : x, y <= 0.0f ? INFINITY : y, &layout);
  return layout;
}
void* FG_FASTCALL fgFontFormatD2D(void* font, const int* text, size_t srclen, float lineheight, float letterspacing, AbsRect* area, fgFlag flags, void* cache)
{
  IDWriteTextLayout* layout = (IDWriteTextLayout*)cache;
  if(layout)
    layout->Release();
  layout = CreateD2DLayout((IDWriteTextFormat*)font, text, srclen, area);
  if(!layout) return 0;
  FLOAT linespacing;
  FLOAT baseline;
  DWRITE_LINE_SPACING_METHOD method;
  layout->GetLineSpacing(&method, &linespacing, &baseline);
  layout->SetLineSpacing(DWRITE_LINE_SPACING_METHOD_UNIFORM, lineheight, baseline * (lineheight / linespacing));
  layout->SetWordWrapping((flags&(FGTEXT_CHARWRAP | FGTEXT_WORDWRAP)) ? DWRITE_WORD_WRAPPING_WRAP : DWRITE_WORD_WRAPPING_NO_WRAP);
  DWRITE_TEXT_METRICS metrics;
  layout->GetMetrics(&metrics);
  area->right = area->left + metrics.width;
  area->bottom = area->top + metrics.height;
  return layout;
}
void FG_FASTCALL fgFontGetD2D(void* font, struct _FG_FONT_DESC* desc)
{
  IDWriteTextFormat* format = (IDWriteTextFormat*)font;
  if(desc)
  {
    FLOAT lineheight;
    FLOAT baseline;
    DWRITE_LINE_SPACING_METHOD method;
    format->GetLineSpacing(&method, &lineheight, &baseline);
    desc->lineheight = lineheight;
    desc->pt = format->GetFontSize();
    desc->dpi = { 0,0 };
  }
}
size_t FG_FASTCALL fgFontIndexD2D(void* font, const int* text, size_t srclen, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, AbsVec pos, AbsVec* cursor, void* cache)
{
  IDWriteTextLayout* layout = (IDWriteTextLayout*)cache;
  if(!layout)
    layout = CreateD2DLayout((IDWriteTextFormat*)font, text, srclen, area);
  if(!layout)
    return 0;

  BOOL trailing;
  BOOL inside;
  DWRITE_HIT_TEST_METRICS hit;
  layout->HitTestPoint(pos.x, pos.y, &trailing, &inside, &hit);

  cursor->x = hit.left;
  cursor->y = hit.top;
  return hit.textPosition;
}
AbsVec FG_FASTCALL fgFontPosD2D(void* font, const int* text, size_t srclen, float lineheight, float letterspacing, const AbsRect* area, fgFlag flags, size_t index, void* cache)
{
  IDWriteTextLayout* layout = (IDWriteTextLayout*)cache;
  if(!layout)
    layout = CreateD2DLayout((IDWriteTextFormat*)font, text, srclen, area);
  if(!layout)
    return AbsVec{ 0,0 };

  FLOAT x, y;
  DWRITE_HIT_TEST_METRICS hit;
  layout->HitTestTextPosition(index, false, &x, &y, &hit);
  return AbsVec{ x, y };
}

void* FG_FASTCALL fgCreateResourceD2D(fgFlag flags, const char* data, size_t length)
{
  IWICBitmapDecoder *decoder = NULL;
  IWICBitmapFrameDecode *source = NULL;
  IWICStream *stream = NULL;

  HRESULT hr = fgDirect2D::instance->wicfactory->CreateStream(&stream);
  if(SUCCEEDED(hr))
    stream->InitializeFromMemory((BYTE*)data, length);
  if(SUCCEEDED(hr))
    hr = fgDirect2D::instance->wicfactory->CreateDecoderFromStream(stream, NULL, WICDecodeMetadataCacheOnLoad, &decoder);
  if(SUCCEEDED(hr))
    hr = decoder->GetFrame(0, &source);

  if(stream) stream->Release();
  if(decoder) decoder->Release();
  if(source) source->Release();
  return source;

  //if(data[0] == 0xFF && data[1] == 0xD8) // JPEG SOI header
  //else if(data[0] == 'B' && data[1] == 'M') // BMP header
  //else if(data[0] == 137 && data[1] == 80 && data[2] == 78 && data[3] == 71) // PNG file signature
  //else if(data[0] == 'G' && data[1] == 'I' && data[2] == 'F') // GIF header
  //else if((data[0] == 'I' && data[1] == 'I' && data[2] == '*' && data[3] == 0) || (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && data[3] == '*')) // TIFF header
}
void* FG_FASTCALL fgCloneResourceD2D(void* res, fgElement* src)
{ 
  fgWindowD2D* window = GetElementWindow(src);
  if(!window)
  {
    ((IUnknown*)res)->AddRef();
    return res;
  }
  assert(window->window->destroy == (fgDestroy)fgWindowD2D_Destroy);
  ID2D1HwndRenderTarget* target = window->target;

  IWICFormatConverter *conv = NULL;
  ID2D1Bitmap* bitmap = NULL;
  HRESULT hr = fgDirect2D::instance->wicfactory->CreateFormatConverter(&conv);

  if(SUCCEEDED(hr)) // Convert the image format to 32bppPBGRA (DXGI_FORMAT_B8G8R8A8_UNORM + D2D1_ALPHA_MODE_PREMULTIPLIED).
    hr = conv->Initialize((IWICBitmapSource*)res, GUID_WICPixelFormat32bppPBGRA, WICBitmapDitherTypeNone, NULL, 0.f, WICBitmapPaletteTypeMedianCut);
  if(SUCCEEDED(hr))
    hr = target->CreateBitmapFromWicBitmap(conv, NULL, &bitmap);

  return bitmap;
}
void FG_FASTCALL fgDestroyResourceD2D(void* res) { ((IUnknown*)res)->Release(); }
void FG_FASTCALL fgDrawResourceD2D(void* res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags, const fgDrawAuxData* data)
{
  GETEXDATA(data);
  assert(exdata->window != 0);
  assert(exdata->window->target != 0);

  ID2D1Bitmap* tex = 0;
  if(res)
  {
    ((IUnknown*)res)->QueryInterface<ID2D1Bitmap>(&tex);
    assert(tex);
  }

  D2D1_RECT_F uvresolve;
  if(tex != 0)
  {
    auto sz = tex->GetPixelSize();
    uvresolve = D2D1::RectF(uv->left.rel + (uv->left.abs / sz.width),
      uv->top.rel + (uv->top.abs / sz.height),
      uv->right.rel + (uv->right.abs / sz.width),
      uv->bottom.rel + (uv->bottom.abs / sz.height));
  }
  else
    uvresolve = D2D1::RectF(uv->left.abs, uv->top.abs, uv->right.abs, uv->bottom.abs );

  //psRectRotate rect(area->left, area->top, area->right, area->bottom, rotation, psVec(center->x - area->left, center->y - area->top));
  D2D1_RECT_F rect = D2D1::RectF(area->left, area->top, area->right, area->bottom);
  exdata->window->color->SetColor(ToD2Color(color));
  exdata->window->edgecolor->SetColor(ToD2Color(edge));

  if((flags&FGRESOURCE_SHAPEMASK) == FGRESOURCE_ROUNDRECT)
  {
    auto rrect = D2D1::RoundedRect(rect, uvresolve.left, uvresolve.top);
    exdata->window->target->FillRoundedRectangle(rrect, exdata->window->color);
    exdata->window->target->DrawRoundedRectangle(rrect, exdata->window->edgecolor, outline);
  }
  else if((flags&FGRESOURCE_SHAPEMASK) == FGRESOURCE_CIRCLE) {}
  //  psRenderCircle::DrawCircle(driver->library.CIRCLE, STATEBLOCK_LIBRARY::PREMULTIPLIED, rect, uvresolve, 0, psColor32(color), psColor32(edge), outline);
  else if((flags&FGRESOURCE_SHAPEMASK) == FGRESOURCE_TRIANGLE) {}
  //  psRoundTri::DrawRoundTri(driver->library.ROUNDTRI, STATEBLOCK_LIBRARY::PREMULTIPLIED, rect, uvresolve, 0, psColor32(color), psColor32(edge), outline);
  else
    exdata->window->target->DrawBitmap(tex, rect, (color >> 24) / 255.0f, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, uvresolve);
}

void FG_FASTCALL fgResourceSizeD2D(void* res, const CRect* uv, AbsVec* dim, fgFlag flags)
{
  ID2D1Bitmap* tex = (ID2D1Bitmap*)res;
  auto sz = tex->GetPixelSize();
  D2D1_RECT_F uvresolve = D2D1::RectF(uv->left.rel + (uv->left.abs / sz.width),
    uv->top.rel + (uv->top.abs / sz.height),
    uv->right.rel + (uv->right.abs / sz.width),
    uv->bottom.rel + (uv->bottom.abs / sz.height));
  dim->x = uvresolve.right - uvresolve.left;
  dim->y = uvresolve.bottom - uvresolve.top;
}

void FG_FASTCALL fgDrawLinesD2D(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center, const fgDrawAuxData* data)
{
  GETEXDATA(data);
  exdata->window->color->SetColor(ToD2Color(color));
  for(size_t i = 1; i < n; ++i)
    exdata->window->target->DrawLine(D2D1_POINT_2F{ p[i - 1].x, p[i - 1].y }, D2D1_POINT_2F{ p[i].x, p[i].y }, exdata->window->color, 1.0F, 0);
  //bss_util::Matrix<float, 4, 4>::AffineTransform_T(translate->x, translate->y, 0, rotation, center->x, center->y, m);
}

void FG_FASTCALL fgPushClipRectD2D(const AbsRect* clip, const fgDrawAuxData* data)
{
  GETEXDATA(data);
  AbsRect cliprect = exdata->window->cliprect.top();
  cliprect = { bssmax(clip->left, cliprect.left), bssmax(clip->top, cliprect.top), bssmin(clip->right, cliprect.right), bssmin(clip->bottom, cliprect.bottom) };
  exdata->window->cliprect.push(cliprect);
  exdata->window->target->PushAxisAlignedClip(D2D1::RectF(cliprect.left, cliprect.top, cliprect.right, cliprect.bottom), D2D1_ANTIALIAS_MODE_ALIASED);
}

AbsRect FG_FASTCALL fgPeekClipRectD2D(const fgDrawAuxData* data)
{
  if(data->fgSZ != sizeof(fgDrawAuxDataEx)) return AbsRect{ 0,0,0,0 };
  fgDrawAuxDataEx* exdata = (fgDrawAuxDataEx*)data;
  return exdata->window->cliprect.top();
}

void FG_FASTCALL fgPopClipRectD2D(const fgDrawAuxData* data)
{
  GETEXDATA(data);
  exdata->window->cliprect.pop();
  exdata->window->target->PopAxisAlignedClip();
}

void FG_FASTCALL fgDirtyElementD2D(fgElement* e)
{
  fgWindowD2D* window = GetElementWindow(e);
  if(window)
    InvalidateRect(window->handle, NULL, FALSE);
}

void FG_FASTCALL fgSetCursorD2D(uint32_t type, void* custom)
{
  static HCURSOR hArrow = LoadCursor(NULL, IDC_ARROW);
  static HCURSOR hIBeam = LoadCursor(NULL, IDC_IBEAM);
  static HCURSOR hCross = LoadCursor(NULL, IDC_CROSS);
  static HCURSOR hWait = LoadCursor(NULL, IDC_WAIT);
  static HCURSOR hHand = LoadCursor(NULL, IDC_HAND);
  static HCURSOR hSizeNS = LoadCursor(NULL, IDC_SIZENS);
  static HCURSOR hSizeWE = LoadCursor(NULL, IDC_SIZEWE);
  static HCURSOR hSizeNWSE = LoadCursor(NULL, IDC_SIZENWSE);
  static HCURSOR hSizeNESW = LoadCursor(NULL, IDC_SIZENESW);
  static HCURSOR hSizeAll = LoadCursor(NULL, IDC_SIZEALL);
  static HCURSOR hNo = LoadCursor(NULL, IDC_NO);
  static HCURSOR hHelp = LoadCursor(NULL, IDC_HELP);
  static HCURSOR hDrag = hSizeAll;

  switch(type)
  {
  case FGCURSOR_ARROW: SetCursor(hArrow); break;
  case FGCURSOR_IBEAM: SetCursor(hIBeam); break;
  case FGCURSOR_CROSS: SetCursor(hCross); break;
  case FGCURSOR_WAIT: SetCursor(hWait); break;
  case FGCURSOR_HAND: SetCursor(hHand); break;
  case FGCURSOR_RESIZENS: SetCursor(hSizeNS); break;
  case FGCURSOR_RESIZEWE: SetCursor(hSizeWE); break;
  case FGCURSOR_RESIZENWSE: SetCursor(hSizeNWSE); break;
  case FGCURSOR_RESIZENESW: SetCursor(hSizeNESW); break;
  case FGCURSOR_RESIZEALL: SetCursor(hSizeAll); break;
  case FGCURSOR_NO: SetCursor(hNo); break;
  case FGCURSOR_HELP: SetCursor(hHelp); break;
  case FGCURSOR_DRAG: SetCursor(hDrag); break;
  }
}

void FG_FASTCALL fgClipboardCopyD2D(uint32_t type, const void* data, size_t length)
{
  OpenClipboard(GetActiveWindow());
  if(data != 0 && length > 0 && EmptyClipboard())
  {
    if(type == FGCLIPBOARD_TEXT)
    {
      length /= sizeof(int);
      size_t len = fgUTF32toUTF8((const int*)data, length, 0, 0);
      size_t unilen = fgUTF32toUTF16((const int*)data, length, 0, 0);
      HGLOBAL unimem = GlobalAlloc(GMEM_MOVEABLE, unilen * sizeof(wchar_t));
      if(unimem)
      {
        wchar_t* uni = (wchar_t*)GlobalLock(unimem);
        size_t sz = fgUTF32toUTF16((const int*)data, length, uni, unilen);
        if(sz < unilen) // ensure we have a null terminator
          uni[sz] = 0;
        GlobalUnlock(unimem);
        SetClipboardData(CF_UNICODETEXT, unimem);
      }
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, len);
      if(gmem)
      {
        char* mem = (char*)GlobalLock(gmem);
        size_t sz = fgUTF32toUTF8((const int*)data, length, mem, len);
        if(sz < len)
          mem[sz] = 0;
        GlobalUnlock(gmem);
        SetClipboardData(CF_TEXT, gmem);
      }
    }
    else
    {
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, length);
      if(gmem)
      {
        void* mem = GlobalLock(gmem);
        MEMCPY(mem, length, data, length);
        GlobalUnlock(gmem);
        UINT format = CF_PRIVATEFIRST;
        switch(type)
        {
        case FGCLIPBOARD_WAVE: format = CF_WAVE; break;
        case FGCLIPBOARD_BITMAP: format = CF_BITMAP; break;
        }
        SetClipboardData(format, gmem);
      }
    }
  }
  CloseClipboard();
}

char FG_FASTCALL fgClipboardExistsD2D(uint32_t type)
{
  switch(type)
  {
  case FGCLIPBOARD_TEXT:
    return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT);
  case FGCLIPBOARD_WAVE:
    return IsClipboardFormatAvailable(CF_WAVE);
  case FGCLIPBOARD_BITMAP:
    return IsClipboardFormatAvailable(CF_BITMAP);
  case FGCLIPBOARD_CUSTOM:
    return IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  case FGCLIPBOARD_ALL:
    return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT) | IsClipboardFormatAvailable(CF_WAVE) | IsClipboardFormatAvailable(CF_BITMAP) | IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  }
  return 0;
}

const void* FG_FASTCALL fgClipboardPasteD2D(uint32_t type, size_t* length)
{
  OpenClipboard(GetActiveWindow());
  UINT format = CF_PRIVATEFIRST;
  switch(type)
  {
  case FGCLIPBOARD_TEXT:
    if(IsClipboardFormatAvailable(CF_UNICODETEXT))
    {
      HANDLE gdata = GetClipboardData(CF_UNICODETEXT);
      const wchar_t* str = (const wchar_t*)GlobalLock(gdata);
      SIZE_T size = GlobalSize(gdata) / 2;
      SIZE_T len = fgUTF16toUTF32(str, size, 0, 0);
      int* ret = (int*)malloc(len * sizeof(int));
      *length = fgUTF16toUTF32(str, size, ret, len);
      GlobalUnlock(gdata);
      CloseClipboard();
      return ret;
    }
    {
      HANDLE gdata = GetClipboardData(CF_TEXT);
      const char* str = (const char*)GlobalLock(gdata);
      SIZE_T size = GlobalSize(gdata);
      SIZE_T len = fgUTF8toUTF32(str, size, 0, 0);
      int* ret = (int*)malloc(len*sizeof(int));
      *length = fgUTF8toUTF32(str, size, ret, len);
      GlobalUnlock(gdata);
      CloseClipboard();
      return ret;
    }
    return 0;
  case FGCLIPBOARD_WAVE: format = CF_WAVE; break;
  case FGCLIPBOARD_BITMAP: format = CF_BITMAP; break;
  }
  HANDLE gdata = GetClipboardData(format);
  void* data = GlobalLock(gdata);
  *length = GlobalSize(gdata);
  void* ret = malloc(*length);
  MEMCPY(ret, *length, data, *length);
  GlobalUnlock(gdata);
  CloseClipboard();
  return ret;
}

void FG_FASTCALL fgClipboardFreeD2D(const void* mem)
{
  free(const_cast<void*>(mem));
}

void FG_FASTCALL fgDragStartD2D(char type, void* data, fgElement* draw)
{
  fgRoot* root = fgSingleton();
  root->dragtype = type;
  root->dragdata = data;
  root->dragdraw = draw;
}

char FG_FASTCALL fgProcessMessagesDefault()
{
  MSG msg;

  while(PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
  {
    LRESULT r = DispatchMessageW(&msg);

    switch(msg.message)
    {
    case WM_SYSKEYUP:
    case WM_SYSKEYDOWN:
    case WM_KEYUP:
    case WM_KEYDOWN:
      if(!r) // if the return value is zero, we already processed the keydown message successfully, so DON'T turn it into a character.
        break;
    default:
      TranslateMessage(&msg);
      break;
    case WM_QUIT:
      return 1;
    }
  }

  return 0;
}

struct _FG_ROOT* FG_FASTCALL fgInitialize()
{
  static fgBackend BACKEND = {
    &fgBehaviorHookListener,
    &fgCreateFontD2D,
    &fgCloneFontD2D,
    &fgDestroyFontD2D,
    &fgDrawFontD2D,
    &fgFontFormatD2D,
    &fgFontGetD2D,
    &fgCreateResourceD2D,
    &fgCloneResourceD2D,
    &fgDestroyResourceD2D,
    &fgDrawResourceD2D,
    &fgResourceSizeD2D,
    &fgFontIndexD2D,
    &fgFontPosD2D,
    &fgDrawLinesD2D,
    &fgCreateD2D,
    &fgMessageMapDefault,
    &fgUserDataMapCallbacks,
    &fgPushClipRectD2D,
    &fgPeekClipRectD2D,
    &fgPopClipRectD2D,
    &fgDragStartD2D,
    &fgSetCursorD2D,
    &fgClipboardCopyD2D,
    &fgClipboardExistsD2D,
    &fgClipboardPasteD2D,
    &fgClipboardFreeD2D,
    &fgDirtyElementD2D,
    &fgProcessMessagesDefault,
    &fgLoadExtensionDefault,
    &fgTerminateD2D,
  };

  typedef BOOL(WINAPI *tGetPolicy)(LPDWORD lpFlags);
  typedef BOOL(WINAPI *tSetPolicy)(DWORD dwFlags);
  const DWORD EXCEPTION_SWALLOWING = 0x1;
  DWORD dwFlags;

  HMODULE kernel32 = LoadLibraryA("kernel32.dll");
  assert(kernel32 != 0);
  tGetPolicy pGetPolicy = (tGetPolicy)GetProcAddress(kernel32, "GetProcessUserModeExceptionPolicy");
  tSetPolicy pSetPolicy = (tSetPolicy)GetProcAddress(kernel32, "SetProcessUserModeExceptionPolicy");
  if(pGetPolicy && pSetPolicy && pGetPolicy(&dwFlags))
    pSetPolicy(dwFlags & ~EXCEPTION_SWALLOWING); // Turn off the filter 

#ifdef BSS_DEBUG
  HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);
#endif
  if(FAILED(CoInitialize(NULL)))
    return 0;

  AbsRect extent = { GetSystemMetrics(SM_XVIRTUALSCREEN), GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN), GetSystemMetrics(SM_YVIRTUALSCREEN) };
  extent.right += extent.left;
  extent.bottom += extent.top;

  fgDirect2D* root = reinterpret_cast<fgDirect2D*>(calloc(1, sizeof(fgDirect2D)));
  fgDirect2D::instance = root;

  HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory, (LPVOID*)&root->wicfactory);
  if(SUCCEEDED(hr))
    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&root->writefactory));
  if(SUCCEEDED(hr))
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &root->factory);
  if(FAILED(hr))
  {
    if(root->wicfactory) root->wicfactory->Release();
    if(root->factory) root->factory->Release();
    if(root->writefactory) root->writefactory->Release();
    free(root);
    return 0;
  }

  fgWindowD2D::WndRegister(); // Register window class

  FLOAT dpix;
  FLOAT dpiy;
  root->factory->GetDesktopDpi(&dpix, &dpiy);
  fgIntVec dpi = { dpix, dpiy };
  fgRoot_Init(&root->root, &extent, &dpi, &BACKEND);
  EnumDisplayMonitors(0, 0, SpawnMonitorsProc, (LPARAM)root);
  return &root->root;
}