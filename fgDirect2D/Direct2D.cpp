// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "Direct2D.h"
#include "feather/outline.h"
#include "feather/rtree.h"
#include "util.h"
#include <stdint.h>
#include <wincodec.h>
#include <dwrite_1.h>
#include <malloc.h>
#include <signal.h>
#include <codecvt>
#include <filesystem>
#include <dwmapi.h>
#include <shellscalingapi.h>
#include "RoundRect.h"
#include "Circle.h"
#include "Triangle.h"
#include "Modulation.h"

using namespace D2D;
namespace fs = std::filesystem;

typedef HRESULT(STDAPICALLTYPE* DWMCOMPENABLE)(BOOL*);
typedef HRESULT(STDAPICALLTYPE* DWMBLURBEHIND)(HWND, const DWM_BLURBEHIND*);
typedef HRESULT(STDAPICALLTYPE* GETDPIFORMONITOR)(HMONITOR, int, UINT*, UINT*);
typedef HRESULT(STDAPICALLTYPE *GETSCALEFACTORFORMONITOR)(HMONITOR, int*);
static float PI = 3.14159265359f;

static std::unique_ptr<struct HINSTANCE__, void(*)(struct HINSTANCE__*)> shcoreD2D(LoadLibraryW(L"Shcore.dll"), [](HMODULE h) { FreeLibrary(h); });

fgError DrawFontD2D(struct FG__BACKEND* self, void* data, const fgFont* font, fgFontLayout fontlayout, const fgRect* area, fgColor color, float lineHeight, float letterSpacing, float blur, enum FG_TEXT_ANTIALIASING aa)
{
  if(!fontlayout)
    return -1;
  auto instance = (Direct2D*)self;
  auto context = reinterpret_cast<Context*>(data);

  IDWriteTextLayout* layout = (IDWriteTextLayout*)fontlayout;
  context->color->SetColor(ToD2Color(color.color));

  layout->SetMaxWidth(area->right - area->left);
  layout->SetMaxHeight(area->bottom - area->top);
  context->target->DrawTextLayout(D2D1::Point2F(area->left, area->top), layout, context->color, D2D1_DRAW_TEXT_OPTIONS_CLIP);
  return 0;
}

template<int N, typename Arg, typename... Args>
inline fgError DrawEffectD2D(const Context* ctx, ID2D1Effect* effect, const fgRect& area, const Arg arg, const Args&... args)
{
  effect->SetValue<Arg, int>(N, arg);
  if constexpr(sizeof...(args) > 0)
    return DrawEffectD2D<N + 1, Args...>(ctx, effect, area, args...);

  D2D1_RECT_F rect = D2D1::RectF(area.left, area.top, area.right, area.bottom);
  ctx->context->DrawImage(effect, &D2D1::Point2F(area.left, area.top), &rect, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER);
  return 0;
}

fgError DrawAssetD2D(struct FG__BACKEND* self, void* data, const fgAsset* asset, const fgRect* area, const fgRect* source, fgColor color, float time)
{
  auto instance = (Direct2D*)self;
  auto context = reinterpret_cast<Context*>(data);
  fgassert(context != 0);
  fgassert(context->target != 0);

  ID2D1Bitmap* bitmap = context->GetBitmapFromSource((fgAssetD2D*)asset);
  fgassert(bitmap);

  D2D1_RECT_F uvresolve = D2D1::RectF(source->left, source->top, source->right, source->bottom);
  D2D1_RECT_F rect = D2D1::RectF(area->left, area->top, area->right, area->bottom);

  auto scale = D2D1::Vector2F((rect.right - rect.left) / (uvresolve.right - uvresolve.left), (rect.bottom - rect.top) / (uvresolve.bottom - uvresolve.top));
  if(scale.x == 1.0f && scale.y == 1.0f)
    context->target->DrawBitmap(bitmap, rect, color.a / 255.0f, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, &uvresolve);
  else
  {
    auto e = context->scale;
    e->SetValue(D2D1_SCALE_PROP_SCALE, scale);
    e->SetValue(D2D1_SCALE_PROP_INTERPOLATION_MODE, D2D1_SCALE_INTERPOLATION_MODE_ANISOTROPIC);
    e->SetInput(0, bitmap);
    context->context->DrawImage(e, D2D1::Point2F(rect.left, rect.top), D2D1::RectF(floorf(uvresolve.left * scale.x), floorf(uvresolve.top * scale.y), ceilf(uvresolve.right * scale.x), ceilf(uvresolve.bottom * scale.y + 1.0f)));
  }

  bitmap->Release();
  return 0;
}

fgError DrawRectD2D(struct FG__BACKEND* self, void* data, const fgRect* area, const fgRect* corners, fgColor fillColor, float border, fgColor borderColor, float blur)
{
  auto context = reinterpret_cast<Context*>(data);
  fgassert(context != 0);
  fgassert(context->target != 0);

  DrawEffectD2D<0>(
    context,
    context->roundrect,
    *area,
    D2D1::Vector4F(area->left, area->top, area->right, area->bottom),
    D2D1::Vector4F(corners->left, corners->top, corners->right, corners->bottom),
    fillColor,
    borderColor,
    border);
  return 0;
}

fgError DrawCircleD2D(struct FG__BACKEND* self, void* data, const fgRect* area, const fgRect* arcs, fgColor fillColor, float border, fgColor borderColor, float blur)
{
  auto context = reinterpret_cast<Context*>(data);
  fgassert(context != 0);
  fgassert(context->target != 0);

  DrawEffectD2D<0>(
    context,
    context->circle,
    *area,
    D2D1::Vector4F(area->left, area->top, area->right, area->bottom),
    D2D1::Vector4F(arcs->left, arcs->top, arcs->right, arcs->bottom),
    fillColor,
    borderColor,
    border);
  return 0;
}

fgError DrawLinesD2D(struct FG__BACKEND* self, void* data, const fgVec* points, size_t count, fgColor color)
{
  auto context = reinterpret_cast<Context*>(data);

  context->color->SetColor(ToD2Color(color.color));
  for(size_t i = 1; i < count; ++i)
    context->target->DrawLine(D2D1_POINT_2F{ points[i - 1].x, points[i - 1].y }, D2D1_POINT_2F{ points[i].x, points[i].y }, context->color, 1.0F, 0);
  return 0;
}

fgError DrawCurveD2D(struct FG__BACKEND* self, void* data, const fgVec* anchors, size_t count, fgColor fillColor, float stroke, fgColor strokeColor)
{
  auto instance = (Direct2D*)self;

  return -1;
}

// fgError DrawShaderD2D(struct FG__BACKEND* self, fgShader);
fgError PushLayerD2D(struct FG__BACKEND* self, void* data, fgRect area, const float* transform, float opacity)
{
  auto context = reinterpret_cast<Context*>(data);
  context->layers.push(0);

  // TODO: Properly project 3D transform into 2D transform
  context->target->SetTransform(D2D1::Matrix3x2F(
    transform[0], transform[1],
    transform[4], transform[5], 
    transform[3], transform[7]));

  // We only need a proper layer if we are doing opacity, otherwise the transform is sufficient
  if(opacity != 1.0)
  {
    D2D1_LAYER_PARAMETERS params = {
      D2D1::RectF(area.left, area.top, area.right, area.bottom),
      0,
      D2D1_ANTIALIAS_MODE_ALIASED,
      D2D1::IdentityMatrix(),
      opacity,
      0,
      D2D1_LAYER_OPTIONS_INITIALIZE_FOR_CLEARTYPE,
    };

    context->target->CreateLayer(NULL, &context->layers.top());
    context->target->PushLayer(params, !context->layers.top() ? NULL : context->layers.top());
  }
  else
    context->layers.top() = (ID2D1Layer*)~0;
  return 0;
}

fgError PopLayerD2D(struct FG__BACKEND* self, void* data)
{
  auto context = reinterpret_cast<Context*>(data);
  auto p = context->layers.top();
  context->layers.pop();
  if(p != (ID2D1Layer*)~0)
  {
    context->target->PopLayer();
    if(p)
      p->Release();
  }
  return 0;
}

fgError PushClipD2D(struct FG__BACKEND* self, void* data, fgRect area)
{
  reinterpret_cast<Context*>(data)->PushClip(area);
  return 0;
}

fgError PopClipD2D(struct FG__BACKEND* self, void* data)
{
  reinterpret_cast<Context*>(data)->PopClip();
  return 0;
}

fgError DirtyRectD2D(struct FG__BACKEND* self, void* data, fgRect area)
{
  auto instance = (Direct2D*)self;
  return 0;
}

fgFont* CreateFontD2D(struct FG__BACKEND* self, const char* family, unsigned short weight, bool italic, unsigned int pt, fgVec dpi)
{
  auto instance = (Direct2D*)self;
  size_t len = fgUTF8toUTF16(family, -1, 0, 0);
  auto wtext = (wchar_t*)ALLOCA(sizeof(wchar_t) * len);
  fgUTF8toUTF16(family, -1, wtext, len);

  IDWriteTextFormat* format = 0;
  wchar_t wlocale[LOCALE_NAME_MAX_LENGTH];
  GetSystemDefaultLocaleName(wlocale, LOCALE_NAME_MAX_LENGTH);
  LOGFAILURERET(instance->writefactory->CreateTextFormat(wtext, 0, DWRITE_FONT_WEIGHT(weight), italic ? DWRITE_FONT_STYLE_ITALIC : DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, pt * (dpi.x / 72.0f), wlocale, &format), 0, "CreateTextFormat failed with error code %li", hr);

  if(!format) return 0;
  TCHAR name[64];
  UINT32 findex;
  BOOL exists;
  IDWriteFontCollection* collection;
  format->GetFontFamilyName(name, 64);
  format->GetFontCollection(&collection);
  collection->FindFamilyName(name, &findex, &exists);
  if(!exists) // CreateTextFormat always succeeds even for invalid font names so we have to check to see if we actually loaded a real font
  {
    format->Release();
    collection->Release();
    return 0;
  }

  IDWriteFontFamily* ffamily;
  collection->GetFontFamily(findex, &ffamily);
  IDWriteFont* font;
  ffamily->GetFirstMatchingFont(format->GetFontWeight(), format->GetFontStretch(), format->GetFontStyle(), &font);
  DWRITE_FONT_METRICS metrics;
  font->GetMetrics(&metrics);
  float ratio = format->GetFontSize() / (float)metrics.designUnitsPerEm;
  FLOAT linespacing = (metrics.ascent + metrics.descent + metrics.lineGap) * ratio;
  FLOAT baseline = metrics.ascent * ratio;
  ffamily->Release();
  font->Release();
  format->SetLineSpacing(DWRITE_LINE_SPACING_METHOD_DEFAULT, linespacing, baseline);
  collection->Release();

  return new fgFont{ format, dpi, baseline, linespacing, pt };
}

fgError DestroyFontD2D(struct FG__BACKEND* self, fgFont* font)
{
  if(!reinterpret_cast<IDWriteTextFormat*>(font->data)->Release())
    delete font;
  else
    return -1;
  return 0;
}

fgFontLayout FontLayoutD2D(struct FG__BACKEND* self, fgFont* font, const char* text, fgRect* area, float lineHeight, float letterSpacing, fgFontLayout prev, fgVec dpi)
{
  auto instance = (Direct2D*)self;
  fgassert(font);
  IDWriteTextLayout* layout = (IDWriteTextLayout*)prev;
  if(layout)
    layout->Release();
  if(!area)
    return 0;

  std::wstring utf;
  utf.reserve(MultiByteToWideChar(CP_UTF8, 0, text, -1, 0, 0));
  utf.resize(MultiByteToWideChar(CP_UTF8, 0, text, -1, utf.data(), utf.capacity()));

  if(!text) return 0;
  float x = area->right - area->left;
  float y = area->bottom - area->top;
  LOGFAILURE(instance->writefactory->CreateTextLayout(utf.c_str(), utf.size(), reinterpret_cast<IDWriteTextFormat*>(font->data), (x <= 0.0f ? INFINITY : x), (y <= 0.0f ? INFINITY : y), &layout), "CreateTextLayout failed with error code %li", hr);

  if(!layout)
  {
    area->right = area->left;
    area->bottom = area->top;
    return 0;
  }
  FLOAT linespacing;
  FLOAT baseline;
  DWRITE_LINE_SPACING_METHOD method;
  layout->GetLineSpacing(&method, &linespacing, &baseline);
  if(lineHeight > 0.0f)
    layout->SetLineSpacing(DWRITE_LINE_SPACING_METHOD_UNIFORM, lineHeight, baseline * (lineHeight / linespacing));
  /*layout->SetWordWrapping((flags & (FGTEXT_CHARWRAP | FGTEXT_WORDWRAP)) ? DWRITE_WORD_WRAPPING_WRAP : DWRITE_WORD_WRAPPING_NO_WRAP);
  layout->SetReadingDirection((flags & FGTEXT_RTL) ? DWRITE_READING_DIRECTION_RIGHT_TO_LEFT : DWRITE_READING_DIRECTION_LEFT_TO_RIGHT);
  if(flags & FGTEXT_RIGHTALIGN)
    layout->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_FAR);
  if(flags & FGTEXT_CENTER)
    layout->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);*/
  if(letterSpacing > 0.0f)
  {
    DWRITE_TEXT_RANGE range;
    FLOAT leading, trailing, minimum;
    IDWriteTextLayout1* layout1 = 0;
    layout->QueryInterface<IDWriteTextLayout1>(&layout1);
    layout1->GetCharacterSpacing(0, &leading, &trailing, &minimum, &range);
    layout1->SetCharacterSpacing(leading, trailing + letterSpacing, minimum, range);
    layout1->Release();
  }

  DWRITE_TEXT_METRICS metrics;
  layout->GetMetrics(&metrics);
  if(area->right <= area->left) area->right = area->left + metrics.width;
  if(area->bottom <= area->top) area->bottom = area->top + metrics.height;
  layout->SetMaxWidth(area->right - area->left);
  layout->SetMaxHeight(area->bottom - area->top);
  return layout;
}

size_t FontIndexD2D(struct FG__BACKEND* self, fgFont* font, fgFontLayout fontlayout, const fgRect* area, float lineHeight, float letterSpacing, fgVec pos, fgVec* cursor, fgVec dpi)
{
  fgassert(font != 0);
  IDWriteTextLayout* layout = (IDWriteTextLayout*)fontlayout;
  if(!layout)
    return 0;

  BOOL trailing;
  BOOL inside;
  DWRITE_HIT_TEST_METRICS hit;
  layout->HitTestPoint(pos.x - area->left, pos.y - area->top, &trailing, &inside, &hit);

  cursor->x = hit.left;
  if(trailing) cursor->x += hit.width;
  cursor->y = hit.top;
  return hit.textPosition + trailing;
}

fgVec FontPosD2D(struct FG__BACKEND* self, fgFont* font, fgFontLayout fontlayout, const fgRect* area, float lineHeight, float letterSpacing, size_t index, fgVec dpi)
{
  IDWriteTextLayout* layout = (IDWriteTextLayout*)fontlayout;
  if(!layout)
    return fgVec{ 0,0 };

  FLOAT x, y;
  DWRITE_HIT_TEST_METRICS hit;
  layout->HitTestTextPosition(index, false, &x, &y, &hit);
  return fgVec{ x + area->left, y + area->top };
}

inline fgAsset* Direct2D::LoadAsset(const char* data, size_t count)
{
  IWICBitmapDecoder* decoder = nullptr;
  IWICBitmapFrameDecode* source = nullptr;
  IWICFormatConverter* conv = nullptr;
  IWICStream* stream = nullptr;
  HRESULT hr = 0;

  if(!count && data)
  {
    fs::path p = fs::u8path(data);
    hr = wicfactory->CreateDecoderFromFilename(p.wstring().c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder);
  }
  else
  {
    hr = wicfactory->CreateStream(&stream);
    if(SUCCEEDED(hr))
      stream->InitializeFromMemory((BYTE*)data, count);
    if(SUCCEEDED(hr)) //WICDecodeMetadataCacheOnDemand
      hr = wicfactory->CreateDecoderFromStream(stream, NULL, WICDecodeMetadataCacheOnLoad, &decoder);
  }
  if(SUCCEEDED(hr))
    hr = decoder->GetFrame(0, &source);
  if(SUCCEEDED(hr))
    hr = wicfactory->CreateFormatConverter(&conv);
  if(SUCCEEDED(hr)) // Convert the image format to 32bppPBGRA (DXGI_FORMAT_B8G8R8A8_UNORM + D2D1_ALPHA_MODE_PREMULTIPLIED).
    hr = conv->Initialize(source, GUID_WICPixelFormat32bppPBGRA, WICBitmapDitherTypeNone, nullptr, 0.0f, WICBitmapPaletteTypeCustom);
  if(FAILED(hr))
  {
    fgLog(root, FGLOG_ERROR, "fgCreateAssetD2D failed with error code %li", hr);
    return 0;
  }

  fgAsset* asset = new fgAsset();
  asset->data = conv;
  asset->format = BACKEND_UNKNOWN;

  GUID format;
  decoder->GetContainerFormat(&format);
  
  if(format == GUID_ContainerFormatBmp)
    asset->format = BACKEND_BMP;
  else if(format == GUID_ContainerFormatPng)
    asset->format = BACKEND_PNG;
  else if(format == GUID_ContainerFormatIco)
    asset->format = BACKEND_ICO;
  else if(format == GUID_ContainerFormatJpeg)
    asset->format = BACKEND_JPG;
  else if(format == GUID_ContainerFormatTiff)
    asset->format = BACKEND_TIFF;
  else if(format == GUID_ContainerFormatGif)
    asset->format = BACKEND_GIF;
  else if(format == GUID_ContainerFormatWebp)
    asset->format = BACKEND_WEBP;

  if(stream) stream->Release();
  if(decoder) decoder->Release();
  if(source) source->Release();

  D2D1_SIZE_U sz = { 0 };
  double dpix, dpiy;

  conv->GetSize(&sz.width, &sz.height);
  conv->GetResolution(&dpix, &dpiy);

  asset->dpi = { (float)dpix, (float)dpiy };
  asset->size = { (int)sz.width, (int)sz.height };
  return asset;

  //if(data[0] == 0xFF && data[1] == 0xD8) // JPEG SOI header
  //else if(data[0] == 'B' && data[1] == 'M') // BMP header
  //else if(data[0] == 137 && data[1] == 80 && data[2] == 78 && data[3] == 71) // PNG file signature
  //else if(data[0] == 'G' && data[1] == 'I' && data[2] == 'F') // GIF header
  //else if((data[0] == 'I' && data[1] == 'I' && data[2] == '*' && data[3] == 0) || (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && data[3] == '*')) // TIFF header
}

fgAsset* CreateAssetD2D(struct FG__BACKEND* self, const char* data, size_t count, enum FG_BACKEND_FORMATS format)
{
  return reinterpret_cast<Direct2D*>(self)->LoadAsset(data, count);
}

fgError DestroyAssetD2D(struct FG__BACKEND* self, fgAsset* asset)
{
  fgError e = reinterpret_cast<IUnknown*>(asset->data)->Release();
  free(asset);
  return e;
}

fgError PutClipboardD2D(struct FG__BACKEND* self, enum FG_CLIPBOARD kind, const char* data, size_t count)
{
  OpenClipboard(GetActiveWindow());
  if(data != 0 && count > 0 && EmptyClipboard())
  {
    if(kind == CLIPBOARD_TEXT)
    {
      size_t unilen = fgUTF8toUTF16(data, count, 0, 0);
      HGLOBAL unimem = GlobalAlloc(GMEM_MOVEABLE, unilen * sizeof(wchar_t));
      if(unimem)
      {
        wchar_t* uni = (wchar_t*)GlobalLock(unimem);
        size_t sz = fgUTF8toUTF16(data, count, uni, unilen);
        if(sz < unilen) // ensure we have a null terminator
          uni[sz] = 0;
        GlobalUnlock(unimem);
        SetClipboardData(CF_UNICODETEXT, unimem);
      }
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, count + 1);
      if(gmem)
      {
        char* mem = (char*)GlobalLock(gmem);
        MEMCPY(mem, count + 1, data, count);
        mem[count] = 0;
        GlobalUnlock(gmem);
        SetClipboardData(CF_TEXT, gmem);
      }
    }
    else
    {
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, count);
      if(gmem)
      {
        void* mem = GlobalLock(gmem);
        MEMCPY(mem, count, data, count);
        GlobalUnlock(gmem);
        UINT format = CF_PRIVATEFIRST;
        switch(kind)
        {
        case CLIPBOARD_WAVE: format = CF_WAVE; break;
        case CLIPBOARD_BITMAP: format = CF_BITMAP; break;
        }
        SetClipboardData(format, gmem);
      }
    }
  }
  CloseClipboard();
  return 0;
}

size_t GetClipboardD2D(struct FG__BACKEND* self, enum FG_CLIPBOARD kind, void* target, size_t count)
{
  OpenClipboard(GetActiveWindow());
  UINT format = CF_PRIVATEFIRST;
  switch(kind)
  {
  case CLIPBOARD_TEXT:
    if(IsClipboardFormatAvailable(CF_UNICODETEXT))
    {
      HANDLE gdata = GetClipboardData(CF_UNICODETEXT);
      SIZE_T size = GlobalSize(gdata) / 2;
      const wchar_t* str = (const wchar_t*)GlobalLock(gdata);
      SIZE_T len = fgUTF16toUTF8(str, size, 0, 0);

      if(target && count >= len)
        len = fgUTF16toUTF8(str, size, (char*)target, count);

      GlobalUnlock(gdata);
      CloseClipboard();
      return len;
    }
    format = CF_TEXT;
    break;
  case CLIPBOARD_WAVE: format = CF_WAVE; break;
  case CLIPBOARD_BITMAP: format = CF_BITMAP; break;
  }
  HANDLE gdata = GetClipboardData(format);
  SIZE_T size = GlobalSize(gdata);
  if(target && count >= size)
  {
    void* data = GlobalLock(gdata);
    MEMCPY(target, count, data, size);
    GlobalUnlock(gdata);
  }
  CloseClipboard();
  return size;
}

bool CheckClipboardD2D(struct FG__BACKEND* self, enum FG_CLIPBOARD kind)
{
  switch(kind)
  {
  case CLIPBOARD_TEXT:
    return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT);
  case CLIPBOARD_WAVE:
    return IsClipboardFormatAvailable(CF_WAVE);
  case CLIPBOARD_BITMAP:
    return IsClipboardFormatAvailable(CF_BITMAP);
  case CLIPBOARD_CUSTOM:
    return IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  case CLIPBOARD_ALL:
    return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT) | IsClipboardFormatAvailable(CF_WAVE) | IsClipboardFormatAvailable(CF_BITMAP) | IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  }
  return 0;
}

fgError ClearClipboardD2D(struct FG__BACKEND* self, enum FG_CLIPBOARD kind)
{
  return !EmptyClipboard() ? -1 : 0;
}

fgOutlineNode* Direct2D::GetDisplay(struct HMONITOR__* monitor)
{
  for(unsigned int i = 0; i < base.n_displays; ++i)
    if(base.displays[i]->statedata == monitor)
      return base.displays[i];

  return nullptr;
}

fgError ProcessMessagesD2D(struct FG__ROOT* root, struct FG__BACKEND* self)
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
      return 0;
    }
  }

  return 1;
}

fgError SetCursorD2D(struct FG__BACKEND* self, void* data, enum FG_CURSOR cursor)
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

  switch(cursor)
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
  default:
    return -1;
  }

  return 0;
}

void DestroyD2D(struct FG__BACKEND* self)
{
  PostQuitMessage(0);
  auto d2d = reinterpret_cast<Direct2D*>(self);
  if(!d2d)
    return;

  d2d->~Direct2D();
  CoUninitialize();
  free(d2d);
}

Direct2D::~Direct2D()
{
  if(factory)
    factory->Release();
  if(wicfactory)
    wicfactory->Release();
  if(writefactory)
    writefactory->Release();
  WipeMonitors();
  if(base.displays)
    free(base.displays);
  base.n_displays = 0;
}

inline fgVec operator-(const fgVec& l, const fgVec& r) { return fgVec{ l.x - r.x, l.y - r.y }; }

BOOL __stdcall EnumerateMonitorsProc(HMONITOR monitor, HDC hdc, LPRECT, LPARAM lparam)
{
  Direct2D* root = reinterpret_cast<Direct2D*>(lparam);
  fgOutlineNode* display = nullptr;
  for(unsigned int i = 0; i < root->base.n_displays; ++i)
  {
    if(root->base.displays[i]->statedata == monitor)
    {
      display = root->base.displays[i];
      break;
    }
  }

  if(!display)
  {
    display = (fgOutlineNode*)calloc(1, sizeof(fgOutlineNode));
    display->auxdata = calloc(1, sizeof(DisplayData));
    display->statedata = monitor;
    root->base.displays = (fgOutlineNode**)realloc(root->base.displays, (++root->base.n_displays) * sizeof(fgOutlineNode*));
    root->base.displays[root->base.n_displays - 1] = display;
  }

  DisplayData& data = *reinterpret_cast<DisplayData*>(display->auxdata);
  MONITORINFO info = { sizeof(MONITORINFO), 0 };
  GetMonitorInfoW(monitor, &info);
  if(root->getDpiForMonitor)
  {
    UINT x, y;
    (*root->getDpiForMonitor)(monitor, 0, &x, &y);
    data.dpi = { (float)x, (float)y };
  }
  else
    data.dpi = root->base.dpi;
  
  if(root->getScaleFactorForMonitor)
  {
    static_assert(sizeof(DEVICE_SCALE_FACTOR) == sizeof(int));
    DEVICE_SCALE_FACTOR factor;
    (*root->getScaleFactorForMonitor)(monitor, (int*)&factor);
    switch(factor)
    {
    case DEVICE_SCALE_FACTOR_INVALID:
    case SCALE_100_PERCENT: data.scale = 1.0f; break;
    case SCALE_120_PERCENT: data.scale = 1.2f; break;
    case SCALE_125_PERCENT: data.scale = 1.25f; break;
    case SCALE_140_PERCENT: data.scale = 1.4f; break;
    case SCALE_150_PERCENT: data.scale = 1.5f; break;
    case SCALE_160_PERCENT: data.scale = 1.6f; break;
    case SCALE_175_PERCENT: data.scale = 1.75f; break;
    case SCALE_180_PERCENT: data.scale = 1.8f; break;
    case SCALE_200_PERCENT: data.scale = 2.0f; break;
    case SCALE_225_PERCENT: data.scale = 2.25f; break;
    case SCALE_250_PERCENT: data.scale = 2.5f; break;
    case SCALE_300_PERCENT: data.scale = 3.0f; break;
    case SCALE_350_PERCENT: data.scale = 3.5f; break;
    case SCALE_400_PERCENT: data.scale = 4.0f; break;
    case SCALE_450_PERCENT: data.scale = 4.5f; break;
    case SCALE_500_PERCENT: data.scale = 5.0f; break;
    }
  }
  else
    data.scale = 1.0f;
  

  display->area.abs = { (float)info.rcMonitor.left, (float)info.rcMonitor.top, (float)info.rcMonitor.right, (float)info.rcMonitor.bottom };
  display->layoutFlags = ((info.dwFlags & MONITORINFOF_PRIMARY) != 0) | Direct2D::VALID_DISPLAY;
  display->behavior = &DisplayBehavior;

  // TODO: Trigger the width/height listeners if something changed
  if(!display->node)
  {
    display->node = fgAllocDocumentNode();
    display->node->outline = display;
    display->node->area.bottomright = display->area.abs.topleft - display->area.abs.bottomright;
    display->node->extent.bottomright = display->area.abs.topleft - display->area.abs.bottomright;
  }

  if(!display->node->rtnode)
  {
    display->node->rtnode = fgAllocRTNode();
    display->node->rtnode->area = display->node->area;
    display->node->rtnode->extent = display->node->extent;
    display->node->rtnode->top_zindex = INT_MAX;
    display->node->rtnode->bottom_zindex = INT_MIN;
  }
  return TRUE;
}

void Direct2D::WipeMonitors()
{
  for(unsigned int i = 0; i < base.n_displays; ++i)
  {
    if(base.displays[i]->layoutFlags & Direct2D::VALID_DISPLAY)
      base.displays[i]->layoutFlags &= ~Direct2D::VALID_DISPLAY;
    else
    {
      free(base.displays[i]->auxdata);
      base.displays[i]->auxdata = nullptr;
      base.displays[i]->statedata = nullptr;
      fgDestroyOutlineNode(root, base.displays[i], 0, fgVec{});
      free(base.displays[i]);
      if(i != --base.n_displays)
        base.displays[i--] = base.displays[base.n_displays];
    }
  }
}
void Direct2D::RefreshMonitors()
{
  EnumDisplayMonitors(0, 0, EnumerateMonitorsProc, (LPARAM)this);
  WipeMonitors(); // Delete any monitors that no longer exist
}

fgError RequestAnimationFrameD2D(struct FG__BACKEND* self, struct FG__DOCUMENT_NODE* node, unsigned long long microdelay)
{
  //if(context->nextframe < 0 || context->nextframe > microdelay)
  //  context->nextframe = microdelay;
  return 0;
}

/*void fgDirtyElementD2D(fgElement* e)
{
  static fgElement* lasttop = 0;

  if(e->flags&FGELEMENT_SILENT)
    return;
  if(!Direct2D::instance->root.topmost && lasttop != 0)
  {
    ReleaseCapture();
    ShowWindow(Direct2D::instance->tophwnd, SW_HIDE);
    lasttop = 0;
  }
  while(e && e->destroy != (fgDestroy)fgWindowD2D_Destroy && e != Direct2D::instance->root.topmost && e->destroy != (fgDestroy)fgDebug_Destroy)
    e = e->parent;

  if(Direct2D::instance->root.topmost != 0 && e == Direct2D::instance->root.topmost)
  {
    fgRect& toprect = Direct2D::instance->toprect;
    ResolveNoClipRect(Direct2D::instance->root.topmost, &toprect, 0, 0);
    const fgVec& dpi = Direct2D::instance->root.topmost->GetDPI();
    fgScaleRectDPI(&toprect, dpi.x, dpi.y); // SetWindowPos will resize the direct2D background via the WndProc callback
    bool ignore = (Direct2D::instance->root.topmost->flags&FGELEMENT_IGNORE) != 0;
    SetWindowLong(Direct2D::instance->tophwnd, GWL_EXSTYLE, WS_EX_COMPOSITED | WS_EX_TOOLWINDOW | (ignore ? (WS_EX_LAYERED | WS_EX_TRANSPARENT) : 0));
    SetWindowPos(Direct2D::instance->tophwnd, HWND_TOP, toprect.left, toprect.top, toprect.right - toprect.left, toprect.bottom - toprect.top, SWP_NOSENDCHANGING | SWP_NOACTIVATE | (ignore ? SWP_NOACTIVATE : 0));
    fgInvScaleRectDPI(&toprect, dpi.x, dpi.y);
    ShowWindow(Direct2D::instance->tophwnd, SW_SHOW);
    if(lasttop != Direct2D::instance->root.topmost)
    {
      lasttop = Direct2D::instance->root.topmost;
      if(!ignore)
        SetCapture(Direct2D::instance->tophwnd);
    }
    Direct2D::instance->topcontext.InvalidateHWND(Direct2D::instance->tophwnd);
  }
  else if(e != 0 && e->destroy == (fgDestroy)fgDebug_Destroy)
    Direct2D::instance->debugcontext.InvalidateHWND(Direct2D::instance->debughwnd);
  else if(e)
    ((fgWindowD2D*)e)->context.InvalidateHWND(((fgWindowD2D*)e)->handle);
}*/

/*longptr_t __stdcall Direct2D::TopmostWndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  Direct2D* self = reinterpret_cast<Direct2D*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  if(self)
  {
    switch(message)
    {
    case WM_PAINT:
      if(self->root.topmost != 0)
      {
        fgRect area;
        ResolveRect(self->root.topmost, &area);
        fgDrawAuxDataEx exdata;
        self->topcontext.BeginDraw(self->tophwnd, self->root.topmost, &self->toprect, exdata, 0);
        self->root.topmost->Draw(&area, &exdata.data);
        self->topcontext.EndDraw();
      }
      return 0;
    case WM_MOUSEMOVE:
      if(self->root.topmost != 0)
        return self->topcontext.WndProc(hWnd, message, wParam, lParam, self->root.topmost);
    }

    if(self->root.topmost != 0)
      return self->topcontext.WndProc(hWnd, message, wParam, lParam, self->root.topmost);
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}*/

int64_t GetRegistryValueW(HKEY__* hKeyRoot, const wchar_t* szKey, const wchar_t* szValue, unsigned char* data, unsigned long sz)
{
  HKEY__* hKey;
  LRESULT e = RegOpenKeyExW(hKeyRoot, szKey, 0, KEY_READ, &hKey);
  if(!hKey) return -2;
  LSTATUS r = RegQueryValueExW(hKey, szValue, 0, 0, data, &sz);
  RegCloseKey(hKey);
  if(r == ERROR_SUCCESS)
    return sz;
  return (r == ERROR_MORE_DATA) ? sz : -1;
}

extern "C" FG_COMPILER_DLLEXPORT fgBackend* fgDirect2D(struct FG__ROOT* root)
{
  Direct2D* backend = reinterpret_cast<Direct2D*>(calloc(1, sizeof(Direct2D)));
  new(backend) Direct2D();

  backend->root = root;
  backend->base = fgBackend{
    &DrawFontD2D,
    &DrawAssetD2D,
    &DrawRectD2D,
    &DrawCircleD2D,
    &DrawLinesD2D,
    &DrawCurveD2D,
    // &DrawShaderD2D,
    &PushLayerD2D,
    &PopLayerD2D,
    &PushClipD2D,
    &PopClipD2D,
    &DirtyRectD2D,
    &CreateFontD2D,
    &DestroyFontD2D,
    &FontLayoutD2D,
    &FontIndexD2D,
    &FontPosD2D,
    &CreateAssetD2D,
    &DestroyAssetD2D,
    &PutClipboardD2D,
    &GetClipboardD2D,
    &CheckClipboardD2D,
    &ClearClipboardD2D,
    &ProcessMessagesD2D,
    &SetCursorD2D,
    &RequestAnimationFrameD2D,
    &DestroyD2D,
  };

  typedef BOOL(WINAPI *tGetPolicy)(LPDWORD lpFlags);
  typedef BOOL(WINAPI *tSetPolicy)(DWORD dwFlags);
  const DWORD EXCEPTION_SWALLOWING = 0x1;
  DWORD dwFlags;

  HMODULE kernel32 = LoadLibraryA("kernel32.dll");
  fgassert(kernel32 != 0);
  tGetPolicy pGetPolicy = (tGetPolicy)GetProcAddress(kernel32, "GetProcessUserModeExceptionPolicy");
  tSetPolicy pSetPolicy = (tSetPolicy)GetProcAddress(kernel32, "SetProcessUserModeExceptionPolicy");
  if(pGetPolicy && pSetPolicy && pGetPolicy(&dwFlags))
    pSetPolicy(dwFlags & ~EXCEPTION_SWALLOWING); // Turn off the filter 

  HRESULT hr = CoInitialize(NULL); // If this fails for some reason we can't even log an error
  if(FAILED(hr))
    return 0;

  if(FAILED(hr = CoInitializeSecurity(NULL, -1, NULL, NULL, RPC_C_AUTHN_LEVEL_PKT_PRIVACY, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_DYNAMIC_CLOAKING, NULL)))
    return 0;
  IGlobalOptions *pGlobalOptions;
  hr = CoCreateInstance(CLSID_GlobalOptions, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pGlobalOptions));
  if(SUCCEEDED(hr))
  {
    hr = pGlobalOptions->Set(COMGLB_EXCEPTION_HANDLING, COMGLB_EXCEPTION_DONOT_HANDLE);
    pGlobalOptions->Release();
  }

  //backend->base.extent = { (float)GetSystemMetrics(SM_XVIRTUALSCREEN), (float)GetSystemMetrics(SM_YVIRTUALSCREEN), (float)GetSystemMetrics(SM_CXVIRTUALSCREEN), (float)GetSystemMetrics(SM_CYVIRTUALSCREEN) };
  //backend->base.extent.right += backend->base.extent.left;
  //backend->base.extent.bottom += backend->base.extent.top;

  HDC hdc = GetDC(NULL);
  backend->base.dpi = { (float)GetDeviceCaps(hdc, LOGPIXELSX), (float)GetDeviceCaps(hdc, LOGPIXELSY) };
  ReleaseDC(NULL, hdc);

  fgLog(backend->root, FGLOG_NONE, "Initializing fgDirect2D...");

#ifdef FG_DEBUG
  HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);
#endif

  hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory, (LPVOID*)&backend->wicfactory);
  if(SUCCEEDED(hr))
    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory1), reinterpret_cast<IUnknown**>(&backend->writefactory));
  else
    fgLog(backend->root, FGLOG_ERROR, "CoCreateInstance() failed with error: %li", hr);

  D2D1_FACTORY_OPTIONS d2dopt = { D2D1_DEBUG_LEVEL_NONE };
#ifdef FG_DEBUG
  d2dopt.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif

  if(SUCCEEDED(hr))
    hr = D2D1CreateFactory<ID2D1Factory1>(D2D1_FACTORY_TYPE_SINGLE_THREADED, d2dopt, &backend->factory);
  else
    fgLog(backend->root, FGLOG_ERROR, "DWriteCreateFactory() failed with error: %li", hr);

  if(FAILED(hr))
  {
    fgLog(backend->root, FGLOG_ERROR, "D2D1CreateFactory() failed with error: %li", hr);
    DestroyD2D(&backend->base);
    return 0;
  }

  if(FAILED(RoundRect::Register(backend->factory)))
    fgLog(backend->root, FGLOG_ERROR, "Failed to register RoundRect", hr);
  if(FAILED(Circle::Register(backend->factory)))
    fgLog(backend->root, FGLOG_ERROR, "Failed to register Circle", hr);
  if(FAILED(Triangle::Register(backend->factory)))
    fgLog(backend->root, FGLOG_ERROR, "Failed to register Triangle", hr);
  if(FAILED(Modulation::Register(backend->factory)))
    fgLog(backend->root, FGLOG_ERROR, "Failed to register Modulation", hr);

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
    backend->dwmblurbehind = (DWMBLURBEHIND)GetProcAddress(dwm, "DwmEnableBlurBehindWindow");

    if(!backend->dwmblurbehind)
    {
      FreeLibrary(dwm);
      dwm = 0;
    }
  }

  Context::WndRegister(Context::TopWndProc, L"WindowD2D"); // Register window class
  Context::WndRegister(Context::WndProc, L"BaseWindowD2D");

  backend->factory->GetDesktopDpi(&backend->base.dpi.x, &backend->base.dpi.y);
  
  if(shcoreD2D)
  {
    backend->getDpiForMonitor = (GETDPIFORMONITOR)GetProcAddress(shcoreD2D.get(), "GetDpiForMonitor");
    backend->getScaleFactorForMonitor = (GETSCALEFACTORFORMONITOR)GetProcAddress(shcoreD2D.get(), "GetScaleFactorForMonitor");
  }
  backend->RefreshMonitors();

  DWORD blinkrate = 0;
  int64_t sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate", 0, 0);
  if(sz > 0)
  {
    std::wstring buf;
    buf.resize(sz / 2);
    sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate", reinterpret_cast<unsigned char*>(buf.data()), (unsigned long)sz);
    if(sz > 0)
      backend->base.cursorblink = _wtoi(buf.data());
  }
  else
    fgLog(backend->root, FGLOG_WARNING, "Couldn't get user's cursor blink rate.");
  sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Mouse", L"MouseHoverTime", 0, 0);
  if(sz > 0)
  {
    std::wstring buf;
    buf.resize(sz / 2);
    sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Mouse", L"MouseHoverTime", reinterpret_cast<unsigned char*>(buf.data()), (unsigned long)sz);
    if(sz > 0)
      backend->base.tooltipdelay = _wtoi(buf.data());
  }
  else
    fgLog(backend->root, FGLOG_WARNING, "Couldn't get user's mouse hover time.");

  return &backend->base;
}