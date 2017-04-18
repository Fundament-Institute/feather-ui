// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgDirect2D.h"
#include "fgWindowD2D.h"
#include "fgMonitor.h"
#include "fgResource.h"
#include "fgText.h"
#include "fgDebug.h"
#include "bss-util/bss_util.h"
#include "util.h"
#include <stdint.h>
#include <wincodec.h>
#include <dwrite_1.h>
#include <malloc.h>
#include <assert.h>
#include <signal.h>
#include "fgRoundRect.h"
#include "fgCircle.h"
#include "fgTriangle.h"

#define GETEXDATA(data) assert(data->fgSZ == sizeof(fgDrawAuxDataEx)); if(data->fgSZ != sizeof(fgDrawAuxDataEx)) return; fgDrawAuxDataEx* exdata = (fgDrawAuxDataEx*)data;

typedef HRESULT(STDAPICALLTYPE *GETDPIFORMONITOR)(HMONITOR, int, UINT*, UINT*);
GETDPIFORMONITOR pGetDpiForMonitor = 0;
fgDirect2D* fgDirect2D::instance = 0;
static float PI = 3.14159265359f;

BOOL __stdcall SpawnMonitorsProc(HMONITOR monitor, HDC hdc, LPRECT, LPARAM lparam)
{
  static fgMonitor* prev = 0;
  fgDirect2D* root = reinterpret_cast<fgDirect2D*>(lparam);
  MONITORINFO info = { sizeof(MONITORINFO), 0 };
  GetMonitorInfoW(monitor, &info);
  fgIntVec dpi = { 0, 0 };
  if(pGetDpiForMonitor)
  {
    UINT x, y;
    pGetDpiForMonitor(monitor, 0, &x, &y);
    dpi.x = x;
    dpi.y = y;
  }
  
  AbsRect area = { info.rcMonitor.left, info.rcMonitor.top, info.rcMonitor.right, info.rcMonitor.bottom };
  fgMonitor* cur = reinterpret_cast<fgMonitor*>(calloc(1, sizeof(fgMonitor)));
  fgMonitor_Init(cur, FGFLAGS_INTERNAL, &root->root, (!info.rcMonitor.left && !info.rcMonitor.top) ? 0 : prev, &area, &dpi); // Attempt to identify the primary monitor and make it the first monitor in the list
  ((fgElement*)cur)->free = &free;
  prev = cur;
  return TRUE;
}

void fgTerminateD2D()
{
  PostQuitMessage(0);
  fgDirect2D* d2d = fgDirect2D::instance;
  assert(d2d);
  DestroyWindow(d2d->tophwnd);
  fgContext_Destroy(&d2d->topcontext);
  if(d2d->debughwnd)
  {
    DestroyWindow(d2d->debughwnd);
    fgContext_Destroy(&d2d->debugcontext);
  }
  VirtualFreeChild(d2d->root.gui);
  if(d2d->factory)
    d2d->factory->Release();
  if(d2d->wicfactory)
    d2d->wicfactory->Release();
  if(d2d->writefactory)
    d2d->writefactory->Release();

  CoUninitialize();

  fgDirect2D::instance = 0;
  d2d->~fgDirect2D();
  free(d2d);
}

fgWindowD2D* GetElementWindow(fgElement* cur)
{
  while(cur && cur->destroy != (fgDestroy)fgWindowD2D_Destroy) cur = cur->parent;
  return (fgWindowD2D*)cur;
}

bool IsChildOf(fgElement* cur, fgElement* target)
{
  while(cur && cur != target) cur = cur->parent;
  return cur == target;
}

IDWriteTextLayout* CreateD2DLayout(IDWriteTextFormat* format, const wchar_t* text, size_t len, const AbsRect* area)
{
  if(!text) return 0;
  IDWriteTextLayout* layout = 0;
  FABS x = area->right - area->left;
  FABS y = area->bottom - area->top;
  LOGFAILURE(fgDirect2D::instance->writefactory->CreateTextLayout(text, len, format, (x <= 0.0f ? INFINITY : x), (y <= 0.0f ? INFINITY : y), &layout), "CreateTextLayout failed with error code %li", hr);
  return layout;
}

longptr_t __stdcall fgDirect2D::DebugWndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  fgDebug* self = reinterpret_cast<fgDebug*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  if(self)
  {
    switch(message)
    {
    case WM_PAINT:
    {
      AbsRect area;
      ResolveRect(*self, &area);
      fgDrawAuxDataEx exdata;
      fgDirect2D::instance->debugcontext.BeginDraw(hWnd, *self, &area, exdata);
      //fgDirect2D::instance->debugcontext.target->Clear(D2D1::ColorF(0, 1.0));
      AbsRect test = exdata.context->cliprect.top();
      self->tabs->Draw(&area, &exdata.data);
      fgDirect2D::instance->debugcontext.EndDraw();
    }
    return 0;
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
      SetCapture(hWnd);
      break;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
      ReleaseCapture();
      break;
    }

    return fgDirect2D::instance->debugcontext.WndProc(hWnd, message, wParam, lParam, *self);
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

size_t fgDebugD2D_Message(fgDebug* self, const FG_Msg* msg)
{
  fgFlag otherint = (fgFlag)msg->u;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgDebug_Message(self, msg);
    return FG_ACCEPT;
  case FG_SETPARENT:
    return fgDebug_Message(self, msg);
  case FG_SETFLAG: // Do the same thing fgElement does to resolve a SETFLAG into SETFLAGS
    otherint = bss_util::bssSetBit<fgFlag>(self->tabs->flags, otherint, msg->u2 != 0);
  case FG_SETFLAGS:
    if((otherint^self->tabs->flags) & FGELEMENT_HIDDEN)
      ShowWindow(fgDirect2D::instance->debughwnd, (otherint&FGELEMENT_HIDDEN) ? SW_HIDE : SW_SHOW);
    break;
  }
  return fgDebug_Message(self, msg);
}

void fgDebugD2D_Init(fgDebug* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  if(!fgDirect2D::instance->root.fgCaptureWindow)
    fgDirect2D::instance->debugtarget = fgWindowD2D::windowlist;
  else
    fgDirect2D::instance->debugtarget = GetElementWindow(fgDirect2D::instance->root.fgCaptureWindow);

  fgTransform tf = {0};
  if(fgDirect2D::instance->debugtarget)
  {
    AbsRect area;
    ResolveRect(fgDirect2D::instance->debugtarget->window, &area);
    tf.area.left.abs = area.right;
    tf.area.top.abs = area.top;
    tf.area.right.abs = area.right + 300;
    tf.area.bottom.abs = area.bottom;
    transform = &tf;
    parent = fgDirect2D::instance->debugtarget->window->parent;
  }

  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgDebug_Destroy, (fgMessage)&fgDebugD2D_Message);

  AbsRect r = { 0 };
  ResolveRect(*self, &r);
  fgDirect2D::instance->debughwnd = fgContext::WndCreate(r, WS_EX_TOOLWINDOW, self, L"fgDebugD2D", fgDirect2D::instance->root.dpi);
  ShowWindow(fgDirect2D::instance->debughwnd, SW_SHOW);
  UpdateWindow(fgDirect2D::instance->debughwnd);
  //SetWindowPos(fgDirect2D::instance->debughwnd, HWND_TOP, INT(wleft), INT(wtop), INT(rwidth), INT(rheight), SWP_NOCOPYBITS | SWP_NOMOVE | SWP_NOACTIVATE);
  fgDirect2D::instance->debugcontext.CreateResources(fgDirect2D::instance->debughwnd);
}

template<class T, void (*INIT)(T* BSS_RESTRICT, fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, const char*, fgFlag, const fgTransform*, unsigned short)>
BSS_FORCEINLINE fgElement* d2d_create_default(fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units, const char* file, size_t line)
{
  T* r = (T*)malloc(sizeof(T));
  INIT(r, parent, next, name, flags, transform, units);
  ((fgElement*)r)->free = &free;
  return (fgElement*)r;
}

fgElement* fgCreateD2D(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  if(!_stricmp(type, "window")) // Neither of these have default flags so we don't bother with them.
    return d2d_create_default<fgWindowD2D, fgWindowD2D_Init>(parent, next, name, flags, transform, units, __FILE__, __LINE__);
  if(!_stricmp(type, "debug"))
    return d2d_create_default<fgDebug, fgDebugD2D_Init>(parent, next, name, flags, transform, units, __FILE__, __LINE__);
  return fgCreateDefault(type, parent, next, name, flags, transform, units);
}

struct fgFontD2D
{
  fgFontD2D(IDWriteTextFormat* _f, fgIntVec _dpi, unsigned int _pt) : format(_f), dpi(_dpi), pt(_pt) {}
  IDWriteTextFormat* format;
  fgIntVec dpi;
  unsigned int pt;
};

fgFont fgCreateFontD2D(fgFlag flags, const char* family, short weight, char italic, unsigned int pt, const fgIntVec* dpi)
{
  size_t len = fgUTF8toUTF16(family, -1, 0, 0);
  DYNARRAY(wchar_t, wtext, len);
  fgUTF8toUTF16(family, -1, wtext, len);
  IDWriteTextFormat* format = 0;
  LOGFAILURERET(fgDirect2D::instance->writefactory->CreateTextFormat(wtext, 0, DWRITE_FONT_WEIGHT(weight), italic ? DWRITE_FONT_STYLE_ITALIC : DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, pt * (dpi->x / 72.0f), L"en-us", &format), 0, "CreateTextFormat failed with error code %li", hr);
  if(!format) return 0;
  return new fgFontD2D(format, *dpi, pt);
}

fgFont fgCloneFontD2D(fgFont font, const struct _FG_FONT_DESC* desc)
{
  fgFontD2D* f = (fgFontD2D*)font;
  if(!desc || (desc->pt == f->pt && desc->dpi.x == f->dpi.x))
    f->format->AddRef();
  else
  {
    DYNARRAY(wchar_t, wtext, f->format->GetFontFamilyNameLength()+1);
    f->format->GetFontFamilyName(wtext, f->format->GetFontFamilyNameLength() + 1);
    IDWriteTextFormat* format = 0;
    LOGFAILURERET(fgDirect2D::instance->writefactory->CreateTextFormat(wtext, 0, f->format->GetFontWeight(), f->format->GetFontStyle(), f->format->GetFontStretch(), desc->pt * (desc->dpi.x / 72.0f), L"en-us", &format), 0, "CreateTextFormat failed with error code %li", hr);
    if(!format) return 0;
    f = new fgFontD2D(format, desc->dpi, desc->pt);
  }
  return f;
}
void fgDestroyFontD2D(fgFont font) { fgFontD2D* f = (fgFontD2D*)font; if(!f->format->Release()) delete f; }
void fgDrawFontD2D(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, unsigned int color, const AbsRect* dpiarea, FABS rotation, const AbsVec* dpicenter, fgFlag flags, const fgDrawAuxData* data, void* cache)
{
  if(!len) return;
  fgFontD2D* f = (fgFontD2D*)font;
  AbsRect area;
  AbsVec center;
  fgResolveDrawRect(dpiarea, &area, dpicenter, &center, flags, data);
  GETEXDATA(data);
  IDWriteTextLayout* layout = (IDWriteTextLayout*)cache;
  exdata->context->color->SetColor(ToD2Color(color));
  
  if(!layout)
  { // We CANNOT input the string directly from the DLL for some unbelievably stupid reason. We must make a copy in this DLL and pass that to Direct2D.
    DYNARRAY(wchar_t, wtext, len);
    memcpy_s(wtext, len*sizeof(wchar_t), text, len * sizeof(wchar_t));
    exdata->context->target->DrawTextW(wtext, len, f->format, D2D1::RectF(area.left, area.top, area.right, area.bottom), exdata->context->color, (flags&FGELEMENT_NOCLIP) ? D2D1_DRAW_TEXT_OPTIONS_NONE : D2D1_DRAW_TEXT_OPTIONS_CLIP);
  }
  else
    exdata->context->target->DrawTextLayout(D2D1::Point2F(area.left, area.top), layout, exdata->context->color, D2D1_DRAW_TEXT_OPTIONS_CLIP);
}
void* fgFontFormatD2D(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, AbsRect* area, fgFlag flags, const fgIntVec* dpi, void* cache)
{
  fgFontD2D* f = (fgFontD2D*)font;
  IDWriteTextLayout* layout = (IDWriteTextLayout*)cache;
  if(layout)
    layout->Release();
  if(!area)
    return 0;
  fgScaleRectDPI(area, dpi->x, dpi->y);
  layout = CreateD2DLayout(f->format, (const wchar_t*)text, len, area);
  if(!layout) return 0;
  FLOAT linespacing;
  FLOAT baseline;
  DWRITE_LINE_SPACING_METHOD method;
  layout->GetLineSpacing(&method, &linespacing, &baseline);
  if(lineheight > 0.0f)
    layout->SetLineSpacing(DWRITE_LINE_SPACING_METHOD_UNIFORM, lineheight, baseline * (lineheight / linespacing));
  layout->SetWordWrapping((flags&(FGTEXT_CHARWRAP | FGTEXT_WORDWRAP)) ? DWRITE_WORD_WRAPPING_WRAP : DWRITE_WORD_WRAPPING_NO_WRAP);
  layout->SetReadingDirection((flags&FGTEXT_RTL) ? DWRITE_READING_DIRECTION_RIGHT_TO_LEFT : DWRITE_READING_DIRECTION_LEFT_TO_RIGHT);
  if(flags&FGTEXT_RIGHTALIGN)
    layout->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_FAR);
  if(flags&FGTEXT_CENTER)
    layout->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
  if(letterspacing > 0.0f)
  {
    DWRITE_TEXT_RANGE range;
    FLOAT leading, trailing, minimum;
    IDWriteTextLayout1* layout1 = 0;
    layout->QueryInterface<IDWriteTextLayout1>(&layout1);
    layout1->GetCharacterSpacing(0, &leading, &trailing, &minimum, &range);
    layout1->SetCharacterSpacing(leading, trailing + letterspacing, minimum, range);
  }

  DWRITE_TEXT_METRICS metrics;
  layout->GetMetrics(&metrics);
  area->right = area->left + metrics.width;
  area->bottom = area->top + metrics.height;
  fgInvScaleRectDPI(area, dpi->x, dpi->y);
  layout->SetMaxWidth(metrics.width);
  layout->SetMaxHeight(metrics.height);
  return layout;
}
void fgFontGetD2D(fgFont font, struct _FG_FONT_DESC* desc)
{
  fgFontD2D* f = (fgFontD2D*)font;
  if(desc)
  {
    FLOAT lineheight;
    FLOAT baseline;
    DWRITE_LINE_SPACING_METHOD method;
    f->format->GetLineSpacing(&method, &lineheight, &baseline);
    desc->lineheight = lineheight;
    desc->pt = f->pt;
    desc->dpi = f->dpi;
  }
}
size_t fgFontIndexD2D(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* dpiarea, fgFlag flags, AbsVec pos, AbsVec* cursor, const fgIntVec* dpi, void* cache)
{
  assert(font != 0);
  fgFontD2D* f = (fgFontD2D*)font;
  IDWriteTextLayout* layout = (IDWriteTextLayout*)cache;
  AbsRect area = *dpiarea;
  fgScaleRectDPI(&area, dpi->x, dpi->y);
  if(!layout)
    layout = CreateD2DLayout(f->format, (const wchar_t*)text, len, &area);
  if(!layout)
    return 0;

  BOOL trailing;
  BOOL inside;
  DWRITE_HIT_TEST_METRICS hit;
  fgScaleVecDPI(&pos, dpi->x, dpi->y);
  layout->HitTestPoint(pos.x, pos.y, &trailing, &inside, &hit);

  cursor->x = hit.left;
  cursor->y = hit.top;
  fgInvScaleVecDPI(cursor, dpi->x, dpi->y);
  return hit.textPosition;
}
AbsVec fgFontPosD2D(fgFont font, const void* text, size_t len, float lineheight, float letterspacing, const AbsRect* dpiarea, fgFlag flags, size_t index, const fgIntVec* dpi, void* cache)
{
  fgFontD2D* f = (fgFontD2D*)font;
  IDWriteTextLayout* layout = (IDWriteTextLayout*)cache;
  AbsRect area = *dpiarea;
  fgScaleRectDPI(&area, dpi->x, dpi->y);
  if(!layout)
    layout = CreateD2DLayout(f->format, (const wchar_t*)text, len, &area);
  if(!layout)
    return AbsVec{ 0,0 };

  FLOAT x, y;
  DWRITE_HIT_TEST_METRICS hit;
  layout->HitTestTextPosition(index, false, &x, &y, &hit);
  AbsVec p = { x, y };
  fgInvScaleVecDPI(&p, dpi->x, dpi->y);
  return p;
}

void* fgCreateAssetD2D(fgFlag flags, const char* data, size_t length)
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
  if(FAILED(hr))
    fgLog("fgCreateAssetD2D failed with error code %li", hr);

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
fgAsset fgCloneAssetD2D(fgAsset asset, fgElement* src)
{ 
  fgWindowD2D* window = GetElementWindow(src);
  if(!window)
  {
    ((IUnknown*)asset)->AddRef();
    return asset;
  }
  assert(window->window->destroy == (fgDestroy)fgWindowD2D_Destroy);
  ID2D1HwndRenderTarget* target = window->context.target;

  IWICFormatConverter *conv = NULL;
  ID2D1Bitmap* bitmap = NULL;
  HRESULT hr = fgDirect2D::instance->wicfactory->CreateFormatConverter(&conv);

  if(SUCCEEDED(hr)) // Convert the image format to 32bppPBGRA (DXGI_FORMAT_B8G8R8A8_UNORM + D2D1_ALPHA_MODE_PREMULTIPLIED).
    hr = conv->Initialize((IWICBitmapSource*)asset, GUID_WICPixelFormat32bppPBGRA, WICBitmapDitherTypeNone, NULL, 0.f, WICBitmapPaletteTypeMedianCut);
  if(SUCCEEDED(hr))
    hr = target->CreateBitmapFromWicBitmap(conv, NULL, &bitmap);
  if(FAILED(hr))
    fgLog("fgCloneAssetD2D failed with error code %li", hr);

  return bitmap;
}
void fgDestroyAssetD2D(fgAsset asset) { ((IUnknown*)asset)->Release(); }
void fgDrawAssetD2D(fgAsset asset, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* dpiarea, FABS rotation, const AbsVec* dpicenter, fgFlag flags, const fgDrawAuxData* data)
{
  AbsRect area;
  AbsVec center;
  fgResolveDrawRect(dpiarea, &area, dpicenter, &center, flags, data);
  GETEXDATA(data);
  assert(exdata->context != 0);
  assert(exdata->context->target != 0);

  D2D1_MATRIX_3X2_F world;
  exdata->context->target->GetTransform(&world);
  if(rotation != 0) // WHY THE FUCK DOES THIS TAKE DEGREES?!
    exdata->context->target->SetTransform(D2D1::Matrix3x2F::Rotation(rotation * 180.0f / PI, D2D1::Point2F(center.x, center.y))*world);
  
  ID2D1Bitmap* tex = 0;
  if(asset)
  {
    ((IUnknown*)asset)->QueryInterface<ID2D1Bitmap>(&tex);
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
  D2D1_RECT_F rect = D2D1::RectF(area.left, area.top, area.right, area.bottom);
  exdata->context->color->SetColor(ToD2Color(color));
  exdata->context->edgecolor->SetColor(ToD2Color(edge));
  ID2D1Effect* e = 0;

  switch(flags&FGRESOURCE_SHAPEMASK)
  {
    case FGRESOURCE_RECT:
      e = exdata->context->roundrect;
      break;
    case FGRESOURCE_CIRCLE:
      e = exdata->context->circle;
      break;
    case FGRESOURCE_TRIANGLE:
      e = exdata->context->triangle;
      break;
  }

  if(e)
  {
    e->SetValue(0, D2D1::Vector4F(area.left, area.top, area.right, area.bottom));
    e->SetValue(1, D2D1::Vector4F(uvresolve.left, uvresolve.top, uvresolve.right, uvresolve.bottom));
    e->SetValue(2, color);
    e->SetValue(3, edge);
    e->SetValue(4, outline);
    exdata->context->context->DrawImage(e, &D2D1::Point2F(rect.left, rect.top), &rect);
  }
  else
    exdata->context->target->DrawBitmap(tex, rect, (color >> 24) / 255.0f, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, uvresolve);

  exdata->context->target->SetTransform(world);
}

void fgAssetSizeD2D(fgAsset asset, const CRect* uv, AbsVec* dim, fgFlag flags)
{
  ID2D1Bitmap* tex = (ID2D1Bitmap*)asset;
  auto sz = tex->GetPixelSize();
  D2D1_RECT_F uvresolve = D2D1::RectF(uv->left.rel + (uv->left.abs / sz.width),
    uv->top.rel + (uv->top.abs / sz.height),
    uv->right.rel + (uv->right.abs / sz.width),
    uv->bottom.rel + (uv->bottom.abs / sz.height));
  dim->x = uvresolve.right - uvresolve.left;
  dim->y = uvresolve.bottom - uvresolve.top;
}

void fgDrawLinesD2D(const AbsVec* p, size_t n, unsigned int color, const AbsVec* translate, const AbsVec* scale, FABS rotation, const AbsVec* center, const fgDrawAuxData* data)
{
  AbsVec t = *translate;
  fgScaleVecDPI(&t, data->dpi.x, data->dpi.y);
  GETEXDATA(data);
  D2D1_MATRIX_3X2_F world;
  exdata->context->target->GetTransform(&world);
  exdata->context->target->SetTransform(
    D2D1::Matrix3x2F::Rotation(rotation * 180.0f / PI, D2D1::Point2F(center->x, center->y))*
    D2D1::Matrix3x2F::Scale(scale->x * (data->dpi.x / 96.0f), scale->y * (data->dpi.y / 96.0f))*
    world*
    D2D1::Matrix3x2F::Translation(t.x + 0.5, t.y + 0.5));

  exdata->context->color->SetColor(ToD2Color(color));
  for(size_t i = 1; i < n; ++i)
    exdata->context->target->DrawLine(D2D1_POINT_2F{ p[i - 1].x, p[i - 1].y }, D2D1_POINT_2F{ p[i].x, p[i].y }, exdata->context->color, 1.0F, 0);
  //bss_util::Matrix<float, 4, 4>::AffineTransform_T(translate->x, translate->y, 0, rotation, center->x, center->y, m);
  exdata->context->target->SetTransform(world);
}

void fgPushClipRectD2D(const AbsRect* clip, const fgDrawAuxData* data)
{
  GETEXDATA(data);
  assert(exdata->context->cliprect.size() > 0);
  AbsRect cliprect = exdata->context->cliprect.top();
  cliprect = { bssmax(floor(clip->left), cliprect.left), bssmax(floor(clip->top), cliprect.top), bssmin(ceil(clip->right), cliprect.right), bssmin(ceil(clip->bottom), cliprect.bottom) };
  exdata->context->cliprect.push(cliprect);
  fgScaleRectDPI(&cliprect, data->dpi.x, data->dpi.y);
  exdata->context->target->PushAxisAlignedClip(D2D1::RectF(cliprect.left, cliprect.top, cliprect.right, cliprect.bottom), D2D1_ANTIALIAS_MODE_ALIASED);
}

AbsRect fgPeekClipRectD2D(const fgDrawAuxData* data)
{
  if(data->fgSZ != sizeof(fgDrawAuxDataEx)) return AbsRect{ 0,0,0,0 };
  fgDrawAuxDataEx* exdata = (fgDrawAuxDataEx*)data;
  return exdata->context->cliprect.top();
}

void fgPopClipRectD2D(const fgDrawAuxData* data)
{
  GETEXDATA(data);
  exdata->context->cliprect.pop();
  exdata->context->target->PopAxisAlignedClip();
  assert(exdata->context->cliprect.size() > 0);
}

void fgDirtyElementD2D(fgElement* e)
{
  static fgElement* lasttop = 0;

  if(e->flags&FGELEMENT_SILENT)
    return;
  if(!fgDirect2D::instance->root.topmost && lasttop != 0)
  {
    ReleaseCapture();
    ShowWindow(fgDirect2D::instance->tophwnd, SW_HIDE);
    lasttop = 0;
  }
  while(e && e->destroy != (fgDestroy)fgWindowD2D_Destroy && e != fgDirect2D::instance->root.topmost && e->destroy != (fgDestroy)fgDebug_Destroy)
    e = e->parent;

  if(fgDirect2D::instance->root.topmost != 0 && e == fgDirect2D::instance->root.topmost)
  {
    AbsRect& toprect = fgDirect2D::instance->toprect;
    ResolveNoClipRect(fgDirect2D::instance->root.topmost, &toprect, 0, 0);
    fgIntVec& dpi = fgDirect2D::instance->root.topmost->GetDPI();
    fgScaleRectDPI(&toprect, dpi.x, dpi.y); // SetWindowPos will resize the direct2D background via the WndProc callback
    bool ignore = (fgDirect2D::instance->root.topmost->flags&FGELEMENT_IGNORE) != 0;
    SetWindowLong(fgDirect2D::instance->tophwnd, GWL_EXSTYLE, WS_EX_COMPOSITED | WS_EX_TOOLWINDOW | (ignore ? (WS_EX_LAYERED | WS_EX_TRANSPARENT) : 0));
    SetWindowPos(fgDirect2D::instance->tophwnd, HWND_TOP, toprect.left, toprect.top, toprect.right - toprect.left, toprect.bottom - toprect.top, SWP_NOSENDCHANGING | SWP_NOACTIVATE | (ignore ? SWP_NOACTIVATE : 0));
    fgInvScaleRectDPI(&toprect, dpi.x, dpi.y);
    ShowWindow(fgDirect2D::instance->tophwnd, SW_SHOW);
    if(lasttop != fgDirect2D::instance->root.topmost)
    {
      lasttop = fgDirect2D::instance->root.topmost;
      if(!ignore)
        SetCapture(fgDirect2D::instance->tophwnd);
    }
    fgDirect2D::instance->topcontext.InvalidateHWND(fgDirect2D::instance->tophwnd);
  }
  else if(e != 0 && e->destroy == (fgDestroy)fgDebug_Destroy)
    fgDirect2D::instance->debugcontext.InvalidateHWND(fgDirect2D::instance->debughwnd);
  else if(e)
    ((fgWindowD2D*)e)->context.InvalidateHWND(((fgWindowD2D*)e)->handle);
}

void fgSetCursorD2D(uint32_t type, void* custom)
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

void fgClipboardCopyD2D(uint32_t type, const void* data, size_t length)
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

char fgClipboardExistsD2D(uint32_t type)
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

const void* fgClipboardPasteD2D(uint32_t type, size_t* length)
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

void fgClipboardFreeD2D(const void* mem)
{
  free(const_cast<void*>(mem));
}

void fgDragStartD2D(char type, void* data, fgElement* draw)
{
  fgRoot* root = fgSingleton();
  root->dragtype = type;
  root->dragdata = data;
  root->dragdraw = draw;
}

char fgProcessMessagesD2D()
{
  fgProcessMessagesDefault();
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
      fgDebug_Hide();
      return 0;
    }
  }

  return 1;
}

longptr_t __stdcall fgDirect2D::TopmostWndProc(HWND__* hWnd, unsigned int message, size_t wParam, longptr_t lParam)
{
  fgDirect2D* self = reinterpret_cast<fgDirect2D*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  if(self)
  {
    switch(message)
    {
    case WM_PAINT:
      if(self->root.topmost != 0)
      {
        AbsRect area;
        ResolveRect(self->root.topmost, &area);
        fgDrawAuxDataEx exdata;
        self->topcontext.BeginDraw(self->tophwnd, self->root.topmost, &self->toprect, exdata);
        self->root.topmost->Draw(&area, &exdata.data);
        self->topcontext.EndDraw();
      }
      return 0;
    }

    if(self->root.topmost != 0)
      return self->topcontext.WndProc(hWnd, message, wParam, lParam, self->root.topmost);
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

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

struct _FG_ROOT* fgInitialize()
{
  static fgBackend BACKEND = {
    FGTEXTFMT_UTF16,
    &fgCreateFontD2D,
    &fgCloneFontD2D,
    &fgDestroyFontD2D,
    &fgDrawFontD2D,
    &fgFontFormatD2D,
    &fgFontGetD2D,
    &fgFontIndexD2D,
    &fgFontPosD2D,
    &fgCreateAssetD2D,
    &fgCloneAssetD2D,
    &fgDestroyAssetD2D,
    &fgDrawAssetD2D,
    &fgAssetSizeD2D,
    &fgDrawLinesD2D,
    &fgCreateD2D,
    &fgFlagMapDefault,
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
    &fgBehaviorHookListener,
    &fgProcessMessagesD2D,
    &fgLoadExtensionDefault,
    &fgLogHookDefault,
    &fgTerminateD2D,
  };

  signal(SIGABRT, [](int) {*((int*)0) = 0; });
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

  AbsRect extent = { GetSystemMetrics(SM_XVIRTUALSCREEN), GetSystemMetrics(SM_YVIRTUALSCREEN), GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN) };
  extent.right += extent.left;
  extent.bottom += extent.top;

  fgDirect2D* root = reinterpret_cast<fgDirect2D*>(calloc(1, sizeof(fgDirect2D)));
  new(root) fgDirect2D();
  fgDirect2D::instance = root;
  HDC hdc = GetDC(NULL);
  fgIntVec dpi = { GetDeviceCaps(hdc, LOGPIXELSX), GetDeviceCaps(hdc, LOGPIXELSY) };
  ReleaseDC(NULL, hdc);
  fgRoot_Init(&root->root, &extent, &dpi, &BACKEND);

  fgLog("Initializing fgDirect2D...");

#ifdef BSS_DEBUG
  HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);
#endif

  hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory, (LPVOID*)&root->wicfactory);
  if(SUCCEEDED(hr))
    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory1), reinterpret_cast<IUnknown**>(&root->writefactory));
  else
    fgLog("CoCreateInstance() failed with error: %li", hr);

  D2D1_FACTORY_OPTIONS d2dopt = { D2D1_DEBUG_LEVEL_NONE };
#ifdef BSS_DEBUG
  d2dopt.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif

  if(SUCCEEDED(hr))
    hr = D2D1CreateFactory<ID2D1Factory1>(D2D1_FACTORY_TYPE_SINGLE_THREADED, d2dopt, &root->factory);
  else
    fgLog("DWriteCreateFactory() failed with error: %li", hr);

  if(FAILED(hr))
  {
    fgLog("D2D1CreateFactory() failed with error: %li", hr);
    if(root->wicfactory) root->wicfactory->Release();
    if(root->factory) root->factory->Release();
    if(root->writefactory) root->writefactory->Release();
    fgRoot_Destroy(&root->root);
    free(root);
    return 0;
  }

  if(FAILED(fgRoundRect::Register(root->factory)))
    fgLog("Failed to register fgRoundRect", hr);
  if(FAILED(fgCircle::Register(root->factory)))
    fgLog("Failed to register fgCircle", hr);
  if(FAILED(fgTriangle::Register(root->factory)))
    fgLog("Failed to register fgTriangle", hr);

  fgContext_Construct(&root->topcontext);
  fgContext::SetDWMCallbacks();
  fgContext::WndRegister(fgWindowD2D::WndProc, L"fgWindowD2D"); // Register window class
  fgContext::WndRegister(fgDirect2D::TopmostWndProc, L"fgDirectD2Dtopmost"); // Register topmost class
  fgContext::WndRegister(fgDirect2D::DebugWndProc, L"fgDebugD2D"); // Register topmost class

  AbsRect empty = { 0 };
  root->tophwnd = fgContext::WndCreate(empty, WS_EX_TOOLWINDOW, root, L"fgDirectD2Dtopmost", dpi);
  ShowWindow(root->tophwnd, SW_SHOW);
  UpdateWindow(root->tophwnd);
  //SetWindowPos(root->topmost, HWND_TOP, INT(wleft), INT(wtop), INT(rwidth), INT(rheight), SWP_NOCOPYBITS | SWP_NOMOVE | SWP_NOACTIVATE);
  root->topcontext.CreateResources(root->tophwnd);

  FLOAT dpix;
  FLOAT dpiy;
  root->factory->GetDesktopDpi(&dpix, &dpiy);
  root->root.dpi.x = (int)dpix;
  root->root.dpi.y = (int)dpiy;

  HMODULE shcore = LoadLibraryW(L"Shcore.dll"); // If we're on windows 8.1 or above, the user can set per-monitor DPI
  if(shcore)
    pGetDpiForMonitor = (GETDPIFORMONITOR)GetProcAddress(shcore, "GetDpiForMonitor");
  EnumDisplayMonitors(0, 0, SpawnMonitorsProc, (LPARAM)root);
  if(shcore)
    FreeLibrary(shcore);

  DWORD blinkrate = 0;
  int64_t sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate", 0, 0);
  if(sz > 0)
  {
    DYNARRAY(wchar_t, buf, sz / 2);
    sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate", (unsigned char*)buf, sz);
    if(sz > 0)
      root->root.cursorblink = _wtoi(buf) / 1000.0;
  }
  else
    fgLog("Couldn't get user's cursor blink rate.");

  return &root->root;
}
