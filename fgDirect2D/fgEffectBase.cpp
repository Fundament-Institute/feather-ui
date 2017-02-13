// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "util.h"
#include <initguid.h>
#include "fgEffectBase.h"
#include "bss-util/bss_defines.h"
#include <utility>
#include <memory>

#define XML(x) TEXT(#x)

fgEffectBase::fgEffectBase() : _ref(1), _drawInfo(0)
{ 
  assert(sizeof(_constants) == sizeof(float)*(4*4 + 1));
  memset(&_constants, 0, sizeof(_constants));
}
fgEffectBase::~fgEffectBase()
{
  if(_drawInfo)
    _drawInfo->Release();
}
IFACEMETHODIMP fgEffectBase::PrepareForRender(D2D1_CHANGE_TYPE changeType)
{
  return UpdateConstants();
}
IFACEMETHODIMP fgEffectBase::SetGraph(_In_ ID2D1TransformGraph* pGraph) { return E_NOTIMPL; }
IFACEMETHODIMP_(ULONG) fgEffectBase::AddRef() { return ++_ref; }
IFACEMETHODIMP_(ULONG) fgEffectBase::Release()
{
  if(--_ref > 0)
    return _ref;
  delete this;
  return 0;
}
IFACEMETHODIMP fgEffectBase::QueryInterface(_In_ REFIID riid, _Outptr_ void** ppOutput)
{
  *ppOutput = nullptr;
  HRESULT hr = S_OK;

  if(riid == __uuidof(ID2D1EffectImpl))
  {
    *ppOutput = reinterpret_cast<ID2D1EffectImpl*>(this);
  }
  else if(riid == __uuidof(ID2D1DrawTransform))
  {
    *ppOutput = static_cast<ID2D1DrawTransform*>(this);
  }
  else if(riid == __uuidof(ID2D1Transform))
  {
    *ppOutput = static_cast<ID2D1Transform*>(this);
  }
  else if(riid == __uuidof(ID2D1TransformNode))
  {
    *ppOutput = static_cast<ID2D1TransformNode*>(this);
  }
  else if(riid == __uuidof(IUnknown))
  {
    *ppOutput = this;
  }
  else
  {
    hr = E_NOINTERFACE;
  }

  if(*ppOutput != nullptr)
  {
    AddRef();
  }

  return hr;
}

HRESULT fgEffectBase::SetRect(D2D_VECTOR_4F rect)
{
  _rect = D2D1::RectL(rect.x, rect.y, rect.z, rect.w);
  _constants.rect = rect;
  return S_OK;
}
D2D_VECTOR_4F fgEffectBase::GetRect() const { return _constants.rect; }

HRESULT fgEffectBase::SetCorners(D2D_VECTOR_4F corners) { _constants.corners = corners; return S_OK; }
D2D_VECTOR_4F fgEffectBase::GetCorners() const { return _constants.corners; }

HRESULT fgEffectBase::SetOutline(FLOAT outline) { _constants.outline = outline; return S_OK; }
FLOAT fgEffectBase::GetOutline() const { return _constants.outline; }

HRESULT fgEffectBase::SetColor(UINT color)
{
  _color = color;
  _constants.color = ToD2Color(color);
  return S_OK;
}
UINT fgEffectBase::GetColor() const { return _color; }

HRESULT fgEffectBase::SetOutlineColor(UINT color)
{
  _outlinecolor = color;
  _constants.outlinecolor = ToD2Color(color);
  return S_OK;
}
UINT fgEffectBase::GetOutlineColor() const { return _outlinecolor; }

IFACEMETHODIMP fgEffectBase::MapOutputRectToInputRects(_In_ const D2D1_RECT_L* pOutputRect, _Out_writes_(inputRectCount) D2D1_RECT_L* pInputRects, UINT32 inputRectCount) const
{
  return E_INVALIDARG;
}
IFACEMETHODIMP fgEffectBase::MapInputRectsToOutputRect(_In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputRects, _In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputOpaqueSubRects, UINT32 inputRectCount, _Out_ D2D1_RECT_L* pOutputRect, _Out_ D2D1_RECT_L* pOutputOpaqueSubRect)
{
  *pOutputRect = _rect;
  ZeroMemory(pOutputOpaqueSubRect, sizeof(*pOutputOpaqueSubRect)); // Entire image might be transparent
  return S_OK;
}
IFACEMETHODIMP fgEffectBase::MapInvalidRect(UINT32 inputIndex, D2D1_RECT_L invalidInputRect, _Out_ D2D1_RECT_L* pInvalidOutputRect) const { *pInvalidOutputRect = _rect; return S_OK; }
IFACEMETHODIMP_(UINT32) fgEffectBase::GetInputCount() const { return 0; }

HRESULT fgEffectBase::UpdateConstants()
{
  return _drawInfo->SetPixelShaderConstantBuffer(reinterpret_cast<BYTE*>(&_constants), sizeof(_constants));
}