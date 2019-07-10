// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "util.h"
#include <initguid.h>
#include "ImageEffectBase.h"
#include <utility>
#include <memory>

using namespace D2D;

ImageEffectBase::ImageEffectBase() : _ref(1), _drawInfo(0)
{
  fgassert(sizeof(_constants) == sizeof(float)*4);
  memset(&_constants, 0, sizeof(_constants));
}
ImageEffectBase::~ImageEffectBase()
{
  if(_drawInfo)
    _drawInfo->Release();
}
IFACEMETHODIMP ImageEffectBase::PrepareForRender(D2D1_CHANGE_TYPE changeType)
{
  return UpdateConstants();
}
IFACEMETHODIMP ImageEffectBase::SetGraph(_In_ ID2D1TransformGraph* pGraph) { return E_NOTIMPL; }
IFACEMETHODIMP_(ULONG) ImageEffectBase::AddRef() { return ++_ref; }
IFACEMETHODIMP_(ULONG) ImageEffectBase::Release()
{
  if(--_ref > 0)
    return _ref;
  delete this;
  return 0;
}
IFACEMETHODIMP ImageEffectBase::QueryInterface(_In_ REFIID riid, _Outptr_ void** ppOutput)
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

HRESULT ImageEffectBase::SetColor(UINT color)
{
  _color = color;
  _constants.color = ToD2Color(color);
  return S_OK;
}
UINT ImageEffectBase::GetColor() const { return _color; }

IFACEMETHODIMP ImageEffectBase::MapOutputRectToInputRects(_In_ const D2D1_RECT_L* pOutputRect, _Out_writes_(inputRectCount) D2D1_RECT_L* pInputRects, UINT32 inputRectCount) const
{
  if(inputRectCount != 1)
    return E_INVALIDARG;

  pInputRects[0] = *pOutputRect;
  return S_OK;
}
IFACEMETHODIMP ImageEffectBase::MapInputRectsToOutputRect(_In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputRects, _In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputOpaqueSubRects, UINT32 inputRectCount, _Out_ D2D1_RECT_L* pOutputRect, _Out_ D2D1_RECT_L* pOutputOpaqueSubRect)
{    
  if(inputRectCount != 1)
    return E_INVALIDARG;

  *pOutputRect = pInputRects[0];
  if((_color >> 24) != 0xFF)
    ZeroMemory(pOutputOpaqueSubRect, sizeof(*pOutputOpaqueSubRect)); // Entire image might be transparent
  else
    *pOutputOpaqueSubRect = pInputOpaqueSubRects[0]; // Otherwise maintain transparency
  _inputRect = pInputRects[0];

  return S_OK;
}
IFACEMETHODIMP ImageEffectBase::MapInvalidRect(UINT32 inputIndex, D2D1_RECT_L invalidInputRect, _Out_ D2D1_RECT_L* pInvalidOutputRect) const
{  
  if(inputIndex != 0)
    return E_INVALIDARG;
  *pInvalidOutputRect = invalidInputRect;
  return S_OK;
}
IFACEMETHODIMP_(UINT32) ImageEffectBase::GetInputCount() const { return 1; }

HRESULT ImageEffectBase::UpdateConstants()
{
  return _drawInfo->SetPixelShaderConstantBuffer(reinterpret_cast<BYTE*>(&_constants), sizeof(_constants));
}