// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgRoundRect.h"
#include "fgDirect2D_fgRoundRect.h"
#include "bss-util/bss_defines.h"
#include <initguid.h>
#include <utility>
#include <memory>

#define XML(x) TEXT(#x)
DEFINE_GUID(CLSID_fgRoundRect, 0xC58204BE, 0x9FA1, 0x4884, 0xB1, 0x82, 0xBF, 0xE8, 0x5D, 0x65, 0x90, 0xDC);
DEFINE_GUID(CLSID_fgRoundRectPixelShader, 0x0050F29E, 0xF56A, 0xAC, 0xB2, 0x5F, 0x23, 0x90, 0x46, 0x94, 0xAA, 0x5F);

fgRoundRect::fgRoundRect() : _ref(0) {}

IFACEMETHODIMP fgRoundRect::Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph)
{
  _effectContext = pContextInternal;
  HRESULT hr = pContextInternal->LoadPixelShader(CLSID_fgRoundRectPixelShader, fgRoundRect_main, sizeof(fgRoundRect_main));

  if(SUCCEEDED(hr))
    hr = pTransformGraph->SetSingleTransformNode(this);

  return hr;
}
IFACEMETHODIMP fgRoundRect::PrepareForRender(D2D1_CHANGE_TYPE changeType)
{
  return UpdateConstants();
}
IFACEMETHODIMP fgRoundRect::SetGraph(_In_ ID2D1TransformGraph* pGraph) { return E_NOTIMPL; }
HRESULT fgRoundRect::Register(_In_ ID2D1Factory1* pFactory)
{
  PCWSTR pszXml =
    XML(
      <? xml version = '1.0' ?>
      <Effect>
      <!--System Properties-->
      <Property name = 'DisplayName' type = 'string' value = 'Round Rectangle' / >
      <Property name = 'Author' type = 'string' value = 'Black Sphere Studios' / >
      <Property name = 'Category' type = 'string' value = 'Vector' / >
      <Property name = 'Description' type = 'string' value = 'Draws a rounded rectangle with an outline.' / >
      <Inputs>
      <Input name = 'Source' / >
      </Inputs>
      <!--Custom Properties go here-->
      <Property name = 'Corners' type = 'vector4'>
      <Property name = 'DisplayName' type = 'string' value = 'Corner Radii' / >
      <Property name = 'Min' type = 'float' value = '0.0' / >
      <Property name = 'Default' type = 'float' value = '0.0' / >
      </Property>
      <Property name = 'Outline' type = 'float'>
      <Property name = 'DisplayName' type = 'string' value = 'Outline Width' / >
      <Property name = 'Min' type = 'float' value = '0.0' / >
      <Property name = 'Default' type = 'float' value = '0.0' / >
      </Property>
      <Property name = 'Color' type = 'uint32'>
      <Property name = 'DisplayName' type = 'string' value = 'Color' / >
      </Property>
      <Property name = 'OutlineColor' type = 'uint32'>
      <Property name = 'DisplayName' type = 'string' value = 'Outline Color' / >
      </Property>
      </Effect>
    );

  // This defines the bindings from specific properties to the callback functions
  // on the class that ID2D1Effect::SetValue() & GetValue() will call.
  const D2D1_PROPERTY_BINDING bindings[] =
  {
    D2D1_VALUE_TYPE_BINDING(L"Corners", &SetCorners, &GetCorners),
    D2D1_VALUE_TYPE_BINDING(L"Outline", &SetOutline, &GetOutline),
    D2D1_VALUE_TYPE_BINDING(L"Color", &SetColor, &GetColor),
    D2D1_VALUE_TYPE_BINDING(L"OutlineColor", &SetOutlineColor, &GetOutlineColor),
  };
  return pFactory->RegisterEffectFromString(CLSID_fgRoundRect, L"< ? xml version = '1.0' ? ><Effect><Property name = 'DisplayName' type = 'string' value = 'Round Rectangle' / ></Effect>", 0, 0, CreateEffect);
  
  return pFactory->RegisterEffectFromString(
    CLSID_fgRoundRect,
    pszXml,
    bindings,
    ARRAYSIZE(bindings),
    CreateEffect
  );
}

HRESULT __stdcall fgRoundRect::CreateEffect(_Outptr_ IUnknown** ppEffectImpl)
{
  // This code assumes that the effect class initializes its reference count to 1.
  *ppEffectImpl = static_cast<ID2D1EffectImpl*>(new fgRoundRect());

  if(*ppEffectImpl == nullptr)
    return E_OUTOFMEMORY;
  return S_OK;
}
IFACEMETHODIMP_(ULONG) fgRoundRect::AddRef() { return ++_ref; }
IFACEMETHODIMP_(ULONG) fgRoundRect::Release()
{ 
  if(--_ref > 0)
    return _ref;
  delete this;
  return 0;
}
IFACEMETHODIMP fgRoundRect::QueryInterface(_In_ REFIID riid, _Outptr_ void** ppOutput)
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


HRESULT fgRoundRect::SetCorners(D2D_VECTOR_4F corners) { _constants.corners = corners; return S_OK; }
D2D_VECTOR_4F fgRoundRect::GetCorners() const { return _constants.corners; }

HRESULT fgRoundRect::SetOutline(FLOAT outline) { _constants.outline = outline; return S_OK; }
FLOAT fgRoundRect::GetOutline() const { return _constants.outline; }

HRESULT fgRoundRect::SetColor(UINT color) { _constants.color = color; return S_OK; }
UINT fgRoundRect::GetColor() const { return _constants.color; }

HRESULT fgRoundRect::SetOutlineColor(UINT color) { _constants.outlinecolor = color; return S_OK; }
UINT fgRoundRect::GetOutlineColor() const { return _constants.outlinecolor; }

IFACEMETHODIMP fgRoundRect::SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo)
{
  _drawInfo = pRenderInfo;
  return _drawInfo->SetPixelShader(CLSID_fgRoundRectPixelShader);
}
IFACEMETHODIMP fgRoundRect::MapOutputRectToInputRects(_In_ const D2D1_RECT_L* pOutputRect, _Out_writes_(inputRectCount) D2D1_RECT_L* pInputRects, UINT32 inputRectCount) const
{
  if(inputRectCount != 1)
    return E_INVALIDARG;
  pInputRects[0] = *pOutputRect;
  return S_OK;
}
IFACEMETHODIMP fgRoundRect::MapInputRectsToOutputRect(_In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputRects, _In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputOpaqueSubRects, UINT32 inputRectCount, _Out_ D2D1_RECT_L* pOutputRect, _Out_ D2D1_RECT_L* pOutputOpaqueSubRect)
{
  if(inputRectCount != 1)
    return E_INVALIDARG;
  *pOutputRect = pInputRects[0];
  _inputRect = pInputRects[0];
  ZeroMemory(pOutputOpaqueSubRect, sizeof(*pOutputOpaqueSubRect)); // Entire image might be transparent
  return S_OK;
}
IFACEMETHODIMP fgRoundRect::MapInvalidRect(UINT32 inputIndex, D2D1_RECT_L invalidInputRect, _Out_ D2D1_RECT_L* pInvalidOutputRect) const { *pInvalidOutputRect = _inputRect; return S_OK; }
IFACEMETHODIMP_(UINT32) fgRoundRect::GetInputCount() const { return 1; }

HRESULT fgRoundRect::UpdateConstants()
{
  return _drawInfo->SetPixelShaderConstantBuffer(reinterpret_cast<BYTE*>(&_constants), sizeof(_constants));
}