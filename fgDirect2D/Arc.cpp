// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "win32_includes.h"
#include <d2d1effecthelpers.h>
#include <initguid.h>
#include "Arc.h"
#include "Direct2D_Arc.h"
#include <utility>
#include <memory>

#define XML(x) TEXT(#x)

using namespace D2D;

Arc::Arc() {}
Arc::~Arc() {}
IFACEMETHODIMP Arc::Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph)
{
  HRESULT hr = pContextInternal->LoadPixelShader(CLSID_ArcPixelShader, Arc_main, sizeof(Arc_main));

  if(SUCCEEDED(hr))
    hr = pTransformGraph->SetSingleTransformNode(this);

  return hr;
}
HRESULT Arc::Register(_In_ ID2D1Factory1* pFactory)
{
  /* clang-format off */
  PCWSTR pszXml =
    XML(
      <?xml version='1.0'?>
      <Effect>
        <!-- System Properties -->
        <Property name = 'DisplayName' type = 'string' value = 'Arc' />
        <Property name = 'Author' type = 'string' value = 'Fundament Software' />
        <Property name = 'Category' type = 'string' value = 'Vector' />
        <Property name = 'Description' type = 'string' value = 'Draws an arc segment with an outline.' />
        <Inputs>
        </Inputs>
        <Property name='Rect' type='vector4'>
            <Property name='DisplayName' type='string' value='Output Rect'/>
            <Property name='Default' type='vector4' value='(0.0, 0.0, 0.0, 0.0)' />
        </Property>
        <Property name='InnerRadius' type='float'>
            <Property name='DisplayName' type='string' value='Inner Radius'/>
            <Property name='Default' type='float' value='0.0' />
        </Property>
        <Property name='Arcs' type='vector2'>
            <Property name='DisplayName' type='string' value='Arcs'/>
            <Property name='Default' type='vector2' value='(0.0, 0.0)' />
        </Property>
        <Property name='Fill' type='uint32'>
            <Property name='DisplayName' type='string' value='Fill'/>
            <Property name='Default' type='uint32' value='0' />
        </Property>
        <Property name='Outline' type='uint32'>
            <Property name='DisplayName' type='string' value='Outline Color'/>
            <Property name='Default' type='uint32' value='0' />
        </Property>
        <Property name='Border' type='float'>
            <Property name='DisplayName' type='string' value='Outline Width'/>
            <Property name='Min' type='float' value='0.0' />
            <Property name='Default' type='float' value='0.0' />
        </Property>
        <Property name='Blur' type='float'>
            <Property name='DisplayName' type='string' value='Blur Amount'/>
            <Property name='Min' type='float' value='0.0' />
            <Property name='Default' type='float' value='0.0' />
        </Property>
      </Effect>
      );
  /* clang-format on */

  // This defines the bindings from specific properties to the callback functions
  // on the class that ID2D1Effect::SetValue() & GetValue() will call.
  const D2D1_PROPERTY_BINDING bindings[] = {
    D2D1_VALUE_TYPE_BINDING(L"Rect", &SetRect, &GetRect),
    D2D1_VALUE_TYPE_BINDING(L"InnerRadius", &SetInnerRadius, &GetInnerRadius),
    D2D1_VALUE_TYPE_BINDING(L"Arcs", &SetArcs, &GetArcs),
    D2D1_VALUE_TYPE_BINDING(L"Fill", &SetFill, &GetFill),
    D2D1_VALUE_TYPE_BINDING(L"Outline", &SetOutline, &GetOutline),
    D2D1_VALUE_TYPE_BINDING(L"Border", &SetBorder, &GetBorder),
    D2D1_VALUE_TYPE_BINDING(L"Blur", &SetBlur, &GetBlur),
  };

  return pFactory->RegisterEffectFromString(CLSID_Arc, pszXml, bindings, ARRAYSIZE(bindings), CreateEffect);
}

HRESULT Arc::SetArcs(D2D_VECTOR_2F arcs)
{
  _constants.corners.x = arcs.x;
  _constants.corners.y = arcs.y;
  return S_OK;
}
D2D_VECTOR_2F Arc::GetArcs() const { return D2D1::Vector2F(_constants.corners.x, _constants.corners.y); }

HRESULT Arc::SetInnerRadius(FLOAT r)
{
  _constants.corners.z = r;
  return S_OK;
}
FLOAT Arc::GetInnerRadius() const { return _constants.corners.z; }

HRESULT __stdcall Arc::CreateEffect(_Outptr_ IUnknown** ppEffectImpl)
{
  // This code assumes that the effect class initializes its reference count to 1.
  *ppEffectImpl = static_cast<ID2D1EffectImpl*>(new Arc());

  if(*ppEffectImpl == nullptr)
    return E_OUTOFMEMORY;
  return S_OK;
}

IFACEMETHODIMP Arc::SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo)
{
  _drawInfo = pRenderInfo;
  _drawInfo->AddRef();
  return _drawInfo->SetPixelShader(CLSID_ArcPixelShader);
}