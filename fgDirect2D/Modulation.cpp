// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "win32_includes.h"
#include <d2d1effecthelpers.h>
#include <initguid.h>
#include "Modulation.h"
#include "Direct2D_Modulation.h"
#include <utility>
#include <memory>

#define XML(x) TEXT(#x)

using namespace D2D;

Modulation::Modulation() {}
Modulation::~Modulation()
{}
IFACEMETHODIMP Modulation::Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph)
{
  HRESULT hr = pContextInternal->LoadPixelShader(CLSID_ModulationPixelShader, Modulation_main, sizeof(Modulation_main));

  if(SUCCEEDED(hr))
    hr = pTransformGraph->SetSingleTransformNode(this);

  return hr;
}
HRESULT Modulation::Register(_In_ ID2D1Factory1* pFactory)
{
  PCWSTR pszXml =
    XML(
      <?xml version='1.0'?>
      <Effect>
        <!-- System Properties -->
        <Property name = 'DisplayName' type = 'string' value = 'Color Modulation' />
        <Property name = 'Author' type = 'string' value = 'Fundament Software' />
        <Property name = 'Category' type = 'string' value = 'Bitmap' />
        <Property name = 'Description' type = 'string' value = 'Draws an image with color modulation.' />
        <Inputs>
          <Input name='Image'/>
        </Inputs>
        <Property name='Color' type='uint32'>
          <Property name='DisplayName' type='string' value='Color'/>
          <Property name='Default' type='uint32' value='0' />
        </Property>
      </Effect>
    );

  // This defines the bindings from specific properties to the callback functions
  // on the class that ID2D1Effect::SetValue() & GetValue() will call.
  const D2D1_PROPERTY_BINDING bindings[] =
  {
    D2D1_VALUE_TYPE_BINDING(L"Color", &SetColor, &GetColor),
  };

  return pFactory->RegisterEffectFromString(
    CLSID_Modulation,
    pszXml,
    bindings,
    ARRAYSIZE(bindings),
    CreateEffect
  );
}

HRESULT __stdcall Modulation::CreateEffect(_Outptr_ IUnknown** ppEffectImpl)
{
  // This code assumes that the effect class initializes its reference count to 1.
  *ppEffectImpl = static_cast<ID2D1EffectImpl*>(new Modulation());

  if(*ppEffectImpl == nullptr)
    return E_OUTOFMEMORY;
  return S_OK;
}

IFACEMETHODIMP Modulation::SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo)
{
  _drawInfo = pRenderInfo;
  _drawInfo->AddRef();
  return _drawInfo->SetPixelShader(CLSID_ModulationPixelShader);
}