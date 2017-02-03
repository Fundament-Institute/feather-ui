// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "fgRoundRect.h"

#define XML(x) TEXT(#x)

fgRoundRect::fgRoundRect() : _ref(0) {}

IFACEMETHODIMP fgRoundRect::Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph)
{
  ID2D1OffsetTransform* p;
    HRESULT hr = pContextInternal->CreateOffsetTransform(
      D2D1::Point2L(100, 100),  // Offsets the input by 100px in each axis.
      &p
    );
  
    if(SUCCEEDED(hr))
      hr = pTransformGraph->SetSingleTransformNode(p);
  
    return hr;
}
IFACEMETHODIMP fgRoundRect::PrepareForRender(D2D1_CHANGE_TYPE changeType)
{
    //D2D1_POINT_2L pixelOffset;
    //pixelOffset.x = static_cast<LONG>(m_offset.x * (m_dpiX / 96.0f));
    //pixelOffset.y = static_cast<LONG>(m_offset.y * (m_dpiY / 96.0f));
    //m_pOffsetTransform->SetOffset(pixelOffset);
  
    return S_OK;
}
IFACEMETHODIMP fgRoundRect::SetGraph(_In_ ID2D1TransformGraph* pGraph) { return E_NOTIMPL; }
HRESULT fgRoundRect::Register(_In_ ID2D1Factory1* pFactory)
{
  /*
  PCWSTR pszXml =
    XML(
      < ? xml version = '1.0' ? >
      <Effect>
      <!--System Properties-->
      <Property name = 'DisplayName' type = 'string' value = 'Ripple' / >
      <Property name = 'Author' type = 'string' value = 'Microsoft Corporation' / >
      <Property name = 'Category' type = 'string' value = 'Stylize' / >
      <Property name = 'Description' type = 'string' value = 'Adds a ripple effect that can be animated' / >
      <Inputs>
      <Input name = 'Source' / >
      < / Inputs>
      <!--Custom Properties go here-->
      <Property name = 'Frequency' type = 'float'>
      <Property name = 'DisplayName' type = 'string' value = 'Frequency' / >
      <Property name = 'Min' type = 'float' value = '0.0' / >
      <Property name = 'Max' type = 'float' value = '1000.0' / >
      <Property name = 'Default' type = 'float' value = '0.0' / >
      < / Property>
      <Property name = 'Phase' type = 'float'>
      <Property name = 'DisplayName' type = 'string' value = 'Phase' / >
      <Property name = 'Min' type = 'float' value = '-100.0' / >
      <Property name = 'Max' type = 'float' value = '100.0' / >
      <Property name = 'Default' type = 'float' value = '0.0' / >
      < / Property>
      <Property name = 'Amplitude' type = 'float'>
      <Property name = 'DisplayName' type = 'string' value = 'Amplitude' / >
      <Property name = 'Min' type = 'float' value = '0.0001' / >
      <Property name = 'Max' type = 'float' value = '1000.0' / >
      <Property name = 'Default' type = 'float' value = '0.0' / >
      < / Property>
      <Property name = 'Spread' type = 'float'>
      <Property name = 'DisplayName' type = 'string' value = 'Spread' / >
      <Property name = 'Min' type = 'float' value = '0.0001' / >
      <Property name = 'Max' type = 'float' value = '1000.0' / >
      <Property name = 'Default' type = 'float' value = '0.0' / >
      < / Property>
      <Property name = 'Center' type = 'vector2'>
      <Property name = 'DisplayName' type = 'string' value = 'Center' / >
      <Property name = 'Default' type = 'vector2' value = '(0.0, 0.0)' / >
      < / Property>
      < / Effect>
    );

  // This defines the bindings from specific properties to the callback functions
  // on the class that ID2D1Effect::SetValue() & GetValue() will call.
  const D2D1_PROPERTY_BINDING bindings[] =
  {
    D2D1_VALUE_TYPE_BINDING(L"Frequency", &SetFrequency, &GetFrequency),
    D2D1_VALUE_TYPE_BINDING(L"Phase", &SetPhase, &GetPhase),
    D2D1_VALUE_TYPE_BINDING(L"Amplitude", &SetAmplitude, &GetAmplitude),
    D2D1_VALUE_TYPE_BINDING(L"Spread", &SetSpread, &GetSpread),
    D2D1_VALUE_TYPE_BINDING(L"Center", &SetCenter, &GetCenter),
  };

  // This registers the effect with the factory, which will make the effect
  // instantiatable.
  return pFactory->RegisterEffectFromString(
    CLSID_CustomRippleEffect,
    pszXml,
    bindings,
    ARRAYSIZE(bindings),
    CreateRippleImpl
  );*/

  return S_OK;
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
  //else if(riid == __uuidof(ID2D1DrawTransform))
  //{
  //  *ppOutput = static_cast<ID2D1DrawTransform*>(this);
  //}
  //else if(riid == __uuidof(ID2D1Transform))
  //{
  //  *ppOutput = static_cast<ID2D1Transform*>(this);
  //}
  //else if(riid == __uuidof(ID2D1TransformNode))
  //{
  //  *ppOutput = static_cast<ID2D1TransformNode*>(this);
  //}
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
