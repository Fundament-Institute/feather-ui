// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__CIRCLE_H
#define D2D__CIRCLE_H

#include "EffectBase.h"

namespace D2D {
  DEFINE_GUID(CLSID_Circle, 0x8C08CBDF, 0x092B, 0x4123, 0xA7, 0x4C, 0x92, 0xC7, 0x5B, 0xDA, 0x47, 0x01);
  DEFINE_GUID(CLSID_CirclePixelShader, 0x4BDBC599, 0xDD68, 0x4199, 0x83, 0x5E, 0xD8, 0x39, 0xA0, 0x9D, 0xE7, 0x32);

  class Circle : public EffectBase
  {
  public:
    IFACEMETHODIMP Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph);
    static HRESULT Register(_In_ ID2D1Factory1* pFactory);
    static HRESULT __stdcall CreateEffect(_Outptr_ IUnknown** ppEffectImpl);

    IFACEMETHODIMP SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo);

  private:
    Circle();
    ~Circle();
  };
}

#endif
