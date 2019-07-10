// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__ROUNDRECT_H
#define D2D__ROUNDRECT_H

#include "EffectBase.h"

namespace D2D {
  DEFINE_GUID(CLSID_RoundRect, 0xC58204BE, 0x9FA1, 0x4884, 0xB1, 0x82, 0xBF, 0xE8, 0x5D, 0x65, 0x90, 0xDC);
  DEFINE_GUID(CLSID_RoundRectPixelShader, 0x0050F29E, 0xF56A, 0xAC, 0xB2, 0x5F, 0x23, 0x90, 0x46, 0x94, 0xAA, 0x5F);

  class RoundRect : public EffectBase
  {
  public:
    IFACEMETHODIMP Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph);
    static HRESULT Register(_In_ ID2D1Factory1* pFactory);
    static HRESULT __stdcall CreateEffect(_Outptr_ IUnknown** ppEffectImpl);

    IFACEMETHODIMP SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo);

  private:
    RoundRect();
    ~RoundRect();
  };
}

#endif
