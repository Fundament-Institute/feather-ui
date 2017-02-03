// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_ROUNDRECT_D2D_H__
#define __FG_ROUNDRECT_D2D_H__

#include <d2d1_1.h>
#include <d2d1effectauthor.h>  
#include <d2d1effecthelpers.h>

DEFINE_GUID(CLSID_fgRoundRect, 0xC58204BE, 0x9FA1, 0x4884, 0xB1, 0x82, 0xBF, 0xE8, 0x5D, 0x65, 0x90, 0xDC);

class fgRoundRect : public ID2D1EffectImpl
{
public:
  IFACEMETHODIMP Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph);
  IFACEMETHODIMP PrepareForRender(D2D1_CHANGE_TYPE changeType);
  IFACEMETHODIMP SetGraph(_In_ ID2D1TransformGraph* pGraph);
  static HRESULT Register(_In_ ID2D1Factory1* pFactory);
  static HRESULT CreateEffect(_Outptr_ IUnknown** ppEffectImpl);
  IFACEMETHODIMP_(ULONG) AddRef();
  IFACEMETHODIMP_(ULONG) Release();
  IFACEMETHODIMP QueryInterface(_In_ REFIID riid, _Outptr_ void** ppOutput);

private:
  fgRoundRect();

  LONG _ref;
};

#endif