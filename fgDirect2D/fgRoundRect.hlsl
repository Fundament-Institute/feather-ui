// Copyright ©2017 Black Sphere Studios

#define D2D_INPUT_COUNT 0           // The pixel shader takes 1 input texture.
#define D2D_INPUT0_SIMPLE
#define D2D_REQUIRES_SCENE_POSITION // The pixel shader requires the SCENE_POSITION input.

#include "d2d1effecthelpers.hlsli"

cbuffer constants : register(b0)
{
    float4 corners : packoffset(c0);
    float outline : packoffset(c1.x);
    uint color : packoffset(c1.y);
    uint outlinecolor : packoffset(c1.z);
};

D2D_PS_ENTRY(main)
{
  return float4(1,0,0,1);
}
 