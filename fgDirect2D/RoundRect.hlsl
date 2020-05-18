// Copyright (c)2020 Fundament Software

#define D2D_INPUT_COUNT 0
#define D2D_INPUT0_SIMPLE
#define D2D_REQUIRES_SCENE_POSITION

//#include "d2d1effecthelpers.hlsli"

cbuffer constants : register(b0)
{
  float4 rect : packoffset(c0);
  float4 corners : packoffset(c1);
  float4 fill : packoffset(c2);
  float4 outline : packoffset(c3);
  float border : packoffset(c4.x);
  float blur : packoffset(c4.y);
};

float rectangle(float2 pos, float2 halfSize, float4 edges){
    float edge = 20.0;
    if(pos.x > 0.0)
      edge = (pos.y < 0.0) ? edges.y : edges.z;
    else
      edge = (pos.y < 0.0) ? edges.x : edges.w;
    
    float2 componentWiseEdgeDistance = abs(pos) - halfSize + float2(edge, edge);
    float outsideDistance = length(max(componentWiseEdgeDistance, 0.0f));
    float insideDistance = min(max(componentWiseEdgeDistance.x, componentWiseEdgeDistance.y), 0.0f);
    return outsideDistance + insideDistance - edge;
}

float4 main(float4 pos : SV_POSITION, float4 posScene : SCENE_POSITION) : SV_TARGET
{
  float2 d = (rect.zw - rect.xy) / 2.0f;
  float2 p = posScene.xy - rect.xy - d.xy;
  d -= float2(blur, blur);
  float w = (float)(fwidth(posScene.xy) * 0.5) + blur;
  float dist = rectangle(p,d,corners);

  float alpha = smoothstep(0.0 + w, 0.0 - w, dist);
  float s = smoothstep(-border.x + w, -border.x - w, dist);
  return (float4(fill.rgb, 1)*fill.a*s) + (float4(outline.rgb, 1)*outline.a*saturate(alpha - s));
}
 