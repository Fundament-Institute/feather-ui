// Copyright (c)2020 Fundament Software

#define D2D_INPUT_COUNT 0
#define D2D_INPUT0_SIMPLE
#define D2D_REQUIRES_SCENE_POSITION

//#include "d2d1effecthelpers.hlsli"

cbuffer constants : register(b0)
{
  float4 rect : packoffset(c0);
  float4 corners : packoffset(c1);
  float4 color : packoffset(c2);
  float4 outlinecolor : packoffset(c3);
  float outline : packoffset(c4.x);
};

float linetopoint(float2 p1, float2 p2, float2 p)
{
  float2 n = p2 - p1;
  n = float2(n.y, -n.x);
  return dot(normalize(n), p1 - p);
}
float4 main(float4 pos : SV_POSITION, float4 posScene : SCENE_POSITION) : SV_TARGET
{
  float2 p = posScene.xy - rect.xy;
  float2 d = rect.zw - rect.xy;
  float4 c = corners;
  float2 dist;
  float2 p2 = float2(c.w*d.x,0);
  float r1 = linetopoint(p2, float2(0, d.y), p);
  float r2 = -linetopoint(p2, d, p);
  float r = max(r1, r2);
  r = max(r, p.y - d.y);

  float w = fwidth((p.x + p.y / 2));
  float s = 1 - smoothstep(1 - outline - w, 1 - outline, r);
  float alpha = smoothstep(1, 1 - w, r);
  float4 fill = float4(color.rgb, 1);
  float4 edge = float4(outlinecolor.rgb, 1);
  return (fill*color.a*s) + (edge*outlinecolor.a*saturate(alpha - s));
}