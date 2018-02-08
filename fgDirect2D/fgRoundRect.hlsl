// Copyright ©2018 Black Sphere Studios

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

float4 main(float4 pos : SV_POSITION, float4 posScene : SCENE_POSITION) : SV_TARGET
{
  //float2 p = (posScene.xy - rect.xy) / (rect.zw - rect.xy);

  float2 p = posScene.xy - rect.xy;
  float2 d = rect.zw - rect.xy;
  float2 dd = abs(rcp(d));
  float4 r = corners;
  float2 dist;
  float w = fwidth(posScene.xy) * 0.5;

  //return float4(p.x/d.x, p.y/d.y, 1, 1);

  if(p.x < r[0] && p.y < r[0]) {
    dist = float2(distance(p, float2(r[0],r[0])), outline) / r[0];
    //w = fwidth(dist.x)*0.5;
    w = w / r[0];
  }
  else if(p.x > d.x - r[1] && p.y < r[1]) {
    dist = float2(distance(p, float2(d.x - r[1],r[1])), outline) / r[1];
    w = w / r[1];
  }
  else if(p.x < r[2] && p.y > d.y - r[2]) {
    dist = float2(distance(p, float2(r[2],d.y - r[2])), outline) / r[2];
    w = w / r[2];
  }
  else if(p.x > d.x - r[3] && p.y > d.y - r[3]) {
    dist = float2(distance(p, float2(d.x - r[3],d.y - r[3])), outline) / r[3];
    w = w / r[3];
  }
  else {
    dist = min(p, d - p);
    dist.x = 1 - (min(dist.x,dist.y) / outline);
    dist.y = 1;
    w = w * 0.5;
    //w = (((abs(ddx(p.x)) + abs(ddy(p.y))) / 2) / outline)*0.5; // Don't use dist.x here because it's not continuous, which breaks ddx and ddy
  }

  float4 c = color;
  float4 o = outlinecolor;
  float s = 1 - smoothstep(1 - dist.y - w, 1 - dist.y + w, dist.x);
  float alpha = smoothstep(1 + w, 1 - w, dist.x);
  float4 fill = float4(c.rgb, 1);
  float4 edge = float4(o.rgb, 1);
  return (fill*c.a*s) + (edge*o.a*saturate(alpha - s));
  //return c = (fill*c.a*s) + (edge*o.a*saturate(alpha - s));
  return float4(1, p.x, p.y, 1);
}
 