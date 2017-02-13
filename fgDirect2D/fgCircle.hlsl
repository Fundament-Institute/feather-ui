// Copyright ©2017 Black Sphere Studios

#define D2D_INPUT_COUNT 0
#define D2D_INPUT0_SIMPLE
#define D2D_REQUIRES_SCENE_POSITION

//#include "d2d1effecthelpers.hlsli"

cbuffer constants : register(b0)
{
  float4 rect : packoffset(c0);
  float4 arcs : packoffset(c1);
  float4 color : packoffset(c2);
  float4 outlinecolor : packoffset(c3);
  float outline : packoffset(c4.x);
};

float fmod(float n, float d)
{
  return n - floor(n/d)*d;
}

float getarc(float2 arcs, float angle, float anglew)
{
  const float PI = 3.14159265359;
  if(arcs.y > PI) // we can only deal with an empty circle, not a full circle, so we invert the arc if it's greater than PI
  {
    arcs.x += arcs.y; // Note: this is probably mathematically equivelent to constructing something using abs()
    arcs.y = PI*2 - arcs.y;
    // Copy paste from below becuase no recursive functions in HLSL :C
    angle = fmod(angle-arcs.x+anglew, PI*2) - anglew;
    return 1.0 - (smoothstep(-anglew, anglew, angle) - smoothstep(arcs.y-anglew, arcs.y+anglew, angle));
  }
  angle = fmod(angle-arcs.x+anglew, PI*2) - anglew; // Instead of enclosing within [0, 2PI], we shift to [-anglew, PI*2-anglew] which prevents the smoothstep from breaking
  return smoothstep(-anglew, anglew, angle) - smoothstep(arcs.y-anglew, arcs.y+anglew, angle);
}

float4 main(float4 pos : SV_POSITION, float4 posScene : SCENE_POSITION) : SV_TARGET
{
  float2 d = rect.zw - rect.xy;
  float2 p = (posScene.xy - rect.xy)/d;
  float angle = atan2(- p.y + 0.5, p.x - 0.5);
  float anglew = fwidth(angle)*0.5;
  float r = distance(p, float2(0.50, 0.50))*2;
  float w = fwidth(r)*0.5;
  float ol = (outline / d.x)*2;
  
  float s = 1 - smoothstep(1 - ol - w, 1 - ol + w, r);
  float alpha = smoothstep(1+w, 1-w, r);
  float4 fill = float4(color.rgb, 1);
  float4 edge = float4(outlinecolor.rgb, 1);
  return (fill*color.a*s*getarc(arcs.xy, angle, anglew)) + (edge*outlinecolor.a*saturate(alpha-s)*getarc(arcs.zw, angle, anglew));
}