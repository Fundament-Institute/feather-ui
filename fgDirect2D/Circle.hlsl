// Copyright (c)2021 Fundament Software

#define D2D_INPUT_COUNT 0
#define D2D_INPUT0_SIMPLE
#define D2D_REQUIRES_SCENE_POSITION

//#include "d2d1effecthelpers.hlsli"

static const float PI = 3.14159265359;
  
cbuffer constants : register(b0)
{
  float4 rect : packoffset(c0);
  float4 angles : packoffset(c1);
  float4 color : packoffset(c2);
  float4 outlinecolor : packoffset(c3);
  float outline : packoffset(c4.x);
  float blur : packoffset(c4.y);
};

float linearstep(float low, float high, float x) { return saturate((x - low) / (high - low)); }

float4 main(float4 pos : SV_POSITION, float4 posScene : SCENE_POSITION) : SV_TARGET
{
  float2 dim = rect.zw - rect.xy;
  float2 p = (posScene.xy - rect.xy)/dim;

  float l = (dim.x + dim.y) * 0.5;
  float2 uv = (p*2.0) - 1.0;
  float w1 = (1.0 + blur)*fwidth(p.x); 

  float border = (outline / l) * 2.0; // double because UV is in range [-1,1], not [0,1]
  float t = 0.50 - (angles.x / l);
  // We have to compensate for needing to do smoothstep starting from 0, which combined with abs()
  // acts as a ceil() function, creating one extra half pixel.
  float r = 1.0 - t - w1;
  
  // SDF for circle
  float inner = (angles.y / l) * 2.0;
  float d0 = abs(length(uv) - r + (border*0.5) - (inner*0.5)) - t + (border*0.5) + (inner*0.5);
  float d1 = abs(length(uv) - r) - t;
  float s = linearstep(w1*2.0, 0.0, d0); 
  float alpha = linearstep(w1*2.0, 0.0, d1); 
  
  // Output to screen 
  return (float4(color.rgb, 1)*color.a*s) + (float4(outlinecolor.rgb, 1)*outlinecolor.a*saturate(alpha - s));
}