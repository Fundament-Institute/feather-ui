// Copyright (c)2020 Fundament Software

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

float4 main(float4 pos : SV_POSITION, float4 posScene : SCENE_POSITION) : SV_TARGET
{
  float2 d = rect.zw - rect.xy;
  float2 p = (posScene.xy - rect.xy)/d;

  float l = (d.x + d.y) * 0.5;
  float2 uv = (p*2.0) - 1.0;
  
  float t = 0.50 - (angles.z / l) * 2.0;
  float r = 1.0 - t;
 
  float2 up = float2(cos(angles.x), sin(angles.x));
  
  float outer = (outline / l) * 2.0; // double because UV is in range [-1,1], not [0,1]
  float inner = (angles.w / l) * 2.0;
  float b = 0.701 + (blur * 0.5); // halve blur amount because of UV again
  
  // SDF for circle
  float w1 = fwidth(length(uv))*b; // Don't take fwidth of abs() because it's discontinuous
  inner -= w1; // Prevent antialiasing from creating a hole in the center
  float d0 = abs(length(uv) - r + (outer*0.5) - (inner*0.5)) - t + (outer*0.5) + (inner*0.5);
  float d1 = abs(length(uv) - r) - t;
  float s1 = smoothstep(w1, -w1, d0); 
  float alpha1 = smoothstep(w1, -w1, d1); 

  // SDF for arc
  float d2 = dot(up, normalize(uv)) - cos(angles.y);
  float w2 = fwidth(d2)*b;
  float alpha2 = smoothstep(w2, -w2, d2); 
  //float s2 = smoothstep(w2 + border, -w2 + border, d2); // doesn't work
  alpha2 *= smoothstep(0.0, w1, abs(angles.y - PI));
  
  // Merge alpha results of both SDFs
  float s = s1*(1.0 - alpha2);
  float alpha = alpha1*(1.0 - alpha2);
  
  return (float4(color.rgb, 1)*color.a*s) + (float4(outlinecolor.rgb, 1)*outlinecolor.a*saturate(alpha - s));
}