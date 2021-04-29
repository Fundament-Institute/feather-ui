// Copyright (c)2021 Fundament Software

#define D2D_INPUT_COUNT 1
#define D2D_INPUT1_SIMPLE
#define D2D_REQUIRES_SCENE_POSITION

Texture2D InputTexture : register(t0);

SamplerState InputSampler : register(s0);

cbuffer constants : register(b0)
{
  float4 color : packoffset(c0);
};

float4 main(
    float4 pos      : SV_POSITION,
    float4 posScene : SCENE_POSITION,
    float4 uv0      : TEXCOORD0
    ) : SV_Target
{            
    return InputTexture.Sample(InputSampler, uv0.xy)*color;
}