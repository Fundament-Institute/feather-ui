/* backend.h - Standard C interface for feather backends
Copyright (c)2022 Fundament Software

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef BACKEND_H
#define BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>  // for integers
#include <stdbool.h> // for bool

enum FG_PixelFormat
{
  FG_PixelFormat_UNKNOWN = 0,
  FG_PixelFormat_R8G8B8A8_UINT,
  FG_PixelFormat_A8_UNORM,
  FG_PixelFormat_A8P8,
  FG_PixelFormat_AI44,
  FG_PixelFormat_AYUV,
  FG_PixelFormat_B4G4R4A4_UNORM,
  FG_PixelFormat_B5G5R5A1_UNORM,
  FG_PixelFormat_B5G6R5_UNORM,
  FG_PixelFormat_B8G8R8A8_TYPELESS,
  FG_PixelFormat_B8G8R8A8_UNORM,
  FG_PixelFormat_B8G8R8A8_UNORM_SRGB,
  FG_PixelFormat_B8G8R8X8_TYPELESS,
  FG_PixelFormat_B8G8R8X8_UNORM,
  FG_PixelFormat_B8G8R8X8_UNORM_SRGB,
  FG_PixelFormat_BC1_TYPELESS,
  FG_PixelFormat_BC1_UNORM,
  FG_PixelFormat_BC1_UNORM_SRGB,
  FG_PixelFormat_BC2_TYPELESS,
  FG_PixelFormat_BC2_UNORM,
  FG_PixelFormat_BC2_UNORM_SRGB,
  FG_PixelFormat_BC3_TYPELESS,
  FG_PixelFormat_BC3_UNORM,
  FG_PixelFormat_BC3_UNORM_SRGB,
  FG_PixelFormat_BC4_SNORM,
  FG_PixelFormat_BC4_TYPELESS,
  FG_PixelFormat_BC4_UNORM,
  FG_PixelFormat_BC5_SNORM,
  FG_PixelFormat_BC5_TYPELESS,
  FG_PixelFormat_BC5_UNORM,
  FG_PixelFormat_BC6H_SF16,
  FG_PixelFormat_BC6H_TYPELESS,
  FG_PixelFormat_BC6H_UF16,
  FG_PixelFormat_BC7_TYPELESS,
  FG_PixelFormat_BC7_UNORM,
  FG_PixelFormat_BC7_UNORM_SRGB,
  FG_PixelFormat_D16_UNORM,
  FG_PixelFormat_D24_UNORM_S8_UINT,
  FG_PixelFormat_D24_UNORM_X8_TYPELESS,
  FG_PixelFormat_D32_FLOAT,
  FG_PixelFormat_D32_FLOAT_S8X24_UINT,
  FG_PixelFormat_G8R8_G8B8_UNORM,
  FG_PixelFormat_IA44,
  FG_PixelFormat_NV11,
  FG_PixelFormat_NV12,
  FG_PixelFormat_P010,
  FG_PixelFormat_P016,
  FG_PixelFormat_P208,
  FG_PixelFormat_P8,
  FG_PixelFormat_R1_UNORM,
  FG_PixelFormat_R10G10B10_XR_BIAS_A2_UNORM,
  FG_PixelFormat_R10G10B10A2_TYPELESS,
  FG_PixelFormat_R10G10B10A2_UINT,
  FG_PixelFormat_R10G10B10A2_UNORM,
  FG_PixelFormat_R11G11B10_FLOAT,
  FG_PixelFormat_R16_FLOAT,
  FG_PixelFormat_R16_INT,
  FG_PixelFormat_R16_SNORM,
  FG_PixelFormat_R16_TYPELESS,
  FG_PixelFormat_R16_UINT,
  FG_PixelFormat_R16_UNORM,
  FG_PixelFormat_R16G16_FLOAT,
  FG_PixelFormat_R16G16_INT,
  FG_PixelFormat_R16G16_SNORM,
  FG_PixelFormat_R16G16_TYPELESS,
  FG_PixelFormat_R16G16_UINT,
  FG_PixelFormat_R16G16_UNORM,
  FG_PixelFormat_R16G16B16_FLOAT,
  FG_PixelFormat_R16G16B16_INT,
  FG_PixelFormat_R16G16B16_SNORM,
  FG_PixelFormat_R16G16B16_TYPELESS,
  FG_PixelFormat_R16G16B16_UINT,
  FG_PixelFormat_R16G16B16_UNORM,
  FG_PixelFormat_R16G16B16A16_FLOAT,
  FG_PixelFormat_R16G16B16A16_INT,
  FG_PixelFormat_R16G16B16A16_SNORM,
  FG_PixelFormat_R16G16B16A16_TYPELESS,
  FG_PixelFormat_R16G16B16A16_UINT,
  FG_PixelFormat_R16G16B16A16_UNORM,
  FG_PixelFormat_R24_UNORM_X8_TYPELESS,
  FG_PixelFormat_R24G8_TYPELESS,
  FG_PixelFormat_R32_FLOAT,
  FG_PixelFormat_R32_FLOAT_X8X24_TYPELESS,
  FG_PixelFormat_R32_INT,
  FG_PixelFormat_R32_TYPELESS,
  FG_PixelFormat_R32_UINT,
  FG_PixelFormat_R32G32_FLOAT,
  FG_PixelFormat_R32G32_INT,
  FG_PixelFormat_R32G32_TYPELESS,
  FG_PixelFormat_R32G32_UINT,
  FG_PixelFormat_R32G32B32_FLOAT,
  FG_PixelFormat_R32G32B32_INT,
  FG_PixelFormat_R32G32B32_TYPELESS,
  FG_PixelFormat_R32G32B32_UINT,
  FG_PixelFormat_R32G32B32A32_FLOAT,
  FG_PixelFormat_R32G32B32A32_INT,
  FG_PixelFormat_R32G32B32A32_TYPELESS,
  FG_PixelFormat_R32G32B32A32_UINT,
  FG_PixelFormat_R32G8X24_TYPELESS,
  FG_PixelFormat_R5G5B5A1_UNORM,
  FG_PixelFormat_R5G6B5_UNORM,
  FG_PixelFormat_R8_INT,
  FG_PixelFormat_R8_SNORM,
  FG_PixelFormat_R8_TYPELESS,
  FG_PixelFormat_R8_UINT,
  FG_PixelFormat_R8_UNORM,
  FG_PixelFormat_R8G8_B8G8_UNORM,
  FG_PixelFormat_R8G8_INT,
  FG_PixelFormat_R8G8_SNORM,
  FG_PixelFormat_R8G8_TYPELESS,
  FG_PixelFormat_R8G8_UINT,
  FG_PixelFormat_R8G8_UNORM,
  FG_PixelFormat_R8G8B8A8_INT,
  FG_PixelFormat_R8G8B8A8_SNORM,
  FG_PixelFormat_R8G8B8A8_TYPELESS,
  FG_PixelFormat_R8G8B8A8_UNORM,
  FG_PixelFormat_R8G8B8A8_UNORM_SRGB,
  FG_PixelFormat_R8G8B8X8_INT,
  FG_PixelFormat_R8G8B8X8_SNORM,
  FG_PixelFormat_R8G8B8X8_TYPELESS,
  FG_PixelFormat_R8G8B8X8_UINT,
  FG_PixelFormat_R8G8B8X8_UNORM,
  FG_PixelFormat_R9G9B9E5_SHAREDEXP,
  FG_PixelFormat_V208,
  FG_PixelFormat_V408,
  FG_PixelFormat_X24_TYPELESS_G8_UINT,
  FG_PixelFormat_X32_TYPELESS_G8X24_UINT,
  FG_PixelFormat_Y210,
  FG_PixelFormat_Y216,
  FG_PixelFormat_Y410,
  FG_PixelFormat_Y416,
  FG_PixelFormat_YUY2,
};

enum FG_Type
{
  FG_Type_Unknown = 0,
  FG_Type_VertexData,
  FG_Type_VertexIndices,
  FG_Type_PixelRead,
  FG_Type_PixelWrite,
  FG_Type_CopyRead,
  FG_Type_CopyWrite,
  FG_Type_TextureBuffer,
  FG_Type_Transform,
  FG_Type_Uniform,
  FG_Type_Texture1D,
  FG_Type_Texture2D,
  FG_Type_Texture3D,
  FG_Type_Texture2D_Multisample,
  FG_Type_Texture2D_Multisample_Proxy,
};

enum FG_ShaderStage
{
  FG_ShaderStage_Pixel = 0,
  FG_ShaderStage_Vertex,
  FG_ShaderStage_Geometry,
  FG_ShaderStage_Hull,
  FG_ShaderStage_Domain,
  FG_ShaderStage_Compute,
  FG_ShaderStage_COUNT
};

enum FG_Feature
{
  FG_Feature_API_OPENGL_ES      = 1,
  FG_Feature_API_OPENGL         = 2,
  FG_Feature_API_DIRECTX        = 3,
  FG_Feature_API_VULKAN         = 4,
  FG_Feature_API_METAL          = 5,
  FG_Feature_IMMEDIATE_MODE     = (1 << 4),
  FG_Feature_RENDER_TARGET      = (1 << 5),
  FG_Feature_BLEND_EX           = (1 << 6),
  FG_Feature_BACKGROUND_OPACITY = (1 << 7),
  FG_Feature_LINES_ALPHA        = (1 << 8),
  FG_Feature_BLEND_GAMMA        = (1 << 9),
  FG_Feature_INSTANCING         = (1 << 10),
  FG_Feature_MULTITHREADING     = (1 << 11),
  FG_Feature_COMMAND_BUNDLES    = (1 << 12),
  FG_Feature_INDEPENDENT_BLEND  = (1 << 13),
  FG_Feature_COMPUTE_SHADER     = (1 << 14),
};

typedef struct FG_Vec3i__
{
  int x;
  int y;
  int z;
} FG_Vec3i;

typedef struct FG_Vec3__
{
  float x;
  float y;
  float z;
} FG_Vec3;

typedef struct FG_Vec2i__
{
  int x;
  int y;
} FG_Vec2i;

typedef struct FG_Vec2__
{
  float x;
  float y;
} FG_Vec2;

typedef union FG_Rect__
{
  struct
  {
    float left;
    float top;
    float right;
    float bottom;
  };

  float ltrb[4];
} FG_Rect;

typedef union FG_Color8__
{
  uint32_t v;
  uint8_t colors[4];
  struct
  {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
  };
} FG_Color8;

typedef union FG_Color16__
{
  uint64_t v;
  uint16_t colors[4];
  struct
  {
    uint16_t b;
    uint16_t g;
    uint16_t r;
    uint16_t a;
  };
} FG_Color;

// This can hold caps for either OpenGL or OpenGL ES. These have different version numbers.
typedef struct FG_OpenGL_Caps__
{
  int features;
  int version; // 2.1 is 21, 4.2 is 42, etc.
  int glsl;    // Version 1.10 is 110, version 4.40 is 440
  int max_textures;
  int max_rendertargets;
} FG_OpenGL_Caps;

typedef struct FG_DirectX_Caps__
{
  int features;
  int version; // 10.0 is 100, 11.1 is 111, etc.
  int hlsl;
  int max_textures;
  int max_rendertargets;
} FG_DirectX_Caps;

typedef union FG_Caps__
{
  int features;
  FG_OpenGL_Caps OpenGL;
  FG_OpenGL_Caps OpenGL_ES;
  FG_DirectX_Caps DirectX;
} FG_Caps;

typedef void FG_Resource;

enum FG_Primitive
{
  FG_Primitive_POINT                    = 0,
  FG_Primitive_LINE                     = 1,
  FG_Primitive_TRIANGLE                 = 2,
  FG_Primitive_LINE_STRIP               = 3,
  FG_Primitive_TRIANGLE_STRIP           = 4,
  FG_Primitive_LINE_ADJACENCY           = 5,
  FG_Primitive_TRIANGLE_ADJACENCY       = 6,
  FG_Primitive_LINE_STRIP_ADJACENCY     = 7,
  FG_Primitive_TRIANGLE_STRIP_ADJACENCY = 8,
};

enum FG_COMPARISON
{
  FG_COMPARISON_DISABLED      = 0,
  FG_COMPARISON_NEVER         = 1,
  FG_COMPARISON_LESS          = 2,
  FG_COMPARISON_EQUAL         = 3,
  FG_COMPARISON_LESS_EQUAL    = 4,
  FG_COMPARISON_GREATER       = 5,
  FG_COMPARISON_NOT_EQUAL     = 6,
  FG_COMPARISON_GREATER_EQUAL = 7,
  FG_COMPARISON_ALWAYS        = 8
};

enum FG_STRIP_CUT_VALUE
{
  FG_STRIP_CUT_VALUE_DISABLED   = 0,
  FG_STRIP_CUT_VALUE_0xFFFF     = 1,
  FG_STRIP_CUT_VALUE_0xFFFFFFFF = 2
};

enum FG_BLEND
{
  FG_BLEND_ZERO             = 1,
  FG_BLEND_ONE              = 2,
  FG_BLEND_SRC_COLOR        = 3,
  FG_BLEND_INV_SRC_COLOR    = 4,
  FG_BLEND_SRC_ALPHA        = 5,
  FG_BLEND_INV_SRC_ALPHA    = 6,
  FG_BLEND_DEST_ALPHA       = 7,
  FG_BLEND_INV_DEST_ALPHA   = 8,
  FG_BLEND_DEST_COLOR       = 9,
  FG_BLEND_INV_DEST_COLOR   = 10,
  FG_BLEND_SRC_ALPHA_SAT    = 11,
  FG_BLEND_BLEND_FACTOR     = 14,
  FG_BLEND_INV_BLEND_FACTOR = 15,
  FG_BLEND_SRC1_COLOR       = 16,
  FG_BLEND_INV_SRC1_COLOR   = 17,
  FG_BLEND_SRC1_ALPHA       = 18,
  FG_BLEND_INV_SRC1_ALPHA   = 19
};

enum FG_BLEND_OP
{
  FG_BLEND_OP_ADD          = 1,
  FG_BLEND_OP_SUBTRACT     = 2,
  FG_BLEND_OP_REV_SUBTRACT = 3,
  FG_BLEND_OP_MIN          = 4,
  FG_BLEND_OP_MAX          = 5
};

// Each bit here represents an element of the pipelinestate struct that has been set
enum FG_PIPELINE_MEMBER
{
  FG_PIPELINE_MEMBER_PRIMITIVE          = (1 << 0),
  FG_PIPELINE_MEMBER_STENCIL_REF        = (1 << 1),
  FG_PIPELINE_MEMBER_INDEX_BUFFER       = (1 << 2),
  FG_PIPELINE_MEMBER_BLEND_FACTOR       = (1 << 3),
  FG_PIPELINE_MEMBER_VS                 = (1 << 4),
  FG_PIPELINE_MEMBER_GS                 = (1 << 5),
  FG_PIPELINE_MEMBER_HS                 = (1 << 6),
  FG_PIPELINE_MEMBER_DS                 = (1 << 7),
  FG_PIPELINE_MEMBER_PS                 = (1 << 8),
  FG_PIPELINE_MEMBER_CS                 = (1 << 9),
  FG_PIPELINE_MEMBER_BLENDFACTOR        = (1 << 10),
  FG_PIPELINE_MEMBER_FLAGS              = (1 << 11),
  FG_PIPELINE_MEMBER_SAMPLEMASK         = (1 << 12),
  FG_PIPELINE_MEMBER_STENCIL_READ_MASK  = (1 << 13),
  FG_PIPELINE_MEMBER_STENCIL_WRITE_MASK = (1 << 14),
  FG_PIPELINE_MEMBER_DEPTH_FUNC         = (1 << 15),
  FG_PIPELINE_MEMBER_STRIP_CUT_VALUE    = (1 << 16),
  FG_PIPELINE_MEMBER_NUM_RENDER_TARGETS = (1 << 17),
  FG_PIPELINE_MEMBER_RT_FORMATS         = (1 << 18),
  FG_PIPELINE_MEMBER_FILL               = (1 << 19),
  FG_PIPELINE_MEMBER_CULL               = (1 << 20),
  FG_PIPELINE_MEMBER_DEPTH_BIAS         = (1 << 21),
  FG_PIPELINE_MEMBER_DEPTH_BIAS_CLAMP   = (1 << 22),
  FG_PIPELINE_MEMBER_SLOP_ESCALED_BIAS  = (1 << 23),
  FG_PIPELINE_MEMBER_FORCEDSAMPLECOUNT  = (1 << 24),
  FG_PIPELINE_MEMBER_STENCIL            = (1 << 25),
  FG_PIPELINE_MEMBER_DEPTH              = (1 << 26),
  FG_PIPELINE_NODE_MASK                 = (1 << 27),
  FG_PIPELINE_MULTISAMPLE_COUNT         = (1 << 28),
  FG_PIPELINE_MULTISAMPLE_QUALITY       = (1 << 29),
};

enum FG_FILL_MODE
{
  FG_FILL_MODE_FILL  = 0,
  FG_FILL_MODE_LINE  = 1,
  FG_FILL_MODE_POINT = 2,
};

enum FG_CULL_MODE
{
  FG_CULL_MODE_NONE  = 0,
  FG_CULL_MODE_FRONT = 1,
  FG_CULL_MODE_BACK  = 2,
};

enum FG_PIPELINE_FLAGS
{
  FG_PIPELINE_FLAG_ALPHA_TO_COVERAGE_ENABLE = (1 << 0),
  FG_PIPELINE_FLAG_BLEND_ENABLE             = (1 << 1),
  FG_PIPELINE_FLAG_DEPTH_ENABLE             = (1 << 2),
  FG_PIPELINE_FLAG_STENCIL_ENABLE           = (1 << 3),
  FG_PIPELINE_FLAG_DEPTH_WRITE_ENABLE       = (1 << 4),
  FG_PIPELINE_FLAG_CONSERVATIVE_RASTER      = (1 << 5),
  FG_PIPELINE_FLAG_FRONT_COUNTER_CLOCKWISE  = (1 << 6),
  FG_PIPELINE_FLAG_DEPTH_CLIP_ENABLE        = (1 << 7),
  FG_PIPELINE_FLAG_MULTISAMPLE_ENABLE       = (1 << 8),
  FG_PIPELINE_FLAG_ANTIALIASED_LINE_ENABLE  = (1 << 9),
  FG_PIPELINE_FLAG_TOOL_DEBUG               = (1 << 10),
  FG_PIPELINE_FLAG_INDEPENDENT_BLEND_ENABLE = (1 << 11),
  FG_PIPELINE_FLAG_RENDERTARGET_SRGB_ENABLE = (1 << 12),
};

enum FG_STENCIL_OP
{
  FG_STENCIL_OP_NONE      = 0,
  FG_STENCIL_OP_KEEP      = 1,
  FG_STENCIL_OP_ZERO      = 2,
  FG_STENCIL_OP_REPLACE   = 3,
  FG_STENCIL_OP_INCR_SAT  = 4,
  FG_STENCIL_OP_DECR_SAT  = 5,
  FG_STENCIL_OP_INVERT    = 6,
  FG_STENCIL_OP_INCR_WRAP = 7,
  FG_STENCIL_OP_DECR_WRAP = 8
};

typedef struct FG_PipelineState__
{
  uint64_t Members; // Each bit represents a particular member that has been set
  void* Shaders[FG_ShaderStage_COUNT];
  FG_Resource* DepthStencil;
  FG_Color BlendFactor;
  uint16_t Flags;
  uint32_t SampleMask;
  uint32_t StencilRef;
  uint8_t StencilReadMask;
  uint8_t StencilWriteMask;
  uint8_t StencilFailOp;
  uint8_t StencilDepthFailOp;
  uint8_t StencilPassOp;
  uint8_t StencilFunc;
  uint8_t DepthFunc;
  uint8_t RenderTargetsCount;
  uint8_t RTFormats[8];
  uint8_t FillMode;
  uint8_t CullMode;
  uint8_t Primitive;
  int DepthBias;
  float SlopeScaledDepthBias;
  uint32_t NodeMask;
} FG_PipelineState;

/* typedef struct FG_VertexStream__
{
  uint32_t Stream;
  const char* SemanticName;
  uint32_t SemanticIndex;
  uint8_t StartComponent;
  uint8_t ComponentCount;
  uint8_t OutputSlot;
  uint32_t Stride;
  uint32_t RasterizedStream;
} FG_VertexStream;*/

enum FG_FILTER
{
  FG_FILTER_MIN_MAG_MIP_POINT                          = 0,
  FG_FILTER_MIN_MAG_POINT_MIP_LINEAR                   = 0x1,
  FG_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT             = 0x4,
  FG_FILTER_MIN_POINT_MAG_MIP_LINEAR                   = 0x5,
  FG_FILTER_MIN_LINEAR_MAG_MIP_POINT                   = 0x10,
  FG_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR            = 0x11,
  FG_FILTER_MIN_MAG_LINEAR_MIP_POINT                   = 0x14,
  FG_FILTER_MIN_MAG_MIP_LINEAR                         = 0x15,
  FG_FILTER_ANISOTROPIC                                = 0x55,
  FG_FILTER_COMPARISON_MIN_MAG_MIP_POINT               = 0x80,
  FG_FILTER_COMPARISON_MIN_MAG_POINT_MIP_LINEAR        = 0x81,
  FG_FILTER_COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT  = 0x84,
  FG_FILTER_COMPARISON_MIN_POINT_MAG_MIP_LINEAR        = 0x85,
  FG_FILTER_COMPARISON_MIN_LINEAR_MAG_MIP_POINT        = 0x90,
  FG_FILTER_COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR = 0x91,
  FG_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT        = 0x94,
  FG_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR              = 0x95,
  FG_FILTER_COMPARISON_ANISOTROPIC                     = 0xd5,
};

enum FG_TEXTURE_ADDRESS_MODE
{
  FG_TEXTURE_ADDRESS_MODE_WRAP        = 1,
  FG_TEXTURE_ADDRESS_MODE_MIRROR      = 2,
  FG_TEXTURE_ADDRESS_MODE_CLAMP       = 3,
  FG_TEXTURE_ADDRESS_MODE_BORDER      = 4,
  FG_TEXTURE_ADDRESS_MODE_MIRROR_ONCE = 5
};

typedef struct FG_Sampler__
{
  uint8_t Filter;        // FG_FILTER
  uint8_t Addressing[3]; // enum FG_TEXTURE_ADDRESS_MODE
  uint8_t MaxAnisotropy;
  uint8_t Comparison; // FG_COMPARISON
  float MipBias;
  float MinLOD;
  float MaxLOD;
  FG_Color BorderColor;
} FG_Sampler;

enum FG_LOGIC_OP
{
  FG_LOGIC_OP_CLEAR = 0,
  FG_LOGIC_OP_SET,
  FG_LOGIC_OP_COPY,
  FG_LOGIC_OP_COPY_INVERTED,
  FG_LOGIC_OP_NOOP,
  FG_LOGIC_OP_INVERT,
  FG_LOGIC_OP_AND,
  FG_LOGIC_OP_NAND,
  FG_LOGIC_OP_OR,
  FG_LOGIC_OP_NOR,
  FG_LOGIC_OP_XOR,
  FG_LOGIC_OP_EQUIV,
  FG_LOGIC_OP_AND_REVERSE,
  FG_LOGIC_OP_AND_INVERTED,
  FG_LOGIC_OP_OR_REVERSE,
  FG_LOGIC_OP_OR_INVERTED
};

// Array
typedef struct FG_Blend__
{
  uint8_t SrcBlend;
  uint8_t DestBlend;
  uint8_t BlendOp;
  uint8_t SrcBlendAlpha;
  uint8_t DestBlendAlpha;
  uint8_t BlendOpAlpha;
  uint8_t RenderTargetWriteMask;
} FG_Blend;

// Array
typedef struct FG_Viewport__
{
  FG_Vec3 pos;
  FG_Vec3 dim;
  FG_Rect scissor;
} FG_Viewport;

enum FG_ShaderType
{
  FG_ShaderType_HALF    = 0,
  FG_ShaderType_FLOAT   = 1,
  FG_ShaderType_DOUBLE  = 2,
  FG_ShaderType_INT     = 3,
  FG_ShaderType_UINT    = 4,
  FG_ShaderType_COLOR32 = 5,
  FG_ShaderType_TEXTURE = 6,
  FG_ShaderType_TEXCUBE = 7,
};

// Array
typedef struct FG_ShaderParameter__
{
  const char* name;
  uint32_t offset;
  uint32_t length;
  uint32_t multi;
  uint8_t index;
  uint8_t type;      // enum FG_ShaderType
  bool per_instance; // false if per_vertex
  uint32_t step;
} FG_ShaderParameter;

typedef struct FG_Display__
{
  FG_Vec2i size;
  FG_Vec2i offset;
  FG_Vec2 dpi;
  float scale;
  void* handle;
  bool primary;
} FG_Display;

enum FG_Clipboard
{
  FG_Clipboard_NONE    = 0,
  FG_Clipboard_ALL     = 7,
  FG_Clipboard_CUSTOM  = 6,
  FG_Clipboard_WAVE    = 2,
  FG_Clipboard_BITMAP  = 3,
  FG_Clipboard_ELEMENT = 5,
  FG_Clipboard_FILE    = 4,
  FG_Clipboard_TEXT    = 1
};

enum FG_Cursor
{
  FG_Cursor_NONE       = 0,
  FG_Cursor_RESIZEWE   = 7,
  FG_Cursor_RESIZEALL  = 10,
  FG_Cursor_RESIZENS   = 6,
  FG_Cursor_CUSTOM     = 14,
  FG_Cursor_IBEAM      = 2,
  FG_Cursor_ARROW      = 1,
  FG_Cursor_DRAG       = 13,
  FG_Cursor_RESIZENESW = 9,
  FG_Cursor_NO         = 11,
  FG_Cursor_RESIZENWSE = 8,
  FG_Cursor_HELP       = 12,
  FG_Cursor_WAIT       = 4,
  FG_Cursor_HAND       = 5,
  FG_Cursor_CROSS      = 3
};

enum FG_Kind
{
  FG_Kind_MOUSEOFF       = 16,
  FG_Kind_TOUCHEND       = 23,
  FG_Kind_KEYCHAR        = 9,
  FG_Kind_KEYDOWN        = 10,
  FG_Kind_MOUSEDBLCLICK  = 13,
  FG_Kind_ACTION         = 0,
  FG_Kind_GOTFOCUS       = 4,
  FG_Kind_JOYBUTTONDOWN  = 6,
  FG_Kind_DROP           = 2,
  FG_Kind_SETWINDOWFLAGS = 20,
  FG_Kind_MOUSEON        = 17,
  FG_Kind_DRAW           = 1,
  FG_Kind_MOUSEMOVE      = 15,
  FG_Kind_SETWINDOWRECT  = 21,
  FG_Kind_JOYBUTTONUP    = 7,
  FG_Kind_MOUSEDOWN      = 14,
  FG_Kind_MOUSESCROLL    = 18,
  FG_Kind_JOYAXIS        = 5,
  FG_Kind_TOUCHBEGIN     = 22,
  FG_Kind_JOYORIENTATION = 8,
  FG_Kind_TOUCHMOVE      = 24,
  FG_Kind_LOSTFOCUS      = 12,
  FG_Kind_MOUSEUP        = 19,
  FG_Kind_GETWINDOWFLAGS = 3,
  FG_Kind_KEYUP          = 11
};

enum FG_Keys
{
  FG_Keys_4               = 52,
  FG_Keys_S               = 83,
  FG_Keys_PAGEDOWN        = 34,
  FG_Keys_EXECUTE         = 43,
  FG_Keys_F7              = 118,
  FG_Keys_PAGEUP          = 33,
  FG_Keys_8               = 56,
  FG_Keys_W               = 87,
  FG_Keys_NUMPAD_EQUAL    = 146,
  FG_Keys_6               = 54,
  FG_Keys_F16             = 127,
  FG_Keys_NUMPAD5         = 101,
  FG_Keys_Y               = 89,
  FG_Keys_RETURN          = 13,
  FG_Keys_F9              = 120,
  FG_Keys_LSUPER          = 91,
  FG_Keys_KANA            = 21,
  FG_Keys_BACK            = 8,
  FG_Keys_GRAVE           = 192,
  FG_Keys_DELETE          = 46,
  FG_Keys_LSHIFT          = 160,
  FG_Keys_F14             = 125,
  FG_Keys_JUNJA           = 23,
  FG_Keys_RSUPER          = 92,
  FG_Keys_FINAL           = 24,
  FG_Keys_B               = 66,
  FG_Keys_LCONTROL        = 162,
  FG_Keys_INSERT          = 45,
  FG_Keys_F15             = 126,
  FG_Keys_APPS            = 93,
  FG_Keys_XBUTTON1        = 5,
  FG_Keys_SELECT          = 41,
  FG_Keys_H               = 72,
  FG_Keys_F               = 70,
  FG_Keys_F21             = 132,
  FG_Keys_J               = 74,
  FG_Keys_L               = 76,
  FG_Keys_NUMLOCK         = 144,
  FG_Keys_RSHIFT          = 161,
  FG_Keys_COMMA           = 188,
  FG_Keys_F20             = 131,
  FG_Keys_NUMPAD1         = 97,
  FG_Keys_LEFT_BRACKET    = 219,
  FG_Keys_SPACE           = 32,
  FG_Keys_F18             = 129,
  FG_Keys_F23             = 134,
  FG_Keys_SEMICOLON       = 186,
  FG_Keys_MODECHANGE      = 31,
  FG_Keys_MENU            = 18,
  FG_Keys_NUMPAD9         = 105,
  FG_Keys_5               = 53,
  FG_Keys_R               = 82,
  FG_Keys_F19             = 130,
  FG_Keys_F6              = 117,
  FG_Keys_T               = 84,
  FG_Keys_NUMPAD_MULTIPLY = 106,
  FG_Keys_ACCEPT          = 30,
  FG_Keys_F22             = 133,
  FG_Keys_UP              = 38,
  FG_Keys_NUMPAD2         = 98,
  FG_Keys_CANCEL          = 3,
  FG_Keys_9               = 57,
  FG_Keys_X               = 88,
  FG_Keys_F25             = 136,
  FG_Keys_NUMPAD0         = 96,
  FG_Keys_V               = 86,
  FG_Keys_DOWN            = 40,
  FG_Keys_SNAPSHOT        = 44,
  FG_Keys_NUMPAD3         = 99,
  FG_Keys_F24             = 135,
  FG_Keys_NUMPAD8         = 104,
  FG_Keys_APOSTROPHE      = 222,
  FG_Keys_A               = 65,
  FG_Keys_NUMPAD6         = 102,
  FG_Keys_TAB             = 9,
  FG_Keys_LBUTTON         = 1,
  FG_Keys_PLUS            = 187,
  FG_Keys_C               = 67,
  FG_Keys_RIGHT_BRACKET   = 221,
  FG_Keys_BACKSLASH       = 220,
  FG_Keys_SLASH           = 191,
  FG_Keys_PERIOD          = 190,
  FG_Keys_HOME            = 36,
  FG_Keys_NUMPAD4         = 100,
  FG_Keys_F1              = 112,
  FG_Keys_LMENU           = 164,
  FG_Keys_E               = 69,
  FG_Keys_RCONTROL        = 163,
  FG_Keys_D               = 68,
  FG_Keys_NONCONVERT      = 29,
  FG_Keys_F11             = 122,
  FG_Keys_F8              = 119,
  FG_Keys_7               = 55,
  FG_Keys_NUMPAD7         = 103,
  FG_Keys_G               = 71,
  FG_Keys_3               = 51,
  FG_Keys_F4              = 115,
  FG_Keys_RIGHT           = 39,
  FG_Keys_RMENU           = 165,
  FG_Keys_XBUTTON2        = 6,
  FG_Keys_SCROLL          = 145,
  FG_Keys_CAPITAL         = 20,
  FG_Keys_F12             = 123,
  FG_Keys_I               = 73,
  FG_Keys_RBUTTON         = 2,
  FG_Keys_1               = 49,
  FG_Keys_OEM_8           = 223,
  FG_Keys_P               = 80,
  FG_Keys_U               = 85,
  FG_Keys_Z               = 90,
  FG_Keys_NULL            = 0,
  FG_Keys_CONTROL         = 17,
  FG_Keys_NUMPAD_DECIMAL  = 110,
  FG_Keys_K               = 75,
  FG_Keys_CLEAR           = 12,
  FG_Keys_M               = 77,
  FG_Keys_F2              = 113,
  FG_Keys_KANJI           = 25,
  FG_Keys_ESCAPE          = 27,
  FG_Keys_SHIFT           = 16,
  FG_Keys_F13             = 124,
  FG_Keys_MBUTTON         = 4,
  FG_Keys_LEFT            = 37,
  FG_Keys_HELP            = 47,
  FG_Keys_MINUS           = 189,
  FG_Keys_N               = 78,
  FG_Keys_0               = 48,
  FG_Keys_CONVERT         = 28,
  FG_Keys_O               = 79,
  FG_Keys_PRINT           = 42,
  FG_Keys_SLEEP           = 95,
  FG_Keys_NUMPAD_ADD      = 107,
  FG_Keys_NUMPAD_DIVIDE   = 111,
  FG_Keys_END             = 35,
  FG_Keys_F10             = 121,
  FG_Keys_NUMPAD_SUBTRACT = 109,
  FG_Keys_SEPARATOR       = 108,
  FG_Keys_Q               = 81,
  FG_Keys_F5              = 116,
  FG_Keys_2               = 50,
  FG_Keys_F3              = 114,
  FG_Keys_PAUSE           = 19,
  FG_Keys_F17             = 128
};

enum FG_ModKey
{
  FG_ModKey_CONTROL  = 2,
  FG_ModKey_SHIFT    = 1,
  FG_ModKey_NUMLOCK  = 32,
  FG_ModKey_HELD     = 64,
  FG_ModKey_ALT      = 4,
  FG_ModKey_CAPSLOCK = 16,
  FG_ModKey_SUPER    = 8
};

enum FG_MouseButton
{
  FG_MouseButton_X2 = 16,
  FG_MouseButton_M  = 4,
  FG_MouseButton_X4 = 64,
  FG_MouseButton_X3 = 32,
  FG_MouseButton_R  = 2,
  FG_MouseButton_X5 = 128,
  FG_MouseButton_L  = 1,
  FG_MouseButton_X1 = 8
};

enum FG_JoyAxis
{
  FG_JoyAxis_X       = 0,
  FG_JoyAxis_INVALID = 65535,
  FG_JoyAxis_U       = 4,
  FG_JoyAxis_Y       = 1,
  FG_JoyAxis_V       = 5,
  FG_JoyAxis_Z       = 2,
  FG_JoyAxis_R       = 3
};
enum FG_Joy
{
  FG_Joy_Button4  = 3,
  FG_Joy_Button21 = 20,
  FG_Joy_Button11 = 10,
  FG_Joy_Button12 = 11,
  FG_Joy_Button25 = 24,
  FG_Joy_Button3  = 2,
  FG_Joy_Button15 = 14,
  FG_Joy_ID2      = 256,
  FG_Joy_Button31 = 30,
  FG_Joy_Button32 = 31,
  FG_Joy_Button2  = 1,
  FG_Joy_Button28 = 27,
  FG_Joy_Button18 = 17,
  FG_Joy_ID7      = 1536,
  FG_Joy_ID15     = 3584,
  FG_Joy_ID14     = 3328,
  FG_Joy_Button22 = 21,
  FG_Joy_Button9  = 8,
  FG_Joy_ID12     = 2816,
  FG_Joy_ID3      = 512,
  FG_Joy_ID11     = 2560,
  FG_Joy_ID10     = 2304,
  FG_Joy_ID9      = 2048,
  FG_Joy_ID1      = 0,
  FG_Joy_ID8      = 1792,
  FG_Joy_Button20 = 19,
  FG_Joy_ID16     = 3840,
  FG_Joy_Button8  = 7,
  FG_Joy_ID6      = 1280,
  FG_Joy_Button26 = 25,
  FG_Joy_ID5      = 1024,
  FG_Joy_Button23 = 22,
  FG_Joy_Button1  = 0,
  FG_Joy_Button24 = 23,
  FG_Joy_ID4      = 768,
  FG_Joy_Button29 = 28,
  FG_Joy_Button16 = 15,
  FG_Joy_Button7  = 6,
  FG_Joy_Button10 = 9,
  FG_Joy_Button27 = 26,
  FG_Joy_Button30 = 29,
  FG_Joy_Button5  = 4,
  FG_Joy_Button17 = 16,
  FG_Joy_Button19 = 18,
  FG_Joy_ID13     = 3072,
  FG_Joy_Button14 = 13,
  FG_Joy_Button13 = 12,
  FG_Joy_Button6  = 5
};

struct FG_Msg_TouchMove
{
  float x;
  float y;
  float z;
  float r;
  float pressure;
  uint16_t index;
  uint8_t flags;
  uint8_t modkeys;
};
struct FG_Msg_MouseScroll
{
  float x;
  float y;
  float delta;
  float hdelta;
};
struct FG_Msg_MouseOn
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
};
struct FG_Msg_JoyButtonUp
{
  uint16_t index;
  uint16_t button;
  uint8_t modkeys;
};
struct FG_Msg_MouseOff
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
};
struct FG_Msg_JoyButtonDown
{
  uint16_t index;
  uint16_t button;
  uint8_t modkeys;
};
struct FG_Msg_GetWindowFlags
{
  char __padding;
};
struct FG_Msg_KeyDown
{
  uint8_t key;
  uint8_t modkeys;
  uint16_t scancode;
};
struct FG_Msg_Action
{
  int32_t subkind;
};
struct FG_Msg_KeyChar
{
  int32_t unicode;
  uint8_t modkeys;
};
struct FG_Msg_KeyUp
{
  uint8_t key;
  uint8_t modkeys;
  uint16_t scancode;
};
struct FG_Msg_TouchEnd
{
  float x;
  float y;
  float z;
  float r;
  float pressure;
  uint16_t index;
  uint8_t flags;
  uint8_t modkeys;
};
struct FG_Msg_TouchBegin
{
  float x;
  float y;
  float z;
  float r;
  float pressure;
  uint16_t index;
  uint8_t flags;
  uint8_t modkeys;
};
struct FG_Msg_SetWindowFlags
{
  uint32_t flags;
};
struct FG_Msg_MouseUp
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
  uint8_t button;
};
struct FG_Msg_MouseDblClick
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
  uint8_t button;
};
struct FG_Msg_JoyAxis
{
  uint16_t index;
  float value;
  uint16_t axis;
  uint8_t modkeys;
};
struct FG_Msg_LostFocus
{
  char __padding;
};
struct FG_Msg_MouseDown
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
  uint8_t button;
};
struct FG_Msg_JoyOrientation
{
  uint16_t index;
  FG_Vec3 velocity;
  FG_Vec3 rotation;
};
struct FG_Msg_Draw
{
  FG_Rect area;
};
struct FG_Msg_Drop
{
  int32_t kind;
  void* target;
  uint32_t count;
};
struct FG_Msg_SetWindowRect
{
  FG_Rect rect;
};
struct FG_Msg_GotFocus
{
  char __padding;
};
struct FG_Msg_MouseMove
{
  float x;
  float y;
  uint8_t all;
  uint8_t modkeys;
};

typedef struct FG_Msg__
{
  uint16_t kind;
  union
  {
    struct FG_Msg_TouchMove touchMove;
    struct FG_Msg_MouseScroll mouseScroll;
    struct FG_Msg_MouseOn mouseOn;
    struct FG_Msg_JoyButtonUp joyButtonUp;
    struct FG_Msg_MouseOff mouseOff;
    struct FG_Msg_JoyButtonDown joyButtonDown;
    struct FG_Msg_GetWindowFlags getWindowFlags;
    struct FG_Msg_KeyDown keyDown;
    struct FG_Msg_Action action;
    struct FG_Msg_KeyChar keyChar;
    struct FG_Msg_KeyUp keyUp;
    struct FG_Msg_TouchEnd touchEnd;
    struct FG_Msg_TouchBegin touchBegin;
    struct FG_Msg_SetWindowFlags setWindowFlags;
    struct FG_Msg_MouseUp mouseUp;
    struct FG_Msg_MouseDblClick mouseDblClick;
    struct FG_Msg_JoyAxis joyAxis;
    struct FG_Msg_LostFocus lostFocus;
    struct FG_Msg_MouseDown mouseDown;
    struct FG_Msg_JoyOrientation joyOrientation;
    struct FG_Msg_Draw draw;
    struct FG_Msg_Drop drop;
    struct FG_Msg_SetWindowRect setWindowRect;
    struct FG_Msg_GotFocus gotFocus;
    struct FG_Msg_MouseMove mouseMove;
  };
} FG_Msg;

typedef union FG_Result__
{
  int32_t touchMove;
  int32_t mouseScroll;
  int32_t mouseOn;
  int32_t joyButtonUp;
  int32_t mouseOff;
  int32_t joyButtonDown;
  uint32_t getWindowFlags;
  int32_t keyDown;
  int32_t action;
  int32_t keyChar;
  int32_t keyUp;
  int32_t touchEnd;
  int32_t touchBegin;
  int32_t setWindowFlags;
  int32_t mouseUp;
  int32_t mouseDblClick;
  int32_t joyAxis;
  int32_t lostFocus;
  int32_t mouseDown;
  int32_t joyOrientation;
  int32_t draw;
  void* drop;
  int32_t setWindowRect;
  int32_t gotFocus;
  int32_t mouseMove;
} FG_Result;

enum FG_Level
{
  FG_Level_NONE    = -1,
  FG_Level_FATAL   = 0,
  FG_Level_DEBUG   = 4,
  FG_Level_WARNING = 2,
  FG_Level_ERROR   = 1,
  FG_Level_NOTICE  = 3
};

enum FG_LogType
{
  FG_LogType_Boolean,
  FG_LogType_I32,
  FG_LogType_U32,
  FG_LogType_I64,
  FG_LogType_U64,
  FG_LogType_F32,
  FG_LogType_F64,
  FG_LogType_String,
  FG_LogType_OwnedString,
};

typedef struct FG_LogValue__
{
  enum FG_LogType type;
  union
  {
    bool bit;
    int i32;
    unsigned int u32;
    long long i64;
    unsigned long long u64;
    float f32;
    double f64;
    const char* string;
    char* owned;
  };
} FG_LogValue;

typedef void FG_Context;
typedef void FG_Element;
typedef void (*FG_Log)(void*, enum FG_Level, const char*, int, const char*, FG_LogValue*, int, void (*)(char*));
typedef FG_Result (*FG_Behavior)(FG_Element*, FG_Context*, void*, FG_Msg*);

enum FG_WindowFlag
{
  FG_WindowFlag_NOBORDER    = 16,
  FG_WindowFlag_NOCAPTION   = 8,
  FG_WindowFlag_FULLSCREEN  = 256,
  FG_WindowFlag_MINIMIZED   = 32,
  FG_WindowFlag_CLOSED      = 128,
  FG_WindowFlag_MAXIMIZABLE = 2,
  FG_WindowFlag_MINIMIZABLE = 1,
  FG_WindowFlag_RESIZABLE   = 4,
  FG_WindowFlag_MAXIMIZED   = 64
};

typedef struct FG_Window__
{
  void* handle;
  FG_Context* context;
  FG_Element* element;
} FG_Window;

typedef union FG_ShaderValue__
{
  float f32;
  double f64;
  int32_t i32;
  uint32_t u32;
  float* pf32;
  double* pf64;
  int32_t* pi32;
  uint32_t* pu32;
  FG_Resource* resource;
} FG_ShaderValue;

enum FG_AccessFlags
{
  FG_AccessFlag_READ = (1 << 0),
  FG_AccessFlag_WRITE = (1 << 1),
  FG_AccessFlag_PERSISTENT = (1 << 2),
  FG_AccessFlag_INVALIDATE_RANGE = (1 << 3),
  FG_AccessFlag_INVALIDATE_BUFFER = (1 << 4),
  FG_AccessFlag_UNSYNCHRONIZED    = (1 << 5),
};

struct FG_Backend
{
  FG_Caps (*getCaps)(FG_Backend* self);
  void* (*compileShader)(FG_Backend* self, FG_Context* context, enum FG_ShaderStage stage, const char* source);
  int (*destroyShader)(FG_Backend* self, FG_Context* context, void* shader);
  void* (*createCommandList)(FG_Backend* self, FG_Context* context, bool bundle);
  int (*destroyCommandList)(FG_Backend* self, void* commands);
  // clear 0 clears both, clear 1 is just depth, clear -1 is just stencil
  int (*clearDepthStencil)(FG_Backend* self, void* commands, FG_Resource* depthstencil, char clear, uint8_t stencil,
                           float depth, uint32_t num_rects, FG_Rect* rects);
  int (*clearRenderTarget)(FG_Backend* self, void* commands, FG_Resource* rendertarget, FG_Color RGBA, uint32_t num_rects,
                           FG_Rect* rects);
  int (*copyResource)(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest);
  int (*copySubresource)(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest, unsigned long srcoffset,
                         unsigned long destoffset, unsigned long bytes);
  int (*copyResourceRegion)(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest, FG_Vec3i srcoffset,
                            FG_Vec3i destoffset, FG_Vec3i size);
  int (*draw)(FG_Backend* self, void* commands, uint32_t vertexcount, uint32_t instancecount, uint32_t startvertex,
              uint32_t startinstance);
  int (*drawIndexed)(FG_Backend* self, void* commands, uint32_t indexcount, uint32_t instancecount, uint32_t startindex,
                     int startvertex, uint32_t startinstance);
  int (*setPipelineState)(FG_Backend* self, void* commands, void* state);
  int (*setDepthStencil)(FG_Backend* self, void* commands, bool Front, uint8_t StencilFailOp, uint8_t StencilDepthFailOp,
                         uint8_t StencilPassOp, uint8_t StencilFunc);
  int (*setViewports)(FG_Backend* self, void* commands, FG_Viewport* viewports, uint32_t count);
  int (*setShaderConstants)(FG_Backend* self, void* commands, const FG_ShaderParameter* uniforms,
                            const FG_ShaderValue* values, uint32_t count);
  int (*execute)(FG_Backend* self, FG_Context* context, void* commands);
  void* (*createPipelineState)(FG_Backend* self, FG_Context* context, FG_PipelineState* pipelinestate,
                               FG_Resource** rendertargets, uint32_t n_targets, FG_Blend* blends,
                               FG_Resource** vertexbuffer, int* strides, uint32_t n_buffers, FG_ShaderParameter* attributes,
                               uint32_t n_attributes, FG_Resource* indexbuffer, uint8_t indexstride);
  int (*destroyPipelineState)(FG_Backend* self, FG_Context* context, void* state);
  FG_Resource* (*createBuffer)(FG_Backend* self, FG_Context* context, void* data, uint32_t bytes, enum FG_Type usage);
  FG_Resource* (*createTexture)(FG_Backend* self, FG_Context* context, FG_Vec2i size, enum FG_Type usage,
                                enum FG_PixelFormat format, FG_Sampler* sampler, void* data, int MultiSampleCount);
  FG_Resource* (*createRenderTarget)(FG_Backend* self, FG_Context* context, FG_Resource* texture);
  int (*destroyResource)(FG_Backend* self, FG_Context* context, FG_Resource* resource);
  void* (*mapResource)(FG_Backend* self, FG_Context* context, FG_Resource* resource, uint32_t offset, uint32_t length,
                       enum FG_Type usage, uint32_t access);
  int (*unmapResource)(FG_Backend* self, FG_Context* context, FG_Resource* resource, enum FG_Type usage);
  FG_Window* (*createWindow)(FG_Backend* self, FG_Element* element, FG_Display* display, FG_Vec2* pos, FG_Vec2* dim,
                             const char* caption, uint64_t flags);
  int (*setWindow)(FG_Backend* self, FG_Window* window, FG_Element* element, FG_Display* display, FG_Vec2* pos,
                   FG_Vec2* dim, const char* caption, uint64_t flags);
  int (*destroyWindow)(FG_Backend* self, FG_Window* window);
  int (*beginDraw)(FG_Backend* self, FG_Context* context, FG_Rect* area);
  int (*endDraw)(FG_Backend* self, FG_Context* context);
  int (*putClipboard)(FG_Backend* self, FG_Window* window, enum FG_Clipboard kind, const char* data, uint32_t count);
  uint32_t (*getClipboard)(FG_Backend* self, FG_Window* window, enum FG_Clipboard kind, void* target, uint32_t count);
  bool (*checkClipboard)(FG_Backend* self, FG_Window* window, enum FG_Clipboard kind);
  int (*clearClipboard)(FG_Backend* self, FG_Window* window, enum FG_Clipboard kind);
  int (*processMessages)(FG_Backend* self, FG_Window* window);
  int (*getMessageSyncObject)(FG_Backend* self, FG_Window* window);
  int (*setCursor)(FG_Backend* self, FG_Window* window, enum FG_Cursor cursor);
  int (*getDisplayIndex)(FG_Backend* self, unsigned int index, FG_Display* out);
  int (*getDisplay)(FG_Backend* self, void* handle, FG_Display* out);
  int (*getDisplayWindow)(FG_Backend* self, FG_Window* window, FG_Display* out);
  int (*createSystemControl)(FG_Backend* self, FG_Context* context, const char* id, FG_Rect* area, ...);
  int (*setSystemControl)(FG_Backend* self, FG_Context* context, void* control, FG_Rect* area, ...);
  int (*destroySystemControl)(FG_Backend* self, FG_Context* context, void* control);
  int (*destroy)(FG_Backend* self);

  int cursorblink;
  int tooltipdelay;
};

typedef FG_Backend* (*FG_InitBackend)(void*, FG_Log, FG_Behavior);

#ifdef __cplusplus
}
#endif

#endif
