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
  FG_PixelFormat_Unknown = 0,
  FG_PixelFormat_R8G8B8A8_UInt,
  FG_PixelFormat_A8_UNorm,
  FG_PixelFormat_A8P8,
  FG_PixelFormat_AI44,
  FG_PixelFormat_AYUV,
  FG_PixelFormat_B4G4R4A4_UNorm,
  FG_PixelFormat_B5G5R5A1_UNorm,
  FG_PixelFormat_B5G6R5_UNorm,
  FG_PixelFormat_B8G8R8A8_Typeless,
  FG_PixelFormat_B8G8R8A8_UNorm,
  FG_PixelFormat_B8G8R8A8_UNorm_SRGB,
  FG_PixelFormat_B8G8R8X8_Typeless,
  FG_PixelFormat_B8G8R8X8_UNorm,
  FG_PixelFormat_B8G8R8X8_UNorm_SRGB,
  FG_PixelFormat_BC1_Typeless,
  FG_PixelFormat_BC1_UNorm,
  FG_PixelFormat_BC1_UNorm_SRGB,
  FG_PixelFormat_BC2_Typeless,
  FG_PixelFormat_BC2_UNorm,
  FG_PixelFormat_BC2_UNorm_SRGB,
  FG_PixelFormat_BC3_Typeless,
  FG_PixelFormat_BC3_UNorm,
  FG_PixelFormat_BC3_UNorm_SRGB,
  FG_PixelFormat_BC4_SNorm,
  FG_PixelFormat_BC4_Typeless,
  FG_PixelFormat_BC4_UNorm,
  FG_PixelFormat_BC5_SNorm,
  FG_PixelFormat_BC5_Typeless,
  FG_PixelFormat_BC5_UNorm,
  FG_PixelFormat_BC6H_SF16,
  FG_PixelFormat_BC6H_Typeless,
  FG_PixelFormat_BC6H_UF16,
  FG_PixelFormat_BC7_Typeless,
  FG_PixelFormat_BC7_UNorm,
  FG_PixelFormat_BC7_UNorm_SRGB,
  FG_PixelFormat_D16_UNorm,
  FG_PixelFormat_D24_UNorm_S8_UInt,
  FG_PixelFormat_D24_UNorm_X8_Typeless,
  FG_PixelFormat_D32_Float,
  FG_PixelFormat_D32_Float_S8X24_UInt,
  FG_PixelFormat_G8R8_G8B8_UNorm,
  FG_PixelFormat_IA44,
  FG_PixelFormat_NV11,
  FG_PixelFormat_NV12,
  FG_PixelFormat_P010,
  FG_PixelFormat_P016,
  FG_PixelFormat_P208,
  FG_PixelFormat_P8,
  FG_PixelFormat_R1_UNorm,
  FG_PixelFormat_R10G10B10_XR_BIAS_A2_UNorm,
  FG_PixelFormat_R10G10B10A2_Typeless,
  FG_PixelFormat_R10G10B10A2_UInt,
  FG_PixelFormat_R10G10B10A2_UNorm,
  FG_PixelFormat_R11G11B10_Float,
  FG_PixelFormat_R16_Float,
  FG_PixelFormat_R16_Int,
  FG_PixelFormat_R16_SNorm,
  FG_PixelFormat_R16_Typeless,
  FG_PixelFormat_R16_UInt,
  FG_PixelFormat_R16_UNorm,
  FG_PixelFormat_R16G16_Float,
  FG_PixelFormat_R16G16_Int,
  FG_PixelFormat_R16G16_SNorm,
  FG_PixelFormat_R16G16_Typeless,
  FG_PixelFormat_R16G16_UInt,
  FG_PixelFormat_R16G16_UNorm,
  FG_PixelFormat_R16G16B16_Float,
  FG_PixelFormat_R16G16B16_Int,
  FG_PixelFormat_R16G16B16_SNorm,
  FG_PixelFormat_R16G16B16_Typeless,
  FG_PixelFormat_R16G16B16_UInt,
  FG_PixelFormat_R16G16B16_UNorm,
  FG_PixelFormat_R16G16B16A16_Float,
  FG_PixelFormat_R16G16B16A16_Int,
  FG_PixelFormat_R16G16B16A16_SNorm,
  FG_PixelFormat_R16G16B16A16_Typeless,
  FG_PixelFormat_R16G16B16A16_UInt,
  FG_PixelFormat_R16G16B16A16_UNorm,
  FG_PixelFormat_R24_UNorm_X8_Typeless,
  FG_PixelFormat_R24G8_Typeless,
  FG_PixelFormat_R32_Float,
  FG_PixelFormat_R32_Float_X8X24_Typeless,
  FG_PixelFormat_R32_Int,
  FG_PixelFormat_R32_Typeless,
  FG_PixelFormat_R32_UInt,
  FG_PixelFormat_R32G32_Float,
  FG_PixelFormat_R32G32_Int,
  FG_PixelFormat_R32G32_Typeless,
  FG_PixelFormat_R32G32_UInt,
  FG_PixelFormat_R32G32B32_Float,
  FG_PixelFormat_R32G32B32_Int,
  FG_PixelFormat_R32G32B32_Typeless,
  FG_PixelFormat_R32G32B32_UInt,
  FG_PixelFormat_R32G32B32A32_Float,
  FG_PixelFormat_R32G32B32A32_Int,
  FG_PixelFormat_R32G32B32A32_Typeless,
  FG_PixelFormat_R32G32B32A32_UInt,
  FG_PixelFormat_R32G8X24_Typeless,
  FG_PixelFormat_R5G5B5A1_UNorm,
  FG_PixelFormat_R5G6B5_UNorm,
  FG_PixelFormat_R8_Int,
  FG_PixelFormat_R8_SNorm,
  FG_PixelFormat_R8_Typeless,
  FG_PixelFormat_R8_UInt,
  FG_PixelFormat_R8_UNorm,
  FG_PixelFormat_R8G8_B8G8_UNorm,
  FG_PixelFormat_R8G8_Int,
  FG_PixelFormat_R8G8_SNorm,
  FG_PixelFormat_R8G8_Typeless,
  FG_PixelFormat_R8G8_UInt,
  FG_PixelFormat_R8G8_UNorm,
  FG_PixelFormat_R8G8B8A8_Int,
  FG_PixelFormat_R8G8B8A8_SNorm,
  FG_PixelFormat_R8G8B8A8_Typeless,
  FG_PixelFormat_R8G8B8A8_UNorm,
  FG_PixelFormat_R8G8B8A8_UNorm_SRGB,
  FG_PixelFormat_R8G8B8X8_Int,
  FG_PixelFormat_R8G8B8X8_SNorm,
  FG_PixelFormat_R8G8B8X8_Typeless,
  FG_PixelFormat_R8G8B8X8_UInt,
  FG_PixelFormat_R8G8B8X8_UNorm,
  FG_PixelFormat_R9G9B9E5_SHAREDEXP,
  FG_PixelFormat_V208,
  FG_PixelFormat_V408,
  FG_PixelFormat_X24_Typeless_G8_UInt,
  FG_PixelFormat_X32_Typeless_G8X24_UInt,
  FG_PixelFormat_Y210,
  FG_PixelFormat_Y216,
  FG_PixelFormat_Y410,
  FG_PixelFormat_Y416,
  FG_PixelFormat_YUY2,
};

enum FG_Usage
{
  FG_Usage_Unknown = 0,
  FG_Usage_Vertex_Data,
  FG_Usage_Vertex_Indices,
  FG_Usage_Pixel_Read,
  FG_Usage_Pixel_Write,
  FG_Usage_Copy_Read,
  FG_Usage_Copy_Write,
  FG_Usage_Texture_Buffer,
  FG_Usage_Transform,
  FG_Usage_Uniform,
  FG_Usage_Texture1D,
  FG_Usage_Texture2D,
  FG_Usage_Texture3D,
  FG_Usage_Texture2D_Multisample,
  FG_Usage_Texture2D_Multisample_Proxy,
  FG_Usage_Renderbuffer,
  FG_Usage_Storage_Buffer,
};

enum FG_ShaderStage
{
  FG_ShaderStage_Pixel = 0,
  FG_ShaderStage_Vertex,
  FG_ShaderStage_Geometry,
  FG_ShaderStage_Hull,
  FG_ShaderStage_Domain,
  FG_ShaderStage_Compute,
  FG_ShaderStage_Mesh,
  FG_ShaderStage_Task,
  FG_ShaderStage_Count
};

enum FG_Feature
{
  FG_Feature_API_OpenGL_ES      = 1,
  FG_Feature_API_OpenGL         = 2,
  FG_Feature_API_DirectX        = 3,
  FG_Feature_API_Vulkan         = 4,
  FG_Feature_API_Metal          = 5,
  FG_Feature_Immediate_Mode     = (1 << 4),
  FG_Feature_RenderTarget       = (1 << 5),
  FG_Feature_Blend_EX           = (1 << 6),
  FG_Feature_Background_Opacity = (1 << 7),
  FG_Feature_Lines_Alpha        = (1 << 8),
  FG_Feature_Blend_Gamma        = (1 << 9),
  FG_Feature_Instancing         = (1 << 10),
  FG_Feature_Multithreading     = (1 << 11),
  FG_Feature_Command_Bundles    = (1 << 12),
  FG_Feature_Independent_Blend  = (1 << 13),
  FG_Feature_Compute_Shader     = (1 << 14),
  FG_Feature_Mesh_Shader        = (1 << 15),
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
  FG_Vec3i max_workgroups;
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
  FG_OpenGL_Caps openGL;
  FG_OpenGL_Caps openGL_ES;
  FG_DirectX_Caps directX;
} FG_Caps;

typedef uintptr_t FG_Resource;
typedef uintptr_t FG_Shader;

enum FG_Primitive
{
  FG_Primitive_Point                    = 0,
  FG_Primitive_Line                     = 1,
  FG_Primitive_Triangle                 = 2,
  FG_Primitive_Line_Strip               = 3,
  FG_Primitive_Triangle_Strip           = 4,
  FG_Primitive_Line_Adjacency           = 5,
  FG_Primitive_Triangle_Adjacency       = 6,
  FG_Primitive_Line_Strip_Adjacency     = 7,
  FG_Primitive_Triangle_Strip_Adjacency = 8,
};

enum FG_Comparison
{
  FG_Comparison_Disabled      = 0,
  FG_Comparison_Never         = 1,
  FG_Comparison_Less          = 2,
  FG_Comparison_Equal         = 3,
  FG_Comparison_Less_Equal    = 4,
  FG_Comparison_Greater       = 5,
  FG_Comparison_Not_Equal     = 6,
  FG_Comparison_Greater_Equal = 7,
  FG_Comparison_Always        = 8
};

enum FG_Strip_Cut_Value
{
  FG_Strip_Cut_Value_Disabled   = 0,
  FG_Strip_Cut_Value_0xFFFF     = 1,
  FG_Strip_Cut_Value_0xFFFFFFFF = 2
};

enum FG_Blend_Operand
{
  FG_Blend_Operand_Zero             = 1,
  FG_Blend_Operand_One              = 2,
  FG_Blend_Operand_Src_Color        = 3,
  FG_Blend_Operand_Inv_Src_Color    = 4,
  FG_Blend_Operand_Src_Alpha        = 5,
  FG_Blend_Operand_Inv_Src_Alpha    = 6,
  FG_Blend_Operand_Dest_Alpha       = 7,
  FG_Blend_Operand_Inv_Dest_Alpha   = 8,
  FG_Blend_Operand_Dest_Color       = 9,
  FG_Blend_Operand_Inv_Dest_Color   = 10,
  FG_Blend_Operand_Src_Alpha_SAT    = 11,
  FG_Blend_Operand_Blend_FACTOR     = 14,
  FG_Blend_Operand_Inv_Blend_FACTOR = 15,
  FG_Blend_Operand_Src1_Color       = 16,
  FG_Blend_Operand_Inv_Src1_Color   = 17,
  FG_Blend_Operand_Src1_Alpha       = 18,
  FG_Blend_Operand_Inv_Src1_Alpha   = 19
};

enum FG_Blend_OP
{
  FG_Blend_OP_Add          = 1,
  FG_Blend_OP_Subtract     = 2,
  FG_Blend_OP_Rev_Subtract = 3,
  FG_Blend_OP_Min          = 4,
  FG_Blend_OP_Max          = 5
};

// Each bit here represents an element of the pipelinestate struct that has been set
enum FG_Pipeline_Member
{
  FG_Pipeline_Member_VS                 = (1 << 0),
  FG_Pipeline_Member_GS                 = (1 << 1),
  FG_Pipeline_Member_HS                 = (1 << 2),
  FG_Pipeline_Member_DS                 = (1 << 3),
  FG_Pipeline_Member_PS                 = (1 << 4),
  FG_Pipeline_Member_CS                 = (1 << 5),
  FG_Pipeline_Member_MS                 = (1 << 6),
  FG_Pipeline_Member_TS                 = (1 << 7),
  FG_Pipeline_Member_Blend_Factor       = (1 << 8),
  FG_Pipeline_Member_Flags              = (1 << 9),
  FG_Pipeline_Member_Sample_Mask        = (1 << 10),
  FG_Pipeline_Member_Stencil_Ref        = (1 << 11),
  FG_Pipeline_Member_Stencil_Read_Mask  = (1 << 12),
  FG_Pipeline_Member_Stencil_Write_Mask = (1 << 13),
  FG_Pipeline_Member_Stencil_OP       = (1 << 14),
  FG_Pipeline_Member_Stencil_Func       = (1 << 17),
  FG_Pipeline_Member_Depth_Func         = (1 << 18),
  FG_Pipeline_Member_Fill               = (1 << 19),
  FG_Pipeline_Member_Cull               = (1 << 20),
  FG_Pipeline_Member_Primitive          = (1 << 21),
  FG_Pipeline_Member_Depth_Slope_Bias         = (1 << 22),
  FG_Pipeline_Member_Node_Mask          = (1 << 24),
};

enum FG_Fill_Mode
{
  FG_Fill_Mode_Fill  = 0,
  FG_Fill_Mode_Line  = 1,
  FG_Fill_Mode_Point = 2,
};

enum FG_Cull_Mode
{
  FG_Cull_Mode_None  = 0,
  FG_Cull_Mode_Front = 1,
  FG_Cull_Mode_Back  = 2,
};

enum FG_Pipeline_Flags
{
  FG_Pipeline_Flag_Alpha_To_Coverage_Enable = (1 << 0),
  FG_Pipeline_Flag_Blend_Enable             = (1 << 1),
  FG_Pipeline_Flag_Depth_Enable             = (1 << 2),
  FG_Pipeline_Flag_Stencil_Enable           = (1 << 3),
  FG_Pipeline_Flag_Depth_Write_Enable       = (1 << 4),
  FG_Pipeline_Flag_Conservative_Raster      = (1 << 5),
  FG_Pipeline_Flag_Front_Counter_Clockwise  = (1 << 6),
  FG_Pipeline_Flag_Depth_Clip_Enable        = (1 << 7),
  FG_Pipeline_Flag_Multisample_Enable       = (1 << 8),
  FG_Pipeline_Flag_Antialiased_Line_Enable  = (1 << 9),
  FG_Pipeline_Flag_Tool_Debug               = (1 << 10),
  FG_Pipeline_Flag_Independent_Blend_Enable = (1 << 11),
  FG_Pipeline_Flag_RenderTarget_SRGB_Enable = (1 << 12),
};

enum FG_Stencil_OP
{
  FG_Stencil_OP_None           = 0,
  FG_Stencil_OP_Keep           = 1,
  FG_Stencil_OP_Zero           = 2,
  FG_Stencil_OP_Replace        = 3,
  FG_Stencil_OP_Increment_Sat  = 4,
  FG_Stencil_OP_Decrement_Sat  = 5,
  FG_Stencil_OP_Invert         = 6,
  FG_Stencil_OP_Increment_Wrap = 7,
  FG_Stencil_OP_Decrement_Wrap = 8
};

typedef struct FG_PipelineState__
{
  uint64_t members; // Each bit represents a particular member that has been set
  FG_Shader shaders[FG_ShaderStage_Count];
  FG_Color blendFactor;
  uint16_t flags;
  uint32_t sampleMask;
  uint32_t stencilRef;
  uint8_t stencilReadMask;
  uint8_t stencilWriteMask;
  uint8_t stencilFailOp;
  uint8_t stencilDepthFailOp;
  uint8_t stencilPassOp;
  uint8_t stencilFunc;
  uint8_t depthFunc;
  uint8_t fillMode;
  uint8_t cullMode;
  uint8_t primitive;
  int depthBias;
  float slopeScaledDepthBias;
  uint32_t nodeMask;
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

enum FG_Filter
{
  FG_Filter_Min_Mag_Mip_Point                          = 0,
  FG_Filter_Min_Mag_Point_Mip_Linear                   = 0x1,
  FG_Filter_Min_Point_Mag_Linear_Mip_Point             = 0x4,
  FG_Filter_Min_Point_Mag_Mip_Linear                   = 0x5,
  FG_Filter_Min_Linear_Mag_Mip_Point                   = 0x10,
  FG_Filter_Min_Linear_Mag_Point_Mip_Linear            = 0x11,
  FG_Filter_Min_Mag_Linear_Mip_Point                   = 0x14,
  FG_Filter_Min_Mag_Mip_Linear                         = 0x15,
  FG_Filter_Anisotropic                                = 0x55,
  FG_Filter_Comparison_Min_Mag_Mip_Point               = 0x80,
  FG_Filter_Comparison_Min_Mag_Point_Mip_Linear        = 0x81,
  FG_Filter_Comparison_Min_Point_Mag_Linear_Mip_Point  = 0x84,
  FG_Filter_Comparison_Min_Point_Mag_Mip_Linear        = 0x85,
  FG_Filter_Comparison_Min_Linear_Mag_Mip_Point        = 0x90,
  FG_Filter_Comparison_Min_Linear_Mag_Point_Mip_Linear = 0x91,
  FG_Filter_Comparison_Min_Mag_Linear_Mip_Point        = 0x94,
  FG_Filter_Comparison_Min_Mag_Mip_Linear              = 0x95,
  FG_Filter_Comparison_Anisotropic                     = 0xd5,
};

enum FG_Texture_Address_Mode
{
  FG_Texture_Address_Mode_Wrap        = 1,
  FG_Texture_Address_Mode_Mirror      = 2,
  FG_Texture_Address_Mode_Clamp       = 3,
  FG_Texture_Address_Mode_Border      = 4,
  FG_Texture_Address_Mode_Mirror_Once = 5
};

typedef struct FG_Sampler__
{
  uint8_t filter;        // FG_Filter
  uint8_t addressing[3]; // enum FG_Texture_Address_Mode
  uint8_t max_anisotropy;
  uint8_t comparison; // FG_Comparison
  float mip_bias;
  float min_lod;
  float max_lod;
  FG_Color border_color;
} FG_Sampler;

enum FG_Logic_OP
{
  FG_Logic_OP_Clear = 0,
  FG_Logic_OP_Set,
  FG_Logic_OP_Copy,
  FG_Logic_OP_Copy_Inverted,
  FG_Logic_OP_NOOP,
  FG_Logic_OP_Invert,
  FG_Logic_OP_And,
  FG_Logic_OP_NAnd,
  FG_Logic_OP_Or,
  FG_Logic_OP_Nor,
  FG_Logic_OP_Xor,
  FG_Logic_OP_Equiv,
  FG_Logic_OP_And_Reverse,
  FG_Logic_OP_And_Inverted,
  FG_Logic_OP_Or_Reverse,
  FG_Logic_OP_Or_Inverted
};

// Array
typedef struct FG_Blend__
{
  uint8_t src_blend;
  uint8_t dest_blend;
  uint8_t blend_op;
  uint8_t src_blend_alpha;
  uint8_t dest_blend_alpha;
  uint8_t blend_op_alpha;
  uint8_t rendertarget_write_mask;
} FG_Blend;

// Array
typedef struct FG_Viewport__
{
  FG_Vec3 pos;
  FG_Vec3 dim;
  FG_Rect scissor;
} FG_Viewport;

enum FG_Vertex_Type
{
  FG_Vertex_Type_Half = 0,
  FG_Vertex_Type_Float,
  FG_Vertex_Type_Double,
  FG_Vertex_Type_Byte,
  FG_Vertex_Type_UByte,
  FG_Vertex_Type_Short,
  FG_Vertex_Type_UShort,
  FG_Vertex_Type_Int,
  FG_Vertex_Type_UInt,
  // FG_VertexType_Int_2_10_10_10,
  // FG_VertexType_UInt_2_10_10_10,
};

// Array
typedef struct FG_VertexParameter__
{
  const char* name;
  uint32_t offset;
  uint32_t step;
  uint8_t length;
  uint8_t type; // enum FG_ShaderType
  uint8_t index;
  bool per_instance; // false if per_vertex
} FG_VertexParameter;

enum FG_Shader_Type
{
  FG_Shader_Type_Half         = 0,
  FG_Shader_Type_Float        = 1,
  FG_Shader_Type_Double       = 2,
  FG_Shader_Type_Int          = 3,
  FG_Shader_Type_UInt         = 4,
  FG_Shader_Type_Color32      = 5,
  FG_Shader_Type_Texture      = 6,
  FG_Shader_Type_Texture_Cube = 7,
  FG_Shader_Type_Buffer       = 8,
};
// Array
typedef struct FG_ShaderParameter__
{
  const char* name;
  uint32_t length;
  uint32_t width; // or offset for buffers
  uint32_t count; // or index for buffers
  uint8_t type;   // enum FG_ShaderType
} FG_ShaderParameter;

typedef struct FG_Display__
{
  FG_Vec2i size;
  FG_Vec2i offset;
  FG_Vec2 dpi;
  float scale;
  uintptr_t handle;
  bool primary;
} FG_Display;

enum FG_Clipboard
{
  FG_Clipboard_None    = 0,
  FG_Clipboard_Text    = 1,
  FG_Clipboard_Wave    = 2,
  FG_Clipboard_Bitmap  = 3,
  FG_Clipboard_File    = 4,
  FG_Clipboard_Element = 5,
  FG_Clipboard_Custom  = 6,
  FG_Clipboard_All     = 7,
};

enum FG_Cursor
{
  FG_Cursor_None       = 0,
  FG_Cursor_Arrow      = 1,
  FG_Cursor_IBeam      = 2,
  FG_Cursor_Cross      = 3,
  FG_Cursor_Wait       = 4,
  FG_Cursor_Hand       = 5,
  FG_Cursor_ResizeNS   = 6,
  FG_Cursor_ResizeWE   = 7,
  FG_Cursor_ResizeNWSE = 8,
  FG_Cursor_ResizeNESW = 9,
  FG_Cursor_ResizeALL  = 10,
  FG_Cursor_No         = 11,
  FG_Cursor_Help       = 12,
  FG_Cursor_Drag       = 13,
  FG_Cursor_Custom     = 14,
};

enum FG_Event_Kind
{
  FG_Event_Kind_Action         = 0,
  FG_Event_Kind_Draw           = 1,
  FG_Event_Kind_Drop           = 2,
  FG_Event_Kind_GetWindowFlags = 3,
  FG_Event_Kind_GotFocus       = 4,
  FG_Event_Kind_JoyAxis        = 5,
  FG_Event_Kind_JoyButtonDown  = 6,
  FG_Event_Kind_JoyButtonUp    = 7,
  FG_Event_Kind_JoyOrientation = 8,
  FG_Event_Kind_KeyChar        = 9,
  FG_Event_Kind_KeyDown        = 10,
  FG_Event_Kind_KeyUp          = 11,
  FG_Event_Kind_LostFocus      = 12,
  FG_Event_Kind_MouseDblClick  = 13,
  FG_Event_Kind_MouseDown      = 14,
  FG_Event_Kind_MouseMove      = 15,
  FG_Event_Kind_MouseOff       = 16,
  FG_Event_Kind_MouseOn        = 17,
  FG_Event_Kind_MouseScroll    = 18,
  FG_Event_Kind_MouseUp        = 19,
  FG_Event_Kind_SetWindowFlags = 20,
  FG_Event_Kind_SetWindowRect  = 21,
  FG_Event_Kind_TouchBegin     = 22,
  FG_Event_Kind_TouchEnd       = 23,
  FG_Event_Kind_TouchMove      = 24,
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
  FG_ModKey_Shift    = 1,
  FG_ModKey_Control  = 2,
  FG_ModKey_Alt      = 4,
  FG_ModKey_Super    = 8,
  FG_ModKey_Capslock = 16,
  FG_ModKey_Numlock  = 32,
  FG_ModKey_Held     = 64,
};

enum FG_MouseButton
{
  FG_MouseButton_L  = 1,
  FG_MouseButton_R  = 2,
  FG_MouseButton_M  = 4,
  FG_MouseButton_X1 = 8,
  FG_MouseButton_X2 = 16,
  FG_MouseButton_X3 = 32,
  FG_MouseButton_X4 = 64,
  FG_MouseButton_X5 = 128,
};

enum FG_JoyAxis
{
  FG_JoyAxis_X       = 0,
  FG_JoyAxis_Y       = 1,
  FG_JoyAxis_Z       = 2,
  FG_JoyAxis_R       = 3,
  FG_JoyAxis_U       = 4,
  FG_JoyAxis_V       = 5,
  FG_JoyAxis_Invalid = 65535,
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
  uintptr_t drop;
  int32_t setWindowRect;
  int32_t gotFocus;
  int32_t mouseMove;
} FG_Result;

enum FG_Level
{
  FG_Level_Fatal   = 0,
  FG_Level_Error   = 1,
  FG_Level_Warning = 2,
  FG_Level_Notice  = 3,
  FG_Level_Ddebug  = 4,
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
  FG_WindowFlag_Minimizable = 1,
  FG_WindowFlag_Maximizable = 2,
  FG_WindowFlag_Resizable   = 4,
  FG_WindowFlag_No_Caption  = 8,
  FG_WindowFlag_No_Border   = 16,
  FG_WindowFlag_Minimized   = 32,
  FG_WindowFlag_Maximized   = 64,
  FG_WindowFlag_Closed      = 128,
  FG_WindowFlag_Fullscreen  = 256,
};

typedef struct FG_Window__
{
  uintptr_t handle;
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
  FG_Resource resource;
} FG_ShaderValue;

enum FG_AccessFlags
{
  FG_AccessFlag_Read              = (1 << 0),
  FG_AccessFlag_Write             = (1 << 1),
  FG_AccessFlag_Persistent        = (1 << 2),
  FG_AccessFlag_Invalidate_Range  = (1 << 3),
  FG_AccessFlag_Invalidate_Buffer = (1 << 4),
  FG_AccessFlag_Unsynchronized    = (1 << 5),
};

enum FG_BarrierFlags
{
  FG_BarrierFlags_Vertex             = (1 << 0),
  FG_BarrierFlags_Element            = (1 << 1),
  FG_BarrierFlags_Uniform            = (1 << 2),
  FG_BarrierFlags_Texture_Fetch      = (1 << 3),
  FG_BarrierFlags_Texture_Update     = (1 << 4),
  FG_BarrierFlags_Image_Access       = (1 << 5),
  FG_BarrierFlags_Command            = (1 << 6),
  FG_BarrierFlags_Pixel              = (1 << 7),
  FG_BarrierFlags_Buffer             = (1 << 8),
  FG_BarrierFlags_RenderTarget       = (1 << 9),
  FG_BarrierFlags_Storage_Buffer     = (1 << 10),
  FG_BarrierFlags_Transform_Feedback = (1 << 11),
  FG_BarrierFlags_Atomic_Counter     = (1 << 12),
};

enum FG_ClearFlags
{
  FG_ClearFlag_Color       = (1 << 0),
  FG_ClearFlag_Depth       = (1 << 1),
  FG_ClearFlag_Stencil     = (1 << 2),
  FG_ClearFlag_Accumulator = (1 << 3),
};

struct FG_Backend
{
  FG_Caps (*getCaps)(FG_Backend* self);
  FG_Shader (*compileShader)(FG_Backend* self, FG_Context* context, enum FG_ShaderStage stage, const char* source);
  int (*destroyShader)(FG_Backend* self, FG_Context* context, FG_Shader shader);
  void* (*createCommandList)(FG_Backend* self, FG_Context* context, bool bundle);
  int (*destroyCommandList)(FG_Backend* self, void* commands);
  int (*clear)(FG_Backend* self, void* commands, uint8_t clearbits, FG_Color RGBA, uint8_t stencil, float depth,
               uint32_t num_rects, FG_Rect* rects);
  int (*copyResource)(FG_Backend* self, void* commands, FG_Resource src, FG_Resource dest);
  int (*copySubresource)(FG_Backend* self, void* commands, FG_Resource src, FG_Resource dest, unsigned long srcoffset,
                         unsigned long destoffset, unsigned long bytes);
  int (*copyResourceRegion)(FG_Backend* self, void* commands, FG_Resource src, FG_Resource dest, FG_Vec3i srcoffset,
                            FG_Vec3i destoffset, FG_Vec3i size);
  int (*draw)(FG_Backend* self, void* commands, uint32_t vertexcount, uint32_t instancecount, uint32_t startvertex,
              uint32_t startinstance);
  int (*drawIndexed)(FG_Backend* self, void* commands, uint32_t indexcount, uint32_t instancecount, uint32_t startindex,
                     int startvertex, uint32_t startinstance);
  int (*dispatch)(FG_Backend* self, void* commands);
  int (*syncPoint)(FG_Backend* self, void* commands, uint32_t barrier_flags);
  int (*setPipelineState)(FG_Backend* self, void* commands, uintptr_t state);
  int (*setDepthStencil)(FG_Backend* self, void* commands, bool Front, uint8_t StencilFailOp, uint8_t StencilDepthFailOp,
                         uint8_t StencilPassOp, uint8_t StencilFunc);
  int (*setViewports)(FG_Backend* self, void* commands, FG_Viewport* viewports, uint32_t count);
  int (*setShaderConstants)(FG_Backend* self, void* commands, const FG_ShaderParameter* uniforms,
                            const FG_ShaderValue* values, uint32_t count);
  int (*execute)(FG_Backend* self, FG_Context* context, void* commands);
  uintptr_t (*createPipelineState)(FG_Backend* self, FG_Context* context, FG_PipelineState* pipelinestate,
                                   FG_Resource rendertarget, FG_Blend* blends, FG_Resource* vertexbuffer, int* strides,
                                   uint32_t n_buffers, FG_VertexParameter* attributes, uint32_t n_attributes,
                                   FG_Resource indexbuffer, uint8_t indexstride);
  uintptr_t (*createComputePipeline)(FG_Backend* self, FG_Context* context, FG_Shader computeshader, FG_Vec3i workgroup,
                                     uint32_t flags);
  int (*destroyPipelineState)(FG_Backend* self, FG_Context* context, uintptr_t state);
  FG_Resource (*createBuffer)(FG_Backend* self, FG_Context* context, void* data, uint32_t bytes, enum FG_Usage usage);
  FG_Resource (*createTexture)(FG_Backend* self, FG_Context* context, FG_Vec2i size, enum FG_Usage usage,
                               enum FG_PixelFormat format, FG_Sampler* sampler, void* data, int MultiSampleCount);
  FG_Resource (*createRenderTarget)(FG_Backend* self, FG_Context* context, FG_Resource depthstencil, FG_Resource* textures,
                                    uint32_t n_textures, int attachments);
  int (*destroyResource)(FG_Backend* self, FG_Context* context, FG_Resource resource);
  void* (*mapResource)(FG_Backend* self, FG_Context* context, FG_Resource resource, uint32_t offset, uint32_t length,
                       enum FG_Usage usage, uint32_t access);
  int (*unmapResource)(FG_Backend* self, FG_Context* context, FG_Resource resource, enum FG_Usage usage);
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
  int (*getDisplay)(FG_Backend* self, uintptr_t handle, FG_Display* out);
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
