/* graphics_interface.h - Standard C interface for feather graphics interfaces
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

#ifndef GRAPHICS_INTERFACE_H
#define GRAPHICS_INTERFACE_H

#include "shared_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  FG_Feature_Anisotropic_Filter = (1 << 10),
  FG_Feature_Half_Float         = (1 << 11),
  FG_Feature_Atomic_Counters    = (1 << 12),
  FG_Feature_StorageBuffer      = (1 << 13),
  FG_Feature_Sync               = (1 << 14),
  FG_Feature_Instancing         = (1 << 15),
  FG_Feature_Multithreading     = (1 << 16),
  FG_Feature_Command_Bundles    = (1 << 17),
  FG_Feature_Independent_Blend  = (1 << 18),
  FG_Feature_Compute_Shader     = (1 << 19),
  FG_Feature_Tesselation_Shader = (1 << 20),
  FG_Feature_Mesh_Shader        = (1 << 21),
  FG_Feature_Headless           = (1 << 22),
};

// This can hold caps for either OpenGL or OpenGL ES. These have different version numbers.
typedef struct FG_OpenGL_Caps__
{
  int features;
  int version; // 2.1 is 21, 4.2 is 42, etc.
  int glsl;    // Version 1.10 is 110, version 4.40 is 440
  int max_texture_size;
  FG_Vec2i max_viewport_size;
  int max_3d_texture_size;
  int max_vertex_buffer_size;
  int max_index_buffer_size;
  int max_cube_texture_size;
  float max_texture_bias;
  int max_vertex_attributes;
  int max_texture_samplers;
  int max_fragment_uniforms;
  int max_vertex_uniforms;
  int max_varying_outputs;
  int max_vertex_texture_samplers;
  int max_samplers;
  int max_texture_levels;
  int max_texel_offset;
  int max_renderbuffer_size;
  int max_rendertargets;
  int max_multisampling;
  int max_texturebuffer_size;
  int max_vertex_uniform_blocks;
  int max_geometry_uniform_blocks;
  int max_geometry_texture_samplers;
  int max_texture_coordinates;
  int max_geometry_uniforms;
  int max_geometry_output_vertices;
  int max_geometry_total_output;
  int max_vertex_output_components;
  int max_geometry_input_components;
  int max_geometry_output_components;
  int max_fragment_input_components;
  int max_multisample_mask_words;
  int max_multisample_color_samples;
  int max_multisample_depth_samples;
  int max_dual_source_rendertargets;
  int max_compute_uniform_blocks;
  int max_compute_samplers;
  int max_compute_image_uniforms;
  int max_compute_shared_memory;
  int max_compute_uniforms;
  int max_compute_atomic_counter_buffers;
  int max_compute_atomic_counters;
  int max_compute_components;
  int max_work_group_invocations;
  FG_Vec3i max_work_group_count;
  FG_Vec3i max_work_group_size;
  int max_atomic_counter_buffers;
  int max_atomic_counters;
  int max_image_uniforms;
  int max_storage_blocks;
  int max_storage_buffers;
  int max_storage_block_size;
  int max_shader_output_resources;
  int max_texture_anisotropy;
  int max_patch_vertices;
  int max_tessellation_level;
  int max_tess_control_uniforms;
  int max_tess_evaluation_uniforms;
  int max_tess_control_components;
  int max_tess_evaluation_components;
  int max_tess_control_samplers;
  int max_tess_evaluation_samplers;
  int max_tess_control_output_components;
  int max_tess_evaluation_output_components;
  int max_control_outputs;
  int max_patch_components;
  int max_tess_control_inputs;
  int max_tess_evaluation_inputs;
  int max_images;
  int max_mesh_samplers;
  int max_mesh_uniforms;
  int max_mesh_image_uniforms;
  int max_mesh_atomic_counter_buffers;
  int max_mesh_atomic_counters;
  int max_mesh_components;
  int max_task_samplers;
  int max_task_image_uniforms;
  int max_task_uniform_components;
  int max_task_atomic_counter_buffers;
  int max_task_atomic_counters;
  int max_task_components;
  int max_mesh_work_group_invocations;
  int max_task_work_group_invocations;
  int max_mesh_memory;
  int max_task_memory;
  int max_mesh_output_vertices;
  int max_mesh_output_primitives;
  int max_task_output;
  int max_mesh_draw_tasks;
  int max_mesh_views;
  FG_Vec3i max_mesh_work_group_size;
  FG_Vec3i max_task_work_group_size;
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

enum FG_Blend_Op
{
  FG_Blend_Op_Add          = 1,
  FG_Blend_Op_Subtract     = 2,
  FG_Blend_Op_Rev_Subtract = 3,
  FG_Blend_Op_Min          = 4,
  FG_Blend_Op_Max          = 5
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
  FG_Pipeline_Member_Stencil_OP         = (1 << 14),
  FG_Pipeline_Member_Stencil_Func       = (1 << 17),
  FG_Pipeline_Member_Depth_Func         = (1 << 18),
  FG_Pipeline_Member_Fill               = (1 << 19),
  FG_Pipeline_Member_Cull               = (1 << 20),
  FG_Pipeline_Member_Primitive          = (1 << 21),
  FG_Pipeline_Member_Depth_Slope_Bias   = (1 << 22),
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
  FG_Pipeline_Flag_Scissor_Enable           = (1 << 2),
  FG_Pipeline_Flag_Depth_Enable             = (1 << 3),
  FG_Pipeline_Flag_Stencil_Enable           = (1 << 4),
  FG_Pipeline_Flag_Depth_Write_Enable       = (1 << 5),
  FG_Pipeline_Flag_Conservative_Raster      = (1 << 6),
  FG_Pipeline_Flag_Front_Counter_Clockwise  = (1 << 7),
  FG_Pipeline_Flag_Depth_Clip_Enable        = (1 << 8),
  FG_Pipeline_Flag_Multisample_Enable       = (1 << 9),
  FG_Pipeline_Flag_Antialiased_Line_Enable  = (1 << 10),
  FG_Pipeline_Flag_Tool_Debug               = (1 << 11),
  FG_Pipeline_Flag_Independent_Blend_Enable = (1 << 12),
  FG_Pipeline_Flag_RenderTarget_SRGB_Enable = (1 << 13),
};

enum FG_Stencil_Op
{
  FG_Stencil_Op_None           = 0,
  FG_Stencil_Op_Keep           = 1,
  FG_Stencil_Op_Zero           = 2,
  FG_Stencil_Op_Replace        = 3,
  FG_Stencil_Op_Increment_Sat  = 4,
  FG_Stencil_Op_Decrement_Sat  = 5,
  FG_Stencil_Op_Invert         = 6,
  FG_Stencil_Op_Increment_Wrap = 7,
  FG_Stencil_Op_Decrement_Wrap = 8
};

typedef struct FG_PipelineState__
{
  uint64_t members; // Each bit represents a particular member that has been set
  FG_Shader shaders[FG_ShaderStage_Count];
  FG_Color16 blendFactor;
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
  FG_Color16 border_color;
} FG_Sampler;

enum FG_Logic_Op
{
  FG_Logic_Op_Clear = 0,
  FG_Logic_Op_Set,
  FG_Logic_Op_Copy,
  FG_Logic_Op_Copy_Inverted,
  FG_Logic_Op_NOOP,
  FG_Logic_Op_Invert,
  FG_Logic_Op_And,
  FG_Logic_Op_NAnd,
  FG_Logic_Op_Or,
  FG_Logic_Op_Nor,
  FG_Logic_Op_Xor,
  FG_Logic_Op_Equiv,
  FG_Logic_Op_And_Reverse,
  FG_Logic_Op_And_Inverted,
  FG_Logic_Op_Or_Reverse,
  FG_Logic_Op_Or_Inverted
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

typedef void FG_CommandList;

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

enum FG_AccessFlag
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
  FG_BarrierFlag_Vertex             = (1 << 0),
  FG_BarrierFlag_Element            = (1 << 1),
  FG_BarrierFlag_Uniform            = (1 << 2),
  FG_BarrierFlag_Texture_Fetch      = (1 << 3),
  FG_BarrierFlag_Texture_Update     = (1 << 4),
  FG_BarrierFlag_Image_Access       = (1 << 5),
  FG_BarrierFlag_Command            = (1 << 6),
  FG_BarrierFlag_Pixel              = (1 << 7),
  FG_BarrierFlag_Buffer             = (1 << 8),
  FG_BarrierFlag_RenderTarget       = (1 << 9),
  FG_BarrierFlag_Storage_Buffer     = (1 << 10),
  FG_BarrierFlag_Transform_Feedback = (1 << 11),
  FG_BarrierFlag_Atomic_Counter     = (1 << 12),
};

enum FG_ClearFlag
{
  FG_ClearFlag_Color       = (1 << 0),
  FG_ClearFlag_Depth       = (1 << 1),
  FG_ClearFlag_Stencil     = (1 << 2),
  FG_ClearFlag_Accumulator = (1 << 3),
};

struct FG_GraphicsInterface
{
  FG_Caps (*getCaps)(struct FG_GraphicsInterface* self);
  FG_Context* (*createContext)(struct FG_GraphicsInterface* self, FG_Vec2i size, enum FG_PixelFormat backbuffer);
  int (*resizeContext)(struct FG_GraphicsInterface* self, FG_Context* context, FG_Vec2i size);
  FG_Shader (*compileShader)(struct FG_GraphicsInterface* self, FG_Context* context, enum FG_ShaderStage stage,
                             const char* source);
  int (*destroyShader)(struct FG_GraphicsInterface* self, FG_Context* context, FG_Shader shader);
  FG_CommandList* (*createCommandList)(struct FG_GraphicsInterface* self, FG_Context* context, bool bundle);
  int (*destroyCommandList)(struct FG_GraphicsInterface* self, FG_CommandList* commands);
  int (*clear)(struct FG_GraphicsInterface* self, FG_CommandList* commands, uint8_t clearbits, FG_Color16 RGBA,
               uint8_t stencil, float depth, uint32_t num_rects, FG_Rect* rects);
  int (*copyResource)(struct FG_GraphicsInterface* self, FG_CommandList* commands, FG_Resource src, FG_Resource dest,
                      FG_Vec3i size, int mipmaplevel);
  int (*copySubresource)(struct FG_GraphicsInterface* self, FG_CommandList* commands, FG_Resource src, FG_Resource dest,
                         unsigned long srcoffset, unsigned long destoffset, unsigned long bytes);
  int (*copyResourceRegion)(struct FG_GraphicsInterface* self, FG_CommandList* commands, FG_Resource src, FG_Resource dest,
                            int level, FG_Vec3i srcoffset, FG_Vec3i destoffset, FG_Vec3i size);
  int (*draw)(struct FG_GraphicsInterface* self, FG_CommandList* commands, uint32_t vertexcount, uint32_t instancecount,
              uint32_t startvertex, uint32_t startinstance);
  int (*drawIndexed)(struct FG_GraphicsInterface* self, FG_CommandList* commands, uint32_t indexcount,
                     uint32_t instancecount, uint32_t startindex, int startvertex, uint32_t startinstance);
  int (*drawMesh)(struct FG_GraphicsInterface* self, FG_CommandList* commands, uint32_t first, uint32_t count);
  int (*dispatch)(struct FG_GraphicsInterface* self, FG_CommandList* commands);
  int (*syncPoint)(struct FG_GraphicsInterface* self, FG_CommandList* commands, uint32_t barrier_flags);
  int (*setPipelineState)(struct FG_GraphicsInterface* self, FG_CommandList* commands, uintptr_t state);
  int (*setViewports)(struct FG_GraphicsInterface* self, FG_CommandList* commands, FG_Viewport* viewports, uint32_t count);
  int (*setScissors)(struct FG_GraphicsInterface* self, FG_CommandList* commands, FG_Rect* rects, uint32_t count);
  int (*setShaderConstants)(struct FG_GraphicsInterface* self, FG_CommandList* commands, const FG_ShaderParameter* uniforms,
                            const FG_ShaderValue* values, uint32_t count);
  int (*execute)(struct FG_GraphicsInterface* self, FG_Context* context, FG_CommandList* commands);
  uintptr_t (*createPipelineState)(struct FG_GraphicsInterface* self, FG_Context* context, FG_PipelineState* pipelinestate,
                                   FG_Resource rendertarget, FG_Blend* blends, FG_Resource* vertexbuffer, int* strides,
                                   uint32_t n_buffers, FG_VertexParameter* attributes, uint32_t n_attributes,
                                   FG_Resource indexbuffer, uint8_t indexstride);
  uintptr_t (*createComputePipeline)(struct FG_GraphicsInterface* self, FG_Context* context, FG_Shader computeshader,
                                     FG_Vec3i workgroup, uint32_t flags);
  int (*destroyPipelineState)(struct FG_GraphicsInterface* self, FG_Context* context, uintptr_t state);
  FG_Resource (*createBuffer)(struct FG_GraphicsInterface* self, FG_Context* context, void* data, uint32_t bytes,
                              enum FG_Usage usage);
  FG_Resource (*createTexture)(struct FG_GraphicsInterface* self, FG_Context* context, FG_Vec2i size, enum FG_Usage usage,
                               enum FG_PixelFormat format, FG_Sampler* sampler, void* data, int MultiSampleCount);
  FG_Resource (*createRenderTarget)(struct FG_GraphicsInterface* self, FG_Context* context, FG_Resource depthstencil,
                                    FG_Resource* textures, uint32_t n_textures, int attachments);
  int (*destroyResource)(struct FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource);
  void* (*mapResource)(struct FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource, uint32_t offset,
                       uint32_t length, enum FG_Usage usage, uint32_t access);
  int (*unmapResource)(struct FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource, enum FG_Usage usage);
  int (*destroy)(struct FG_GraphicsInterface* self);
};

typedef struct FG_GraphicsInterface* (*FG_InitGraphics)(void*, FG_Log);

#ifdef __cplusplus
}
#endif

#endif
