// TODO: AUTOGENERATE THIS FILE
#ifndef BACKEND_H
#define BACKEND_H

#include <stdint.h> // for integers
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
  FG_Feature_RENDER_TARGET     = (1 << 5),
  FG_Feature_BLEND_EX           = (1 << 6),
  FG_Feature_BACKGROUND_OPACITY = (1 << 7),
  FG_Feature_LINES_ALPHA        = (1 << 8),
  FG_Feature_BLEND_GAMMA        = (1 << 9),
  FG_Feature_INSTANCING         = (1 << 10),
  FG_Feature_INDEPENDENT_BLEND  = (1 << 11),
  FG_Feature_COMPUTE_SHADER     = (1 << 12),
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
  int version;     // 2.1 is 21, 4.2 is 42, etc.
  int glsl; // Version 1.10 is 110, version 4.40 is 440
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

enum FG_COMMAND_TYPE
{
  FG_COMMAND_UNKNOWN = 0,
  FG_COMMAND_CLEAR_DEPTH_STENCIL,
  FG_COMMAND_CLEAR_RENDER_TARGET,
  FG_COMMAND_COPY_RESOURCE,
  FG_COMMAND_COPY_SUBRESOURCE,
  FG_COMMAND_COPY_RESOURCE_REGION,
  FG_COMMAND_DRAW,
  FG_COMMAND_DRAW_INDEXED,
  FG_COMMAND_PIPELINE_STATE,
  FG_COMMAND_DEPTH_STENCIL,
  FG_COMMAND_VERTEX_BUFFER_ARRAY,
  FG_COMMAND_VERTEX_STREAM,
  FG_COMMAND_VIEWPORT_ARRAY,
};
// Used at the start of every command bytestream
typedef struct FG_Command_Header__
{
  uint16_t type;
  uint16_t size; // Size of a single element
} FG_Command_Header;

// Header that includes a count for array command types
typedef struct FG_Command_Header_Array__
{
  FG_Command_Header header;
  uint32_t count;
} FG_Command_Header_Array;

typedef void FG_Resource;

typedef struct FG_Command_ClearDepthStencil__
{
  FG_Resource* depthstencil;
  char clear; // If 0, clears both. If -1, clears stencil, if 1, clears depth.
  uint8_t stencil;
  float depth;
  uint32_t num_rects;
} FG_Command_ClearDepthStencil;


typedef struct FG_Command_ClearRenderTarget__
{
  FG_Resource* rendertarget;
  FG_Color RGBA;
  uint32_t num_rects;
} FG_Command_ClearRenderTarget;

typedef struct FG_Command_CopyResource__
{
  FG_Resource* src;
  FG_Resource* dest;
} FG_Command_CopyResource;

typedef struct FG_Command_CopySubresource__
{
  FG_Resource* src;
  FG_Resource* dest;
  unsigned long destoffset;
  unsigned long srcoffset;
  unsigned long bytes;
} FG_Command_CopySubresource;

typedef struct FG_Command_CopyResourceRegion__
{
  FG_Resource* src;
  FG_Resource* dest;
  FG_Vec3i destoffset;
  FG_Vec3i srcoffset;
  FG_Vec3i size;
} FG_Command_CopyResourceRegion;

typedef struct FG_Command_Draw__
{
  uint32_t vertexcount;
  uint32_t instancecount; // Can be 0, in which case instancing is not used
  uint32_t startvertex;
  uint32_t startinstance;
} FG_Command_Draw;

typedef struct FG_Command_DrawIndexed__
{
  uint32_t indexcount;
  uint32_t instancecount; // Can be 0, in which case instancing is not used
  uint32_t startindex;
  int startvertex;
  uint32_t startinstance;
} FG_Command_DrawIndexed;

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
  FG_CULL_MODE_NONE = 0,
  FG_CULL_MODE_FRONT = 1,
  FG_CULL_MODE_BACK = 2,
};

enum FG_PIPELINE_FLAGS
{
  FG_PIPELINE_FLAG_ALPHA_TO_COVERAGE_ENABLE = (1 << 0),
  FG_PIPELINE_FLAG_INDEPENDENT_BLEND_ENABLE = (1 << 1),
  FG_PIPELINE_FLAG_DEPTH_ENABLE             = (1 << 2),
  FG_PIPELINE_FLAG_STENCIL_ENABLE           = (1 << 3),
  FG_PIPELINE_FLAG_DEPTH_WRITE_ENABLE       = (1 << 4),
  FG_PIPELINE_FLAG_CONSERVATIVE_RASTER      = (1 << 5),
  FG_PIPELINE_FLAG_FRONT_COUNTER_CLOCKWISE  = (1 << 6),
  FG_PIPELINE_FLAG_DEPTH_CLIP_ENABLE        = (1 << 7),
  FG_PIPELINE_FLAG_MULTISAMPLE_ENABLE       = (1 << 8),
  FG_PIPELINE_FLAG_ANTIALIASED_LINE_ENABLE  = (1 << 9),
  FG_PIPELINE_FLAG_TOOL_DEBUG               = (1 << 10),
};

typedef struct FG_PipelineState__
{
  uint64_t Members; // Each bit represents a particular member that has been set
  uint8_t Primitive;
  uint32_t StencilRef;
  FG_Resource* IndexBuffer;
  FG_Resource* DepthStencil;
  void* Shaders[FG_ShaderStage_COUNT];
  FG_Color BlendFactor;
  uint16_t Flags;
  uint32_t SampleMask;
  uint8_t StencilReadMask;
  uint8_t StencilWriteMask;
  uint8_t DepthFunc;
  uint8_t StripCutValue;
  uint8_t RenderTargetsCount;
  uint8_t RTFormats[8];
  uint8_t FillMode;
  uint8_t CullMode;
  int DepthBias;
  float DepthBiasClamp;
  float SlopeScaledDepthBias;
  uint32_t ForcedSampleCount;
  uint32_t NodeMask;
  uint32_t MultisampleCount;
  uint32_t MultisampleQuality;
} FG_PipelineState;

typedef struct FG_Command_PipelineState__
{
  void* state;
} FG_Command_PipelineState;

enum FG_STENCIL_OP
{
  FG_STENCIL_OP_KEEP     = 1,
  FG_STENCIL_OP_ZERO     = 2,
  FG_STENCIL_OP_REPLACE  = 3,
  FG_STENCIL_OP_INCR_SAT = 4,
  FG_STENCIL_OP_DECR_SAT = 5,
  FG_STENCIL_OP_INVERT   = 6,
  FG_STENCIL_OP_INCR     = 7,
  FG_STENCIL_OP_DECR     = 8
};

typedef struct FG_Command_DepthStencil
{
  bool Front; // front if true, back if false
  uint8_t StencilFailOp;
  uint8_t StencilDepthFailOp;
  uint8_t StencilPassOp;
  uint8_t StencilFunc;
} FG_Command_DepthStencil;

// Array
typedef struct FG_Command_VertexBuffer__
{
  FG_Resource* vertexbuffer; 
} FG_Command_VertexBuffer;

typedef struct FG_Command_VertexStream__
{
  uint32_t Stream;
  const char* SemanticName;
  uint32_t SemanticIndex;
  uint8_t StartComponent;
  uint8_t ComponentCount;
  uint8_t OutputSlot;
  uint32_t Stride;
  uint32_t RasterizedStream;
} FG_Command_VertexStream;

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

typedef enum FG_TEXTURE_ADDRESS_MODE
{
  FG_TEXTURE_ADDRESS_MODE_WRAP        = 1,
  FG_TEXTURE_ADDRESS_MODE_MIRROR      = 2,
  FG_TEXTURE_ADDRESS_MODE_CLAMP       = 3,
  FG_TEXTURE_ADDRESS_MODE_BORDER      = 4,
  FG_TEXTURE_ADDRESS_MODE_MIRROR_ONCE = 5
};

typedef struct FG_Sampler__
{
  uint8_t Filter; // FG_FILTER
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
typedef struct FG_Command_Blend__
{
  uint8_t SrcBlend;
  uint8_t DestBlend;
  uint8_t BlendOp;
  uint8_t SrcBlendAlpha;
  uint8_t DestBlendAlpha;
  uint8_t BlendOpAlpha;
  uint8_t LogicOp;
  uint8_t RenderTargetWriteMask;
  bool BlendEnable;
  bool LogicOpEnable;
} FG_Command_Blend;

// Array
typedef struct FG_Command_Viewport__
{
  FG_Vec3 pos;
  FG_Vec3 dim;
  FG_Rect scissor;
} FG_Command_Viewport;

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
  uint8_t semantic;
  uint8_t slot;
  uint8_t type; // enum FG_ShaderType
  bool per_instance; // false if per_vertex
  uint32_t offset;
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

typedef struct FG_Msg__
{
  uint16_t kind;
  union
  {
    struct FG_TouchMove
    {
      float x;
      float y;
      float z;
      float r;
      float pressure;
      uint16_t index;
      uint8_t flags;
      uint8_t modkeys;
    } touchMove;
    struct FG_MouseScroll
    {
      float x;
      float y;
      float delta;
      float hdelta;
    } mouseScroll;
    struct FG_MouseOn
    {
      float x;
      float y;
      uint8_t all;
      uint8_t modkeys;
    } mouseOn;
    struct FG_JoyButtonUp
    {
      uint16_t index;
      uint16_t button;
      uint8_t modkeys;
    } joyButtonUp;
    struct FG_MouseOff
    {
      float x;
      float y;
      uint8_t all;
      uint8_t modkeys;
    } mouseOff;
    struct FG_JoyButtonDown
    {
      uint16_t index;
      uint16_t button;
      uint8_t modkeys;
    } joyButtonDown;
    struct FG_GetWindowFlags
    {
      char __padding;
    } getWindowFlags;
    struct FG_KeyDown
    {
      uint8_t key;
      uint8_t modkeys;
      uint16_t scancode;
    } keyDown;
    struct FG_Action
    {
      int32_t subkind;
    } action;
    struct FG_KeyChar
    {
      int32_t unicode;
      uint8_t modkeys;
    } keyChar;
    struct FG_KeyUp
    {
      uint8_t key;
      uint8_t modkeys;
      uint16_t scancode;
    } keyUp;
    struct FG_TouchEnd
    {
      float x;
      float y;
      float z;
      float r;
      float pressure;
      uint16_t index;
      uint8_t flags;
      uint8_t modkeys;
    } touchEnd;
    struct FG_TouchBegin
    {
      float x;
      float y;
      float z;
      float r;
      float pressure;
      uint16_t index;
      uint8_t flags;
      uint8_t modkeys;
    } touchBegin;
    struct FG_SetWindowFlags
    {
      uint32_t flags;
    } setWindowFlags;
    struct FG_MouseUp
    {
      float x;
      float y;
      uint8_t all;
      uint8_t modkeys;
      uint8_t button;
    } mouseUp;
    struct FG_MouseDblClick
    {
      float x;
      float y;
      uint8_t all;
      uint8_t modkeys;
      uint8_t button;
    } mouseDblClick;
    struct FG_JoyAxis
    {
      uint16_t index;
      float value;
      uint16_t axis;
      uint8_t modkeys;
    } joyAxis;
    struct FG_LostFocus
    {
      char __padding;
    } lostFocus;
    struct FG_MouseDown
    {
      float x;
      float y;
      uint8_t all;
      uint8_t modkeys;
      uint8_t button;
    } mouseDown;
    struct FG_JoyOrientation
    {
      uint16_t index;
      FG_Vec3 velocity;
      FG_Vec3 rotation;
    } joyOrientation;
    struct FG_Draw
    {
      FG_Rect area;
    } draw;
    struct FG_Drop
    {
      int32_t kind;
      void* target;
      uint32_t count;
    } drop;
    struct FG_SetWindowRect
    {
      FG_Rect rect;
    } setWindowRect;
    struct FG_GotFocus
    {
      char __padding;
    } gotFocus;
    struct FG_MouseMove
    {
      float x;
      float y;
      uint8_t all;
      uint8_t modkeys;
    } mouseMove;
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

typedef void FG_Context;
typedef void FG_Element;
typedef void (*FG_Log)(void*, enum FG_Level, const char*, ...);
typedef FG_Result (*FG_Behavior)(FG_Element*, FG_Context*, void*, FG_Msg*);

typedef struct FG_Window__
{
  void* handle;
  FG_Context* context;
} FG_Window;

struct FG_Backend
{
  FG_Caps (*getCaps)(FG_Backend* self);
  void* (*compileShader)(FG_Backend* self, enum FG_ShaderStage stage, const char* source);
  int (*destroyShader)(FG_Backend* self, void* shader);
  int (*execute)(FG_Backend* self, FG_Context* context, FG_Command_Header* bytestream);
  void* (*createPipelineState)(FG_Backend* self, FG_Context* context, FG_PipelineState* pipelinestate,
                               FG_ShaderParameter* params, uint32_t n_params, FG_Resource* streamoutput, uint32_t n_outputs,
                               FG_Resource* rendertargets, uint32_t n_targets, FG_Command_Blend* blends);
  int (*destroyPipelineState)(FG_Backend* self, FG_Context* context, void* state);
  FG_Resource* (*createBuffer)(FG_Backend* self, FG_Context* context, void* data, uint32_t bytes,
                               enum FG_PixelFormat format);
  FG_Resource* (*createTexture)(FG_Backend* self, FG_Context* context, FG_Vec2i size, enum FG_PixelFormat format,
                                FG_Sampler* sampler);
  void (*destroyResource)(FG_Backend* self, FG_Resource* resource);
  FG_Window* (*createWindow)(FG_Backend* self, FG_Element* element, FG_Display* display, FG_Vec2* pos, FG_Vec2* dim,
                              const char* caption, uint64_t flags);
  int (*setWindow)(FG_Backend* self, FG_Window* window, FG_Element* element, FG_Display* display, FG_Vec2* pos,
                   FG_Vec2* dim, const char* caption, uint64_t flags);
  int (*destroyWindow)(FG_Backend* self, FG_Window* window);
  int (*beginDraw)(FG_Backend* self, FG_Context* context, FG_Rect* area);
  int (*endDraw)(FG_Backend* self, FG_Context* context);
  int (*putClipboard)(FG_Backend* self, FG_Context* context, enum FG_Clipboard kind, const char* data, uint32_t count);
  uint32_t (*getClipboard)(FG_Backend* self, FG_Context* context, enum FG_Clipboard kind, void* target, uint32_t count);
  bool (*checkClipboard)(FG_Backend* self, FG_Context* context, enum FG_Clipboard kind);
  int (*clearClipboard)(FG_Backend* self, FG_Context* context, enum FG_Clipboard kind);
  int (*processMessages)(FG_Backend* self, FG_Context* context);
  int (*getMessageSyncObject)(FG_Backend* self, FG_Context* context);
  int (*setCursorGL)(FG_Backend* self, FG_Context* context, enum FG_Cursor cursor);
  int (*getDisplayIndex)(FG_Backend* self, unsigned int index, FG_Display* out);
  int (*getDisplay)(FG_Backend* self, FG_Context* handle, FG_Display* out);
  int (*getDisplayWindow)(FG_Backend* self, FG_Context* context, FG_Display* out);
  int (*createSystemControl)(FG_Backend* self, FG_Context* context, const char* id, FG_Rect* area, ...);
  int (*setSystemControl)(FG_Backend* self, FG_Context* context, void* control, FG_Rect* area, ...);
  int (*destroySystemControl)(FG_Backend* self, FG_Context* context, void* control);
};

typedef FG_Backend* (*FG_InitBackend)(void*, FG_Log, FG_Behavior);

#endif