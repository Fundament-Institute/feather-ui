/* backendtest - Manual test of feather GUI backends
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

#include "feather/graphics_desktop_bridge.h"
#include "resource.h"
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include <assert.h>
#include <string.h>
#include <malloc.h>

#include "linmath.h"

#define BACKEND fgOpenGL
#define BRIDGE  fgOpenGLDesktopBridge
#define TEST(x)                      \
  if(!(x))                           \
  {                                  \
    printf("Failed test: %s\n", #x); \
  }

extern struct FG_GraphicsInterface* BACKEND(void* root, FG_Log log);
extern struct FG_DesktopInterface* fgGLFW(void* root, FG_Log log, FG_Behavior behavior);
extern struct FG_GraphicsDesktopBridge* BRIDGE(struct FG_GraphicsInterface*, void*, FG_Log);

const char* LEVELS[] = { "FATAL: ", "ERROR: ", "WARNING: ", "NOTICE: ", "DEBUG: " };

void FakeLogArg(const char* str, FG_LogValue v)
{
  switch(v.type)
  {
  case FG_LogType_Boolean: printf(str, v.bit); break;
  case FG_LogType_I32: printf(str, v.i32); break;
  case FG_LogType_U32: printf(str, v.u32); break;
  case FG_LogType_I64: printf(str, v.i64); break;
  case FG_LogType_U64: printf(str, v.u64); break;
  case FG_LogType_F32: printf(str, v.f32); break;
  case FG_LogType_F64: printf(str, v.f64); break;
  case FG_LogType_String: printf(str, v.string); break;
  case FG_LogType_OwnedString: printf(str, v.owned); break;
  }
}

// A logging function that just forwards everything to printf
void FakeLog(void* _, enum FG_Level level, const char* file, int line, const char* msg, const FG_LogValue* values,
             int n_values, void (*free)(char*))
{
  if(level >= 0 && level < sizeof(LEVELS))
    printf("%s [%s:%i] ", LEVELS[level], file, line);

  // Crazy hack to feed values into printf
  if(strchr(msg, '%'))
  {
    int i     = -1;
    char* str = (char*)alloca(strlen(msg) + 1);
    strcpy(str, msg);
    char* cur = strchr(str, '%');

    while(cur)
    {
      cur[0] = 0;
      if(i >= 0)
        FakeLogArg(str, values[i]);
      else
        printf("%s", str);

      cur[0] = '%';
      str    = cur;
      cur    = strchr(str + 1, '%');
      ++i;
    }
    FakeLogArg(str, values[i]);
  }
  else
  {
    printf("%s", msg);
    for(int i = 0; i < n_values; ++i)
    {
      switch(values[i].type)
      {
      case FG_LogType_Boolean: printf(" %s", values[i].bit ? "true" : "false"); break;
      case FG_LogType_I32: printf(" %i", values[i].i32); break;
      case FG_LogType_U32: printf(" %u", values[i].u32); break;
      case FG_LogType_I64: printf(" %lli", values[i].i64); break;
      case FG_LogType_U64: printf(" %llu", values[i].u64); break;
      case FG_LogType_F32: printf(" %g", values[i].f32); break;
      case FG_LogType_F64: printf(" %g", values[i].f64); break;
      case FG_LogType_String: printf(" %s", values[i].string); break;
      case FG_LogType_OwnedString: printf(" %s", values[i].owned); break;
      }
    }
    printf("\n");
  }

  for(int i = 0; i < n_values; ++i)
  {
    if(values[i].type == FG_LogType_OwnedString)
      free(values[i].owned);
  }
}

// This assembles a custom projection matrix specifically designed for 2D drawing.
void mat4x4_proj(mat4x4 M, float l, float r, float b, float t, float n, float f)
{
  memset(M, 0, sizeof(mat4x4));

  M[0][0] = 2.0f / (r - l);
  M[0][1] = 0.f;
  M[0][2] = 0.f;
  M[0][3] = 0.f;

  M[1][0] = 0.f;
  M[1][1] = 2.0f / (t - b);
  M[1][2] = 0.f;
  M[1][3] = 0.f;

  M[2][0] = 0.f;
  M[2][1] = 0.f;
  M[2][2] = -((f + n) / (f - n));
  M[2][3] = -1.0f;

  M[3][0] = -(r + l) / (r - l);
  M[3][1] = -(t + b) / (t - b);
  M[3][2] = -((2.f * f * n) / (f - n));
  M[3][3] = 0.f;

  mat4x4_translate_in_place(M, 0, 0, -1.0f);
}

// This "mock element" serves as a fake UI object that contains our state and will do our drawing.
typedef struct MockElement__
{
  FG_Resource image;
  FG_Shader shader;
  FG_Resource vertices;
  uintptr_t pipeline;
  uintptr_t mesh_pipeline;
  uint64_t flags;
  bool close;
  FG_Caps caps;
} MockElement;

static FG_Vec2i WindowDim = { 800.f, 600.f };

uintptr_t load_mesh(struct FG_GraphicsInterface* b, FG_Window* w)
{
  const char* shader_ms = "#version 450\n"
                          "#extension GL_NV_mesh_shader : require\n"
                          "layout(local_size_x=1) in;\n"
                          "layout(max_vertices=3, max_primitives=1) out;\n"
                          "layout(triangles) out;\n"

                          "out PerVertexData\n"
                          "{\n"
                          "  vec4 color;\n"
                          "} v_out[];   \n"
                          "const vec3 vertices[3] = {vec3(-1,-1,0), vec3(0,1,0), vec3(1,-1,0)};\n"
                          "const vec3 colors[3] = {vec3(1.0,0.0,0.0), vec3(0.0,1.0,0.0), vec3(0.0,0.0,1.0)};\n"
                          "uniform float scale;\n"

                          "void main()\n"
                          "{\n"
                          "  gl_MeshVerticesNV[0].gl_Position = vec4(vertices[0] * scale, 1.0); \n"
                          "  gl_MeshVerticesNV[1].gl_Position = vec4(vertices[1] * scale, 1.0); \n"
                          "  gl_MeshVerticesNV[2].gl_Position = vec4(vertices[2] * scale, 1.0); \n"

                          "  v_out[0].color = vec4(colors[0], 1.0);\n"
                          "  v_out[1].color = vec4(colors[1], 1.0);\n"
                          "  v_out[2].color = vec4(colors[2], 1.0);\n"

                          "  gl_PrimitiveIndicesNV[0] = 0;\n"
                          "  gl_PrimitiveIndicesNV[1] = 1;\n"
                          "  gl_PrimitiveIndicesNV[2] = 2;\n"

                          "  gl_PrimitiveCountNV = 1;\n"
                          "}";
  const char* shader_fs = "#version 450\n"

                          "layout(location = 0) out vec4 FragColor;\n"

                          "in PerVertexData\n"
                          "{\n"
                          "  vec4 color;\n"
                          "} fragIn;  \n"

                          "void main()\n"
                          "{\n"
                          "  FragColor = fragIn.color;\n"
                          "}";

  FG_PipelineState pipeline = { 0 };
  pipeline.members = FG_Pipeline_Member_PS | FG_Pipeline_Member_MS | FG_Pipeline_Member_Flags | FG_Pipeline_Member_Fill |
                     FG_Pipeline_Member_Cull | FG_Pipeline_Member_Primitive;
  pipeline.shaders[FG_ShaderStage_Pixel] = (*b->compileShader)(b, w->context, FG_ShaderStage_Pixel, shader_fs);
  pipeline.shaders[FG_ShaderStage_Mesh]  = (*b->compileShader)(b, w->context, FG_ShaderStage_Mesh, shader_ms);
  pipeline.flags                         = FG_Pipeline_Flag_Blend_Enable |
                   FG_Pipeline_Flag_Scissor_Enable; // | FG_PIPELINE_FLAG_RENDERTARGET_SRGB_ENABLE
  pipeline.fillMode = FG_Fill_Mode_Fill;
  pipeline.cullMode = FG_Cull_Mode_None; // FG_CULL_MODE_BACK

  FG_Blend Premultiply_Blend = {
    FG_Blend_Operand_One,
    FG_Blend_Operand_Inv_Src_Alpha,
    FG_Blend_Op_Add,
    FG_Blend_Operand_One,
    FG_Blend_Operand_Inv_Src_Alpha,
    FG_Blend_Op_Add,
    0b1111,
  };

  return (*b->createPipelineState)(b, w->context, &pipeline, 0, &Premultiply_Blend, 0, 0, 0, 0, 0, 0, 0);
}

void test_mesh(struct FG_GraphicsInterface* b, FG_Context* ctx, void* commands, const MockElement* e)
{
  static const FG_ShaderParameter params[] = { { "scale", 1, 0, 0, FG_Shader_Type_Float } };
  FG_ShaderValue values[1];
  values[0].f32 = 1;

  (*b->setPipelineState)(b, commands, e->mesh_pipeline);
  (*b->setShaderConstants)(b, commands, params, values, 1);
  (*b->drawMesh)(b, commands, 0, 1);
  (*b->execute)(b, ctx, commands);
}

struct UIState
{
  struct FG_GraphicsInterface* graphics;
  struct FG_GraphicsDesktopBridge* bridge;
};

// The behavior function simply processes all window messages from the host OS.
FG_Result behavior(FG_Window* w, FG_Msg* msg, void* ui_context, uintptr_t window_id)
{
  static const FG_ShaderParameter params[] = { { "MVP", 4, 4, 0, FG_Shader_Type_Float },
                                               { 0, 0, 0, 0, FG_Shader_Type_Texture } };
  static int counter                       = 0;

  MockElement* e = (MockElement*)window_id;

  if(msg->kind == FG_Event_Kind_Draw && e->pipeline != 0)
  {
    struct UIState* state          = (struct UIState*)ui_context;
    struct FG_GraphicsInterface* b = state->graphics;
    void* commands                 = (*b->createCommandList)(b, w->context, false);
    assert(commands);

    (*state->bridge->beginDraw)(state->bridge, w, NULL);
    if(e->mesh_pipeline && (e->caps.features & FG_Feature_Mesh_Shader) != 0)
      test_mesh(b, w->context, commands, e);

    // Try drawing a custom shader.
    mat4x4 proj;
    mat4x4_proj(proj, 0, (float)WindowDim.x, (float)WindowDim.y, 0, 0.2f, 1000.0f);
    FG_ShaderValue values[2];
    values[0].pf32     = (float*)proj;
    values[1].resource = e->image;

    (*b->setPipelineState)(b, commands, e->pipeline);
    (*b->setShaderConstants)(b, commands, params, values, 2);
    (*b->draw)(b, commands, 4, 0, 0, 0);
    (*b->execute)(b, w->context, commands);
    (*b->destroyCommandList)(b, w->context, commands);
    (*state->bridge->endDraw)(state->bridge, w);

    FG_Result r = { 0 };
    return r;
  }

  // These are simply some input handling boilerplate code
  if(msg->kind == FG_Event_Kind_GetWindowFlags)
  {
    FG_Result r = { e->close ? FG_WindowFlag_Closed : 0 };
    return r;
  }
  if(msg->kind == FG_Event_Kind_SetWindowFlags)
  {
    e->close = e->close || ((msg->setWindowFlags.flags & FG_WindowFlag_Closed) != 0);
  }
  // We normally close when any keypress happens, but we don't want to close if the user is trying to take a screenshot.
  if((msg->kind == FG_Event_Kind_KeyDown && msg->keyDown.key != FG_Keys_LMENU && msg->keyDown.scancode != 84 &&
      msg->keyDown.scancode != 88) ||
     msg->kind == FG_Event_Kind_MouseDown)
  {
    e->close = true;
  }

  FG_Result r = { -1 };
  return r;
}

void test_compute(struct FG_GraphicsInterface* b, FG_Window* w)
{
  const char* shader_cs = "#version 430\n"
                          "layout(local_size_x=1) in;\n"

                          "layout(location = 0) uniform int dt;\n"
                          "layout(std430, binding=0) buffer inblock { int num[]; };\n"
                          "layout(std430, binding=1) buffer outblock { int process[]; };\n"

                          "void main() {\n"
                          "   int index = int(gl_LocalInvocationIndex + gl_WorkGroupID.x * gl_WorkGroupSize.x);\n"
                          "   process[index] = num[index] * dt;\n"
                          "}\n";

  static const int GROUPSIZE = 1024;
  int* initvals              = (int*)malloc(sizeof(int) * GROUPSIZE);
  for(int i = 0; i < GROUPSIZE; ++i)
  {
    initvals[i] = i;
  }
  int* outvals = (int*)malloc(sizeof(int) * GROUPSIZE);
  memset(outvals, 0, sizeof(int) * GROUPSIZE);
  FG_Resource inbuf  = (*b->createBuffer)(b, w->context, initvals, sizeof(int) * GROUPSIZE, FG_Usage_Storage_Buffer);
  FG_Resource outbuf = (*b->createBuffer)(b, w->context, outvals, sizeof(int) * GROUPSIZE, FG_Usage_Storage_Buffer);
  FG_Vec3i groupsize = { GROUPSIZE, 1, 1 };

  uintptr_t compute_pipeline =
    (*b->createComputePipeline)(b, w->context, (*b->compileShader)(b, w->context, FG_ShaderStage_Compute, shader_cs),
                                groupsize, 0);

  void* commands = (*b->createCommandList)(b, w->context, false);
  assert(commands);

  FG_Resource CopiedInbuf  = (*b->createBuffer)(b, w->context, 0, sizeof(int) * GROUPSIZE, FG_Usage_Storage_Buffer);
  FG_Resource CopiedOutbuf = (*b->createBuffer)(b, w->context, 0, sizeof(int) * GROUPSIZE, FG_Usage_Storage_Buffer);
  TEST((*b->copySubresource)(b, commands, inbuf, CopiedInbuf, 0, 0, sizeof(int) * GROUPSIZE) == 0);
  TEST((*b->copySubresource)(b, commands, outbuf, CopiedOutbuf, 0, 0, sizeof(int) * GROUPSIZE) == 0);

  FG_ShaderValue values[3];
  values[0].i32      = 3;
  values[1].resource = CopiedInbuf;
  values[2].resource = CopiedOutbuf;

  static const FG_ShaderParameter params[] = { { "dt", 1, 0, 0, FG_Shader_Type_Int },
                                               { "inblock", 0, 0, 0, FG_Shader_Type_Buffer },
                                               { "outblock", 0, 0, 1, FG_Shader_Type_Buffer } };
  TEST((*b->setPipelineState)(b, commands, compute_pipeline) == 0);
  TEST((*b->setShaderConstants)(b, commands, params, values, 3) == 0);
  TEST((*b->dispatch)(b, commands) == 0);
  TEST((*b->syncPoint)(b, commands, FG_BarrierFlag_Storage_Buffer) == 0);
  TEST((*b->execute)(b, w->context, commands) == 0);

  int* readbuf = (int*)(*b->mapResource)(b, w->context, CopiedOutbuf, 0, 0, FG_Usage_Storage_Buffer, FG_AccessFlag_Read);
  TEST(readbuf != NULL);

  for(int i = 0; i < GROUPSIZE; ++i)
  {
    TEST(readbuf[i] == initvals[i] * values[0].i32);
  }

  (*b->destroyCommandList)(b, w->context, commands);
  TEST((*b->unmapResource)(b, w->context, CopiedOutbuf, FG_Usage_Storage_Buffer) == 0);
  TEST((*b->destroyResource)(b, w->context, outbuf) == 0);
  TEST((*b->destroyResource)(b, w->context, inbuf) == 0);
  TEST((*b->destroyPipelineState)(b, w->context, compute_pipeline) == 0);
}

int main(int argc, char* argv[])
{
  struct FG_GraphicsInterface* b          = BACKEND(NULL, FakeLog);
  struct FG_DesktopInterface* desktop     = fgGLFW(NULL, FakeLog, behavior);
  struct FG_GraphicsDesktopBridge* bridge = BRIDGE(b, NULL, FakeLog);
  
  struct UIState state = { b, bridge };

  if(!b)
  {
    printf("Failed to load backend!\n");
    return -1;
  }

  MockElement e = { 0 };

  const char* shader_vs = "#version 110\n"
                          "uniform mat4 MVP;\n"
                          "attribute vec2 vPos;\n"
                          "attribute vec2 vUV;\n"
                          "varying vec2 uv;\n"
                          "void main() { gl_Position = MVP * vec4(vPos.xy, -0.75, 1.0); uv = vUV.xy; }";

  const char* shader_fs = "#version 110\n"
                          "varying vec2 uv;\n"
                          "uniform sampler2D texture;\n"
                          "void main() { gl_FragColor = texture2D(texture, uv).rgba; }";

  float verts[4][4] = {
    {
      450.f,
      10.f,
      0.0f,
      0.0f,
    },
    {
      1050.f,
      10.f,
      1.0f,
      0.0f,
    },
    {
      450.f,
      410.f,
      0.0f,
      1.0f,
    },
    {
      1050.f,
      410.f,
      1.0f,
      1.0f,
    },
  };

  FG_Blend Premultiply_Blend = {
    FG_Blend_Operand_One,
    FG_Blend_Operand_Inv_Src_Alpha,
    FG_Blend_Op_Add,
    FG_Blend_Operand_One,
    FG_Blend_Operand_Inv_Src_Alpha,
    FG_Blend_Op_Add,
    0b1111,
  };

  FG_VertexParameter vertparams[] = { { "vPos", 0, 0, 2, FG_Shader_Type_Float },
                                      { "vUV", sizeof(float) * 2, 0, 2, FG_Shader_Type_Float } };

  e.flags = FG_WindowFlag_Resizable;

  FG_Vec2 pos  = { 200.f, 100.f };
  FG_Vec2 dim  = { 800.f, 600.f };
  FG_Window* w = (*desktop->createWindow)(desktop, (uintptr_t)(&e), NULL, &pos, &dim, "Feather Test", e.flags);
  TEST(w != NULL);
  TEST(!(*bridge->emplaceContext)(bridge, w, FG_PixelFormat_R8G8B8A8_Typeless));

  if(!w)
  {
    printf("failed to create window!\n");
    return -1;
  }

  e.caps = (*b->getCaps)(b);

  if(e.caps.features & FG_Feature_Compute_Shader)
    test_compute(b, w);

  FG_Vec2i fakedim = { 200, 200 };
  void* fakedata   = malloc(fakedim.x * (size_t)fakedim.y * 4);
  memset(fakedata, 128, fakedim.x * (size_t)fakedim.y * 4);

  FG_Sampler sampler = { FG_Filter_Min_Mag_Mip_Linear };
  e.image    = (*b->createTexture)(b, w->context, fakedim, FG_Usage_Texture2D, FG_PixelFormat_R8G8B8A8_Typeless, &sampler,
                                fakedata, 0);
  e.vertices = (*b->createBuffer)(b, w->context, verts, sizeof(verts), FG_Usage_Vertex_Data);
  int vertstride = sizeof(verts[0]);

  FG_PipelineState pipeline = { 0 };
  pipeline.members = FG_Pipeline_Member_PS | FG_Pipeline_Member_VS | FG_Pipeline_Member_Flags | FG_Pipeline_Member_Fill |
                     FG_Pipeline_Member_Cull | FG_Pipeline_Member_Primitive;
  pipeline.shaders[FG_ShaderStage_Pixel]  = (*b->compileShader)(b, w->context, FG_ShaderStage_Pixel, shader_fs);
  pipeline.shaders[FG_ShaderStage_Vertex] = (*b->compileShader)(b, w->context, FG_ShaderStage_Vertex, shader_vs);
  pipeline.fillMode                       = FG_Fill_Mode_Fill;
  pipeline.cullMode                       = FG_Cull_Mode_None; // FG_CULL_MODE_BACK
  pipeline.primitive                      = FG_Primitive_Triangle_Strip;
  pipeline.flags                          = FG_Pipeline_Flag_Blend_Enable |
                   FG_Pipeline_Flag_Scissor_Enable; // | FG_PIPELINE_FLAG_RENDERTARGET_SRGB_ENABLE

  FG_Resource RenderTarget0 =
    (*b->createTexture)(b, w->context, WindowDim, FG_Usage_Texture2D, FG_PixelFormat_R8G8B8A8_Typeless, &sampler, NULL, 0);
  FG_Resource rts[1]      = { RenderTarget0 };
  FG_Resource framebuffer = (*b->createRenderTarget)(b, w->context, 0, rts, 1, 0);
  e.pipeline = (*b->createPipelineState)(b, w->context, &pipeline, framebuffer, &Premultiply_Blend, &e.vertices,
                                         &vertstride, 1, vertparams, 2, 0, 0);
  e.close    = false;

  FG_Msg drawmsg = { FG_Event_Kind_Draw };
  behavior(w, &drawmsg, &state, (uintptr_t)(&e));

  FG_Resource CopiedTexture =
    (*b->createTexture)(b, w->context, WindowDim, FG_Usage_Texture2D, FG_PixelFormat_R8G8B8A8_Typeless, &sampler, NULL, 0);
  void* commands = (*b->createCommandList)(b, w->context, false);

  FG_Vec3i vec3zero   = { 0 };
  FG_Vec3i vec3window = { WindowDim.x, WindowDim.y, 0 };
  TEST((*b->copyResourceRegion)(b, commands, RenderTarget0, CopiedTexture, 0, vec3zero, vec3zero, vec3window) == 0);
  (*b->execute)(b, w->context, commands);
  (*b->destroyCommandList)(b, w->context, commands);

  if(e.caps.features & FG_Feature_Mesh_Shader)
    e.mesh_pipeline = load_mesh(b, w);

  e.image    = CopiedTexture;
  e.pipeline = (*b->createPipelineState)(b, w->context, &pipeline, 0, &Premultiply_Blend, &e.vertices, &vertstride, 1,
                                         vertparams, 2, 0, 0);

  TEST((*desktop->setCursor)(desktop, w, FG_Cursor_Cross) == 0);

  TEST((*desktop->setWindow)(desktop, w, NULL, NULL, NULL, NULL, FG_WindowFlag_Resizable) == 0);
  TEST((*desktop->setWindow)(desktop, w, NULL, NULL, NULL, "Feather Test Changed", FG_WindowFlag_Resizable) == 0);
  TEST((*desktop->invalidateWindow)(desktop, w, NULL) == 0);

  TEST((*desktop->processMessages)(desktop, NULL, &state) != 0);

  const char TEST_TEXT[] = "testtext";

  TEST((*desktop->processMessages)(desktop, NULL, &state) != 0);
  TEST((*desktop->clearClipboard)(desktop, w, FG_Clipboard_All) == 0);
  TEST((*desktop->checkClipboard)(desktop, w, FG_Clipboard_Text) == false);
  TEST((*desktop->checkClipboard)(desktop, w, FG_Clipboard_Wave) == false);
  TEST((*desktop->checkClipboard)(desktop, w, FG_Clipboard_All) == false);
  TEST((*desktop->putClipboard)(desktop, w, FG_Clipboard_Text, TEST_TEXT, 9) == 0);
  TEST((*desktop->checkClipboard)(desktop, w, FG_Clipboard_Text) == true);
  TEST((*desktop->checkClipboard)(desktop, w, FG_Clipboard_Wave) == false);
  TEST((*desktop->checkClipboard)(desktop, w, FG_Clipboard_All) == true);

  char hold[10];

  TEST((*desktop->getClipboard)(desktop, w, FG_Clipboard_Text, hold, 10) == 9)
  for(int i = 0; i < sizeof(TEST_TEXT); ++i)
  {
    TEST(TEST_TEXT[i] == hold[i]);
  }

  TEST((*desktop->getClipboard)(desktop, w, FG_Clipboard_Wave, hold, 10) == 0)
  while((*desktop->processMessages)(desktop, NULL, &state) != 0 && e.close == false)
    ;

  TEST((*b->destroyResource)(b, w->context, e.image) == 0);
  TEST((*b->destroyPipelineState)(b, w->context, e.pipeline) == 0);
  TEST((*desktop->destroyWindow)(desktop, w) == 0);
  (*bridge->destroy)(bridge);
  (*desktop->destroy)(desktop);
  (*b->destroy)(b);
}