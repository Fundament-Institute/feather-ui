#include "backend.h"
#include "resource.h"
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

#define BACKEND fgOpenGL
#define TEST(x)                      \
  if(!(x))                           \
  {                                  \
    printf("Failed test: %s\n", #x); \
  }

extern "C" FG_Backend* BACKEND(void* root, FG_Log log, FG_Behavior behavior);

const char* LEVELS[] = { "FATAL: ", "ERROR: ", "WARNING: ", "NOTICE: ", "DEBUG: " };

// A logging function that just forwards everything to printf
void FakeLog(void* root, FG_Level level, const char* f, ...)
{
  if(level >= 0)
    printf("%s", LEVELS[level]);

  va_list args;
  va_start(args, f);
  vprintf(f, args);
  va_end(args);
  printf("\n");
}

// This "mock element" serves as a fake UI object that contains our state and will do our drawing.
struct MockElement : FG_MsgReceiver
{
  FG_Asset* image;
  FG_Font* font;
  FG_Shader* shader;
  FG_Asset* vertices;
  FG_Asset* layer;
  void* layout;
  void* input;
  uint64_t flags;
  bool close;
};

static FG_Command FAKE_CMD;

void SetShape(decltype(FAKE_CMD.shape)& shape, FG_Rect& area, float border, FG_Color fill, FG_Color outline, float blur)
{
  shape.area        = &area;
  shape.border      = border;
  shape.blur        = blur;
  shape.fillColor   = fill;
  shape.borderColor = outline;
  shape.z           = 0.0f;
  shape.asset       = NULL;
}

// The behavior function simply processes all window messages from the host OS.
FG_Result behavior(FG_MsgReceiver* element, FG_Window* w, void* ui, FG_Msg* m)
{
  static int counter = 0;
  MockElement& e     = *static_cast<MockElement*>(element);
  if(m->kind == FG_Kind_DRAW)
  {
    ++counter;
    FG_Backend* b = *(FG_Backend**)ui;
    FG_Clear(b, w, FG_Color{ 0xFF000000 });

    // Here we batch 8 different draw calls all at once. We do not know if the backend is actually capable of drawing
    // all of them in a single draw call, but by batching them together we give the backend the option of doing so. Some
    // backends will only be able to merge the first two rectangle calls. Some backends may be able to merge the rect,
    // arc, circle, and triangle calls. Some might be able to include the first asset draw, but the text draw afterwards
    // can't be batched because the text glyphs would not be in the same texture atlas as the asset draw.
    FG_Command list[9] = {};

    list[0].category = FG_Category_RECT;
    auto r1          = FG_Rect{ 0.f, 0.f, 100.f, 80.f };
    auto c1          = FG_Rect{ 0.f, 4.f, 8.f, 12.f };
    SetShape(list[0].shape, r1, 5.0f, FG_Color{ 0xFF18eac8 }, FG_Color{ 0xFFee46a6 }, 0.0f);
    list[0].shape.rect.corners = &c1;
    
    list[1].category = FG_Category_RECT;
    auto r1b         = FG_Rect{ 300.f, 150.f, 340.f, 190.f };
    auto c1b         = FG_Rect{ 0.0f, 0.0f, 0.0f, 0.0f };
    SetShape(list[1].shape, r1b, 0.0f, FG_Color{ 0xFFFFFFFF }, FG_Color{ 0xFF00FFFF }, 0.0f);
    list[1].shape.rect.corners = &c1b;

    list[2].category = FG_Category_ARC;
    auto r2          = FG_Rect{ 350.f, 100.f, 500.f, 300.f };
    SetShape(list[2].shape, r2, 5.0f, FG_Color{ 0xFF0000FF }, FG_Color{ 0xFFFFFFFF }, 0.0f);
    list[2].shape.arc.angles      = FG_Vec{ 0.f, 3.f };
    list[2].shape.arc.innerRadius = 10.0f;

    list[3].category = FG_Category_CIRCLE;
    auto r2a         = FG_Rect{ 100.f, 150.f, 250.f, 300.f };
    SetShape(list[3].shape, r2a, 10.0f, FG_Color{ 0xFFFFFFFF }, FG_Color{ 0xFF00FFFF }, 0.0f);
    list[3].shape.circle.innerRadius = 10.0f;
    list[3].shape.circle.innerBorder = 2.0f;

    list[4].category = FG_Category_ARC;
    auto r2b         = FG_Rect{ 300.f, 200.f, 380.0f, 280.0f };
    SetShape(list[4].shape, r2b, 5.0f, FG_Color{ 0xFFFFFFFF }, FG_Color{ 0xFFFFFF00 }, 0.0f);
    list[4].shape.arc.angles = FG_Vec{ 0.0f, 3.1416f * (sinf(counter * 0.01f) + 1.0f) };

    list[5].category = FG_Category_TRIANGLE;
    auto r3          = FG_Rect{ 150.f, 300.f, 300.f, 500.f };
    auto c3          = FG_Rect{ 0.f, 4.f, 8.f, 0.5f };
    SetShape(list[5].shape, r3, 5.0f, FG_Color{ 0xFF0000FF }, FG_Color{ 0xFF00FFFF }, 0.0f);
    list[5].shape.triangle.corners = &c3;

    list[6].category    = FG_Category_ASSET;
    auto r4             = FG_Rect{ 300.f, 300.f, 620.f, 580.f };
    list[6].asset.area  = &r4;
    list[6].asset.asset = e.image;
    list[6].asset.color = FG_Color{ 0xFFFFFFFF };

    list[7].category    = FG_Category_TEXT;
    auto r5             = FG_Rect{ 200.f, 10.f, 550.f, 40.f };
    list[7].text.area   = &r5;
    list[7].text.font   = e.font;
    list[7].text.layout = e.layout;
    list[7].text.color  = FG_Color{ 0xFFFFFFFF };

    list[8].category    = FG_Category_LINES;
    FG_Vec lines[]       = { { 100.0f, 100.0f }, { 100.0f, 150.0f }, { 225.0f, 125.0f } };
    list[8].lines.color = FG_Color{ 0xFFFFFFFF };
    list[8].lines.count = sizeof(lines) / sizeof(FG_Vec);
    list[8].lines.points = lines;

    /*list[9].category    = FG_Category_ASSET;
    auto r9             = FG_Rect{ 0.0f, 100.f, 330.f, 320.f };
    list[9].asset.area  = &r9;
    list[9].asset.asset = e.gamma1;
    list[9].asset.color = FG_Color{ 0xFFFFFFFF };

    list[10].category    = FG_Category_ASSET;
    list[10].asset.area  = &r9;
    list[10].asset.asset = e.gamma2;
    list[10].asset.color = FG_Color{ 0xFFFFFFFF };*/

    // Now that we've assembled our draw call list, send it to the backend
    FG_Draw(b, w, list, sizeof(list) / sizeof(FG_Command), nullptr);

    // Try drawing a custom shader.
    float proj[4 * 4];
    FG_GetProjection(b, w, 0, proj);
    FG_ShaderValue values[2];
    values[0].pf32         = (float*)proj;
    values[1].asset        = e.image;
    FG_Command shader      = { FG_Category_SHADER };
    shader.shader.shader   = e.shader;
    shader.shader.input    = e.input;
    shader.shader.primitive = FG_Primitive_TRIANGLE_STRIP;
    shader.shader.count     = 4;
    shader.shader.values = values;
    FG_Draw(b, w, &shader, 1, nullptr);

    // Try drawing with a custom blend function
    auto r6              = FG_Rect{ 650.f, 125.f, 800.f, 275.f };
    FG_BlendState blend6 = { FG_BlendValue_ZERO,      FG_BlendValue_SRC_ALPHA, FG_BlendOp_ADD, FG_BlendValue_ZERO,
                             FG_BlendValue_SRC_ALPHA, FG_BlendOp_ADD,          0b1111 };
    FG_Command circle    = { FG_Category_CIRCLE };
    SetShape(circle.shape, r6, 30.0f, FG_Color{ 0xFFFFFFFF }, FG_Color{ 0 }, 0.0f);
    FG_Draw(b, w, &circle, 1, &blend6);

    // Draw a layer with partial opacity and a custom transform to ensure it is blended correctly.
    float transform[16] = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 550, 50, 0, 1,
    };
    FG_PushLayer(b, w, e.layer, transform, 0.5f, nullptr);
    FG_Clear(b, w, FG_Color{ 0 });

    FG_Command rect = { FG_Category_RECT };
    auto r7         = FG_Rect{ 0.f, 0.f, 100.f, 80.f };
    SetShape(rect.shape, r7, 5.0f, FG_Color{ 0xFFFF0000 }, FG_Color{ 0xFFFFFF00 }, 0.0f);
    rect.shape.rect.corners = &c1;
    FG_Draw(b, w, &rect, 1, nullptr);

    FG_PopLayer(b, w);

    return FG_Result{ 0 };
  }

  // These are simply some input handling boilerplate code
  if(m->kind == FG_Kind_GETWINDOWFLAGS)
  {
    return e.close ? FG_Result{ FG_WindowFlag_CLOSED } : FG_Result{ 0 };
  }
  if(m->kind == FG_Kind_SETWINDOWFLAGS)
  {
    e.close = e.close || ((m->setWindowFlags.flags & FG_WindowFlag_CLOSED) != 0);
  }
  // We normally close when any keypress happens, but we don't want to close if the user is trying to take a screenshot.
  if((m->kind == FG_Kind_KEYDOWN && m->keyDown.key != FG_Keys_LMENU && m->keyDown.scancode != 84 &&
      m->keyDown.scancode != 88) ||
     m->kind == FG_Kind_MOUSEDOWN)
  {
    e.close = true;
  }
  return FG_Result{ -1 };
}

int main(int argc, char* argv[])
{
  FG_Backend* ui;
  auto b = BACKEND(&ui, FakeLog, behavior);
  if(!b)
  {
    printf("Failed to load backend!\n");
    return -1;
  }

  ui            = b;
  auto textrect = FG_Rect{ 0.f, 0.f, 1000.f, 700.f };
  auto e        = MockElement{};

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
      850.f,
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
      850.f,
      210.f,
      0.0f,
      1.0f,
    },
    {
      1050.f,
      210.f,
      1.0f,
      1.0f,
    },
  };
  FG_ShaderParameter params[]     = { { FG_ShaderType_FLOAT, 4, 4, "MVP" }, { FG_ShaderType_TEXTURE, 0, 0, 0 } };
  FG_ShaderParameter vertparams[] = { { FG_ShaderType_FLOAT, 2, 0, "vPos" }, { FG_ShaderType_FLOAT, 2, 0, "vUV" } };

  e.flags    = FG_WindowFlag_RESIZABLE;
  e.image    = FG_CreateAsset(b, 0, (const char*)EXAMPLE_PNG_ARRAY, sizeof(EXAMPLE_PNG_ARRAY), FG_Format_PNG, 0);
  e.font     = FG_CreateFont(b, "Arial", 700, false, 16, FG_Vec{ 96.f, 96.f }, FG_AntiAliasing_AA);
  e.layout   = FG_FontLayout(b, e.font, "Example Text!", &textrect, 16.f, 0.f, FG_BreakStyle_NONE, nullptr);
  e.shader   = FG_CreateShader(b, shader_fs, shader_vs, 0, 0, 0, 0, params, 2);
  e.vertices = FG_CreateBuffer(b, 0, verts, sizeof(verts), 0, FG_Format_VERTEX_BUFFER);
  e.input    = FG_CreateShaderInput(b, &e.vertices, 1, vertparams, 2);
  e.close    = false;

  auto pos = FG_Vec{ 200.f, 100.f };
  auto dim = FG_Vec{ 800.f, 600.f };
  auto w   = FG_CreateWindow(b, &e, nullptr, &pos, &dim, "Feather Test", e.flags);
  TEST(w != nullptr);

  if(!w)
  {
    printf("failed to create window!\n");
    return -1;
  }

  FG_Vec layerdim = { 200, 100 };
  e.layer         = FG_CreateLayer(b, w, &layerdim, 0);

  TEST(FG_SetCursor(b, w, FG_Cursor_CROSS) == 0);
  TEST(FG_DirtyRect(b, w, 0) == 0);

  TEST(FG_SetWindow(b, w, nullptr, nullptr, nullptr, nullptr, nullptr, FG_WindowFlag_RESIZABLE) == 0);
  TEST(FG_SetWindow(b, w, &e, nullptr, nullptr, nullptr, "Feather Test Changed", FG_WindowFlag_RESIZABLE) == 0);

  TEST(FG_ProcessMessages(b) != 0);

  const char TEST_TEXT[] = "testtext";

  TEST(FG_ProcessMessages(b) != 0);
  TEST(FG_ClearClipboard(b, w, FG_Clipboard_ALL) == 0);
  TEST(FG_CheckClipboard(b, w, FG_Clipboard_TEXT) == false);
  TEST(FG_CheckClipboard(b, w, FG_Clipboard_WAVE) == false);
  TEST(FG_CheckClipboard(b, w, FG_Clipboard_ALL) == false);
  TEST(FG_PutClipboard(b, w, FG_Clipboard_TEXT, TEST_TEXT, 9) == 0);
  TEST(FG_CheckClipboard(b, w, FG_Clipboard_TEXT) == true);
  TEST(FG_CheckClipboard(b, w, FG_Clipboard_WAVE) == false);
  TEST(FG_CheckClipboard(b, w, FG_Clipboard_ALL) == true);

  char hold[10];

  TEST(FG_GetClipboard(b, w, FG_Clipboard_TEXT, hold, 10) == 9)
  for(int i = 0; i < 8; ++i)
  {
    TEST(TEST_TEXT[i] == hold[i]);
  }

  TEST(FG_GetClipboard(b, w, FG_Clipboard_WAVE, hold, 10) == 0)

  while(FG_ProcessMessages(b) != 0 && e.close == false)
  {
    FG_DirtyRect(b, w, 0);
  }

  TEST(FG_DestroyAsset(b, e.layer) == 0); // Must destroy layers before destroying the window
  TEST(FG_DestroyWindow(b, w) == 0);
  TEST(FG_DestroyAsset(b, e.image) == 0);
  TEST(FG_DestroyLayout(b, e.layout) == 0);
  TEST(FG_DestroyFont(b, e.font) == 0);
  TEST(FG_DestroyShader(b, e.shader) == 0);
  TEST(FG_DestroyShaderInput(b, e.input) == 0);
  (*b->destroy)(b);
}
