#include "backend.h"
#include "resource.h"
#include <stdio.h>
#include <stdarg.h>

#define BACKEND fgOpenGL
#define TEST(x)                      \
  if(!(x))                           \
  {                                  \
    printf("Failed test: %s\n", #x); \
  }

extern "C" FG_Backend* BACKEND(void* root, FG_Log log, FG_Behavior behavior);

const char* LEVELS[] = { "FATAL: ", "ERROR: ", "WARNING: ", "NOTICE: ", "DEBUG: " };

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

struct MockElement : FG_Element
{
  FG_Asset* image;
  FG_Font* font;
  FG_Shader* shader;
  FG_Asset* vertices;
  FG_Asset* layer;
  void* layout;
  uint64_t flags;
  bool close;
};

FG_Result behavior(FG_Element* element, void* w, void* ui, FG_Msg* m)
{
  MockElement& e = *static_cast<MockElement*>(element);
  if(m->kind == FG_Kind_DRAW)
  {
    FG_Backend* b = *(FG_Backend**)ui;
    (*b->clear)(b, w, FG_Color{ 0 });

    auto r = FG_Rect{ 50.f, 100.f, 200.f, 300.f };
    auto c = FG_Rect{ 0.f, 4.f, 8.f, 12.f };
    (*b->drawRect)(b, w, &r, &c, FG_Color{ 0xFF0000FF }, 5.f, FG_Color{ 0xFF00FFFF }, 0.f, nullptr, 0.f, 0.f, nullptr);

    auto r2 = FG_Rect{ 350.f, 100.f, 500.f, 300.f };
    auto c2 = FG_Vec{ 0.f, 3.f };
    (*b->drawCircle)(b, w, &r2, &c2, FG_Color{ 0xFF0000FF }, 5.f, FG_Color{ 0xFF00FFFF }, 0.f, 10.f, 2.f, nullptr, 0.f,
                     nullptr);

    auto r3 = FG_Rect{ 150.f, 300.f, 300.f, 500.f };
    auto c3 = FG_Rect{ 0.f, 4.f, 8.f, 0.5f };
    (*b->drawTriangle)(b, w, &r3, &c3, FG_Color{ 0xFF0000FF }, 5.f, FG_Color{ 0xFF00FFFF }, 0.f, nullptr, 0.45f, 0.f,
                       nullptr);

    auto r4 = FG_Rect{ 300.f, 300.f, 620.f, 580.f };
    (*b->drawAsset)(b, w, e.image, &r4, nullptr, FG_Color{ 0xFFFFFFFF }, 0.f, 0.f, 0.f, nullptr);

    auto r5 = FG_Rect{ 200.f, 10.f, 550.f, 40.f };
    (*b->drawText)(b, w, e.font, e.layout, &r5, FG_Color{ 0xFFFFFFFF }, 0.f, 0.f, 0.f, nullptr);

    float proj[4 * 4];
    (*b->getProjection)(b, w, 0, proj);
    (*b->drawShader)(b, w, e.shader, e.vertices, 0, 0, (float*)proj, e.image);

    auto r6              = FG_Rect{ 650.f, 125.f, 800.f, 275.f };
    auto c6              = FG_Vec{ 0.f, 3.14159f };
    FG_BlendState blend6 = { FG_BlendValue_ZERO,      FG_BlendValue_SRC_ALPHA, FG_BlendOp_ADD, FG_BlendValue_ZERO,
                             FG_BlendValue_SRC_ALPHA, FG_BlendOp_ADD,          0b1111 };
    (*b->drawCircle)(b, w, &r6, &c6, FG_Color{ 0xFFFFFFFF }, 30.f, FG_Color{ 0 }, 0.f, 0.f, 0.f, nullptr, 0.f, &blend6);

    float transform[16] = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 550, 50, 0, 1,
    };
    (*b->pushLayer)(b, w, e.layer, transform, 0.5f, nullptr);
    auto r7 = FG_Rect{ 0.f, 0.f, 100.f, 80.f };
    (*b->drawRect)(b, w, &r7, &c, FG_Color{ 0xFFFF0000 }, 5.f, FG_Color{ 0xFFFFFF00 }, 0.f, nullptr, 0.f, 0.f, nullptr);
    (*b->popLayer)(b, w);

    return FG_Result{ 0 };
  }
  if(m->kind == FG_Kind_GETWINDOWFLAGS)
  {
    return e.close ? FG_Result{ FG_Window_CLOSED } : FG_Result{ 0 };
  }
  if(m->kind == FG_Kind_SETWINDOWFLAGS)
  {
    e.close = e.close || ((m->setWindowFlags.flags & FG_Window_CLOSED) != 0);
  }
  if((m->kind == FG_Kind_KEYDOWN && m->keyDown.key != FG_Keys_LMENU && m->keyDown.scancode != 84) ||
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
      850.f,
      210.f,
      0.0f,
      1.0f,
    },
    {
      1050.f,
      10.f,
      1.0f,
      0.0f,
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

  e.flags    = FG_Window_RESIZABLE;
  e.image    = (*b->createAsset)(b, (const char*)EXAMPLE_PNG_ARRAY, sizeof(EXAMPLE_PNG_ARRAY), FG_Format_PNG);
  e.font     = (*b->createFont)(b, "Arial", 700, false, 16, FG_Vec{ 96.f, 96.f }, FG_AntiAliasing_AA);
  e.layout   = (*b->fontLayout)(b, e.font, "Example Text!", &textrect, 16.f, 0.f, FG_BreakStyle_NONE, nullptr);
  e.shader   = (*b->createShader)(b, shader_fs, shader_vs, 0, 0, 0, 0, params, 2);
  e.vertices = (*b->createBuffer)(b, verts, sizeof(verts), FG_Primitive_TRIANGLE_STRIP, vertparams, 2);
  e.close    = false;

  auto pos = FG_Vec{ 200.f, 100.f };
  auto dim = FG_Vec{ 800.f, 600.f };
  auto w   = (*b->createWindow)(b, &e, nullptr, &pos, &dim, "Feather Test", e.flags, nullptr);
  TEST(w != nullptr);

  if(!w)
  {
    printf("failed to create window!\n");
    return -1;
  }

  FG_Vec layerdim = { 200, 100 };
  e.layer = (*b->createLayer)(b, w, &layerdim, false);

  TEST((*b->setCursor)(b, w, FG_Cursor_CROSS) == 0);
  TEST((*b->dirtyRect)(b, w, 0) == 0);

  TEST((*b->setWindow)(b, w, nullptr, nullptr, nullptr, nullptr, nullptr, FG_Window_RESIZABLE) == 0);
  TEST((*b->setWindow)(b, w, &e, nullptr, nullptr, nullptr, "Feather Test Changed", FG_Window_RESIZABLE) == 0);

  TEST((*b->processMessages)(b) != 0);

  const char TEST_TEXT[] = "testtext";

  TEST((*b->processMessages)(b) != 0);
  TEST((*b->clearClipboard)(b, w, FG_Clipboard_ALL) == 0);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_TEXT) == false);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_WAVE) == false);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_ALL) == false);
  TEST((*b->putClipboard)(b, w, FG_Clipboard_TEXT, TEST_TEXT, 9) == 0);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_TEXT) == true);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_WAVE) == false);
  TEST((*b->checkClipboard)(b, w, FG_Clipboard_ALL) == true);

  char hold[10];

  TEST((*b->getClipboard)(b, w, FG_Clipboard_TEXT, hold, 10) == 9)
  for(int i = 0; i < 8; ++i)
  {
    TEST(TEST_TEXT[i] == hold[i]);
  }

  TEST((*b->getClipboard)(b, w, FG_Clipboard_WAVE, hold, 10) == 0)

  while((*b->processMessages)(b) != 0 && e.close == false)
  {
    (*b->dirtyRect)(b, w, 0);
  }

  TEST((*b->destroyAsset)(b, e.layer) == 0); // Must destroy layers before destroying the window
  TEST((*b->destroyWindow)(b, w) == 0);
  TEST((*b->destroyAsset)(b, e.image) == 0);
  TEST((*b->destroyLayout)(b, e.layout) == 0);
  TEST((*b->destroyFont)(b, e.font) == 0);
  TEST((*b->destroyShader)(b, e.shader) == 0);
  (*b->destroy)(b);
}
