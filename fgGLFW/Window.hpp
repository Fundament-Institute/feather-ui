// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__WINDOW_H
#define GL__WINDOW_H

#include "feather/desktop_interface.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace GLFW {
  class Provider;

  struct WindowGL : FG_Window
  {
    WindowGL(Provider* backend, GLFWmonitor* display, uintptr_t window_id, FG_Vec2* pos, FG_Vec2* dim, uint64_t flags,
           const char* caption);
    ~WindowGL();
    static uint8_t GetModKeys(int mods);
    void DirtyRect(const FG_Rect* r);
    uint8_t ScanJoysticks();
    void PollJoysticks();
    void Invalidate(FG_Rect* area);
    inline GLFWwindow* GetWindow() { return _window; }
    FG_COMPILER_DLLEXPORT FG_Vec2i GetSize() const;
    FG_COMPILER_DLLEXPORT void MakeCurrent();
    FG_COMPILER_DLLEXPORT void SwapBuffers();
#ifdef FG_PLATFORM_WIN32
    double TranslateJoyAxis(uint8_t axis, uint8_t index) const;
#endif
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CloseCallback(GLFWwindow* window);
    static void CharCallback(GLFWwindow* window, unsigned int key);
    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void MousePosCallback(GLFWwindow* window, double x, double y);
    static void EnterCallback(GLFWwindow* window, int entered);
    static void ScrollCallback(GLFWwindow* window, double xdelta, double ydelta);
    static void DropCallback(GLFWwindow* window, int count, const char* paths[]);
    static void FocusCallback(GLFWwindow* window, int focused);
    static void SizeCallback(GLFWwindow* window, int width, int height);
    static void RefreshCallback(GLFWwindow* window);
    static void JoystickCallback(int jid, int e);

    static uint8_t KeyMap[512];
    static constexpr uint8_t MAXJOY  = 16;
    static constexpr uint8_t MAXAXIS = 6;

    struct JoyCaps
    {
      uint32_t axes;
      uint32_t buttons;
      uint32_t offset[MAXAXIS];
      uint32_t range[MAXAXIS];
    };

    GLFWwindow* _window;
    Provider* _backend;
    uint64_t _flags;
    WindowGL* _next; // GLFW doesn't let us detect when it destroys a window so we have to do it ourselves.
    WindowGL* _prev;
    uint32_t _joysticks;
#ifdef FG_PLATFORM_WIN32
    JoyCaps _joycaps[MAXJOY];
    uint32_t _allbuttons[MAXJOY];
    uint32_t _alljoyaxis[MAXJOY][MAXAXIS];
#endif
    static void FillKeyMap();
  };
}

#endif
