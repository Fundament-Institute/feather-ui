// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__SHADER_OBJECT_H
#define GL__SHADER_OBJECT_H

#include "Ref.hpp"
#include <string>

struct FG_ShaderParameter__;

namespace GL {
  static bool IsShader(GLuint i) noexcept { return glIsShader(i) == GL_TRUE; };
  static void DeleteShader(GLuint i) noexcept { glDeleteShader(i); };

  struct ShaderObject : public Ref<IsShader,DeleteShader>
  {
    explicit constexpr ShaderObject(GLuint shader) noexcept : Ref(shader) {}
    explicit constexpr ShaderObject(void* shader) noexcept : Ref(shader) {}
    constexpr ShaderObject() noexcept = default;
    constexpr ShaderObject(ShaderObject&& right) noexcept = default;
    constexpr ShaderObject(const ShaderObject&) = delete;
    constexpr ~ShaderObject() noexcept          = default;
    bool is_valid() const noexcept;
    GLExpected<std::string> log() const noexcept;

    ShaderObject& operator=(const ShaderObject&) = delete;
    ShaderObject& operator=(ShaderObject&& right) noexcept = default;

    static GLExpected<ShaderObject> create(const char* src, int type) noexcept;
    static GLenum get_type(const FG_ShaderParameter__& param);
  };
}

#endif
