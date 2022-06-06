// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__SHADER_OBJECT_H
#define GL__SHADER_OBJECT_H

#include "GLRef.h"
#include <string>

namespace GL {
  static constexpr bool IsShader(GLuint i) noexcept { return glIsShader(i) == GL_TRUE; };
  static constexpr void DeleteShader(GLuint i) noexcept { glDeleteShader(i); };

  struct ShaderObject : public GLRef<IsShader,DeleteShader>
  {
    explicit constexpr ShaderObject(GLuint shader) noexcept : GLRef(shader) {}
    explicit constexpr ShaderObject(void* shader) noexcept : GLRef(shader) {}
    constexpr ShaderObject() noexcept = default;
    constexpr ShaderObject(ShaderObject&& right) noexcept = default;
    constexpr ShaderObject(const ShaderObject&) = delete;
    constexpr ~ShaderObject() noexcept          = default;
    bool is_valid() const noexcept;
    GLExpected<std::string> log() const noexcept;
    GLExpected<void> set_uniform(const char* name, GLenum type, float* data) noexcept;

    ShaderObject& operator=(const ShaderObject&) = delete;
    ShaderObject& operator=(ShaderObject&& right) noexcept = default;

    static GLExpected<ShaderObject> create(const char* src, int type) noexcept;
  };
}

#endif
