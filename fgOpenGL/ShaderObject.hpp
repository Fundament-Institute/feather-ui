// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__SHADER_OBJECT_H
#define GL__SHADER_OBJECT_H

#include "Ref.hpp"
#include "feather/graphics_interface.h"
#include <string>

struct FG_ShaderParameter__;

namespace GL {
  class Provider;
  struct ShaderObject : public Ref
  {
    static constexpr DESTROY_FUNC DESTROY = [](GLuint i) { glDeleteShader(i); };

    explicit constexpr ShaderObject(GLuint shader) noexcept : Ref(shader) {}
    explicit constexpr ShaderObject(FG_Shader shader) noexcept : Ref(static_cast<GLuint>(shader))
    {
#ifdef _DEBUG
      if(_ref != 0)
        assert(is_valid());
#endif
    }
    constexpr ShaderObject() noexcept                    = default;
    constexpr ShaderObject(const ShaderObject&) noexcept = default;
    constexpr ~ShaderObject() noexcept                   = default;
    GLExpected<bool> is_valid() const noexcept;
    GLExpected<std::string> log() const noexcept;

    ShaderObject& operator=(const ShaderObject&) noexcept = default;

    static GLExpected<Owned<ShaderObject>> create(const char* src, int type, Provider* backend) noexcept;
    static GLenum get_type(const FG_ShaderParameter__& param);
  };
}

#endif
