// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__PROGRAM_OBJECT_H
#define GL__PROGRAM_OBJECT_H

#include "ShaderObject.hpp"
#include <string>

namespace GL {
  //static bool IsProgramObj(GLuint i) noexcept { return glIsProgram(i) == GL_TRUE; };

  struct ProgramObject : Ref
  {
    static constexpr DESTROY_FUNC DESTROY = [](GLuint i) { glDeleteProgram(i); };

    explicit constexpr ProgramObject(GLuint shader) noexcept : Ref(shader) {}
    constexpr ProgramObject() noexcept                      = default;
    constexpr ProgramObject(const ProgramObject&)           = default;
    constexpr ~ProgramObject() noexcept                   = default;
    GLExpected<void> attach(ShaderObject shader) noexcept;
    GLExpected<void> link() noexcept;
    bool is_valid() const noexcept;
    GLExpected<std::string> log() const noexcept;
    GLExpected<void> set_uniform(const char* name, GLenum type, const float* data, uint32_t count) const noexcept;
    GLExpected<void> set_buffer(GLuint resource, uint32_t index, uint32_t offset, uint32_t length) const noexcept;
    GLExpected<void> set_texture(const char* name, GLenum type, GLuint data) const noexcept;

    ProgramObject& operator=(const ProgramObject&) = default;

    static GLExpected<Owned<ProgramObject>> create() noexcept;
  };
}

#endif
