// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__PROGRAM_OBJECT_H
#define GL__PROGRAM_OBJECT_H

#include "Ref.hpp"
#include <string>

namespace GL {
  static bool IsProgramObj(GLuint i) noexcept { return glIsProgram(i) == GL_TRUE; };
  static void DeleteProgramObj(GLuint i) noexcept { glDeleteProgram(i); };
  struct ShaderObject;

  struct ProgramObject : Ref<IsProgramObj, DeleteProgramObj>
  {
    explicit constexpr ProgramObject(GLuint shader) noexcept : Ref(shader) {}
    explicit constexpr ProgramObject(void* shader) noexcept : Ref(shader) {}
    constexpr ProgramObject() noexcept                      = default;
    constexpr ProgramObject(ProgramObject&& right) noexcept = default;
    constexpr ProgramObject(const ProgramObject&)         = delete;
    constexpr ~ProgramObject() noexcept                   = default;
    GLExpected<void> attach(ShaderObject&& shader) noexcept;
    GLExpected<void> link() noexcept;
    bool is_valid() const noexcept;
    GLExpected<std::string> log() const noexcept;
    GLExpected<void> set_uniform(const char* name, GLenum type, const float* data) const noexcept;

    ProgramObject& operator=(const ProgramObject&) = delete;
    ProgramObject& operator=(ProgramObject&& right) noexcept = default;

    static GLExpected<ProgramObject> create() noexcept;
  };
}

#endif
