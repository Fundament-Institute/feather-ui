// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "ProgramObject.hpp"
#include "ShaderObject.hpp"
#include "BackendGL.hpp"
#include <assert.h>

using namespace GL;

GLExpected<void> ProgramObject::attach(ShaderObject&& shader) noexcept
{
  glAttachShader(_ref, shader.release());
  GL_ERROR("glAttachShader");
  return {};
}

GLExpected<void> ProgramObject::link() noexcept
{
  glLinkProgram(_ref);
  GL_ERROR("glLinkProgram");
  glValidateProgram(_ref);
  GL_ERROR("glValidateProgram");
  return {};
}

bool ProgramObject::is_valid() const noexcept
{
  if(glIsProgram(_ref) == GL_FALSE)
    return false;

  GLint status = GL_FALSE;
  glGetProgramiv(_ref, GL_VALIDATE_STATUS, &status);
  return status == GL_TRUE;
}

GLExpected<std::string> ProgramObject::log() const noexcept
{
  if(glIsProgram(_ref) == GL_FALSE)
    return GLError(ERR_INVALID_PARAMETER, "ShaderObject::Log");

  GLint len;
  glGetProgramiv(_ref, GL_INFO_LOG_LENGTH, &len); // this includes the null terminator
  GL_ERROR("glGetProgramiv");
  std::string log;
  log.resize(len);
  glGetProgramInfoLog(_ref, len, &len, log.data());
  GL_ERROR("glGetProgramInfoLog");
  log.resize(len); // this value doesn't, which is what we want
  return log;
}

GLExpected<ProgramObject> ProgramObject::create() noexcept
{
  auto program = glCreateProgram();
  GL_ERROR("glCreateProgram");
  return ProgramObject(program);
}

GLExpected<void> ProgramObject::set_uniform(const char* name, GLenum type, const float* data) const noexcept
{
  if(type >= GL_TEXTURE0 && type <= GL_TEXTURE31)
  {
    if(name) // Names are optional for textures
    {
      auto loc = glGetUniformLocation(_ref, name);
      GL_ERROR("glGetUniformLocation");
      if(loc > 0)
        type = GL_TEXTURE0 + loc - 1;
    }

    glActiveTexture(type);
    GL_ERROR("glActiveTexture");
    glEnable(GL_TEXTURE_2D);
    GL_ERROR("glEnable");
    glBindTexture(GL_TEXTURE_2D, unpack_ptr<GLuint>((void*)data));
    GL_ERROR("glBindTexture");
  }
  else
  {
    if(!name)
      return GLError(ERR_INVALID_PARAMETER, "name cannot be null");

    auto loc = glGetUniformLocation(_ref, name);
    GL_ERROR("glGetUniformLocation");
    switch(type)
    {
    case GL_FLOAT_MAT2: glUniformMatrix2fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT2x3: glUniformMatrix2x3fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT2x4: glUniformMatrix2x4fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT3x2: glUniformMatrix3x2fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT3: glUniformMatrix3fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT3x4: glUniformMatrix3x4fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT4x2: glUniformMatrix4x2fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT4x3: glUniformMatrix4x3fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT4: glUniformMatrix4fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_VEC2:
    case GL_INT_VEC2:
    case GL_UNSIGNED_INT_VEC2:
    case GL_BOOL_VEC2: glUniform2fv(loc, 1, data); break;
    case GL_FLOAT_VEC3:
    case GL_INT_VEC3:
    case GL_UNSIGNED_INT_VEC3:
    case GL_BOOL_VEC3: glUniform3fv(loc, 1, data); break;
    case GL_FLOAT_VEC4:
    case GL_INT_VEC4:
    case GL_UNSIGNED_INT_VEC4:
    case GL_BOOL_VEC4: glUniform4fv(loc, 1, data); break;
    case GL_DOUBLE: return GLError(ERR_INVALID_PARAMETER, "Cannot assign GL_DOUBLE!");
    default: glUniform1f(loc, data[0]); break;
    }
    GL_ERROR("glUniform1f");
  }

  return {};
}