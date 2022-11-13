// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "ProgramObject.hpp"
#include "ShaderObject.hpp"
#include "ProviderGL.hpp"
#include <cassert>

using namespace GL;

GLExpected<void> ProgramObject::attach(ShaderObject shader) noexcept
{
  RETURN_ERROR(CALLGL(glAttachShader, _ref, shader));
  return {};
}

GLExpected<void> ProgramObject::link() noexcept
{
  RETURN_ERROR(CALLGL(glLinkProgram, _ref));
  RETURN_ERROR(CALLGL(glValidateProgram, _ref));
  return {};
}

GLExpected<bool> ProgramObject::is_valid() const noexcept
{
  if(auto e = CALLGL(glIsProgram, _ref); e.has_error())
    return std::move(e.error());
  else if(e.value() == GL_FALSE)
    return false;

  GLint status = GL_FALSE;
  RETURN_ERROR(CALLGL(glGetProgramiv, _ref, GL_VALIDATE_STATUS, &status));
  return status == GL_TRUE;
}

GLExpected<std::string> ProgramObject::log() const noexcept
{
  if(glIsProgram(_ref) == GL_FALSE)
    return CUSTOM_ERROR(ERR_INVALID_REF, "ProgramObject has invalid _ref");

  GLint len;
  RETURN_ERROR(CALLGL(glGetProgramiv, _ref, GL_INFO_LOG_LENGTH, &len)); // this includes the null terminator
  std::string log;
  log.resize(len);
  glGetProgramInfoLog(_ref, len, &len, log.data());
  log.resize(len); // this value doesn't, which is what we want
  return log;
}

GLExpected<Owned<ProgramObject>> ProgramObject::create() noexcept
{
  auto program = glCreateProgram();
  return Owned<ProgramObject>{ program };
}
GLExpected<void> ProgramObject::set_buffer(GLuint resource, uint32_t index, uint32_t offset, uint32_t length) const noexcept
{
  if(length > 0)
  {
    RETURN_ERROR(CALLGL(glBindBufferRange, GL_SHADER_STORAGE_BUFFER, index, resource, offset, length));
  }
  else
  {
    RETURN_ERROR(CALLGL(glBindBufferBase, GL_SHADER_STORAGE_BUFFER, index, resource));
  }

  return {};
}

GLExpected<void> ProgramObject::set_texture(const char* name, GLenum type, GLuint data) const noexcept
{
  if(type >= GL_TEXTURE0 && type <= GL_TEXTURE31)
  {
    if(name) // Names are optional for textures
    {
      auto loc = CALLGL(glGetUniformLocation, _ref, name);
      if(loc.has_error())
        return std::move(loc.error());
      if(loc.value() > 0)
        type = GL_TEXTURE0 + loc.value() - 1;
    }

    RETURN_ERROR(CALLGL(glActiveTexture, type));
    RETURN_ERROR(CALLGL(glBindTexture, GL_TEXTURE_2D, data));
  }
  else
    return CUSTOM_ERROR(ERR_INVALID_ENUM, "type must be a GL_TEXTURE enum!");
  return {};
}

GLExpected<void> ProgramObject::set_uniform(const char* name, GLenum type, const float* data, uint32_t count) const noexcept
{
  if(!name)
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "name cannot be null");

  auto r = CALLGL(glGetUniformLocation, _ref, name);
  if(r.has_error())
    return std::move(r.error());
  auto loc = r.value();

  switch(type)
  {
  case GL_FLOAT_MAT2: RETURN_ERROR(CALLGL(glUniformMatrix2fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_MAT2x3: RETURN_ERROR(CALLGL(glUniformMatrix2x3fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_MAT2x4: RETURN_ERROR(CALLGL(glUniformMatrix2x4fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_MAT3x2: RETURN_ERROR(CALLGL(glUniformMatrix3x2fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_MAT3: RETURN_ERROR(CALLGL(glUniformMatrix3fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_MAT3x4: RETURN_ERROR(CALLGL(glUniformMatrix3x4fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_MAT4x2: RETURN_ERROR(CALLGL(glUniformMatrix4x2fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_MAT4x3: RETURN_ERROR(CALLGL(glUniformMatrix4x3fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_MAT4: RETURN_ERROR(CALLGL(glUniformMatrix4fv, loc, count, GL_FALSE, data)); break;
  case GL_FLOAT_VEC2: RETURN_ERROR(CALLGL(glUniform2fv, loc, count, data)); break;
  case GL_INT_VEC2:
  case GL_UNSIGNED_INT_VEC2:
  case GL_BOOL_VEC2: glUniform2iv(loc, count, reinterpret_cast<const int*>(data)); break;
  case GL_FLOAT_VEC3: RETURN_ERROR(CALLGL(glUniform3fv, loc, count, data)); break;
  case GL_INT_VEC3:
  case GL_UNSIGNED_INT_VEC3:
  case GL_BOOL_VEC3: glUniform3iv(loc, count, reinterpret_cast<const int*>(data)); break;
  case GL_FLOAT_VEC4: RETURN_ERROR(CALLGL(glUniform4fv, loc, count, data)); break;
  case GL_INT_VEC4:
  case GL_UNSIGNED_INT_VEC4:
  case GL_BOOL_VEC4: glUniform4iv(loc, count, reinterpret_cast<const int*>(data)); break;
  case GL_DOUBLE: return CUSTOM_ERROR(ERR_NOT_IMPLEMENTED, "Cannot assign GL_DOUBLE!");
  case GL_FLOAT: RETURN_ERROR(CALLGL(glUniform1fv, loc, count, data)); break;
  case GL_INT:
  case GL_UNSIGNED_INT:
  case GL_BOOL: glUniform1iv(loc, count, reinterpret_cast<const int*>(data)); break;
  default: return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Unknown OpenGL type!");
  }

  return {};
}