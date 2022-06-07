// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#include "ProgramObject.h"
#include "ShaderObject.h"
#include "BackendGL.h"
#include <assert.h>

using namespace GL;

GLExpected<void> ProgramObject::attach(ShaderObject&& shader) noexcept
{
  glAttachShader(_ref, std::move(shader).take());
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
  return GLExpected<std::string>(log);
}

GLExpected<ProgramObject> ProgramObject::create() noexcept
{
  auto program = glCreateProgram();
  GL_ERROR("glCreateProgram");
  return ProgramObject(program);
}
