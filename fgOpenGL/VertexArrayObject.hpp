// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__VAO_H
#define GL__VAO_H

#include "feather/compiler.h"
#include "feather/graphics_interface.h"
#include "glad/glad.h"
#include <span>
#include <vector>
#include <utility>

namespace GL {
  class VertexArrayObject
  {
  public:
#ifndef USE_EMULATED_VAOS
    VertexArrayObject() : _vaoID(0) {}
    VertexArrayObject(const VertexArrayObject&) = delete;
    VertexArrayObject(VertexArrayObject&& right) noexcept : _vaoID(right._vaoID) { right._vaoID = 0; }
    VertexArrayObject& operator=(VertexArrayObject&& right) noexcept
    {
      this->~VertexArrayObject();
      _vaoID       = right._vaoID;
      right._vaoID = 0;
      return *this;
    }
    VertexArrayObject& operator=(const VertexArrayObject&) noexcept = delete;

  private:
    VertexArrayObject(GLuint id) : _vaoID(id) {}
    /// OpenGL object ID of VertexArrayObject
    GLuint _vaoID;
#else
    VertexArrayObject();
    VertexArrayObject(const VertexArrayObject&)                     = delete;
    VertexArrayObject(VertexArrayObject&&)                          = default;
    VertexArrayObject& operator=(VertexArrayObject&&) noexcept      = default;
    VertexArrayObject& operator=(const VertexArrayObject&) noexcept = delete;

  private:
    struct VertexAttrib
    {
      /// Index for this attribute used by the GLSL vertex shader
      GLint location;

      /// Number of elements per vertex. For example, a position attribute specifying XYZ would have numElements = 3.
      GLint numElements;

      /// OpenGL datatype of attribute elements. GL_FLOAT, GL_INT, GL_UNSIGNED_INT, etc.
      GLenum type;

      /// Attribute divisor for instancing
      uint16_t divisor;
    };

    struct VertexBuffer
    {
      /// object ID of the buffer
      GLuint id;

      /// total stride of all attributes for this buffer
      GLsizei stride;

      /// span view into attribute array
      std::span<VertexAttrib> attributes;
    };

    // Vector of all attributes over all vertex buffers
    std::vector<VertexAttrib> _attribs;

    /// OpenGL object ID and total stride of underlying vertex buffer for the VAO
    std::vector<VertexBuffer> _vertexBuffer;

    /// OpenGL object ID of underlying index buffer for the VAO
    GLuint _indexBuffer;
#endif

  public:
    ~VertexArrayObject();

    GLExpected<void> bind();
    GLExpected<void> unbind();

    static GLExpected<VertexArrayObject> create(GLuint program, std::span<FG_VertexParameter> parameters,
                                                std::span<std::pair<GLuint, GLsizei>> vbuffers, GLuint indices) noexcept;
  };
}

#endif
