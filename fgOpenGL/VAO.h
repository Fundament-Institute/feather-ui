
#ifndef GL__VAO_H
#define GL__VAO_H

#include "compiler.h"
#include "backend.h"
#include "glad/glad.h"
#include "span.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <vector>
#include <utility>

namespace GL {
  class Backend;

  class VAO
  {
  private:
#ifndef USE_EMULATED_VAOS
    /// OpenGL object ID of VertexArrayObject
    GLuint _vaoID;
#else
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

    Backend* _backend;
    GLuint _element;

  public:
    VAO(Backend* backend, GLuint shader, std::span<FG_ShaderParameter> parameters, GLuint* vbuffers, GLsizei* vstrides,
        size_t n_vbuffers, GLuint indices, GLuint element);
    ~VAO();

    void Bind();
    void Unbind();
    inline GLuint ElementType() const { return _element; }
  };
}

#endif
