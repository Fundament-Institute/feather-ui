TXT(#version 110\n
uniform mat4 MVP;\n
attribute vec2 vPos;\n

void main()\n
{\n
  gl_Position = MVP * vec4(vPos.xy, 0, 1);\n
}\n
)