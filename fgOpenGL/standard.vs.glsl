TXT(
#version 110\n
uniform mat4 MVP;\n
attribute vec2 vPos;\n
varying vec2 pos;

void main()\n
{\n
  pos = vPos.xy;\n
  gl_Position = MVP * vec4(vPos, 0, 1);\n
}\n
)