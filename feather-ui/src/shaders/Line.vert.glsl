#version 440  
in vec2 vPos;  
layout(binding=0) uniform mat4 MVP;  
layout(binding=1) uniform vec2 ViewPort; //Width and Height of the viewport  
out vec2 vLineCenter;  
void main(void)  
{  
  vec4 pp = MVP * vec4(vPos.xy, 0, 1);  
  gl_Position = pp;  
  vLineCenter = 0.5*(pp.xy + vec2(1, 1))*ViewPort;  
}
