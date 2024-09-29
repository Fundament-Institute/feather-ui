#version 440  
layout(binding=0) uniform mat4 MVP;  
in vec4 vPosUV;  
in vec4 vColor;  
out vec2 uv;  
out vec4 color;  

void main()  
{  
  gl_Position = MVP * vec4(vPosUV.xy, 0, 1);  
  uv = vPosUV.zw;  
  color = vColor;  
}  
