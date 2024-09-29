#version 440  
layout(binding=0) uniform mat4 MVP;  
layout(binding=1) uniform vec2 Inflate;  
in vec2 vPos;  
out vec2 pos;

void main()  
{  
  pos = vPos.xy;  
  pos -= vec2(0.5,0.5);  
  pos *= Inflate;  
  pos += vec2(0.5,0.5);  
  gl_Position = MVP * vec4(pos, 0, 1);  
}  
