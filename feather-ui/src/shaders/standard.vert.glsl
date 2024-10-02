#version 450  
layout(binding=0) uniform mat4 MVP;  
layout(binding=1) uniform vec4 PosDim;  
in vec2 vPos;  
out vec2 pos;

void main()  
{  
  pos = vPos.xy;
  mat4 move;
  move[0] = vec4(PosDim.z, 0,0,0);
  move[1] = vec4(0, PosDim.w,0,0);
  move[2] = vec4(0, 0,1,0);
  move[3] = vec4(PosDim.x + PosDim.z*0.5, PosDim.y + PosDim.w*0.5,0,1);
  gl_Position = MVP * move * vec4(pos.x - 0.5, pos.y - 0.5, 1, 1);  
}  
