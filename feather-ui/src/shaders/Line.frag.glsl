#version 450
layout(binding=2) uniform vec4 Color;  
in vec2 vLineCenter;  
out vec4 fragColor;

void main(void)  
{  
      vec4 col = Color;          
      float d = length(vLineCenter-gl_FragCoord.xy);  
      float LineWidth = 1.0;  
      col.a *= smoothstep(0.0, 1.0, LineWidth-d);  
      fragColor = vec4(col.rgb*col.a, col.a);  
}
