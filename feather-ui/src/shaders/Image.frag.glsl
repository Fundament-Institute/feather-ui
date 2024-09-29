#version 440  
in vec2 uv;  
in vec4 color;  
out vec4 fragColor;

uniform sampler2D texture;  

void main()  
{  
  vec4 c = texture2D(texture, uv);
  fragColor = vec4(c.rgb * color.rgb * vec3(color.a,color.a,color.a), c.a*color.a);  
  //fragColor = vec4(uv.x, uv.y, 0, 1);  
}  
