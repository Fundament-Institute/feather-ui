
float rectangle(vec2 samplePosition, vec2 halfSize, vec4 edges){
    float edge = 20.0;
    if(samplePosition.x > 0.0)
      edge = (samplePosition.y > 0.0) ? edges.y : edges.z;
    else
      edge = (samplePosition.y > 0.0) ? edges.x : edges.w;
    
    vec2 componentWiseEdgeDistance = abs(samplePosition) - halfSize + vec2(edge);
    float outsideDistance = length(max(componentWiseEdgeDistance, 0.0f));
    float insideDistance = min(max(componentWiseEdgeDistance.x, componentWiseEdgeDistance.y), 0.0f);
    return outsideDistance + insideDistance - edge;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float blur = 0.0;
    float outline = 2.0;
	//float w = (abs(dFdx(fragCoord.x)) + abs(dFdy(fragCoord.y))) * 0.5;
    float w = 0.75 + blur;
    vec2 uv = fragCoord.xy - (iResolution.xy * 0.5);
    
    float dist = rectangle(uv, vec2(200, 100.5), vec4(0.0, 10.0, 40.0, 80.0));
  	float alpha = smoothstep(0.0 + w, 0.0 - w, dist);
    float s = smoothstep(-outline + w, -outline - w, dist);
    
    vec4 fill = vec4(0.7, 0.7, 1.0, 1.0);
    vec4 edge = vec4(1.0, 1.0, 1.0, 1.0);
    
    // Output to screen
    //fragColor = vec4(fill.rgb * alpha,1);
  	fragColor = vec4((fill.rgb*fill.a*s) + (edge.rgb*edge.a*clamp(alpha - s, 0.0, 1.0)), 1.0);
}