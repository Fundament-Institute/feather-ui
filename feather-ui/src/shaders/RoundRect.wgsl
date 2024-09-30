struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

var<private> pos_1: vec2<f32>;
@group(0) @binding(2) 
var<uniform> DimBorderBlur: vec4<f32>;
@group(0) @binding(3) 
var<uniform> Corners: vec4<f32>;
@group(0) @binding(4) 
var<uniform> Fill: vec4<f32>;
@group(0) @binding(5) 
var<uniform> Outline: vec4<f32>;
var<private> fragColor: vec4<f32>;

fn linearstep(low: f32, high: f32, x: f32) -> f32 {
    var low_1: f32;
    var high_1: f32;
    var x_1: f32;

    low_1 = low;
    high_1 = high;
    x_1 = x;
    let _e12 = x_1;
    let _e13 = low_1;
    let _e15 = high_1;
    let _e16 = low_1;
    let _e21 = x_1;
    let _e22 = low_1;
    let _e24 = high_1;
    let _e25 = low_1;
    return clamp(((_e21 - _e22) / (_e24 - _e25)), 0f, 1f);
}

fn rectangle(samplePosition: vec2<f32>, halfSize: vec2<f32>, edges: vec4<f32>) -> f32 {
    var samplePosition_1: vec2<f32>;
    var halfSize_1: vec2<f32>;
    var edges_1: vec4<f32>;
    var edge: f32 = 20f;
    var local: f32;
    var local_1: f32;
    var componentWiseEdgeDistance: vec2<f32>;
    var outsideDistance: f32;
    var insideDistance: f32;

    samplePosition_1 = samplePosition;
    halfSize_1 = halfSize;
    edges_1 = edges;
    let _e14 = samplePosition_1;
    if (_e14.x > 0f) {
        let _e18 = samplePosition_1;
        if (_e18.y < 0f) {
            let _e22 = edges_1;
            local = _e22.y;
        } else {
            let _e24 = edges_1;
            local = _e24.z;
        }
        let _e27 = local;
        edge = _e27;
    } else {
        let _e28 = samplePosition_1;
        if (_e28.y < 0f) {
            let _e32 = edges_1;
            local_1 = _e32.x;
        } else {
            let _e34 = edges_1;
            local_1 = _e34.w;
        }
        let _e37 = local_1;
        edge = _e37;
    }
    let _e39 = samplePosition_1;
    let _e41 = halfSize_1;
    let _e43 = edge;
    componentWiseEdgeDistance = ((abs(_e39) - _e41) + vec2(_e43));
    let _e49 = componentWiseEdgeDistance;
    let _e55 = componentWiseEdgeDistance;
    outsideDistance = length(max(_e55, vec2(0f)));
    let _e61 = componentWiseEdgeDistance;
    let _e63 = componentWiseEdgeDistance;
    let _e65 = componentWiseEdgeDistance;
    let _e67 = componentWiseEdgeDistance;
    let _e71 = componentWiseEdgeDistance;
    let _e73 = componentWiseEdgeDistance;
    let _e75 = componentWiseEdgeDistance;
    let _e77 = componentWiseEdgeDistance;
    insideDistance = min(max(_e75.x, _e77.y), 0f);
    let _e83 = outsideDistance;
    let _e84 = insideDistance;
    let _e86 = edge;
    return ((_e83 + _e84) - _e86);
}

fn main_1() {
    var w: f32;
    var uv: vec2<f32>;
    var dist: f32;
    var alpha: f32;
    var s: f32;

    let _e6 = DimBorderBlur;
    let _e8 = pos_1;
    let _e11 = DimBorderBlur;
    let _e13 = pos_1;
    let _e16 = fwidth((_e11.x * _e13.x));
    let _e20 = DimBorderBlur;
    w = ((_e16 * 0.5f) * (1f + _e20.w));
    let _e25 = pos_1;
    let _e26 = DimBorderBlur;
    let _e29 = DimBorderBlur;
    uv = ((_e25 * _e26.xy) - (_e29.xy * 0.5f));
    let _e36 = DimBorderBlur;
    let _e41 = uv;
    let _e42 = DimBorderBlur;
    let _e46 = Corners;
    let _e47 = rectangle(_e41, (_e42.xy * 0.5f), _e46);
    dist = _e47;
    let _e50 = w;
    let _e53 = w;
    let _e54 = w;
    let _e56 = dist;
    let _e57 = linearstep(_e53, -(_e54), _e56);
    alpha = _e57;
    let _e60 = w;
    let _e62 = dist;
    let _e63 = DimBorderBlur;
    let _e66 = w;
    let _e67 = w;
    let _e69 = dist;
    let _e70 = DimBorderBlur;
    let _e73 = linearstep(_e66, -(_e67), (_e69 + _e70.z));
    s = _e73;
    let _e75 = Fill;
    let _e76 = _e75.xyz;
    let _e83 = Fill;
    let _e86 = s;
    let _e88 = Outline;
    let _e89 = _e88.xyz;
    let _e96 = Outline;
    let _e99 = alpha;
    let _e100 = s;
    let _e104 = alpha;
    let _e105 = s;
    fragColor = (((vec4<f32>(_e76.x, _e76.y, _e76.z, 1f) * _e83.w) * _e86) + ((vec4<f32>(_e89.x, _e89.y, _e89.z, 1f) * _e96.w) * clamp((_e104 - _e105), 0f, 1f)));
    return;
}

@fragment 
fn main(@location(0) pos: vec2<f32>) -> FragmentOutput {
    pos_1 = pos;
    main_1();
    let _e15 = fragColor;
    return FragmentOutput(_e15);
}
