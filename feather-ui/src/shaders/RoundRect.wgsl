struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

var<private> pos_1: vec2<f32>;
@group(0) @binding(1) 
var<uniform> PosDim: vec4<f32>;
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
    let _e13 = x_1;
    let _e14 = low_1;
    let _e16 = high_1;
    let _e17 = low_1;
    let _e22 = x_1;
    let _e23 = low_1;
    let _e25 = high_1;
    let _e26 = low_1;
    return clamp(((_e22 - _e23) / (_e25 - _e26)), 0f, 1f);
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
    let _e15 = samplePosition_1;
    if (_e15.x > 0f) {
        let _e19 = samplePosition_1;
        if (_e19.y < 0f) {
            let _e23 = edges_1;
            local = _e23.y;
        } else {
            let _e25 = edges_1;
            local = _e25.z;
        }
        let _e28 = local;
        edge = _e28;
    } else {
        let _e29 = samplePosition_1;
        if (_e29.y < 0f) {
            let _e33 = edges_1;
            local_1 = _e33.x;
        } else {
            let _e35 = edges_1;
            local_1 = _e35.w;
        }
        let _e38 = local_1;
        edge = _e38;
    }
    let _e40 = samplePosition_1;
    let _e42 = halfSize_1;
    let _e44 = edge;
    componentWiseEdgeDistance = ((abs(_e40) - _e42) + vec2(_e44));
    let _e50 = componentWiseEdgeDistance;
    let _e56 = componentWiseEdgeDistance;
    outsideDistance = length(max(_e56, vec2(0f)));
    let _e62 = componentWiseEdgeDistance;
    let _e64 = componentWiseEdgeDistance;
    let _e66 = componentWiseEdgeDistance;
    let _e68 = componentWiseEdgeDistance;
    let _e72 = componentWiseEdgeDistance;
    let _e74 = componentWiseEdgeDistance;
    let _e76 = componentWiseEdgeDistance;
    let _e78 = componentWiseEdgeDistance;
    insideDistance = min(max(_e76.x, _e78.y), 0f);
    let _e84 = outsideDistance;
    let _e85 = insideDistance;
    let _e87 = edge;
    return ((_e84 + _e85) - _e87);
}

fn main_1() {
    var w: f32;
    var uv: vec2<f32>;
    var dist: f32;
    var alpha: f32;
    var s: f32;

    let _e7 = PosDim;
    let _e9 = pos_1;
    let _e12 = PosDim;
    let _e14 = pos_1;
    let _e17 = fwidth((_e12.z * _e14.x));
    let _e21 = DimBorderBlur;
    w = ((_e17 * 0.5f) * (1f + _e21.w));
    let _e26 = pos_1;
    let _e27 = PosDim;
    let _e30 = PosDim;
    uv = ((_e26 * _e27.zw) - (_e30.zw * 0.5f));
    let _e37 = PosDim;
    let _e42 = uv;
    let _e43 = PosDim;
    let _e47 = Corners;
    let _e48 = rectangle(_e42, (_e43.zw * 0.5f), _e47);
    dist = _e48;
    let _e51 = w;
    let _e54 = w;
    let _e55 = w;
    let _e57 = dist;
    let _e58 = linearstep(_e54, -(_e55), _e57);
    alpha = _e58;
    let _e61 = w;
    let _e63 = dist;
    let _e64 = DimBorderBlur;
    let _e67 = w;
    let _e68 = w;
    let _e70 = dist;
    let _e71 = DimBorderBlur;
    let _e74 = linearstep(_e67, -(_e68), (_e70 + _e71.z));
    s = _e74;
    let _e76 = Fill;
    let _e77 = _e76.xyz;
    let _e84 = Fill;
    let _e87 = s;
    let _e89 = Outline;
    let _e90 = _e89.xyz;
    let _e97 = Outline;
    let _e100 = alpha;
    let _e101 = s;
    let _e105 = alpha;
    let _e106 = s;
    fragColor = (((vec4<f32>(_e77.x, _e77.y, _e77.z, 1f) * _e84.w) * _e87) + ((vec4<f32>(_e90.x, _e90.y, _e90.z, 1f) * _e97.w) * clamp((_e105 - _e106), 0f, 1f)));
    return;
}

@fragment 
fn main(@location(0) pos: vec2<f32>) -> FragmentOutput {
    pos_1 = pos;
    main_1();
    let _e17 = fragColor;
    return FragmentOutput(_e17);
}
