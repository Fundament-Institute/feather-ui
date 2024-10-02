struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

const PI: f32 = 3.1415927f;

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
    let _e14 = x_1;
    let _e15 = low_1;
    let _e17 = high_1;
    let _e18 = low_1;
    let _e23 = x_1;
    let _e24 = low_1;
    let _e26 = high_1;
    let _e27 = low_1;
    return clamp(((_e23 - _e24) / (_e26 - _e27)), 0f, 1f);
}

fn rotate(p: vec2<f32>, a: f32) -> vec2<f32> {
    var p_1: vec2<f32>;
    var a_1: f32;

    p_1 = p;
    a_1 = a;
    let _e12 = p_1;
    let _e15 = a_1;
    let _e18 = p_1;
    let _e21 = a_1;
    let _e25 = p_1;
    let _e28 = a_1;
    let _e31 = p_1;
    let _e34 = a_1;
    return vec2<f32>(((_e12.x * cos(_e15)) + (_e18.y * sin(_e21))), ((_e25.x * sin(_e28)) - (_e31.y * cos(_e34))));
}

fn sRGB(linearRGB: vec4<f32>) -> vec4<f32> {
    var linearRGB_1: vec4<f32>;

    linearRGB_1 = linearRGB;
    let _e11 = linearRGB_1;
    let _e25 = linearRGB_1;
    let _e40 = (1.055f * pow(_e25.xyz, vec3<f32>(0.36166665f, 0.36166665f, 0.36166665f)));
    let _e41 = linearRGB_1;
    return vec4<f32>(_e40.x, _e40.y, _e40.z, _e41.w);
}

fn linearRGB_2(sRGB_1: vec4<f32>) -> vec4<f32> {
    var sRGB_2: vec4<f32>;

    sRGB_2 = sRGB_1;
    let _e10 = sRGB_2;
    let _e20 = sRGB_2;
    let _e30 = pow(((_e20.xyz + vec3(0.055f)) / vec3(1.055f)), vec3(2.4f));
    let _e31 = sRGB_2;
    return vec4<f32>(_e30.x, _e30.y, _e30.z, _e31.w);
}

fn main_1() {
    var l: f32;
    var uv: vec2<f32>;
    var width: f32;
    var w1_: f32;
    var border: f32;
    var t: f32;
    var r: f32;
    var d0_: f32;
    var d1_: f32;
    var omega1_: vec2<f32>;
    var omega2_: vec2<f32>;
    var d: f32;
    var d2_: f32;
    var d3_: f32;
    var s: f32;
    var alpha: f32;

    let _e8 = PosDim;
    let _e10 = PosDim;
    l = ((_e8.z + _e10.w) * 0.5f);
    let _e16 = pos_1;
    uv = ((_e16 * 2f) - vec2(1f));
    let _e23 = pos_1;
    let _e25 = pos_1;
    let _e27 = fwidth(_e25.x);
    width = _e27;
    let _e30 = DimBorderBlur;
    let _e33 = width;
    w1_ = ((1f + _e30.w) * _e33);
    let _e36 = DimBorderBlur;
    let _e38 = l;
    border = ((_e36.z / _e38) * 2f);
    let _e44 = Corners;
    let _e46 = l;
    let _e49 = w1_;
    t = ((0.5f - (_e44.z / _e46)) + (_e49 * 1.5f));
    let _e55 = t;
    let _e57 = w1_;
    r = ((1f - _e55) + _e57);
    let _e61 = uv;
    let _e63 = r;
    let _e66 = uv;
    let _e68 = r;
    let _e71 = t;
    let _e73 = border;
    d0_ = ((abs((length(_e66) - _e68)) - _e71) + _e73);
    let _e77 = uv;
    let _e79 = r;
    let _e82 = uv;
    let _e84 = r;
    let _e87 = t;
    d1_ = (abs((length(_e82) - _e84)) - _e87);
    let _e91 = Corners;
    let _e93 = Corners;
    let _e96 = uv;
    let _e97 = Corners;
    let _e99 = Corners;
    let _e102 = rotate(_e96, (_e97.x - _e99.y));
    omega1_ = _e102;
    let _e105 = Corners;
    let _e107 = Corners;
    let _e110 = uv;
    let _e111 = Corners;
    let _e113 = Corners;
    let _e116 = rotate(_e110, (_e111.x + _e113.y));
    omega2_ = _e116;
    let _e119 = omega1_;
    let _e122 = omega1_;
    let _e126 = omega2_;
    let _e128 = omega2_;
    let _e132 = width;
    if ((abs(-(_e122.y)) + abs(_e128.y)) < _e132) {
        {
            let _e134 = Corners;
            let _e141 = width;
            d = ((((_e134.y / PI) - 0.5f) * 2f) * _e141);
        }
    } else {
        let _e143 = Corners;
        if (_e143.y > 1.5707964f) {
            {
                let _e149 = omega1_;
                let _e152 = omega2_;
                let _e154 = omega1_;
                let _e157 = omega2_;
                d = max(-(_e154.y), _e157.y);
            }
        } else {
            {
                let _e160 = omega1_;
                let _e163 = omega2_;
                let _e165 = omega1_;
                let _e168 = omega2_;
                d = min(-(_e165.y), _e168.y);
            }
        }
    }
    let _e171 = d;
    let _e172 = Corners;
    let _e177 = Corners;
    let _e187 = DimBorderBlur;
    let _e189 = width;
    let _e192 = border;
    d = (_e171 + ((((clamp((_e177.y / PI), 0f, 1f) - 0.5f) * 2f) * (_e187.w * _e189)) + _e192));
    let _e195 = d;
    let _e196 = border;
    let _e198 = w1_;
    d2_ = ((_e195 - _e196) + _e198);
    let _e202 = omega1_;
    let _e204 = Corners;
    let _e207 = d;
    let _e208 = omega1_;
    let _e210 = Corners;
    let _e214 = w1_;
    d3_ = (min(_e207, (_e208.x + _e210.y)) + _e214);
    let _e217 = w1_;
    let _e220 = d0_;
    let _e223 = d0_;
    let _e225 = d2_;
    let _e227 = w1_;
    let _e229 = w1_;
    let _e231 = w1_;
    let _e232 = d0_;
    let _e235 = d0_;
    let _e237 = d2_;
    let _e239 = w1_;
    let _e241 = linearstep(-(_e229), _e231, (min(-(_e235), _e237) - _e239));
    s = _e241;
    let _e243 = w1_;
    let _e246 = d1_;
    let _e249 = d1_;
    let _e251 = d3_;
    let _e253 = w1_;
    let _e255 = w1_;
    let _e257 = w1_;
    let _e258 = d1_;
    let _e261 = d1_;
    let _e263 = d3_;
    let _e265 = w1_;
    let _e267 = linearstep(-(_e255), _e257, (min(-(_e261), _e263) - _e265));
    alpha = _e267;
    let _e269 = Fill;
    let _e270 = _e269.xyz;
    let _e277 = Fill;
    let _e280 = s;
    let _e282 = Outline;
    let _e283 = _e282.xyz;
    let _e290 = Outline;
    let _e293 = alpha;
    let _e294 = s;
    let _e298 = alpha;
    let _e299 = s;
    fragColor = (((vec4<f32>(_e270.x, _e270.y, _e270.z, 1f) * _e277.w) * _e280) + ((vec4<f32>(_e283.x, _e283.y, _e283.z, 1f) * _e290.w) * clamp((_e298 - _e299), 0f, 1f)));
    return;
}

@fragment 
fn main(@location(0) pos: vec2<f32>) -> FragmentOutput {
    pos_1 = pos;
    main_1();
    let _e19 = fragColor;
    return FragmentOutput(_e19);
}
