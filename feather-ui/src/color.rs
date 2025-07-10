// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use num_traits::NumCast;
use wide::f32x4;

macro_rules! gen_channel_accessors {
    ($rgba:ident, $r:ident, $g:ident, $b:ident, $a:ident) => {
        pub fn as_array(&self) -> &[f32; 4] {
            self.$rgba.as_array_ref()
        }

        pub const fn new($r: f32, $g: f32, $b: f32, $a: f32) -> Self {
            Self {
                $rgba: f32x4::new([$r, $g, $b, $a]),
            }
        }

        pub fn $r(&self) -> f32 {
            self.as_array()[0]
        }

        pub fn $g(&self) -> f32 {
            self.as_array()[1]
        }

        pub fn $b(&self) -> f32 {
            self.as_array()[2]
        }

        pub fn $a(&self) -> f32 {
            self.as_array()[3]
        }
    };
}

pub fn mat4_x_vec4(l: f32x4, r: [f32x4; 4]) -> f32x4 {
    let v = l.as_array_ref();
    ((r[0]) * f32x4::splat(v[0]))
        + ((r[1]) * f32x4::splat(v[1]))
        + ((r[2]) * f32x4::splat(v[2]))
        + ((r[3]) * f32x4::splat(v[3]))
}

fn fconv<T: num_traits::ToPrimitive, U: NumCast>(v: T) -> U {
    U::from(v).unwrap()
}

pub fn srgb_to_linear<T: num_traits::Float>(c: T) -> T {
    if c <= fconv(0.04045) {
        c / fconv(12.92)
    } else {
        ((c + fconv(0.055)) / fconv(1.055)).powf(fconv(2.4))
    }
}

pub fn linear_to_srgb<T: num_traits::Float>(c: T) -> T {
    if c < fconv(0.0031308) {
        c * fconv(12.92)
    } else {
        (c.powf(<T as NumCast>::from(1.0).unwrap() / fconv(2.4)) * fconv(1.055)) - fconv(0.055)
    }
}

/// Applies a function to all color channels, leaving the alpha channel untouched
pub fn map_color(c: f32x4, f: impl Fn(f32) -> f32) -> f32x4 {
    let v = c.to_array();
    f32x4::new([f(v[0]), f(v[1]), f(v[2]), v[3]])
}

pub trait ColorSpace: Premultiplied {
    /// Everything must provide a way to translate into standard XYZ space. The 4th component
    /// is alpha, which is left unchanged in all color space transforms
    fn xyz(&self) -> XYZ;

    /// This uses the standard formulation to transform into OkLab from XYZ
    /// but this can be overriden if there's a more efficient pathway
    fn oklab(&self) -> OkLab {
        let xyz = self.xyz();

        let mut lms = mat4_x_vec4(xyz.xyza, XYZ_OKLAB_M1);

        let v = lms.as_array_mut();
        for v in v.iter_mut().take(3) {
            *v = v.powf(1.0 / 3.0);
        }

        OkLab {
            laba: mat4_x_vec4(lms, XYZ_OKLAB_M2),
        }
    }

    /// This pathway goes through the linear Raw_sRGB pathway first, because only linear Raw_sRGB to XYZ
    /// and vice-versa is defined, but a color space can override this with a faster pathway.
    fn srgb(&self) -> Raw_sRGB<false, false> {
        let srgb = self.linear_srgb();

        Raw_sRGB {
            rgba: map_color(srgb.rgba, linear_to_srgb),
        }
    }

    /// This automatically converts from XYZ, but can be overriden if there's a faster pathway
    fn linear_srgb(&self) -> Raw_sRGB<true, false> {
        let xyz = self.xyz();

        let linear_rgba = mat4_x_vec4(xyz.xyza, XYZ_SRGB);
        Raw_sRGB { rgba: linear_rgba }
    }
}

/// Represents a color space that is premultiplied. You cannot un-premultiply a colorspace, so
/// this is a one-way transformation.
pub trait Premultiplied {
    /// Transforms this colorspace into premultiplied non-linear sRGB. This should always
    /// linearize the colorspace before performing the premultiplication, which will then be
    /// delinearized back into sRGB after the premultiplication has happened.
    fn srgb_pre(&self) -> Raw_sRGB<false, true> {
        let srgb = self.linear_srgb_pre();

        Raw_sRGB {
            rgba: map_color(srgb.rgba, linear_to_srgb),
        }
    }

    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true>;
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct XYZ {
    xyza: f32x4,
}

impl XYZ {
    gen_channel_accessors! {xyza, x, y, z, a}
}

impl ColorSpace for XYZ {
    fn xyz(&self) -> XYZ {
        *self
    }
}

impl Premultiplied for XYZ {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        self.linear_srgb().premultiply()
    }
}

const XYZ_OKLAB_M1: [f32x4; 4] = [
    f32x4::new([0.818_933, 0.361_866_74, -0.128_859_71, 0.0]),
    f32x4::new([0.032_984_544, 0.929_311_9, 0.036_145_64, 0.0]),
    f32x4::new([0.048_200_3, 0.264_366_27, 0.633_851_7, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

const OKLAB_XYZ_M1: [f32x4; 4] = [
    f32x4::new([1.22701, -0.5578, 0.281256, 0.0]),
    f32x4::new([-0.0405802, 1.11226, -0.0716767, 0.0]),
    f32x4::new([-0.0763813, -0.421482, 1.58616, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

const XYZ_OKLAB_M2: [f32x4; 4] = [
    f32x4::new([0.210_454_26, 0.793_617_8, -0.004_072_047, 0.0]),
    f32x4::new([1.977_998_5, -2.428_592_2, 0.450_593_7, 0.0]),
    f32x4::new([0.025904037, 0.782_771_77, -0.808_675_77, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

const OKLAB_XYZ_M2: [f32x4; 4] = [
    f32x4::new([1.0, 0.396338, 0.215804, 0.0]),
    f32x4::new([1.0, -0.105561, -0.0638542, 0.0]),
    f32x4::new([1.0, -0.0894842, -1.29149, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct OkLab {
    laba: f32x4,
}

impl OkLab {
    gen_channel_accessors! {laba, l, c, h, a}
}

impl ColorSpace for OkLab {
    fn xyz(&self) -> XYZ {
        let mut lms = mat4_x_vec4(self.laba, OKLAB_XYZ_M2);

        let v = lms.as_array_mut();
        for v in v.iter_mut().take(3) {
            *v = v.powf(3.0);
        }

        XYZ {
            xyza: mat4_x_vec4(lms, OKLAB_XYZ_M1),
        }
    }

    fn oklab(&self) -> OkLab {
        *self
    }

    // Based on the somewhat cursed code provided here: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
    fn linear_srgb(&self) -> Raw_sRGB<true, false> {
        let v = self.laba.as_array_ref();
        let l_ = v[0] + 0.396_337_78 * v[1] + 0.215_803_76 * v[2];
        let m_ = v[0] - 0.105_561_346 * v[1] - 0.063_854_17 * v[2];
        let s_ = v[0] - 0.089_484_18 * v[1] - 1.291_485_5 * v[2];

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        Raw_sRGB {
            rgba: f32x4::new([
                4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
                -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
                -0.0041960863 * l - 0.703_418_6 * m + 1.707_614_7 * s,
                v[3],
            ]),
        }
    }
}

impl Premultiplied for OkLab {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        self.linear_srgb().premultiply()
    }
}

const XYZ_SRGB: [f32x4; 4] = [
    f32x4::new([3.2404542, -1.5371385, -0.4985314, 0.0]),
    f32x4::new([-0.969_266, 1.8760108, 0.0415560, 0.0]),
    f32x4::new([0.0556434, -0.2040259, 1.0572252, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];
const SRGB_XYZ: [f32x4; 4] = [
    f32x4::new([0.4124564, 0.3575761, 0.1804375, 0.0]),
    f32x4::new([0.2126729, 0.7151522, 0.0721750, 0.0]),
    f32x4::new([0.0193339, 0.119_192, 0.9503041, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub struct Raw_sRGB<const LINEAR: bool, const PREMULTIPLY: bool> {
    pub rgba: f32x4,
}

impl<const LINEAR: bool, const PREMULTIPLY: bool> Raw_sRGB<LINEAR, PREMULTIPLY> {
    gen_channel_accessors! {rgba, r, g, b, a}

    // Conveniently, white and black are the same in EVERY sRGB variant.

    /// Returns transparent black (all zeroes)
    pub const fn transparent() -> Self {
        Self { rgba: f32x4::ZERO }
    }

    /// Returns opaque black
    pub const fn black() -> Self {
        Self {
            rgba: f32x4::new([0.0, 0.0, 0.0, 1.0]),
        }
    }

    /// Returns pure white (all ones)
    pub const fn white() -> Self {
        Self { rgba: f32x4::ONE }
    }
}

impl Raw_sRGB<false, false> {
    /// Returns this color as an array of 8-bit channels. This is only implemented
    /// on nonlinear sRGB that hasn't been premultiplied, because otherwise you'll
    /// lose precision when sending the color to a shader, unless you are writing
    /// directly to the texture atlas, in which case you want `as_bgra` implemented
    /// on premultiplied sRGB.
    pub fn as_8bit(&self) -> [u8; 4] {
        self.as_array().map(|x| (x * 255.0).round() as u8)
    }

    /// Returns this color in a 32-bit integer. This is only valid for nonlinear sRGB
    /// that isn't premultiplied to avoid precision loss.
    pub fn as_32bit(&self) -> sRGB32 {
        sRGB32 {
            rgba: u32::from_be_bytes(self.as_8bit()),
        }
    }
}

impl Raw_sRGB<false, true> {
    /// Returns this color as an array of 8-bit channels, with the red and blue channels
    /// swapped. This is used exclusively to write values directly to a premultiplied
    /// sRGB texture like the texture atlas.
    pub fn as_bgra(&self) -> [u8; 4] {
        let mut rgba = self.as_array().map(|x| (x * 255.0).round() as u8);
        rgba.swap(0, 2);
        rgba
    }
}

impl<const LINEAR: bool, const PREMULTIPLY: bool> From<[f32; 4]> for Raw_sRGB<LINEAR, PREMULTIPLY> {
    fn from(value: [f32; 4]) -> Self {
        Self {
            rgba: f32x4::new(value),
        }
    }
}

impl ColorSpace for Raw_sRGB<false, false> {
    fn xyz(&self) -> XYZ {
        self.linear_srgb().xyz()
    }

    fn srgb(&self) -> Raw_sRGB<false, false> {
        *self
    }

    fn linear_srgb(&self) -> Raw_sRGB<true, false> {
        Raw_sRGB {
            rgba: map_color(self.rgba, srgb_to_linear),
        }
    }
}

impl Premultiplied for Raw_sRGB<false, false> {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        self.linear_srgb().premultiply()
    }
}

impl ColorSpace for Raw_sRGB<true, false> {
    fn xyz(&self) -> XYZ {
        // We can't really un-premultiply the alpha here, and XYZ can be premultiplied, so we just leave it
        XYZ {
            xyza: mat4_x_vec4(self.rgba, SRGB_XYZ),
        }
    }

    fn oklab(&self) -> OkLab {
        let v = self.as_array();
        let l = 0.412_221_46 * v[0] + 0.536_332_55 * v[1] + 0.051_445_995 * v[2];
        let m = 0.211_903_5 * v[0] + 0.680_699_5 * v[1] + 0.107_396_96 * v[2];
        let s = 0.088_302_46 * v[0] + 0.281_718_85 * v[1] + 0.629_978_7 * v[2];

        let l_ = l.powf(1.0 / 3.0);
        let m_ = m.powf(1.0 / 3.0);
        let s_ = s.powf(1.0 / 3.0);

        OkLab {
            laba: f32x4::new([
                0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
                1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
                0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
                v[3],
            ]),
        }
    }

    fn linear_srgb(&self) -> Raw_sRGB<true, false> {
        *self
    }
}

impl Premultiplied for Raw_sRGB<true, false> {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        self.premultiply()
    }
}

impl Raw_sRGB<true, false> {
    // Premultiplies the alpha
    fn premultiply(&self) -> Raw_sRGB<true, true> {
        let a = self.a();
        Raw_sRGB {
            rgba: map_color(self.rgba, |x| x * a),
        }
    }
}

impl Premultiplied for Raw_sRGB<false, true> {
    fn srgb_pre(&self) -> Raw_sRGB<false, true> {
        *self
    }

    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        Raw_sRGB {
            rgba: map_color(self.rgba, srgb_to_linear),
        }
    }
}

impl Premultiplied for Raw_sRGB<true, true> {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        *self
    }
}

/// Standard sRGB colorspace
#[allow(non_camel_case_types)]
pub type sRGB = Raw_sRGB<false, false>;
/// Linear sRGB colorspace
#[allow(non_camel_case_types)]
pub type Linear_sRGB = Raw_sRGB<true, false>;
/// Premultiplied sRGB colorspace
#[allow(non_camel_case_types)]
pub type Pre_sRGB = Raw_sRGB<false, true>;
/// Premultiplied Linear sRGB colorspace
#[allow(non_camel_case_types)]
pub type PreLinear_sRGB = Raw_sRGB<true, true>;

// We only implement this conversion for sRGB because cosmic_text expects sRGB colors.
impl From<sRGB> for cosmic_text::Color {
    fn from(val: sRGB) -> Self {
        let v = val.as_8bit();
        cosmic_text::Color::rgba(v[0], v[1], v[2], v[3])
    }
}

/// Represents an sRGB color (not premultiplied) as a 32-bit signed integer
#[allow(non_camel_case_types)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct sRGB32 {
    pub rgba: u32,
}

impl sRGB32 {
    pub const fn as_array(&self) -> [u8; 4] {
        self.rgba.to_be_bytes()
    }

    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            rgba: u32::from_be_bytes([r, g, b, a]),
        }
    }

    /// Returns transparent black (zero)
    pub const fn transparent() -> Self {
        Self { rgba: 0 }
    }

    /// Returns opaque black
    pub const fn black() -> Self {
        Self { rgba: 0x000000FF }
    }

    /// Returns pure white
    pub const fn white() -> Self {
        Self { rgba: u32::MAX }
    }

    pub const fn from_alpha(alpha: u8) -> Self {
        Self {
            rgba: 0xFFFFFF00 | alpha as u32,
        }
    }

    pub const fn r(&self) -> u8 {
        self.as_array()[0]
    }

    pub const fn g(&self) -> u8 {
        self.as_array()[1]
    }

    pub const fn b(&self) -> u8 {
        self.as_array()[2]
    }

    pub const fn a(&self) -> u8 {
        self.as_array()[3]
    }

    pub fn as_f32(&self) -> sRGB {
        sRGB {
            rgba: f32x4::new(self.as_array().map(|x| x as f32 / 255.0)),
        }
    }
}
