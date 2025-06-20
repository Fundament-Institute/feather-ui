const FILES: [(&str, &str); 3] = [
    ("feather.wgsl", include_str!("feather.wgsl")),
    ("shape.wgsl", include_str!("shape.wgsl")),
    ("compositor.wgsl", include_str!("compositor.wgsl")),
];

pub fn load_wgsl(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    /*
    const PREFIX: &str = "#include \"";
    
    let s = src
        .find(PREFIX)
        .and_then(|idx| {
            let start = idx + PREFIX.len();
            src[start..].find('"').and_then(|end| {
                get(&src[start..end]).and_then(|s| {
                    Some(std::borrow::Cow::Owned(
                        String::from_str(&src[..idx]).unwrap() + s + &src[end..],
                    ))
                })
            })
        })
        .unwrap_or_else(|| std::borrow::Cow::Borrowed(src));
    */

    let s = std::borrow::Cow::Borrowed(src);

    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(s),
    })
}

pub fn get(file: &str) -> Option<&str> {
    FILES
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(file))
        .map(|(_, src)| *src)
}
