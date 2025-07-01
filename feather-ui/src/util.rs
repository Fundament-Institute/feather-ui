// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use wgpu::CompilationMessageType;

use crate::graphics::Driver;
use crate::shaders;

pub fn create_hotloader<T: 'static>(
    path: &std::path::Path,
    label: &'static str,
    driver: std::sync::Weak<Driver>,
) -> eyre::Result<notify::RecommendedWatcher> {
    use notify::Watcher;
    let mut prev = std::fs::read_to_string(path)?;
    let pathbuf = path.to_owned();
    let mut watcher = notify::recommended_watcher(move |_| {
        if let Some(driver) = driver.upgrade() {
            let contents = std::fs::read_to_string(&pathbuf).unwrap();
            if contents != prev {
                prev = contents;
                driver
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation);
                let module = shaders::load_wgsl(&driver.device, label, &prev);
                let err = futures_lite::future::block_on(driver.device.pop_error_scope());
                if let Some(e) = err {
                    println!("{e}");
                } else {
                    let info = futures_lite::future::block_on(module.get_compilation_info());

                    let mut errored = false;
                    for m in info.messages {
                        println!("{m:?}");
                        errored = errored || m.message_type == CompilationMessageType::Error;
                    }
                    if !errored {
                        driver.reload_pipeline::<T>(module);
                    }
                }
            }
        }
    })?;

    watcher.watch(path, notify::RecursiveMode::NonRecursive)?;

    Ok(watcher)
}

/// Allocates `&[T]` on stack space.
pub(crate) fn alloca_array<T, R>(n: usize, f: impl FnOnce(&mut [T]) -> R) -> R {
    use std::mem::{align_of, size_of};

    alloca::with_alloca_zeroed(
        (n * size_of::<T>()) + (align_of::<T>() - 1),
        |memory| unsafe {
            let mut raw_memory = memory.as_mut_ptr();
            if raw_memory as usize % align_of::<T>() != 0 {
                raw_memory =
                    raw_memory.add(align_of::<T>() - raw_memory as usize % align_of::<T>());
            }

            f(std::slice::from_raw_parts_mut::<T>(
                raw_memory.cast::<T>(),
                n,
            ))
        },
    )
}
