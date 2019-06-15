const Builder = @import("std").build.Builder;

pub fn build(b: *Builder) void {
    const opts = b.standardReleaseOptions();
    const lib = b.addSharedLibrary("feather", "src/schema.zig", b.version(0, 1, 0));
    lib.setBuildMode(opts);
    // lib.setOutputDir("./lib");
    
    const exe = b.addExecutable("test", "test/test.zig");
    exe.setBuildMode(opts);
    exe.linkLibrary(lib);
    
    b.setInstallPrefix(".");
    b.default_step.dependOn(&exe.step);
    
    const run = exe.run();

    const test_step = b.step("test", "Run the test suite");
    test_step.dependOn(&run.step);
    b.installArtifact(exe);
    b.installArtifact(lib);
}