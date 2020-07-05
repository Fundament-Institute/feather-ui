param([string]$arch="x64")

$platform = $arch
$bin = "bin-$arch"
switch ($arch) {
  "x86" {$platform="Win32"}
}

$cmakever = "3.17.1"

# If we can't find CMake, download it
if(!(Test-Path ".\cmake\bin\cmake.exe" -PathType Leaf)) {
  $cmakeshort = $cmakever.substring(0,4)
  Start-BitsTransfer -DisplayName CMake -Source "https://cmake.org/files/v$cmakeshort/cmake-$cmakever-win32-x86.zip" -Destination .
  Expand-Archive ".\cmake-$cmakever-win32-x86.zip" -DestinationPath .
  Rename-Item -Path "cmake-$cmakever-win32-x86" -NewName "cmake"
  Remove-Item cmake-$cmakever-win32-x86.zip
}

$deps = @("freetype2", "glfw", "harfbuzz", "SOIL")

foreach ($dep in $deps) {
  if(Test-Path -path "./$bin/$dep") {
    Remove-Item -Recurse -Force ./$bin/$dep
  }
}

Start-Sleep -Seconds 1

foreach ($dep in $deps) {
  mkdir -p ./$bin/$dep
  .\cmake\bin\cmake.exe "-Tv142,host=$arch" -DCMAKE_GENERATOR_PLATFORM="$platform" -DCMAKE_CXX_STANDARD="17" -DCMAKE_CXX_FLAGS="/D _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /D _CRT_SECURE_NO_WARNINGS" -B./$bin/$dep -S./deps/$dep

  $env:_CL_="/MT"
  .\cmake\bin\cmake.exe --build ./$bin/$dep --config MinSizeRel
  $env:_CL_="/MTd"
  .\cmake\bin\cmake.exe --build ./$bin/$dep --config Debug
}

$builds = @("MinSizeRel", "Debug")
$srcs = @("freetype2", "glfw\src", "harfbuzz", "SOIL")

foreach ($build in $builds) {
  if(Test-Path -path ".\$bin\$build") {
    Remove-Item -Recurse -Force ./$bin/$build
  }
  mkdir -p ./$bin/$build
  foreach ($src in $srcs) {
    Copy-Item -Path ".\$bin\$src\$build\*" -Destination ".\$bin\$build" -Recurse
  }
}