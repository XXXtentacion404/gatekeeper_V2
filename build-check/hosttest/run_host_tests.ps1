# ============================================================================
# Host-side (x86 Windows) verification runner -- ASCII only on purpose:
# Windows PowerShell 5.1 reads UTF-8-without-BOM .ps1 files as ANSI, which
# breaks on non-ASCII text. Keep this file ASCII.
#
# It compiles and runs the real algorithm sources that only depend on
# OpenCV core/imgproc/video:
#   1) GateIdCounter  multi-region state machine + ID handover (11 scenarios)
#   2) ByteTrack      occlusion re-association (no ID switch on occlusion)
# plus a control run with the re-association pass disabled (must FAIL, which
# proves the test really covers the new code).
#
# The product targets RK3588 + Hikvision SDK and cannot be built on x86; this
# script only builds the pieces that are portable, so the tested code is the
# real thing rather than a model.
#
# Usage: powershell -NoProfile -File build-check/hosttest/run_host_tests.ps1
# ============================================================================
$ErrorActionPreference = 'Continue'   # cmake/g++ write to stderr; do not abort on it
$root  = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$here  = $PSScriptRoot
$gxx   = 'D:\Tools\w64devkit\w64devkit\bin\g++.exe'
$cmake = 'D:\Tools\Cmake\bin\cmake.exe'
$ocvSrc = 'D:\Software\opencv\opencv\sources'       # only used for the fallback build
$ocvInc = 'D:\Software\opencv\opencv\build\include' # 4.10.0 headers, same version as libs
$ocvBuild = Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')) 'opencv-build'
$ocvLib = Join-Path $ocvBuild 'lib'
$utf8 = New-Object System.Text.UTF8Encoding $false

$libs = @(
  (Join-Path $ocvLib 'libopencv_video4100.a'),
  (Join-Path $ocvLib 'libopencv_imgproc4100.a'),
  (Join-Path $ocvLib 'libopencv_core4100.a'),
  (Join-Path $ocvBuild '3rdparty\lib\libzlib.a')
)

# ---- 0. build host OpenCV (core/imgproc/video) when missing ----
if (-not (Test-Path $libs[0])) {
  Write-Host '[setup] building host OpenCV core/imgproc/video (first run, ~5-10 min) ...'
  & $cmake '-S' $ocvSrc '-B' $ocvBuild '-G' 'MinGW Makefiles' `
    '-DCMAKE_BUILD_TYPE=Release' '-DCMAKE_C_COMPILER=gcc' '-DCMAKE_CXX_COMPILER=g++' `
    '-DBUILD_LIST=core,imgproc,video' '-DBUILD_SHARED_LIBS=OFF' `
    '-DWITH_IPP=OFF' '-DWITH_OPENCL=OFF' '-DWITH_TBB=OFF' '-DWITH_EIGEN=OFF' '-DWITH_LAPACK=OFF' `
    '-DWITH_JPEG=OFF' '-DWITH_PNG=OFF' '-DWITH_TIFF=OFF' '-DWITH_OPENEXR=OFF' '-DWITH_WEBP=OFF' `
    '-DWITH_PROTOBUF=OFF' '-DWITH_QUIRC=OFF' '-DWITH_GTK=OFF' '-DWITH_FFMPEG=OFF' '-DWITH_V4L=OFF' `
    '-DWITH_1394=OFF' '-DWITH_OPENNI=OFF' '-DWITH_OPENNI2=OFF' '-DWITH_GSTREAMER=OFF' `
    '-DBUILD_opencv_apps=OFF' '-DBUILD_TESTS=OFF' '-DBUILD_PERF_TESTS=OFF' '-DBUILD_EXAMPLES=OFF' `
    '-DBUILD_DOCS=OFF' '-DBUILD_JAVA=OFF' '-DBUILD_opencv_python2=OFF' '-DBUILD_opencv_python3=OFF' |
    Select-Object -Last 3
  & $cmake '--build' $ocvBuild '-j' 8 | Select-Object -Last 3
}

Set-Location $root
$common = @('-std=gnu++17', '-O1', '-w', ('-I' + $ocvInc), '-Iinclude', '-Iinclude\bytetrack')
$tail = @('-lstdc++', '-lm', '-lpthread', '-static')
$failed = 0

function Invoke-Suite([string]$name, [string]$test, [string[]]$sources) {
  Write-Host "`n########## $name ##########"
  $exe = Join-Path $here ($test -replace '\.cpp$', '.exe')
  & $gxx @common @('-o', $exe, (Join-Path $here $test)) $sources @libs @tail
  if ($LASTEXITCODE -ne 0) { Write-Host "[compile FAILED] $name"; $script:failed++; return }
  & $exe
  if ($LASTEXITCODE -ne 0) { $script:failed++ }
}

# ---- 1. counter state machine ----
Invoke-Suite 'GateIdCounter multi-region state machine + ID handover' `
  'test_gate_counter.cpp' @('src\gate_id_counter.cpp')

# ---- 1b. real cameras_config.txt end-to-end smoke test ----
# Synthetic geometry in suite 1 cannot catch "config -> counter wiring" breakage
# (2026-09-16: zone_line/zone_role were never copied into CameraConfig, so the
#  board drew no zone and emitted zero events while Region/Zone still looked fine).
Invoke-Suite 'Real cameras_config.txt end-to-end (line + ring + crossing)' `
  'test_real_config.cpp' @('src\gate_id_counter.cpp')

# ---- 1c. CameraConfig field plumbing (static check, hand-written move ctor) ----
Write-Host "`n########## CameraConfig field plumbing (static) ##########"
$py = $null
foreach ($c in @('D:\Software\Anaconda3\envs\yolov8\python.exe', 'python')) {
  if ($c -eq 'python' -or (Test-Path $c)) { $py = $c; break }
}
if ($py) {
  & $py (Join-Path $here '_chk_camcfg_wiring.py') (Join-Path $root 'src\main_tracking.cpp')
  if ($LASTEXITCODE -ne 0) { Write-Host '[FAILED] CameraConfig field plumbing'; $failed++ }
} else {
  Write-Host '[skip] no python found for _chk_camcfg_wiring.py'
}

# ---- 2. occlusion re-association ----
Invoke-Suite 'ByteTrack occlusion re-association' `
  'test_tracker_reid.cpp' @('src\tracker.cpp')

# ---- 3. control run: re-association disabled (must fail) ----
Write-Host "`n########## control: occlusion re-association disabled ##########"
$patched = Join-Path $here 'tracker_noreid.cpp'
$text = [System.IO.File]::ReadAllText((Join-Path $root 'src\tracker.cpp'), $utf8)
$text = $text.Replace(
  'reassociateLost(active_tracks_, free_tracks, fdets, fhists, tm, dm);',
  "tm.assign(active_tracks_.size(), -1);`r`n            dm.assign(fdets.size(), -1);")
[System.IO.File]::WriteAllText($patched, $text, $utf8)
$exe = Join-Path $here 'test_tracker_noreid.exe'
& $gxx @common @('-o', $exe, (Join-Path $here 'test_tracker_reid.cpp')) $patched @libs @tail
& $exe | Select-Object -Last 4
if ($LASTEXITCODE -eq 0) {
  Write-Host '[WARN] suite still passes with re-association disabled -- test does not cover it'
  $failed++
} else {
  Write-Host '[ok] suite fails with re-association disabled -- new logic is covered'
}

Write-Host "`n==== runner finished: $failed failing suite(s) ===="
exit $failed
