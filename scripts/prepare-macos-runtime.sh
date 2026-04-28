#!/usr/bin/env sh
set -eu

if [ "$(uname -s)" != "Darwin" ]; then
  exit 0
fi

runtime_dir="src-tauri/macos-runtime"

mkdir -p "$runtime_dir"

for library in \
  libsherpa-onnx-c-api.dylib \
  libsherpa-onnx-cxx-api.dylib \
  libonnxruntime.dylib \
  libonnxruntime.1.24.4.dylib
do
  if [ -f "$runtime_dir/$library" ]; then
    continue
  fi

  source_path=""
  for source_dir in \
    "target/release" \
    "target/debug" \
    "target/${TARGET_TRIPLE:-}/release" \
    "target/${TARGET_TRIPLE:-}/debug" \
    "target/${CARGO_BUILD_TARGET:-}/release" \
    "target/${CARGO_BUILD_TARGET:-}/debug" \
    "target/sherpa-onnx-prebuilt/${SHERPA_PREBUILT_DIR:-}/lib" \
    "target/sherpa-onnx-prebuilt/sherpa-onnx-v1.12.39-osx-arm64-shared-lib/lib"
  do
    if [ -f "$source_dir/$library" ]; then
      source_path="$source_dir/$library"
      break
    fi
  done

  if [ -z "$source_path" ]; then
    echo "Missing macOS runtime library: $library" >&2
    exit 1
  fi
  cp "$source_path" "$runtime_dir/$library"
done
