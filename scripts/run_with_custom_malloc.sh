#!/usr/bin/env bash

# Function to locate a shared library robustly
find_library() {
    local lib_name="$1"

    # Method 1: Query the dynamic linker cache (Linux standard)
    if command -v ldconfig >/dev/null 2>&1; then
        local cache_path
        cache_path=$(ldconfig -p | grep -m 1 "$lib_name" | awk '{print $NF}')
        if [ -n "$cache_path" ] && [ -f "$cache_path" ]; then
            echo "$cache_path"
            return 0
        fi
    fi

    # Method 2: Fallback to common library directories (handles macOS and non-standard Linux)
    local search_paths=(
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib/aarch64-linux-gnu"
        "/usr/lib64"
        "/usr/lib"
        "/usr/local/lib"
        "/opt/homebrew/lib"
    )

    for dir in "${search_paths[@]}"; do
        if [ -f "$dir/$lib_name" ]; then
            echo "$dir/$lib_name"
            return 0
        fi
    done

    return 1
}

# Determine correct extension based on OS
if [ "$(uname)" = "Darwin" ]; then
    EXT="dylib"
    PRELOAD_VAR="DYLD_INSERT_LIBRARIES"
else
    EXT="so"
    PRELOAD_VAR="LD_PRELOAD"
fi

# Attempt to find jemalloc first, then tcmalloc
ALLOC_PATH=$(find_library "libjemalloc.$EXT")
if [ -z "$ALLOC_PATH" ]; then
    ALLOC_PATH=$(find_library "libtcmalloc.$EXT")
fi

# Set the environment variable and execute
if [ -n "$ALLOC_PATH" ]; then
    echo "[Allocator Wrapper] Injecting custom allocator: $ALLOC_PATH" >&2

    # Prepend to existing preload variable if it exists, otherwise set it
    if [ -n "${!PRELOAD_VAR}" ]; then
        export eval $PRELOAD_VAR="$ALLOC_PATH:\${$PRELOAD_VAR}"
    else
        export eval $PRELOAD_VAR="$ALLOC_PATH"
    fi
else
    echo "[Allocator Wrapper] No custom allocator found. Falling back to system default." >&2
fi

# Replace the current shell process with the target command
exec "$@"