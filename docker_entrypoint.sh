#!/bin/bash
set -e

# These are passed as env variables from run_docker.sh
TARGET_UID=${HOST_UID:-1000}
TARGET_GID=${HOST_GID:-1000}
# INTERNAL_HOME is passed by run_docker.sh as /root for Rootless and /home/nnue_user for Standard
export HOME=${INTERNAL_HOME:-/home/nnue_user}

# --- Case A: Rootless Mode (Already UID 0) ---
if [ "$(id -u)" -eq 0 ] && [ "$TARGET_UID" -eq 0 ]; then
    echo "Running in Rootless mode as root (mapped to host user)."
    exec "$@"
fi

# --- Case B: Standard Mode (Start as root, drop to Host UID) ---
if [ "$(id -u)" -eq 0 ] && [ "$TARGET_UID" -ne 0 ]; then
    echo "Running in Standard mode. Mapping host UID $TARGET_UID to nnue_user."

    # Create group if it doesn't exist
    if ! getent group nnue_group >/dev/null; then
        groupadd -g "$TARGET_GID" nnue_group
    fi

    # Create user if it doesn't exist, pointing to the mounted internal home
    if ! getent passwd nnue_user >/dev/null; then
        useradd -u "$TARGET_UID" -g nnue_group -d "$HOME" -s /bin/bash nnue_user
    fi

    # Fix ownership of the mount point if it was created by a different root process
    chown "$TARGET_UID:$TARGET_GID" "$HOME"

    # Drop privileges and execute the command
    exec gosu nnue_user "$@"
fi

# Fallback for manual 'docker run -u' overrides
exec "$@"
