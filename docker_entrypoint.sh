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

    # 1. Resolve Group Conflicts
    if getent group "$TARGET_GID" >/dev/null 2>&1; then
        EXISTING_GROUP=$(getent group "$TARGET_GID" | cut -d: -f1)
        if [ "$EXISTING_GROUP" != "nnue_group" ]; then
            # GID is taken by another name. Free the target name if it exists on a different GID.
            if getent group nnue_group >/dev/null 2>&1; then
                groupdel nnue_group
            fi
            # Rename the existing group to nnue_group
            groupmod -n nnue_group "$EXISTING_GROUP"
        fi
    else
        if getent group nnue_group >/dev/null 2>&1; then
            # nnue_group exists but has the wrong GID
            groupmod -g "$TARGET_GID" nnue_group
        else
            # Neither the GID nor the name exists
            groupadd -g "$TARGET_GID" nnue_group
        fi
    fi

    # 2. Resolve User Conflicts
    if getent passwd "$TARGET_UID" >/dev/null 2>&1; then
        EXISTING_USER=$(getent passwd "$TARGET_UID" | cut -d: -f1)
        if [ "$EXISTING_USER" != "nnue_user" ]; then
            # UID is taken by another name. Free the target name if it exists on a different UID.
            if getent passwd nnue_user >/dev/null 2>&1; then
                userdel nnue_user
            fi
            # Rename the existing user, and update their GID and home directory
            usermod -l nnue_user -g "$TARGET_GID" -d "$HOME" -s /bin/bash "$EXISTING_USER"
        else
            # UID and name match, but ensure GID and home are explicitly set
            usermod -g "$TARGET_GID" -d "$HOME" -s /bin/bash nnue_user
        fi
    else
        if getent passwd nnue_user >/dev/null 2>&1; then
            # nnue_user exists but has the wrong UID
            usermod -u "$TARGET_UID" -g "$TARGET_GID" -d "$HOME" -s /bin/bash nnue_user
        else
            # Neither the UID nor the name exists
            useradd -m -u "$TARGET_UID" -g nnue_group -d "$HOME" -s /bin/bash nnue_user
        fi
    fi

    # Fix ownership of the mount point if it was created by a different root process
    # so files from a previous UID/GID mapping remain accessible.
    mkdir -p "$HOME"
    CURRENT_HOME_UID=$(stat -c '%u' "$HOME")
    CURRENT_HOME_GID=$(stat -c '%g' "$HOME")
    if [ "$CURRENT_HOME_UID" -ne "$TARGET_UID" ] || [ "$CURRENT_HOME_GID" -ne "$TARGET_GID" ]; then
        echo "[DOCKER_ENTRYPOINT] Adjusting ownership of $HOME from $CURRENT_HOME_UID:$CURRENT_HOME_GID to $TARGET_UID:$TARGET_GID"
        chown -R -h "$TARGET_UID:$TARGET_GID" "$HOME"
    fi

    # Drop privileges and execute the command
    exec gosu nnue_user "$@"
fi

# Fallback for manual 'docker run -u' overrides
# When running as an arbitrary UID, the default HOME may point to nnue_user's
# home even though that account/directory was never created in this branch.
# Ensure HOME is usable for tools that need config/cache directories.
if [ ! -d "$HOME" ] || [ ! -w "$HOME" ]; then
    FALLBACK_HOME="/tmp/home-$(id -u)"
    mkdir -p "$FALLBACK_HOME"
    export HOME="$FALLBACK_HOME"
fi
exec "$@"
