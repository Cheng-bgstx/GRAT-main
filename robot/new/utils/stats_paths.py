import os
from os.path import join as pjoin


def normalize_root_vel_frame(root_vel_frame):
    root_vel_frame = (root_vel_frame or "body").lower()
    if root_vel_frame not in ("body", "world"):
        root_vel_frame = "body"
    return root_vel_frame


def get_stats_variant(root_vel_frame):
    return normalize_root_vel_frame(root_vel_frame)


def get_stats_filename(dim, kind, root_vel_frame):
    variant = get_stats_variant(root_vel_frame)
    return f"{kind}_{dim}d_{variant}.npy"


def get_stats_path(data_root, dim, kind, root_vel_frame):
    return pjoin(data_root, get_stats_filename(dim, kind, root_vel_frame))


def resolve_stats_path(data_root, dim, kind, root_vel_frame, allow_legacy=True):
    preferred = get_stats_path(data_root, dim, kind, root_vel_frame)
    if os.path.exists(preferred):
        return preferred
    if allow_legacy:
        legacy = pjoin(data_root, f"{kind}_{dim}d.npy")
        if os.path.exists(legacy):
            return legacy
    return preferred


def require_stats_path(data_root, dim, kind, root_vel_frame, allow_legacy=True):
    path = resolve_stats_path(data_root, dim, kind, root_vel_frame, allow_legacy=allow_legacy)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        "\n".join(
            [
                f"Missing required normalization stats: {kind}_{dim}d ({normalize_root_vel_frame(root_vel_frame)})",
                f"- expected: {os.path.abspath(path)}",
                "Run the matching stats script before training/inference.",
            ]
        )
    )

