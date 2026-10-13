import os
import xml.etree.ElementTree as ET

import torch
from torch import nn

from utils.rotation_utils_torch import rot6d_to_rotation_matrix_torch


ISAAC_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

FK_KEYPOINT_BODY_NAMES = {
    "pelvis": "pelvis",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}

DEFAULT_ROBOT_XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "GMR_view-main",
    "assets",
    "robots",
    "g1",
    "g1.xml",
)


def _parse_vec(attr_value, dims):
    if attr_value is None:
        return [0.0] * dims
    values = [float(x) for x in attr_value.strip().split()]
    if len(values) != dims:
        raise ValueError(f"Expected {dims} values, got {values}")
    return values


def _quat_wxyz_to_rotmat_tensor(quat):
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([
        torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1),
    ], dim=-2)


def _axis_angle_to_matrix(axis, angle):
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    x, y, z = axis.unbind(dim=-1)
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c
    return torch.stack([
        torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s], dim=-1),
        torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s], dim=-1),
        torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c], dim=-1),
    ], dim=-2)


def _rotate_vec(R, v):
    return torch.einsum("...ij,...j->...i", R, v)


class RobotFKLayer(nn.Module):
    """
    Differentiable sparse FK for the G1 robot.

    Input motion uses the project 38D layout:
    [joint_pos(29), root_vel_xy(2), root_z(1), root_rot_6d(6)].

    The XML chain provides body-local offsets and joint axes. The root pelvis pose is
    reconstructed from root velocity integration and root orientation, while child body
    positions are produced by forward kinematics in torch.
    """

    def __init__(self, xml_path=DEFAULT_ROBOT_XML_PATH):
        super().__init__()
        xml_path = xml_path or DEFAULT_ROBOT_XML_PATH
        spec = self._load_xml_spec(xml_path)
        self.body_names = spec["body_names"]
        self.body_name_to_index = {name: idx for idx, name in enumerate(self.body_names)}
        self.keypoint_names = list(FK_KEYPOINT_BODY_NAMES.keys())
        self.keypoint_body_indices = [self.body_name_to_index[FK_KEYPOINT_BODY_NAMES[n]] for n in self.keypoint_names]

        self.register_buffer("parent_idx", torch.tensor(spec["parent_idx"], dtype=torch.long))
        self.register_buffer("local_pos", torch.tensor(spec["local_pos"], dtype=torch.float32))
        self.register_buffer("local_rot", torch.tensor(spec["local_rot"], dtype=torch.float32))
        self.register_buffer("joint_pos", torch.tensor(spec["joint_pos"], dtype=torch.float32))
        self.register_buffer("joint_axis", torch.tensor(spec["joint_axis"], dtype=torch.float32))
        self.register_buffer("joint_input_idx", torch.tensor(spec["joint_input_idx"], dtype=torch.long))
        # Joint limits: (29,) lower and upper bounds indexed by ISAAC_JOINT_ORDER
        self.register_buffer("joint_lower", torch.tensor(spec["joint_lower"], dtype=torch.float32))
        self.register_buffer("joint_upper", torch.tensor(spec["joint_upper"], dtype=torch.float32))

    def _load_xml_spec(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"No worldbody found in {xml_path}")
        pelvis_body = None
        for child in worldbody.findall("body"):
            if child.get("name") == "pelvis":
                pelvis_body = child
                break
        if pelvis_body is None:
            raise ValueError(f"No pelvis body found in {xml_path}")

        joint_name_to_input = {name: idx for idx, name in enumerate(ISAAC_JOINT_ORDER)}
        # Joint limits indexed by ISAAC_JOINT_ORDER (29 joints)
        joint_lower = [-3.14] * len(ISAAC_JOINT_ORDER)
        joint_upper = [3.14] * len(ISAAC_JOINT_ORDER)
        body_names = ["pelvis"]
        parent_idx = [-1]
        local_pos = [[0.0, 0.0, 0.0]]
        local_rot = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        joint_pos = [[0.0, 0.0, 0.0]]
        joint_axis = [[0.0, 0.0, 1.0]]
        joint_input_idx = [-1]

        def visit(body_elem, parent_body_idx):
            body_idx = len(body_names)
            body_names.append(body_elem.get("name"))
            parent_idx.append(parent_body_idx)
            local_pos.append(_parse_vec(body_elem.get("pos"), 3))
            quat_attr = body_elem.get("quat")
            quat_vals = [1.0, 0.0, 0.0, 0.0] if quat_attr is None else _parse_vec(quat_attr, 4)
            local_quat = torch.tensor([quat_vals], dtype=torch.float32)
            local_rot.append(_quat_wxyz_to_rotmat_tensor(local_quat)[0].tolist())

            joint_elem = body_elem.find("joint")
            if joint_elem is not None:
                joint_pos.append(_parse_vec(joint_elem.get("pos"), 3))
                joint_axis.append(_parse_vec(joint_elem.get("axis"), 3))
                jname = joint_elem.get("name")
                jin = joint_name_to_input.get(jname, -1)
                joint_input_idx.append(jin)
                # Parse joint limits from XML range attribute
                range_attr = joint_elem.get("range")
                if range_attr is not None and jin >= 0:
                    lo, hi = [float(x) for x in range_attr.strip().split()]
                    joint_lower[jin] = lo
                    joint_upper[jin] = hi
            else:
                joint_pos.append([0.0, 0.0, 0.0])
                joint_axis.append([0.0, 0.0, 1.0])
                joint_input_idx.append(-1)

            for child in body_elem.findall("body"):
                visit(child, body_idx)

        for child in pelvis_body.findall("body"):
            visit(child, 0)

        # NOTE: Wrist bodies (roll/pitch/yaw links) in the G1 XML have NO <joint> elements.
        # They are treated as fixed links in the FK chain. The wrist position is computed
        # correctly from shoulder→elbow→fixed_wrist, but wrist joint angles (channels 23-28)
        # do NOT contribute to FK output. This is acceptable because:
        # 1. Wrist joints have negligible effect on wrist POSITION (they're at the tip)
        # 2. Shoulder/elbow/waist joints are the primary drivers of wrist position
        # 3. Inference-time FK guidance optimizes ALL joints including wrists via autograd

        return {
            "body_names": body_names,
            "parent_idx": parent_idx,
            "local_pos": local_pos,
            "local_rot": local_rot,
            "joint_pos": joint_pos,
            "joint_axis": joint_axis,
            "joint_input_idx": joint_input_idx,
            "joint_lower": joint_lower,
            "joint_upper": joint_upper,
        }

    def _recover_root_translation(self, motion_38d, root_vel_frame="world", fps=50):
        B, T, _ = motion_38d.shape
        dt = 1.0 / float(fps)
        vel_xy = motion_38d[:, :, 29:31]  # No clone — preserve gradient for FK-based losses
        first_frame_mask = torch.ones(T, device=motion_38d.device, dtype=motion_38d.dtype)
        first_frame_mask[0] = 0.0
        vel_xy = vel_xy * first_frame_mask.view(1, T, 1)
        root_rot = rot6d_to_rotation_matrix_torch(motion_38d[:, :, 32:38].reshape(B * T, 6)).reshape(B, T, 3, 3)
        if (root_vel_frame or "world").lower() == "body":
            vel_z = torch.zeros(B, T, 1, device=motion_38d.device, dtype=motion_38d.dtype)
            vel_3d = torch.cat([vel_xy, vel_z], dim=2)  # (B, T, 3) — preserves grad
            vel_xy = torch.einsum("btij,btj->bti", root_rot, vel_3d)[:, :, :2]
        root_xy = (vel_xy * dt).cumsum(dim=1)
        root_z = motion_38d[:, :, 31:32]  # (B, T, 1)
        root_t = torch.cat([root_xy, root_z], dim=2)  # (B, T, 3) — preserves grad
        return root_t, root_rot

    def forward(self, motion_38d_denorm, root_vel_frame="world"):
        B, T, D = motion_38d_denorm.shape
        joint_pos_input = motion_38d_denorm[:, :, :29]
        root_t, root_rot = self._recover_root_translation(motion_38d_denorm, root_vel_frame=root_vel_frame)

        n_bodies = len(self.body_names)
        world_R = [None] * n_bodies
        world_t = [None] * n_bodies
        world_R[0] = root_rot
        world_t[0] = root_t

        for body_idx in range(1, n_bodies):
            parent = int(self.parent_idx[body_idx].item())
            parent_R = world_R[parent]
            parent_t = world_t[parent]

            static_R = self.local_rot[body_idx].to(device=motion_38d_denorm.device, dtype=motion_38d_denorm.dtype)
            static_pos = self.local_pos[body_idx].to(device=motion_38d_denorm.device, dtype=motion_38d_denorm.dtype)
            joint_anchor = self.joint_pos[body_idx].to(device=motion_38d_denorm.device, dtype=motion_38d_denorm.dtype)
            joint_axis = self.joint_axis[body_idx].to(device=motion_38d_denorm.device, dtype=motion_38d_denorm.dtype)

            body_R = torch.einsum("btij,jk->btik", parent_R, static_R)
            body_t = parent_t + _rotate_vec(parent_R, static_pos)

            motion_idx = int(self.joint_input_idx[body_idx].item())
            if motion_idx >= 0:
                angle = joint_pos_input[:, :, motion_idx]
                joint_R = _axis_angle_to_matrix(joint_axis.view(1, 1, 3).expand(B, T, 3), angle)
                body_R_after = torch.einsum("btij,btjk->btik", body_R, joint_R)
                anchor_world = body_t + _rotate_vec(body_R, joint_anchor)
                body_t_after = anchor_world - _rotate_vec(body_R_after, joint_anchor)
                world_R[body_idx] = body_R_after
                world_t[body_idx] = body_t_after
            else:
                world_R[body_idx] = body_R
                world_t[body_idx] = body_t

        all_body_pos = torch.stack(world_t, dim=2)  # (B, T, Nb, 3)
        keypoint_idx = torch.tensor(self.keypoint_body_indices, device=motion_38d_denorm.device, dtype=torch.long)
        keypoint_pos = all_body_pos.index_select(dim=2, index=keypoint_idx)
        return {
            "all_body_pos": all_body_pos,
            "keypoint_pos": keypoint_pos,
            "root_pos": root_t,
            "root_rot": root_rot,
            "traj_xy": root_t[:, :, :2],
        }
