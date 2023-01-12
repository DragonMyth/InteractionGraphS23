import collections
import numpy as np
from fairmotion.ops import conversions

name = "Baxter"

'''
Mobile means the root joint is moving in the world space
The initial posisition orientation of the root joint are set by root_pos and root_ori, respectively.
'''
mobile = False
root_pos = np.array([0, -0.5, 0.95])
root_ori = conversions.Ax2R(-np.pi*0.5)

''' 
The up direction of the character w.r.t. its root joint.
The up direction in the world frame can be computed by dot(R_root, v_up), 
where R_root is the orientation of the root.
'''
v_up = np.array([0.0, 0.0, 1.0])
''' 
The facing direction of the character w.r.t. its root joint.
The facing direction in the world frame can be computed by dot(R_root, v_face), 
where R_root is the orientation of the root.
'''
v_face = np.array([1.0, 0.0, 0.0])
''' 
The up direction of the world frame, when the character holds its defalult posture (e.g. t-pose).
This information is useful/necessary when comparing a relationship between the character and its environment.
'''
v_up_env = np.array([0.0, 0.0, 1.0])
v_ax1_env = np.array([1.0, 0.0, 0.0])
v_ax2_env = np.array([0.0, 1.0, 0.0])

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
base = -1
torso = 0
left_torso_itb = 1
right_torso_itb = 2
pedestal = 3
head = 4
sonar_ring = 5
screen = 6
display = 7
head_camera = 8
dummyhead1 = 9
collision_head_link_1 = 10
collision_head_link_2 = 11
right_arm_mount = 12
right_upper_shoulder = 13
right_lower_shoulder = 14
right_upper_elbow = 15
right_upper_elbow_visual = 16
right_lower_elbow = 17
right_upper_forearm = 18
right_upper_forearm_visual = 19
right_arm_itb = 20
right_lower_forearm = 21
right_wrist = 22
right_hand = 23
right_hand_camera = 24
right_hand_camera_axis = 25
right_hand_range = 26
right_hand_accelerometer = 27
right_gripper_base = 28
right_gripper = 29
left_arm_mount = 30
left_upper_shoulder = 31
left_lower_shoulder = 32
left_upper_elbow = 33
left_upper_elbow_visual = 34
left_lower_elbow = 35
left_upper_forearm = 36
left_upper_forearm_visual = 37
left_arm_itb = 38
left_lower_forearm = 39
left_wrist = 40
left_hand = 41
left_hand_camera = 42
left_hand_camera_axis = 43
left_hand_range = 44
left_hand_accelerometer = 45
left_gripper_base = 46
left_gripper = 47


''' 
Definition of the root (base) joint
'''
ROOT = base

''' 
Definition of end effectors
'''
end_effector_indices = [
    right_gripper, 
    left_gripper, 
]

''' 
Mapping from joint indicies to names
'''
joint_name = collections.OrderedDict()

joint_name[base] = "base"
joint_name[torso] = "torso"
joint_name[left_torso_itb] = "left_torso_itb"
joint_name[right_torso_itb] = "right_torso_itb"
joint_name[pedestal] = "pedestal"
joint_name[head] = "head"
joint_name[sonar_ring] = "sonar_ring"
joint_name[screen] = "screen"
joint_name[display] = "display"
joint_name[head_camera] = "head_camera"
joint_name[dummyhead1] = "dummyhead1"
joint_name[collision_head_link_1] = "collision_head_link_1"
joint_name[collision_head_link_2] = "collision_head_link_2"
joint_name[right_arm_mount] = "right_arm_mount"
joint_name[right_upper_shoulder] = "right_upper_shoulder"
joint_name[right_lower_shoulder] = "right_lower_shoulder"
joint_name[right_upper_elbow] = "right_upper_elbow"
joint_name[right_upper_elbow_visual] = "right_upper_elbow_visual"
joint_name[right_lower_elbow] = "right_lower_elbow"
joint_name[right_upper_forearm] = "right_upper_forearm"
joint_name[right_upper_forearm_visual] = "right_upper_forearm_visual"
joint_name[right_arm_itb] = "right_arm_itb"
joint_name[right_lower_forearm] = "right_lower_forearm"
joint_name[right_wrist] = "right_wrist"
joint_name[right_hand] = "right_hand"
joint_name[right_hand_camera] = "right_hand_camera"
joint_name[right_hand_camera_axis] = "right_hand_camera_axis"
joint_name[right_hand_range] = "right_hand_range"
joint_name[right_hand_accelerometer] = "right_hand_accelerometer"
joint_name[right_gripper_base] = "right_gripper_base"
joint_name[right_gripper] = "right_gripper"
joint_name[left_arm_mount] = "left_arm_mount"
joint_name[left_upper_shoulder] = "left_upper_shoulder"
joint_name[left_lower_shoulder] = "left_lower_shoulder"
joint_name[left_upper_elbow] = "left_upper_elbow"
joint_name[left_upper_elbow_visual] = "left_upper_elbow_visual"
joint_name[left_lower_elbow] = "left_lower_elbow"
joint_name[left_upper_forearm] = "left_upper_forearm"
joint_name[left_upper_forearm_visual] = "left_upper_forearm_visual"
joint_name[left_arm_itb] = "left_arm_itb"
joint_name[left_lower_forearm] = "left_lower_forearm"
joint_name[left_wrist] = "left_wrist"
joint_name[left_hand] = "left_hand"
joint_name[left_hand_camera] = "left_hand_camera"
joint_name[left_hand_camera_axis] = "left_hand_camera_axis"
joint_name[left_hand_range] = "left_hand_range"
joint_name[left_hand_accelerometer] = "left_hand_accelerometer"
joint_name[left_gripper_base] = "left_gripper_base"
joint_name[left_gripper] = "left_gripper"

''' 
Mapping from joint names to indicies
'''
joint_idx = collections.OrderedDict()

for key, val in joint_name.items():
    joint_idx[val] = key

''' 
Mapping from character's joint indicies to bvh's joint names.
Some entry could have no mapping (by assigning None).
'''
# bvh_map = None
bvh_map = collections.OrderedDict()

# bvh_map[base] = None
# bvh_map[torso] = None
# bvh_map[left_torso_itb] = None
# bvh_map[right_torso_itb] = None
# bvh_map[pedestal] = None
# bvh_map[head] = "Head"
# bvh_map[sonar_ring] = None
# bvh_map[screen] = None
# bvh_map[display] = None
# bvh_map[head_camera] = None
# bvh_map[dummyhead1] = None
# bvh_map[collision_head_link_1] = None
# bvh_map[collision_head_link_2] = None
# bvh_map[right_arm_mount] = None
# bvh_map[right_upper_shoulder] = "RightArm"
# bvh_map[right_lower_shoulder] = "RightArm"
# bvh_map[right_upper_elbow] = "RightForeArm"
# bvh_map[right_upper_elbow_visual] = None
# bvh_map[right_lower_elbow] = "RightForeArm"
# bvh_map[right_upper_forearm] = "RightHand"
# bvh_map[right_upper_forearm_visual] = None
# bvh_map[right_arm_itb] = None
# bvh_map[right_lower_forearm] = "RightHand"
# bvh_map[right_wrist] = None
# bvh_map[right_hand] = None
# bvh_map[right_hand_camera] = None
# bvh_map[right_hand_camera_axis] = None
# bvh_map[right_hand_range] = None
# bvh_map[right_hand_accelerometer] = None
# bvh_map[right_gripper_base] = None
# bvh_map[right_gripper] = None
# bvh_map[left_arm_mount] = None
# bvh_map[left_upper_shoulder] = "LeftArm"
# bvh_map[left_lower_shoulder] = "LeftArm"
# bvh_map[left_upper_elbow] = "LeftForeArm"
# bvh_map[left_upper_elbow_visual] = None
# bvh_map[left_lower_elbow] = "LeftForeArm"
# bvh_map[left_upper_forearm] = "LeftHand"
# bvh_map[left_upper_forearm_visual] = None
# bvh_map[left_arm_itb] = None
# bvh_map[left_lower_forearm] = "LeftHand"
# bvh_map[left_wrist] = None
# bvh_map[left_hand] = None
# bvh_map[left_hand_camera] = None
# bvh_map[left_hand_camera_axis] = None
# bvh_map[left_hand_range] = None
# bvh_map[left_hand_accelerometer] = None
# bvh_map[left_gripper_base] = None
# bvh_map[left_gripper] = None

bvh_map[base] = None
bvh_map[torso] = None
bvh_map[left_torso_itb] = None
bvh_map[right_torso_itb] = None
bvh_map[pedestal] = None
bvh_map[head] = None
bvh_map[sonar_ring] = None
bvh_map[screen] = None
bvh_map[display] = None
bvh_map[head_camera] = None
bvh_map[dummyhead1] = None
bvh_map[collision_head_link_1] = None
bvh_map[collision_head_link_2] = None
bvh_map[right_arm_mount] = None
bvh_map[right_upper_shoulder] = None
bvh_map[right_lower_shoulder] = None
bvh_map[right_upper_elbow] = None
bvh_map[right_upper_elbow_visual] = None
bvh_map[right_lower_elbow] = None
bvh_map[right_upper_forearm] = None
bvh_map[right_upper_forearm_visual] = None
bvh_map[right_arm_itb] = None
bvh_map[right_lower_forearm] = None
bvh_map[right_wrist] = None
bvh_map[right_hand] = None
bvh_map[right_hand_camera] = None
bvh_map[right_hand_camera_axis] = None
bvh_map[right_hand_range] = None
bvh_map[right_hand_accelerometer] = None
bvh_map[right_gripper_base] = None
bvh_map[right_gripper] = None
bvh_map[left_arm_mount] = None
bvh_map[left_upper_shoulder] = None
bvh_map[left_lower_shoulder] = None
bvh_map[left_upper_elbow] = None
bvh_map[left_upper_elbow_visual] = None
bvh_map[left_lower_elbow] = None
bvh_map[left_upper_forearm] = None
bvh_map[left_upper_forearm_visual] = None
bvh_map[left_arm_itb] = None
bvh_map[left_lower_forearm] = None
bvh_map[left_wrist] = None
bvh_map[left_hand] = None
bvh_map[left_hand_camera] = None
bvh_map[left_hand_camera_axis] = None
bvh_map[left_hand_range] = None
bvh_map[left_hand_accelerometer] = None
bvh_map[left_gripper_base] = None
bvh_map[left_gripper] = None

''' 
Mapping from bvh's joint names to character's joint indicies.
Some entry could have no mapping (by assigning None).
'''
# bvh_map_inv = None
bvh_map_inv = collections.OrderedDict()

bvh_map_inv["Hips"] = None
bvh_map_inv["Spine"] = None
bvh_map_inv["Spine1"] = None
bvh_map_inv["Spine2"] = None
bvh_map_inv["Spine3"] = None
bvh_map_inv["Neck"] = None
bvh_map_inv["Head"] = None
bvh_map_inv["RightShoulder"] = None
bvh_map_inv["RightArm"] = None
bvh_map_inv["RightForeArm"] = None
bvh_map_inv["RightHand"] = None
bvh_map_inv["RightHandEnd"] = None
bvh_map_inv["RightHandThumb1"] = None
bvh_map_inv["LeftShoulder"] = None
bvh_map_inv["LeftArm"] = None
bvh_map_inv["LeftForeArm"] = None
bvh_map_inv["LeftHand"] = None
bvh_map_inv["LeftHandEnd"] = None
bvh_map_inv["LeftHandThumb1"] = None
bvh_map_inv["RightUpLeg"] = None
bvh_map_inv["RightLeg"] = None
bvh_map_inv["RightFoot"] = None
bvh_map_inv["RightToeBase"] = None
bvh_map_inv["LeftUpLeg"] = None
bvh_map_inv["LeftLeg"] = None
bvh_map_inv["LeftFoot"] = None
bvh_map_inv["LeftToeBase"] = None

dof = {
    base : 0,
    torso : 0,
    left_torso_itb : 0,
    right_torso_itb : 0,
    pedestal : 0,
    head : 1,
    sonar_ring : 0,
    screen : 0,
    display : 0,
    head_camera : 0,
    dummyhead1 : 0,
    collision_head_link_1 : 0,
    collision_head_link_2 : 0,
    right_arm_mount : 0,
    right_upper_shoulder : 1,
    right_lower_shoulder : 1,
    right_upper_elbow : 1,
    right_upper_elbow_visual : 0,
    right_lower_elbow : 1,
    right_upper_forearm : 1,
    right_upper_forearm_visual : 0,
    right_arm_itb : 0,
    right_lower_forearm : 1,
    right_wrist : 1,
    right_hand : 0,
    right_hand_camera : 0,
    right_hand_camera_axis : 0,
    right_hand_range : 0,
    right_hand_accelerometer : 0,
    right_gripper_base : 0,
    right_gripper : 0,
    left_arm_mount : 0,
    left_upper_shoulder : 1,
    left_lower_shoulder : 1,
    left_upper_elbow : 1,
    left_upper_elbow_visual : 0,
    left_lower_elbow : 1,
    left_upper_forearm : 1,
    left_upper_forearm_visual : 0,
    left_arm_itb : 0,
    left_lower_forearm : 1,
    left_wrist : 1,
    left_hand : 0,
    left_hand_camera : 0,
    left_hand_camera_axis : 0,
    left_hand_range : 0,
    left_hand_accelerometer : 0,
    left_gripper_base : 0,
    left_gripper : 0,
    }

''' 
Definition of PD gains
'''

kp = {}
kd = {}

kp['spd'] = {
    base : 0,
    torso : 0,
    left_torso_itb : 0,
    right_torso_itb : 0,
    pedestal : 0,
    head : 100,
    sonar_ring : 0,
    screen : 0,
    display : 0,
    head_camera : 0,
    dummyhead1 : 0,
    collision_head_link_1 : 0,
    collision_head_link_2 : 0,
    right_arm_mount : 0,
    right_upper_shoulder : 100,
    right_lower_shoulder : 100,
    right_upper_elbow : 100,
    right_upper_elbow_visual : 0,
    right_lower_elbow : 100,
    right_upper_forearm : 100,
    right_upper_forearm_visual : 0,
    right_arm_itb : 0,
    right_lower_forearm : 100,
    right_wrist : 100,
    right_hand : 0,
    right_hand_camera : 0,
    right_hand_camera_axis : 0,
    right_hand_range : 0,
    right_hand_accelerometer : 0,
    right_gripper_base : 0,
    right_gripper : 0,
    left_arm_mount : 0,
    left_upper_shoulder : 100,
    left_lower_shoulder : 100,
    left_upper_elbow : 100,
    left_upper_elbow_visual : 0,
    left_lower_elbow : 100,
    left_upper_forearm : 100,
    left_upper_forearm_visual : 0,
    left_arm_itb : 0,
    left_lower_forearm : 100,
    left_wrist : 100,
    left_hand : 0,
    left_hand_camera : 0,
    left_hand_camera_axis : 0,
    left_hand_range : 0,
    left_hand_accelerometer : 0,
    left_gripper_base : 0,
    left_gripper : 0,
    }

kd['spd'] = {}
for k, v in kp['spd'].items():
    kd['spd'][k] = 0.0 * v
kd['spd'][ROOT] = 0

''' 
Definition of PD gains (tuned for Contrained PD Controller).
"cpd_ratio * kp" and "cpd_ratio * kd" will be used respectively.
'''
cpd_ratio = 0.0002
kp['cpd'] = {}
kd['cpd'] = {}
kp['cp'] = {}
for k, v in kp['spd'].items():
    kp['cpd'][k] = cpd_ratio * v
    kp['cp'][k] = cpd_ratio * v
for k, v in kd['spd'].items():
    kd['cpd'][k] = cpd_ratio * v

''' 
Maximum forces that character can generate when PD controller is used.
'''
max_force = {
    base : 0,
    torso : 0,
    left_torso_itb : 0,
    right_torso_itb : 0,
    pedestal : 0,
    head : 500,
    sonar_ring : 0,
    screen : 0,
    display : 0,
    head_camera : 0,
    dummyhead1 : 0,
    collision_head_link_1 : 0,
    collision_head_link_2 : 0,
    right_arm_mount : 0,
    right_upper_shoulder : 500,
    right_lower_shoulder : 500,
    right_upper_elbow : 500,
    right_upper_elbow_visual : 0,
    right_lower_elbow : 500,
    right_upper_forearm : 500,
    right_upper_forearm_visual : 0,
    right_arm_itb : 0,
    right_lower_forearm : 500,
    right_wrist : 500,
    right_hand : 0,
    right_hand_camera : 0,
    right_hand_camera_axis : 0,
    right_hand_range : 0,
    right_hand_accelerometer : 0,
    right_gripper_base : 0,
    right_gripper : 0,
    left_arm_mount : 0,
    left_upper_shoulder : 500,
    left_lower_shoulder : 500,
    left_upper_elbow : 500,
    left_upper_elbow_visual : 0,
    left_lower_elbow : 500,
    left_upper_forearm : 500,
    left_upper_forearm_visual : 0,
    left_arm_itb : 0,
    left_lower_forearm : 500,
    left_wrist : 500,
    left_hand : 0,
    left_hand_camera : 0,
    left_hand_camera_axis : 0,
    left_hand_range : 0,
    left_hand_accelerometer : 0,
    left_gripper_base : 0,
    left_gripper : 0,
    }

contact_allow_map = {
    }

joint_weight = {
    base : 0,
    torso : 0,
    left_torso_itb : 0,
    right_torso_itb : 0,
    pedestal : 0,
    head : 1,
    sonar_ring : 0,
    screen : 0,
    display : 0,
    head_camera : 0,
    dummyhead1 : 0,
    collision_head_link_1 : 0,
    collision_head_link_2 : 0,
    right_arm_mount : 0,
    right_upper_shoulder : 1,
    right_lower_shoulder : 1,
    right_upper_elbow : 1,
    right_upper_elbow_visual : 0,
    right_lower_elbow : 1,
    right_upper_forearm : 1,
    right_upper_forearm_visual : 0,
    right_arm_itb : 0,
    right_lower_forearm : 1,
    right_wrist : 1,
    right_hand : 0,
    right_hand_camera : 0,
    right_hand_camera_axis : 0,
    right_hand_range : 0,
    right_hand_accelerometer : 0,
    right_gripper_base : 0,
    right_gripper : 0,
    left_arm_mount : 0,
    left_upper_shoulder : 1,
    left_lower_shoulder : 1,
    left_upper_elbow : 1,
    left_upper_elbow_visual : 0,
    left_lower_elbow : 1,
    left_upper_forearm : 1,
    left_upper_forearm_visual : 0,
    left_arm_itb : 0,
    left_lower_forearm : 1,
    left_wrist : 1,
    left_hand : 0,
    left_hand_camera : 0,
    left_hand_camera_axis : 0,
    left_hand_range : 0,
    left_hand_accelerometer : 0,
    left_gripper_base : 0,
    left_gripper : 0,
    }

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight

collison_ignore_pairs = [
    # (Spine, LeftShoulder),
    # (Spine, RightShoulder),
    # (Spine1, LeftShoulder),
    # (Spine1, RightShoulder),
    # (Neck, LeftShoulder),
    # (Neck, RightShoulder),
    # (LowerBack, LeftShoulder),
    # (LowerBack, RightShoulder),
    # (LHipJoint, RHipJoint),
    # (LHipJoint, LowerBack),
    # (RHipJoint, LowerBack),
    # (LHipJoint, Spine),
    # (RHipJoint, Spine),
    # (LeftShoulder, RightShoulder),
    # (Neck, Head),
]

friction_lateral = 0.8
friction_spinning = 0.3
restitution = 0.0