import collections
import numpy as np

name = "Lafan1"

''' 
The up direction of the character w.r.t. its root joint.
The up direction in the world frame can be computed by dot(R_root, v_up), 
where R_root is the orientation of the root.
'''
v_up = np.array([1.0, 0.0, 0.0])
''' 
The facing direction of the character w.r.t. its root joint.
The facing direction in the world frame can be computed by dot(R_root, v_face), 
where R_root is the orientation of the root.
'''
v_face = np.array([0.0, 1.0, 0.0])
''' 
The up direction of the world frame, when the character holds its defalult posture (e.g. t-pose).
This information is useful/necessary when comparing a relationship between the character and its environment.
'''
v_up_env = np.array([0.0, 1.0, 0.0])

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
Hips = -1

LeftUpLeg = 0
LeftLeg = 1
LeftFoot = 2
LeftToe = 3

RightUpLeg = 4
RightLeg = 5
RightFoot = 6
RightToe = 7

Spine = 8
Spine1 = 9
Spine2 = 10
Neck = 11
Head = 12

LeftShoulder = 13
LeftArm = 14
LeftForeArm = 15
LeftHand = 16

RightShoulder = 17
RightArm = 18
RightForeArm = 19
RightHand = 20

''' 
Definition of the root (base) joint
'''
ROOT = Hips

''' 
Definition of end effectors
'''
end_effector_indices = [
    LeftHand, 
    RightHand, 
    LeftFoot, 
    RightFoot
]

'''
Definition of interaction mesh vertex candicates
'''

interaction_joint_candidate = [
    Hips ,

    LeftUpLeg ,
    LeftLeg ,
    LeftFoot,
    LeftToe,

    RightUpLeg ,
    RightLeg,
    RightFoot ,
    RightToe,

    Spine,
    Spine1,
    Spine2,
    Neck,
    Head,

    LeftShoulder ,
    LeftArm,
    LeftForeArm ,
    LeftHand ,

    RightShoulder ,
    RightArm ,
    RightForeArm,
    RightHand ,
]

''' 
Mapping from joint indicies to names
'''
joint_name = collections.OrderedDict()

joint_name[Hips] = "Hips"
joint_name[LeftUpLeg] = "LeftUpLeg"
joint_name[LeftLeg] = "LeftLeg"
joint_name[LeftFoot] = "LeftFoot"
joint_name[LeftToe] = "LeftToe"
joint_name[RightUpLeg] = "RightUpLeg"
joint_name[RightLeg] = "RightLeg"
joint_name[RightFoot] = "RightFoot"
joint_name[RightToe] = "RightToe"
joint_name[Spine] = "Spine"
joint_name[Spine1] = "Spine1"
joint_name[Spine2] = "Spine2"
joint_name[Neck] = "Neck"
joint_name[Head] = "Head"
joint_name[LeftShoulder] = "LeftShoulder"
joint_name[LeftArm] = "LeftArm"
joint_name[LeftForeArm] = "LeftForeArm"
joint_name[LeftHand] = "LeftHand"
joint_name[RightShoulder] = "RightShoulder"
joint_name[RightArm] = "RightArm"
joint_name[RightForeArm] = "RightForeArm"
joint_name[RightHand] = "RightHand"

''' 
Mapping from joint names to indicies
'''
joint_idx = collections.OrderedDict()

joint_idx["Hips"] = Hips
joint_idx["LeftUpLeg"] = LeftUpLeg
joint_idx["LeftLeg"] = LeftLeg
joint_idx["LeftFoot"] = LeftFoot
joint_idx["LeftToe"] = LeftToe
joint_idx["RightUpLeg"] = RightUpLeg
joint_idx["RightLeg"] = RightLeg
joint_idx["RightFoot"] = RightFoot
joint_idx["RightToe"] = RightToe
joint_idx["Spine"] = Spine
joint_idx["Spine1"] = Spine1
joint_idx["Spine2"] = Spine2
joint_idx["Neck"] = Neck
joint_idx["Head"] = Head
joint_idx["LeftShoulder"] = LeftShoulder
joint_idx["LeftArm"] = LeftArm
joint_idx["LeftForeArm"] = LeftForeArm
joint_idx["LeftHand"] = LeftHand
joint_idx["RightShoulder"] = RightShoulder
joint_idx["RightArm"] = RightArm
joint_idx["RightForeArm"] = RightForeArm
joint_idx["RightHand"] = RightHand


''' 
Mapping from character's joint indicies to bvh's joint names.
Some entry could have no mapping (by assigning None).
'''
bvh_map = collections.OrderedDict()

bvh_map[Hips]            = 'Hips'
bvh_map[LeftUpLeg]       = 'LeftUpLeg'
bvh_map[LeftLeg]         = 'LeftLeg'
bvh_map[LeftFoot]        = 'LeftFoot'
bvh_map[LeftToe]         = None
bvh_map[RightUpLeg]      = 'RightUpLeg'
bvh_map[RightLeg]        = 'RightLeg'
bvh_map[RightFoot]       = 'RightFoot'
bvh_map[RightToe]        = None
bvh_map[Spine]           = 'Spine'
bvh_map[Spine1]          = 'Spine1'
bvh_map[Spine2]          = 'Spine2'
bvh_map[Neck]            = 'Neck'
bvh_map[Head]            = 'Head'
bvh_map[LeftShoulder]    = 'LeftShoulder'
bvh_map[LeftArm]         = 'LeftArm'
bvh_map[LeftForeArm]     = 'LeftForeArm'
bvh_map[LeftHand]        = None
bvh_map[RightShoulder]   = 'RightShoulder'
bvh_map[RightArm]        = 'RightArm'
bvh_map[RightForeArm]    = 'RightForeArm'
bvh_map[RightHand]       = None

''' 
Mapping from bvh's joint names to character's joint indicies.
Some entry could have no mapping (by assigning None).
'''
bvh_map_inv = collections.OrderedDict()

bvh_map_inv['Hips']            = Hips
bvh_map_inv['LeftUpLeg']       = LeftUpLeg
bvh_map_inv['LeftLeg']         = LeftLeg
bvh_map_inv['LeftFoot']        = LeftFoot
bvh_map_inv['LeftToe']         = None
bvh_map_inv['RightUpLeg']      = RightUpLeg
bvh_map_inv['RightLeg']        = RightLeg
bvh_map_inv['RightFoot']       = RightFoot
bvh_map_inv['RightToe']        = None
bvh_map_inv['Spine']           = Spine
bvh_map_inv['Spine1']          = Spine1
bvh_map_inv['Spine2']          = Spine2
bvh_map_inv['Neck']            = Neck
bvh_map_inv['Head']            = Head
bvh_map_inv['LeftShoulder']    = LeftShoulder
bvh_map_inv['LeftArm']         = LeftArm
bvh_map_inv['LeftForeArm']     = LeftForeArm
bvh_map_inv['LeftHand']        = None
bvh_map_inv['RightShoulder']   = RightShoulder
bvh_map_inv['RightArm']        = RightArm
bvh_map_inv['RightForeArm']    = RightForeArm
bvh_map_inv['RightHand']       = None

dof = {
    Hips : 6,
    LeftUpLeg : 4,
    LeftLeg : 4,
    LeftFoot : 4,
    LeftToe : 0,
    RightUpLeg : 4,
    RightLeg : 4,
    RightFoot : 4,
    RightToe : 0,
    Spine : 4,
    Spine1 : 4,
    Spine2 : 4,
    Neck : 4,
    Head : 0,
    LeftShoulder : 4,
    LeftArm : 4,
    LeftForeArm : 4,
    LeftHand : 0,
    RightShoulder : 4,
    RightArm : 4,
    RightForeArm : 4,
    RightHand : 0,
    }

''' 
Definition of PD gains (tuned for Stable PD Controller)
'''

kp = {
            "spd":{

    Hips : 0,
    LeftUpLeg : 500,
    LeftLeg : 500,
    LeftFoot : 500,
    LeftToe : 0,
    RightUpLeg : 500,
    RightLeg : 500,
    RightFoot : 500,
    RightToe : 0,
    Spine : 500,
    Spine1 : 500,
    Spine2 : 500,
    Neck : 500,
    Head : 500,
    LeftShoulder : 500,
    LeftArm : 500,
    LeftForeArm : 500,
    LeftHand : 0,
    RightShoulder : 500,
    RightArm : 500,
    RightForeArm : 500,
    RightHand : 0,
    }
}


kd = {"spd":{}}
for k, v in kp['spd'].items():
    kd['spd'][k] = 0.05 * v
kd['spd'][ROOT] = 0

''' 
Definition of PD gains (tuned for Contrained PD Controller).
"cpd_ratio * kp" and "cpd_ratio * kd" will be used respectively.
'''
cpd_ratio = 0.0002


''' 
Maximum forces that character can generate when PD controller is used.
'''
max_force = {
    Hips : 0,
    LeftUpLeg : 1000,
    LeftLeg : 800,
    LeftFoot : 600,
    LeftToe : 0,
    RightUpLeg : 1000,
    RightLeg : 800,
    RightFoot : 600,
    RightToe : 0,
    Spine : 800,
    Spine1 : 800,
    Spine2 : 800,
    Neck : 400,
    Head : 400,
    LeftShoulder : 600,
    LeftArm : 500,
    LeftForeArm : 400,
    LeftHand : 0,
    RightShoulder : 600,
    RightArm : 500,
    RightForeArm : 400,
    RightHand : 0,
    }

contact_allow_map = {
    Hips : False,
    LeftUpLeg : False,
    LeftLeg : False,
    LeftFoot : True,
    LeftToe : True,
    RightUpLeg : False,
    RightLeg : False,
    RightFoot : True,
    RightToe : True,
    Spine : False,
    Spine1 : False,
    Spine2 : False,
    Neck : False,
    Head : False,
    LeftShoulder : False,
    LeftArm : False,
    LeftForeArm : False,
    LeftHand : False,
    RightShoulder : False,
    RightArm : False,
    RightForeArm : False,
    RightHand : False,
    }

joint_weight = {
    Hips : 1.0,
    LeftUpLeg : 0.5,
    LeftLeg : 0.3,
    LeftFoot : 0.2,
    LeftToe : 0.1,
    RightUpLeg : 0.5,
    RightLeg : 0.3,
    RightFoot : 0.2,
    RightToe : 0.1,
    Spine : 0.5,
    Spine1 : 0.3,
    Spine2 : 0.3,
    Neck : 0.3,
    Head : 0.3,
    LeftShoulder : 0.5,
    LeftArm : 0.5,
    LeftForeArm : 0.3,
    LeftHand : 0.2,
    RightShoulder : 0.5,
    RightArm : 0.5,
    RightForeArm : 0.3,
    RightHand : 0.2,
    }

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight

collison_ignore_pairs = [
    (Spine, LeftShoulder),
    (Spine, RightShoulder),
    (Spine1, LeftShoulder),
    (Spine1, RightShoulder),
    (Spine2, LeftShoulder),
    (Spine2, RightShoulder),
    (LeftShoulder, RightShoulder),
]

friction_lateral = 0.8
friction_spinning = 0.0
restitution = 0.0