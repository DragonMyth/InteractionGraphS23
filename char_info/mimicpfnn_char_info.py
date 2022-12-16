import collections
import numpy as np

v_up = np.array([0.0, 1.0, 0.0])
v_face = np.array([0.0, 0.0, 1.0])
v_up_env = np.array([0.0, 1.0, 0.0])

Hips = -1
LHipJoint = 0
LeftUpLeg = 1
LeftLeg = 2
LeftFoot = 3
RHipJoint = 4
RightUpLeg = 5
RightLeg = 6
RightFoot = 7
LowerBack = 8
Spine = 9
Spine1 = 10
Neck = 11
Neck1 = 12
Head = 13
LeftShoulder = 14
LeftArm = 15
LeftForeArm = 16
LeftHand = 17
RightShoulder = 18
RightArm = 19
RightForeArm = 20
RightHand = 21

ROOT = Hips

end_effector_indices = [
    LeftHand, RightHand, LeftFoot, RightFoot
]

bvh_map = collections.OrderedDict()
bvh_map[Hips]            = 'Hips'
bvh_map[LHipJoint]       = 'LHipJoint'
bvh_map[LeftUpLeg]       = 'LeftUpLeg'
bvh_map[LeftLeg]         = 'LeftLeg'
bvh_map[LeftFoot]        = 'LeftFoot'
bvh_map[RHipJoint]       = 'RHipJoint'
bvh_map[RightUpLeg]      = 'RightUpLeg'
bvh_map[RightLeg]        = 'RightLeg'
bvh_map[RightFoot]       = 'RightFoot'
bvh_map[LowerBack]       = 'LowerBack'
bvh_map[Spine]           = 'Spine'
bvh_map[Spine1]          = 'Spine1'
bvh_map[Neck]            = 'Neck'
bvh_map[Neck1]           = 'Neck1'
bvh_map[Head]            = 'Head'
bvh_map[LeftShoulder]    = 'LeftShoulder'
bvh_map[LeftArm]         = 'LeftArm'
bvh_map[LeftForeArm]     = 'LeftForeArm'
bvh_map[LeftHand]        = None
bvh_map[RightShoulder]   = 'RightShoulder'
bvh_map[RightArm]        = 'RightArm'
bvh_map[RightForeArm]    = 'RightForeArm'
bvh_map[RightHand]       = None

bvh_map_inv = collections.OrderedDict()
bvh_map_inv['Hips']            = Hips
bvh_map_inv['LHipJoint']       = LHipJoint
bvh_map_inv['LeftUpLeg']       = LeftUpLeg
bvh_map_inv['LeftLeg']         = LeftLeg
bvh_map_inv['LeftFoot']        = LeftFoot
bvh_map_inv['LeftToeBase']     = None
bvh_map_inv['RHipJoint']       = RHipJoint
bvh_map_inv['RightUpLeg']      = RightUpLeg
bvh_map_inv['RightLeg']        = RightLeg
bvh_map_inv['RightFoot']       = RightFoot
bvh_map_inv['RightToeBase']    = None
bvh_map_inv['LowerBack']       = LowerBack
bvh_map_inv['Spine']           = Spine
bvh_map_inv['Spine1']          = Spine1
bvh_map_inv['Neck']            = Neck
bvh_map_inv['Neck1']           = Neck1
bvh_map_inv['Head']            = Head
bvh_map_inv['LeftShoulder']    = LeftShoulder
bvh_map_inv['LeftArm']         = LeftArm
bvh_map_inv['LeftForeArm']     = LeftForeArm
bvh_map_inv['LeftHand']        = None
bvh_map_inv['LeftFingerBase']  = None
bvh_map_inv['LeftHandIndex1']  = None
bvh_map_inv['LThumb']          = None
bvh_map_inv['RightShoulder']   = RightShoulder
bvh_map_inv['RightArm']        = RightArm
bvh_map_inv['RightForeArm']    = RightForeArm
bvh_map_inv['RightHand']       = None
bvh_map_inv['RightFingerBase'] = None
bvh_map_inv['RightHandIndex1'] = None
bvh_map_inv['RThumb']          = None

dof = {
    Hips : 6,
    LHipJoint : 4,
    LeftUpLeg : 4,
    LeftLeg : 4,
    LeftFoot : 4,
    RHipJoint : 4,
    RightUpLeg : 4,
    RightLeg : 4,
    RightFoot : 4,
    LowerBack : 4,
    Spine : 4,
    Spine1 : 4,
    Neck : 0,
    Neck1 : 4,
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

kp = {
    Hips : 0,
    LHipJoint : 400,
    LeftUpLeg : 400,
    LeftLeg : 200,
    LeftFoot : 100,
    RHipJoint : 400,
    RightUpLeg : 400,
    RightLeg : 200,
    RightFoot : 100,
    LowerBack : 400,
    Spine : 400,
    Spine1 : 400,
    Neck : 100,
    Neck1 : 100,
    Head : 100,
    LeftShoulder : 400,
    LeftArm : 400,
    LeftForeArm : 200,
    LeftHand : 0,
    RightShoulder : 400,
    RightArm : 400,
    RightForeArm : 200,
    RightHand : 0,
    }

kd = {}
for k, v in kp.items():
    kd[k] = 0.05 * v

cpd_ratio = 0.005

max_force = {
    Hips : 0,
    LHipJoint : 400,
    LeftUpLeg : 400,
    LeftLeg : 200,
    LeftFoot : 100,
    RHipJoint : 400,
    RightUpLeg : 400,
    RightLeg : 200,
    RightFoot : 100,
    LowerBack : 600,
    Spine : 600,
    Spine1 : 600,
    Neck : 100,
    Neck1 : 100,
    Head : 100,
    LeftShoulder : 300,
    LeftArm : 300,
    LeftForeArm : 150,
    LeftHand : 0,
    RightShoulder : 300,
    RightArm : 300,
    RightForeArm : 150,
    RightHand : 0,
    }

contact_allow_map = {
    Hips : False,
    LHipJoint : False,
    LeftUpLeg : False,
    LeftLeg : False,
    LeftFoot : True,
    RHipJoint : False,
    RightUpLeg : False,
    RightLeg : False,
    RightFoot : True,
    LowerBack : False,
    Spine : False,
    Spine1 : False,
    Neck : False,
    Neck1 : False,
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
    LHipJoint : 0.5,
    LeftUpLeg : 0.5,
    LeftLeg : 0.3,
    LeftFoot : 0.2,
    RHipJoint : 0.5,
    RightUpLeg : 0.5,
    RightLeg : 0.3,
    RightFoot : 0.2,
    LowerBack : 0.5,
    Spine : 0.5,
    Spine1 : 0.3,
    Neck : 0.3,
    Neck1 : 0.3,
    Head : 0.3,
    LeftShoulder : 0.5,
    LeftArm : 0.5,
    LeftForeArm : 0.3,
    LeftHand : 0.0,
    RightShoulder : 0.5,
    RightArm : 0.5,
    RightForeArm : 0.3,
    RightHand : 0.0,
    }

noise_pose_mu = {
    Hips : 0.0,
    LHipJoint : 0.0,
    LeftUpLeg : 0.0,
    LeftLeg : 0.0,
    LeftFoot : 0.0,
    RHipJoint : 0.0,
    RightUpLeg : 0.0,
    RightLeg : 0.0,
    RightFoot : 0.0,
    LowerBack : 0.0,
    Spine : 0.0,
    Spine1 : 0.0,
    Neck : 0.0,
    Neck1 : 0.0,
    Head : 0.0,
    LeftShoulder : 0.0,
    LeftArm : 0.0,
    LeftForeArm : 0.0,
    LeftHand : 0.0,
    RightShoulder : 0.0,
    RightArm : 0.0,
    RightForeArm : 0.0,
    RightHand : 0.0,
    }

noise_pose_sigma = {
    Hips : 0.1,
    LHipJoint : 0.1,
    LeftUpLeg : 0.1,
    LeftLeg : 0.1,
    LeftFoot : 0.1,
    RHipJoint : 0.1,
    RightUpLeg : 0.1,
    RightLeg : 0.1,
    RightFoot : 0.1,
    LowerBack : 0.1,
    Spine : 0.1,
    Spine1 : 0.1,
    Neck : 0.1,
    Neck1 : 0.1,
    Head : 0.1,
    LeftShoulder : 0.1,
    LeftArm : 0.1,
    LeftForeArm : 0.1,
    LeftHand : 0.1,
    RightShoulder : 0.1,
    RightArm : 0.1,
    RightForeArm : 0.1,
    RightHand : 0.1,
    }

noise_pose_lower = {
    Hips : -0.5,
    LHipJoint : -0.5,
    LeftUpLeg : -0.5,
    LeftLeg : -0.5,
    LeftFoot : -0.5,
    RHipJoint : -0.5,
    RightUpLeg : -0.5,
    RightLeg : -0.5,
    RightFoot : -0.5,
    LowerBack : -0.5,
    Spine : -0.5,
    Spine1 : -0.5,
    Neck : -0.5,
    Neck1 : -0.5,
    Head : -0.5,
    LeftShoulder : -0.5,
    LeftArm : -0.5,
    LeftForeArm : -0.5,
    LeftHand : -0.5,
    RightShoulder : -0.5,
    RightArm : -0.5,
    RightForeArm : -0.5,
    RightHand : -0.5,
    }

noise_pose_upper = {
    Hips : 0.5,
    LHipJoint : 0.5,
    LeftUpLeg : 0.5,
    LeftLeg : 0.5,
    LeftFoot : 0.5,
    RHipJoint : 0.5,
    RightUpLeg : 0.5,
    RightLeg : 0.5,
    RightFoot : 0.5,
    LowerBack : 0.5,
    Spine : 0.5,
    Spine1 : 0.5,
    Neck : 0.5,
    Neck1 : 0.5,
    Head : 0.5,
    LeftShoulder : 0.5,
    LeftArm : 0.5,
    LeftForeArm : 0.5,
    LeftHand : 0.5,
    RightShoulder : 0.5,
    RightArm : 0.5,
    RightForeArm : 0.5,
    RightHand : 0.5,
    }

collison_ignore_pairs = [
    (RightShoulder, LeftShoulder),
    (RightShoulder, Neck),
    (LeftShoulder, Neck),
    (RHipJoint, LHipJoint),
    (RHipJoint, LowerBack),
    (LHipJoint, LowerBack),
    (Spine1, Neck1)
]

friction_lateral = 0.8
friction_spinning = 0.3
restitution = 0.0

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight

interaction_mesh_samples = None