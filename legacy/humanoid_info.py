# import pybullet

# full_names = [
#     "Hips",
#     "LHipJoint",
#     "LeftUpLeg",
#     "LeftLeg",
#     "LeftFoot",
#     "LeftToeBase",
#     "RHipJoint",
#     "RightUpLeg",
#     "RightLeg",
#     "RightFoot",
#     "RightToeBase",
#     "LowerBack",
#     "Spine",
#     "Spine1",
#     "Neck",
#     "Neck1",
#     "Head",
#     "LeftShoulder",
#     "LeftArm",
#     "LeftForeArm",
#     "LeftHand",
#     "LeftFingerBase",
#     "LeftHandIndex1",
#     "LThumb",
#     "RightShoulder",
#     "RightArm",
#     "RightForeArm",
#     "RightHand",
#     "RightFingerBase",
#     "RightHandIndex1",
#     "RThumb",
#     ]

# class HumanoidInfo():
#     def __init__(self, model):
#         joint_indices = {}
#         to_bvh_name = {}

Hips = -1
LHipJoint = 0
LeftUpLeg = 1
LeftLeg = 2
LeftFoot = 3
LeftToeBase = 4
RHipJoint = 5
RightUpLeg = 6
RightLeg = 7
RightFoot = 8
RightToeBase = 9
LowerBack = 10
Spine = 11
Spine1 = 12
Neck = 13
Neck1 = 14
Head = 15
LeftShoulder = 16
LeftArm = 17
LeftForeArm = 18
LeftHand = 19
LeftFingerBase = 20
LeftHandIndex1 = 21
LThumb = 22
RightShoulder = 23
RightArm = 24
RightForeArm = 25
RightHand = 26
RightFingerBase = 27
RightHandIndex1 = 28
RThumb = 29

end_effector_indices = [
    LeftHandIndex1, RightHandIndex1, LeftFoot, RightFoot
]

joint_indices = [
    Hips,
    LHipJoint,
    LeftUpLeg,
    LeftLeg,
    LeftFoot,
    LeftToeBase,
    RHipJoint,
    RightUpLeg,
    RightLeg,
    RightFoot,
    RightToeBase,
    LowerBack,
    Spine,
    Spine1,
    Neck,
    Neck1,
    Head,
    LeftShoulder,
    LeftArm,
    LeftForeArm,
    LeftHand,
    LeftFingerBase,
    LeftHandIndex1,
    LThumb,
    RightShoulder,
    RightArm,
    RightForeArm,
    RightHand,
    RightFingerBase,
    RightHandIndex1,
    RThumb,
]

bvh_map = {
    Hips : 'Hips',
    LHipJoint : 'LHipJoint',
    LeftUpLeg : 'LeftUpLeg',
    LeftLeg : 'LeftLeg',
    LeftFoot : 'LeftFoot',
    LeftToeBase : 'LeftToeBase',
    RHipJoint : 'RHipJoint',
    RightUpLeg : 'RightUpLeg',
    RightLeg : 'RightLeg',
    RightFoot : 'RightFoot',
    RightToeBase : 'RightToeBase',
    LowerBack : 'LowerBack',
    Spine : 'Spine',
    Spine1 : 'Spine1',
    Neck : 'Neck',
    Neck1 : 'Neck1',
    Head : 'Head',
    LeftShoulder : 'LeftShoulder',
    LeftArm : 'LeftArm',
    LeftForeArm : 'LeftForeArm',
    LeftHand : 'LeftHand',
    LeftFingerBase : 'LeftFingerBase',
    LeftHandIndex1 : 'LeftHandIndex1',
    LThumb : 'LThumb',
    RightShoulder : 'RightShoulder',
    RightArm : 'RightArm',
    RightForeArm : 'RightForeArm',
    RightHand : 'RightHand',
    RightFingerBase : 'RightFingerBase',
    RightHandIndex1 : 'RightHandIndex1',
    RThumb : 'RThumb',
    }

dof = {
    Hips : 6,
    LHipJoint : 4,
    LeftUpLeg : 4,
    LeftLeg : 4,
    LeftFoot : 4,
    LeftToeBase : 4,
    RHipJoint : 4,
    RightUpLeg : 4,
    RightLeg : 4,
    RightFoot : 4,
    RightToeBase : 4,
    LowerBack : 4,
    Spine : 4,
    Spine1 : 4,
    Neck : 4,
    Neck1 : 4,
    Head : 4,
    LeftShoulder : 4,
    LeftArm : 4,
    LeftForeArm : 4,
    LeftHand : 4,
    LeftFingerBase : 4,
    LeftHandIndex1 : 4,
    LThumb : 4,
    RightShoulder : 4,
    RightArm : 4,
    RightForeArm : 4,
    RightHand : 4,
    RightFingerBase : 4,
    RightHandIndex1 : 4,
    RThumb : 4,
    }



 # self._kpOrg = [
 #        0, 0, 0, 0, 0, 0, 0, 1000, 1000, 1000, 1000, 100, 100, 100, 100, 500, 500, 500, 500, 500,
 #        400, 400, 400, 400, 400, 400, 400, 400, 300, 500, 500, 500, 500, 500, 400, 400, 400, 400,
 #        400, 400, 400, 400, 300
 #    ]
 #    self._kdOrg = [
 #        0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 10, 10, 10, 10, 50, 50, 50, 50, 50, 40, 40, 40,
 #        40, 40, 40, 40, 40, 30, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 30
 #    ]

kp = {
    Hips : 0,
    LHipJoint : 1000,
    LeftUpLeg : 1000,
    LeftLeg : 1000,
    LeftFoot : 1000,
    LeftToeBase : 1000,
    RHipJoint : 1000,
    RightUpLeg : 1000,
    RightLeg : 1000,
    RightFoot : 1000,
    RightToeBase : 1000,
    LowerBack : 1000,
    Spine : 1000,
    Spine1 : 1000,
    Neck : 1000,
    Neck1 : 1000,
    Head : 1000,
    LeftShoulder : 1000,
    LeftArm : 1000,
    LeftForeArm : 1000,
    LeftHand : 1000,
    LeftFingerBase : 1000,
    LeftHandIndex1 : 1000,
    LThumb : 1000,
    RightShoulder : 1000,
    RightArm : 1000,
    RightForeArm : 1000,
    RightHand : 1000,
    RightFingerBase : 1000,
    RightHandIndex1 : 1000,
    RThumb : 1000,
    }

kd = {
    Hips : 0,
    LHipJoint : 100,
    LeftUpLeg : 100,
    LeftLeg : 100,
    LeftFoot : 100,
    LeftToeBase : 100,
    RHipJoint : 100,
    RightUpLeg : 100,
    RightLeg : 100,
    RightFoot : 100,
    RightToeBase : 100,
    LowerBack : 100,
    Spine : 100,
    Spine1 : 100,
    Neck : 100,
    Neck1 : 100,
    Head : 100,
    LeftShoulder : 100,
    LeftArm : 100,
    LeftForeArm : 100,
    LeftHand : 100,
    LeftFingerBase : 100,
    LeftHandIndex1 : 100,
    LThumb : 100,
    RightShoulder : 100,
    RightArm : 100,
    RightForeArm : 100,
    RightHand : 100,
    RightFingerBase : 100,
    RightHandIndex1 : 100,
    RThumb : 100,
    }

max_force = {
    Hips : 0,
    LHipJoint : 1000,
    LeftUpLeg : 1000,
    LeftLeg : 1000,
    LeftFoot : 1000,
    LeftToeBase : 1000,
    RHipJoint : 1000,
    RightUpLeg : 1000,
    RightLeg : 1000,
    RightFoot : 1000,
    RightToeBase : 1000,
    LowerBack : 1000,
    Spine : 1000,
    Spine1 : 1000,
    Neck : 1000,
    Neck1 : 1000,
    Head : 1000,
    LeftShoulder : 1000,
    LeftArm : 1000,
    LeftForeArm : 1000,
    LeftHand : 1000,
    LeftFingerBase : 1000,
    LeftHandIndex1 : 1000,
    LThumb : 1000,
    RightShoulder : 1000,
    RightArm : 1000,
    RightForeArm : 1000,
    RightHand : 1000,
    RightFingerBase : 1000,
    RightHandIndex1 : 1000,
    RThumb : 1000,
    }

contact_allow_map = {
    Hips : False,
    LHipJoint : False,
    LeftUpLeg : False,
    LeftLeg : False,
    LeftFoot : True,
    LeftToeBase : True,
    RHipJoint : False,
    RightUpLeg : False,
    RightLeg : False,
    RightFoot : True,
    RightToeBase : True,
    LowerBack : False,
    Spine : False,
    Spine1 : False,
    Neck : False,
    Neck1 : False,
    Head : False,
    LeftShoulder : False,
    LeftArm : False,
    LeftForeArm : False,
    LeftHand : True,
    LeftFingerBase : True,
    LeftHandIndex1 : True,
    LThumb : True,
    RightShoulder : False,
    RightArm : False,
    RightForeArm : False,
    RightHand : True,
    RightFingerBase : True,
    RightHandIndex1 : True,
    RThumb : True,
    }

contact_allow_map = {
    Hips : False,
    LHipJoint : False,
    LeftUpLeg : False,
    LeftLeg : False,
    LeftFoot : True,
    LeftToeBase : True,
    RHipJoint : False,
    RightUpLeg : False,
    RightLeg : False,
    RightFoot : True,
    RightToeBase : True,
    LowerBack : False,
    Spine : False,
    Spine1 : False,
    Neck : False,
    Neck1 : False,
    Head : False,
    LeftShoulder : False,
    LeftArm : False,
    LeftForeArm : False,
    LeftHand : True,
    LeftFingerBase : True,
    LeftHandIndex1 : True,
    LThumb : True,
    RightShoulder : False,
    RightArm : False,
    RightForeArm : False,
    RightHand : True,
    RightFingerBase : True,
    RightHandIndex1 : True,
    RThumb : True,
    }

joint_weight = {
    Hips : 1.0,
    LHipJoint : 1.0,
    LeftUpLeg : 1.0,
    LeftLeg : 1.0,
    LeftFoot : 1.0,
    LeftToeBase : 1.0,
    RHipJoint : 1.0,
    RightUpLeg : 1.0,
    RightLeg : 1.0,
    RightFoot : 1.0,
    RightToeBase : 1.0,
    LowerBack : 1.0,
    Spine : 1.0,
    Spine1 : 1.0,
    Neck : 1.0,
    Neck1 : 1.0,
    Head : 1.0,
    LeftShoulder : 1.0,
    LeftArm : 1.0,
    LeftForeArm : 1.0,
    LeftHand : 1.0,
    LeftFingerBase : 1.0,
    LeftHandIndex1 : 1.0,
    LThumb : 1.0,
    RightShoulder : 1.0,
    RightArm : 1.0,
    RightForeArm : 1.0,
    RightHand : 1.0,
    RightFingerBase : 1.0,
    RightHandIndex1 : 1.0,
    RThumb : 1.0,
    }