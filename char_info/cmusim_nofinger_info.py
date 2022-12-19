
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
RightShoulder = 20
RightArm = 21
RightForeArm = 22
RightHand = 23

end_effector_indices = [
    LeftHand, RightHand, LeftFoot, RightFoot
]

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
RightShoulder = 18
RightArm = 19

LeftForeArm = -10
LeftHand = -10
RightForeArm = -10
RightHand = -10

end_effector_indices = [
    LeftArm, RightArm, LeftFoot, RightFoot
]

# joint_indices = [
#     Hips,
#     LHipJoint,
#     LeftUpLeg,
#     LeftLeg,
#     LeftFoot,
#     LeftToeBase,
#     RHipJoint,
#     RightUpLeg,
#     RightLeg,
#     RightFoot,
#     RightToeBase,
#     LowerBack,
#     Spine,
#     Spine1,
#     Neck,
#     Neck1,
#     Head,
#     LeftShoulder,
#     LeftArm,
#     LeftForeArm,
#     LeftHand,
#     RightShoulder,
#     RightArm,
#     RightForeArm,
#     RightHand,
# ]

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
    RightShoulder : 'RightShoulder',
    RightArm : 'RightArm',
    RightForeArm : 'RightForeArm',
    RightHand : 'RightHand',
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
    RightShoulder : 4,
    RightArm : 4,
    RightForeArm : 4,
    RightHand : 4,
    }



 # self._kpOrg = [
 #        0, 0, 0, 0, 0, 0, 0, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
 #        400, 400, 400, 400, 400, 400, 400, 400, 300, 500, 500, 500, 500, 500, 400, 400, 400, 400,
 #        400, 400, 400, 400, 300
 #    ]
 #    self._kdOrg = [
 #        0, 0, 0, 0, 0, 0, 0, 500, 500, 500, 500, 10, 10, 10, 10, 50, 50, 50, 50, 50, 40, 40, 40,
 #        40, 40, 40, 40, 40, 30, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 30
 #    ]

kp = {
    Hips : 0,
    LHipJoint : 500,
    LeftUpLeg : 500,
    LeftLeg : 500,
    LeftFoot : 500,
    LeftToeBase : 500,
    RHipJoint : 500,
    RightUpLeg : 500,
    RightLeg : 500,
    RightFoot : 500,
    RightToeBase : 500,
    LowerBack : 500,
    Spine : 500,
    Spine1 : 500,
    Neck : 500,
    Neck1 : 500,
    Head : 500,
    LeftShoulder : 500,
    LeftArm : 500,
    LeftForeArm : 500,
    LeftHand : 500,
    RightShoulder : 500,
    RightArm : 500,
    RightForeArm : 500,
    RightHand : 500,
    }

kd = {
    Hips : 0,
    LHipJoint : 50,
    LeftUpLeg : 50,
    LeftLeg : 50,
    LeftFoot : 50,
    LeftToeBase : 50,
    RHipJoint : 50,
    RightUpLeg : 50,
    RightLeg : 50,
    RightFoot : 50,
    RightToeBase : 50,
    LowerBack : 50,
    Spine : 50,
    Spine1 : 50,
    Neck : 50,
    Neck1 : 50,
    Head : 50,
    LeftShoulder : 50,
    LeftArm : 50,
    LeftForeArm : 50,
    LeftHand : 50,
    RightShoulder : 50,
    RightArm : 50,
    RightForeArm : 50,
    RightHand : 50,
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
    RightShoulder : 1000,
    RightArm : 1000,
    RightForeArm : 1000,
    RightHand : 1000,
    }

# max_force = {
#     Hips : 0,
#     LHipJoint : 200,
#     LeftUpLeg : 200,
#     LeftLeg : 200,
#     LeftFoot : 100,
#     LeftToeBase : 50,
#     RHipJoint : 200,
#     RightUpLeg : 200,
#     RightLeg : 200,
#     RightFoot : 100,
#     RightToeBase : 50,
#     LowerBack : 200,
#     Spine : 200,
#     Spine1 : 200,
#     Neck : 50,
#     Neck1 : 50,
#     Head : 50,
#     LeftShoulder : 100,
#     LeftArm : 100,
#     LeftForeArm : 50,
#     LeftHand : 50,
#     RightShoulder : 100,
#     RightArm : 100,
#     RightForeArm : 50,
#     RightHand : 50,
#     }

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
    RightShoulder : False,
    RightArm : False,
    RightForeArm : False,
    RightHand : True,
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
    RightShoulder : 1.0,
    RightArm : 1.0,
    RightForeArm : 1.0,
    RightHand : 1.0,
    }