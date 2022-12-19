import collections
import numpy as np

v_up = np.array([0.0, 0.0, 1.0])
v_face = np.array([0.0, 0.0, 1.0])
v_up_env = np.array([0.0, 0.0, 1.0])

pelvis      = -1
lfemur      = 0
ltibia      = 1
lfoot       = 2
rfemur      = 3
rtibia      = 4
rfoot       = 5
lowerback   = 6
upperback   = 7
lclavicle   = 8
lhumerus    = 9
lradius     = 10
lhand       = 11
rclavicle   = 12
rhumerus    = 13
rradius     = 14
rhand       = 15
lowerneck   = 16
neck        = 17

ROOT = pelvis

end_effector_indices = [
    lhand, rhand, lfoot, rfoot,
]

bvh_map = collections.OrderedDict()

bvh_map[pelvis] = "Hips"
bvh_map[lfemur] = "LeftUpLeg"
bvh_map[ltibia] = "LeftLeg"
bvh_map[lfoot] = "LeftFoot"
bvh_map[rfemur] = "RightUpLeg"
bvh_map[rtibia] = "RightLeg"
bvh_map[rfoot] = "RightFoot"
bvh_map[lowerneck] = "Spine3"
bvh_map[neck] = "Head"
bvh_map[lowerback] = "Spine"
bvh_map[upperback] = "Spine2"
bvh_map[lclavicle] = "LeftShoulder"
bvh_map[lhumerus] = "LeftArm"
bvh_map[lradius] = "LeftForeArm"
bvh_map[lhand] = "LeftHand"
bvh_map[rclavicle] = "RightShoulder"
bvh_map[rhumerus] = "RightArm"
bvh_map[rradius] = "RightForeArm"
bvh_map[rhand] = "RightHand"

bvh_map_inv = collections.OrderedDict()

bvh_map_inv["Hips"] = pelvis
bvh_map_inv["Spine"] = lowerback
bvh_map_inv["Spine1"] = None
bvh_map_inv["Spine2"] = upperback
bvh_map_inv["Spine3"] = None
bvh_map_inv["neck"] = None
bvh_map_inv["Head"] = neck
bvh_map_inv["RightShoulder"] = rclavicle
bvh_map_inv["RightArm"] = rhumerus
bvh_map_inv["RightForeArm"] = rradius
bvh_map_inv["RightHand"] = rhand
bvh_map_inv["RightHandEnd"] = None
bvh_map_inv["RightHandThumb1"] = None
bvh_map_inv["LeftShoulder"] = lclavicle
bvh_map_inv["LeftArm"] = lhumerus
bvh_map_inv["LeftForeArm"] = lradius
bvh_map_inv["LeftHand"] = lhand
bvh_map_inv["LeftHandEnd"] = None
bvh_map_inv["LeftHandThumb1"] = None
bvh_map_inv["RightUpLeg"] = rfemur
bvh_map_inv["RightLeg"] = rtibia
bvh_map_inv["RightFoot"] = rfoot
bvh_map_inv["RightToeBase"] = None
bvh_map_inv["LeftUpLeg"] = lfemur
bvh_map_inv["LeftLeg"] = ltibia
bvh_map_inv["LeftFoot"] = lfoot
bvh_map_inv["LeftToeBase"] = None

dof = {
    pelvis : 6,
    lfemur : 3,
    ltibia : 1,
    lfoot : 3,
    rfemur : 3,
    rtibia : 1,
    rfoot : 3,
    lowerneck : 3,
    neck : 0,
    lowerback : 3,
    upperback : 3,
    lclavicle : 3,
    lhumerus : 3,
    lradius : 3,
    lhand : 0,
    rclavicle : 3,
    rhumerus : 3,
    rradius : 3,
    rhand : 0,
    }

kp = {
     "spd":{
    pelvis : 0,
    lfemur : 500,
    ltibia : 400,
    lfoot : 300,
    rfemur : 500,
    rtibia : 400,
    rfoot : 300,
    lowerneck : 200,
    neck : 200,
    lowerback : 500,
    upperback : 500,
    lclavicle : 400,
    lhumerus : 400,
    lradius : 300,
    lhand : 300,
    rclavicle : 400,
    rhumerus : 400,
    rradius : 300,
    rhand : 300,
    }
}

kd = {"spd":{}}
for k, v in kp['spd'].items():
    kd['spd'][k] = 0.1 * v
kd['spd'][ROOT] = 0

cpd_ratio = 0.0002

max_force = {
    pelvis : 0,
    lfemur : 300,
    ltibia : 200,
    lfoot : 100,
    rfemur : 300,
    rtibia : 200,
    rfoot : 100,
    lowerneck : 100,
    neck : 100,
    lowerback : 300,
    upperback : 300,
    lclavicle : 200,
    lhumerus : 200,
    lradius : 150,
    lhand : 100,
    rclavicle : 200,
    rhumerus : 200,
    rradius : 150,
    rhand : 100,
    }

contact_allow_map = {
    pelvis : False,
    lfemur : False,
    ltibia : False,
    lfoot : True,
    rfemur : False,
    rtibia : False,
    rfoot : True,
    lowerneck : False,
    neck : False,
    lowerback : False,
    upperback : False,
    lclavicle : False,
    lhumerus : False,
    lradius : False,
    lhand : False,
    rclavicle : False,
    rhumerus : False,
    rradius : False,
    rhand : False,
    }

joint_weight = {
    pelvis : 1.0,
    lfemur : 0.5,
    ltibia : 0.3,
    lfoot : 0.2,
    rfemur : 0.5,
    rtibia : 0.3,
    rfoot : 0.2,
    lowerneck : 0.3,
    neck : 0.3,
    lowerback : 0.5,
    upperback : 0.5,
    lclavicle : 0.3,
    lhumerus : 0.3,
    lradius : 0.2,
    lhand : 0.0,
    rclavicle : 0.3,
    rhumerus : 0.3,
    rradius : 0.2,
    rhand : 0.0,
    }

noise_pose_mu = {
    pelvis : 0.0,
    lfemur : 0.0,
    ltibia : 0.0,
    lfoot : 0.0,
    rfemur : 0.0,
    rtibia : 0.0,
    rfoot : 0.0,
    lowerneck : 0.0,
    neck : 0.0,
    lowerback : 0.0,
    upperback : 0.0,
    lclavicle : 0.0,
    lhumerus : 0.0,
    lradius : 0.0,
    lhand : 0.0,
    rclavicle : 0.0,
    rhumerus : 0.0,
    rradius : 0.0,
    rhand : 0.0,
    }

noise_pose_sigma = {
    pelvis : 0.1,
    lfemur : 0.1,
    ltibia : 0.1,
    lfoot : 0.1,
    rfemur : 0.1,
    rtibia : 0.1,
    rfoot : 0.1,
    lowerneck : 0.1,
    neck : 0.1,
    lowerback : 0.1,
    upperback : 0.1,
    lclavicle : 0.1,
    lhumerus : 0.1,
    lradius : 0.1,
    lhand : 0.1,
    rclavicle : 0.1,
    rhumerus : 0.1,
    rradius : 0.1,
    rhand : 0.1,
    }

noise_pose_lower = {
    pelvis : -0.5,
    lfemur : -0.5,
    ltibia : -0.5,
    lfoot : -0.5,
    rfemur : -0.5,
    rtibia : -0.5,
    rfoot : -0.5,
    lowerneck : -0.5,
    neck : -0.5,
    lowerback : -0.5,
    upperback : -0.5,
    lclavicle : -0.5,
    lhumerus : -0.5,
    lradius : -0.5,
    lhand : -0.5,
    rclavicle : -0.5,
    rhumerus : -0.5,
    rradius : -0.5,
    rhand : -0.5,
    }

noise_pose_upper = {
    pelvis : 0.5,
    lfemur : 0.5,
    ltibia : 0.5,
    lfoot : 0.5,
    rfemur : 0.5,
    rtibia : 0.5,
    rfoot : 0.5,
    lowerneck : 0.5,
    neck : 0.5,
    lowerback : 0.5,
    upperback : 0.5,
    lclavicle : 0.5,
    lhumerus : 0.5,
    lradius : 0.5,
    lhand : 0.5,
    rclavicle : 0.5,
    rhumerus : 0.5,
    rradius : 0.5,
    rhand : 0.5,
    }

collison_ignore_pairs = [
    (lowerneck, rclavicle),
    (lowerneck, lclavicle),
]

friction_lateral = 0.8
friction_spinning = 0.3
restitution = 0.0

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight

interaction_mesh_samples = [
    # Joints
    (pelvis, None, 1.0),    
    (lfemur, None, 1.0),   
    (ltibia, None, 1.0),   
    (lfoot, None, 1.0),    
    (rfemur, None, 1.0),   
    (rtibia, None, 1.0),   
    (rfoot, None, 1.0),    
    (lowerback, None, 1.0),
    (upperback, None, 1.0),
    (lclavicle, None, 1.0),
    (lhumerus, None, 1.0), 
    (lradius, None, 1.0),   
    (lhand, None, 1.0),     
    (rclavicle, None, 1.0), 
    (rhumerus, None, 1.0),  
    (rradius, None, 1.0),   
    (rhand, None, 1.0),     
    (lowerneck, None, 1.0), 
    (neck, None, 1.0),      
    # Intermediate Points
    # (lfemur, ltibia, 0.5),
    # (ltibia, lfoot, 0.5),
    # (lshoulder, lelbow, 0.5),
    # (lelbow, lwrist, 0.5),
    # (rfemur, rtibia, 0.5),
    # (rtibia, rfoot, 0.5),
    # (rshoulder, relbow, 0.5),
    # (relbow, rwrist, 0.5),
]

interaction_mesh_samples_bvh = []
for j1, j2, w in interaction_mesh_samples:
    j1_bvh = bvh_map[j1] if j1 is not None else j1
    j2_bvh = bvh_map[j2] if j2 is not None else j2
    interaction_mesh_samples_bvh.append((j1_bvh, j2_bvh, w))
