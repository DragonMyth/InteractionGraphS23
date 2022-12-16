import collections
import numpy as np

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
v_up_env = np.array([0.0, 0.0, 1.0])
v_ax1_env = np.array([1.0, 0.0, 0.0])
v_ax2_env = np.array([0.0, 1.0, 0.0])

root      = -1
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
head        = 17

ROOT = root

end_effector_indices = [
    lhand, rhand, lfoot, rfoot,
]

'''
Definition of interaction mesh vertex candicates
'''

interaction_joint_candidate = [

    lowerback ,
    upperback ,
    lowerneck,
    lclavicle,

    rclavicle ,
]
bvh_map = collections.OrderedDict()

bvh_map[root] = "root"
bvh_map[lfemur] = "lfemur"
bvh_map[ltibia] = "ltibia"
bvh_map[lfoot] = "lfoot"
bvh_map[rfemur] = "rfemur"
bvh_map[rtibia] = "rtibia"
bvh_map[rfoot] = "rfoot"
bvh_map[lowerneck] = "lowerneck"
bvh_map[head] = "head"
bvh_map[lowerback] = "lowerback"
bvh_map[upperback] = "upperback"
bvh_map[lclavicle] = "lclavicle"
bvh_map[lhumerus] = "lhumerus"
bvh_map[lradius] = "lradius"
bvh_map[lhand] = "lhand"
bvh_map[rclavicle] = "rclavicle"
bvh_map[rhumerus] = "rhumerus"
bvh_map[rradius] = "rradius"
bvh_map[rhand] = "rhand"

bvh_map_inv = collections.OrderedDict()

bvh_map_inv["root"] = root
bvh_map_inv["lhipjoint"] = None
bvh_map_inv["lfemur"] = lfemur
bvh_map_inv["ltibia"] = ltibia
bvh_map_inv["lfoot"] = lfoot
bvh_map_inv["ltoes"] = None
bvh_map_inv["rhipjoint"] = None
bvh_map_inv["rfemur"] = rfemur
bvh_map_inv["rtibia"] = rtibia
bvh_map_inv["rfoot"] = rfoot
bvh_map_inv["rtoes"] = None
bvh_map_inv["lowerneck"] = lowerneck
bvh_map_inv["upperneck"] = None
bvh_map_inv["head"] = head
bvh_map_inv["lowerback"] = lowerback
bvh_map_inv["upperback"] = upperback
bvh_map_inv["thorax"] = None
bvh_map_inv["lclavicle"] = lclavicle
bvh_map_inv["lhumerus"] = lhumerus
bvh_map_inv["lradius"] = lradius
bvh_map_inv["lwrist"] = None
bvh_map_inv["lhand"] = lhand
bvh_map_inv["lfingers"] = None
bvh_map_inv["lthumb"] = None
bvh_map_inv["rclavicle"] = rclavicle
bvh_map_inv["rhumerus"] = rhumerus
bvh_map_inv["rradius"] = rradius
bvh_map_inv["rwrist"] = None
bvh_map_inv["rhand"] = rhand
bvh_map_inv["rfingers"] = None
bvh_map_inv["rthumb"] = None


dof = {
    root : 6,
    lfemur : 3,
    ltibia : 1,
    lfoot : 3,
    rfemur : 3,
    rtibia : 1,
    rfoot : 3,
    lowerneck : 3,
    head : 0,
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
    root : 0,
    lfemur : 500,
    ltibia : 400,
    lfoot : 300,
    rfemur : 500,
    rtibia : 400,
    rfoot : 300,
    lowerneck : 200,
    head : 200,
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
    root : 0,
    lfemur : 300,
    ltibia : 200,
    lfoot : 100,
    rfemur : 300,
    rtibia : 200,
    rfoot : 100,
    lowerneck : 100,
    head : 100,
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
    root : False,
    lfemur : False,
    ltibia : False,
    lfoot : True,
    rfemur : False,
    rtibia : False,
    rfoot : True,
    lowerneck : False,
    head : False,
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
    root : 1.0,
    lfemur : 0.5,
    ltibia : 0.3,
    lfoot : 0.2,
    rfemur : 0.5,
    rtibia : 0.3,
    rfoot : 0.2,
    lowerneck : 0.3,
    head : 0.3,
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
    root : 0.0,
    lfemur : 0.0,
    ltibia : 0.0,
    lfoot : 0.0,
    rfemur : 0.0,
    rtibia : 0.0,
    rfoot : 0.0,
    lowerneck : 0.0,
    head : 0.0,
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
    root : 0.1,
    lfemur : 0.1,
    ltibia : 0.1,
    lfoot : 0.1,
    rfemur : 0.1,
    rtibia : 0.1,
    rfoot : 0.1,
    lowerneck : 0.1,
    head : 0.1,
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
    root : -0.5,
    lfemur : -0.5,
    ltibia : -0.5,
    lfoot : -0.5,
    rfemur : -0.5,
    rtibia : -0.5,
    rfoot : -0.5,
    lowerneck : -0.5,
    head : -0.5,
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
    root : 0.5,
    lfemur : 0.5,
    ltibia : 0.5,
    lfoot : 0.5,
    rfemur : 0.5,
    rtibia : 0.5,
    rfoot : 0.5,
    lowerneck : 0.5,
    head : 0.5,
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
    (root, None, 1.0),    
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
    (head, None, 1.0),      
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
