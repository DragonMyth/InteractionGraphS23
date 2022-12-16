import collections

vup = 'z'

pelvis = -1
lfemur = 0
ltibia = 1
lfoot = 2
# ltoes = 3
rfemur = 3
rtibia = 4
rfoot = 5
# rtoes = 7
lowerback = 6
upperback = 7
lclavicle = 8
lhumerus = 9
lradius = 10
lhand = 11
rclavicle = 12
rhumerus = 13
rradius = 14
rhand = 15
LowerNeck = 16
Neck = 17

ROOT = pelvis

end_effector_indices = [
    lhand, rhand, lfoot, rfoot,
]

bvh_map = collections.OrderedDict()

bvh_map[pelvis] = "pelvis"
bvh_map[lfemur] = "lfemur"
bvh_map[ltibia] = "ltibia"
bvh_map[lfoot] = "lfoot"
# bvh_map[ltoes] = "ltoes"
bvh_map[rfemur] = "rfemur"
bvh_map[rtibia] = "rtibia"
bvh_map[rfoot] = "rfoot"
# bvh_map[rtoes] = "rtoes"
bvh_map[LowerNeck] = "LowerNeck"
bvh_map[Neck] = "Neck"
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

bvh_map_inv["pelvis"] = pelvis
bvh_map_inv["lfemur"] = lfemur
bvh_map_inv["ltibia"] = ltibia
bvh_map_inv["lfoot"] = lfoot
bvh_map_inv["ltoes"] = None
bvh_map_inv["rfemur"] = rfemur
bvh_map_inv["rtibia"] = rtibia
bvh_map_inv["rfoot"] = rfoot
bvh_map_inv["rtoes"] = None
bvh_map_inv["LowerNeck"] = LowerNeck
bvh_map_inv["Neck"] = Neck
bvh_map_inv["lowerback"] = lowerback
bvh_map_inv["upperback"] = upperback
bvh_map_inv["lclavicle"] = lclavicle
bvh_map_inv["lhumerus"] = lhumerus
bvh_map_inv["lradius"] = lradius
bvh_map_inv["lhand"] = lhand
bvh_map_inv["rclavicle"] = rclavicle
bvh_map_inv["rhumerus"] = rhumerus
bvh_map_inv["rradius"] = rradius
bvh_map_inv["rhand"] = rhand

dof = {
    pelvis : 6,
    lfemur : 3,
    ltibia : 1,
    lfoot : 3,
    # ltoes : 0,
    rfemur : 3,
    rtibia : 1,
    rfoot : 3,
    # rtoes : 0,
    LowerNeck : 3,
    Neck : 0,
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
    pelvis : 0,
    lfemur : 500,
    ltibia : 400,
    lfoot : 300,
    # ltoes : 300,
    rfemur : 500,
    rtibia : 400,
    rfoot : 300,
    # rtoes : 300,
    LowerNeck : 200,
    Neck : 200,
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

kd = {}
for k, v in kp.items():
    kd[k] = 0.05 * v

cpd_ratio = 0.0002

max_force = {
    pelvis : 0,
    lfemur : 300,
    ltibia : 200,
    lfoot : 100,
    # ltoes : 100,
    rfemur : 300,
    rtibia : 200,
    rfoot : 100,
    # rtoes : 100,
    LowerNeck : 100,
    Neck : 100,
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
    # ltoes : True,
    rfemur : False,
    rtibia : False,
    rfoot : True,
    # rtoes : True,
    LowerNeck : False,
    Neck : False,
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
    # ltoes : 0.0,
    rfemur : 0.5,
    rtibia : 0.3,
    rfoot : 0.2,
    # rtoes : 0.0,
    LowerNeck : 0.3,
    Neck : 0.3,
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
    # ltoes : 0.0,
    rfemur : 0.0,
    rtibia : 0.0,
    rfoot : 0.0,
    # rtoes : 0.0,
    LowerNeck : 0.0,
    Neck : 0.0,
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
    # ltoes : 0.1,
    rfemur : 0.1,
    rtibia : 0.1,
    rfoot : 0.1,
    # rtoes : 0.1,
    LowerNeck : 0.1,
    Neck : 0.1,
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
    # ltoes : -0.5,
    rfemur : -0.5,
    rtibia : -0.5,
    rfoot : -0.5,
    # rtoes : -0.5,
    LowerNeck : -0.5,
    Neck : -0.5,
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
    # ltoes : 0.5,
    rfemur : 0.5,
    rtibia : 0.5,
    rfoot : 0.5,
    # rtoes : 0.5,
    LowerNeck : 0.5,
    Neck : 0.5,
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
    (LowerNeck, rclavicle),
    (LowerNeck, lclavicle),
]

friction = 0.9
restitution = 0.0

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight