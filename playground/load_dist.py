import pickle as pkl
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

# log_dir = '/checkpoint/yzhang3027/exp/CMUMotionPriorBoxThrowExp03/DDPPO_HumanoidImitationInteractionGraphTwo_bc436_00000_0_2022-12-12_23-49-56'

# log_dir = '/checkpoint/yzhang3027/exp/CMUMotionPriorBoxThrowExp04/DDPPO_HumanoidImitationInteractionGraphTwo_bce10_00000_0_2022-12-12_23-49-57'
# log_dir = '/home/yunbo/character_interaction/CMUMotionPriorBoxThrow_Exp05_to_Exp10/exp/CMUMotionPriorBoxThrowExp05/DDPPO_HumanoidImitationInteractionGraphTwo_c1090_00000_0_2022-12-22_11-42-22'
# log_dir = "/home/yunbo/character_interaction/CMUMotionPriorBoxThrow_Exp05_to_Exp10/exp/CMUMotionPriorBoxThrowExp10/DDPPO_HumanoidImitationInteractionGraphTwo_368b7_00000_0_2022-12-22_12-21-27"
log_dir = "/home/yunbo/character_interaction/CMUMotionPriorBoxThrow_Exp05_to_Exp10/exp/CMUMotionPriorBoxThrowExp07/DDPPO_HumanoidImitationInteractionGraphTwo_37214_00000_0_2022-12-22_12-21-28"

ranges = np.arange(10)+1
for i in ranges:
    chpt = i*400
    dist_name = 'init_dist_data_%05d.pkl'%(chpt)

    full_dir = os.path.join(log_dir,dist_name)
    with open(full_dir,'rb') as f:
        data = pkl.load(f)

    fig,axes = plt.subplots(2,2,figsize=(7,7))
    fig.suptitle("Iteration: %d"%(chpt))
    time_line = np.arange(len(data['total_counts']))/30
    axes[0,0].bar(time_line,data['total_counts'])
    axes[0,0].set_title("Total Count")
    axes[0,1].bar(time_line,data['total_ratios'])
    axes[0,1].set_title("Sum of Ratios")
    axes[1,0].bar(time_line,data['average_ratios'])
    axes[1,0].set_title("Average Ratios")
    axes[1,1].bar(time_line,data['average_ratio_inverse_dist'])
    axes[1,1].set_title("Initial Sample Distribution")

    fig.tight_layout()
    plt.show()