import pickle as pkl
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')

# log_dir = '/home/yunbo/character_interaction/exp/CMUMotionPriorBoxThrowExp03/DDPPO_HumanoidImitationInteractionGraphTwo_bc436_00000_0_2022-12-12_23-49-56'

# log_dir = '/home/yunbo/character_interaction/exp/CMUMotionPriorBoxThrowExp04/DDPPO_HumanoidImitationInteractionGraphTwo_bce10_00000_0_2022-12-12_23-49-57'
log_dir = '/home/yunbo/character_interaction/exp/CMUMotionPriorBoxThrowExp10/DDPPO_HumanoidImitationInteractionGraphTwo_ed4b2_00000_0_2022-12-21_20-56-23'

ranges = np.arange(1)+1
for i in ranges:
    chpt = 170
    dist_name = 'init_dist_data_%05d.pkl'%(chpt)

    full_dir = os.path.join(log_dir,dist_name)
    with open(full_dir,'rb') as f:
        data = pkl.load(f)

    fig,axes = plt.subplots(2,2,figsize=(7,7))
    fig.suptitle("Iteration: %d"%(chpt))
    time_line = np.arange(len(data['total_counts']))/30
    axes[0,0].plot(time_line,data['total_counts'])
    axes[0,0].set_title("Total Count")
    axes[0,1].plot(time_line,data['total_ratios'])
    axes[0,1].set_title("Sum of Ratios")
    axes[1,0].plot(time_line,data['average_ratios'])
    axes[1,0].set_title("Average Ratios")
    axes[1,1].plot(time_line,data['average_ratio_inverse_dist'])
    axes[1,1].set_title("Initial Sample Distribution")

    fig.tight_layout()
    plt.show()