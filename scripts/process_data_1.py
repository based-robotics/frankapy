import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from transformations import euler_from_matrix
# import matplotlib
# matplotlib.use('TkAgg')

def visualize_cartesian_trajectories(trajectory_times, trajectory, labels=['x','y','z'], title='Position', fig_num=1):
    
    plt.figure(fig_num)
    for i in range(3):
        plt.plot(trajectory_times, trajectory[:,i], label=labels[i])
    plt.legend()
    plt.title(f'Cartesian {title}')
    plt.xlabel('Time (s)')
    plt.ylabel(f'{title} (m)')

def process_cartesian_trajectories(trajectory, axes='sxyz'):
    
    num_trajectory_points = trajectory.shape[0]
    cartesian_trajectory = np.zeros((num_trajectory_points,6))
    cartesian_trajectory[:,:3] = trajectory[:,12:15]
    
    transformation_trajectory_1 = trajectory[:,:4].reshape((-1,4,1))
    transformation_trajectory_2 = trajectory[:,4:8].reshape((-1,4,1))
    transformation_trajectory_3 = trajectory[:,8:12].reshape((-1,4,1))
    transformation_trajectory_4 = trajectory[:,12:].reshape((-1,4,1))
    transformation_trajectory = np.concatenate([transformation_trajectory_1, transformation_trajectory_2,
                                                transformation_trajectory_3, transformation_trajectory_4], axis=2)
    
    for i in range(num_trajectory_points):
        cartesian_trajectory[i,3:] = euler_from_matrix(transformation_trajectory[i,:,:], axes=axes)
        # Don't remember exactly why we did this but most likely we will be changing things in the future.
        cartesian_trajectory[i,3:] *= -1

    # Makes euler angles continuous
    for i in range(3,6):
        cartesian_trajectory[:,i] = np.unwrap(cartesian_trajectory[:,i])

    return cartesian_trajectory

def process_cartesian_velocity(dq, zero_jacobian):

    num_trajectory_points = dq.shape[0]
    cartesian_velocity_trajectory = np.zeros((num_trajectory_points,6))

    for i in range(num_trajectory_points):
        Jt = (zero_jacobian[i,:].reshape(7,6)).T
        # Jt = zero_jacobian[i,:].reshape(6,7)
        cartesian_velocity_trajectory[i,:] = Jt@dq[i,:]
    
    return cartesian_velocity_trajectory

def get_object_trajectories(cartesian_trajectory, cartesian_velocity_trajectory, object_goal_trajectory):
    indx = int(1000*4.7)
    object_trajectory = cartesian_trajectory.copy()
    object_trajectory[:indx, :3] = object_goal_trajectory[1500, :3]

    object_velocity_trajectory = cartesian_velocity_trajectory.copy()
    object_velocity_trajectory[:indx, :3] = 0.
    return object_trajectory, object_velocity_trajectory

def process_object_velocity(x, dt):
    num_trajectory_points = x.shape[0]
    object_velocity = np.zeros(x.shape)

    for i in range(1, num_trajectory_points):
        object_velocity[i,:] = (x[i,:] - x[i-1,:])/dt
        if (object_velocity[i,0] > 20) or (object_velocity[i,1] > 20) or (object_velocity[i,2] > 20):
            object_velocity[i,:] = object_velocity[i-1,:].copy()

    object_velocity[-1,:] = object_velocity[-2,:].copy()
    return object_velocity

if __name__ == '__main__':
    input_file = '/home/saumyas/franka/frankapy/data/robot_state_data_many.pkl'

    state_dict = pickle.load( open( input_file, "rb" ) )
    i=0
    for key in state_dict.keys():
        skill_dict = state_dict[key]
        print(skill_dict["skill_desc"])
        
        if skill_dict["skill_desc"] == "GoToPose":
            if i==7:
                skill_state_dict = skill_dict["skill_state_dict"]
                # import ipdb; ipdb.set_trace()

                time = skill_state_dict['time_since_skill_started']
                T = time.shape[0]
                dt = (time[1] - time[0])[0]

                cartesian_trajectory = process_cartesian_trajectories(skill_state_dict['O_T_EE'])
                cartesian_velocity_trajectory = process_cartesian_velocity(skill_state_dict['dq'], skill_state_dict['zero_jacobian'])
                smooth_cartesian_velocity_trajectory = savgol_filter(cartesian_velocity_trajectory, 101, 3, axis = 0)
                object_trajectory, object_velocity_trajectory = get_object_trajectories(cartesian_trajectory[:,:3], smooth_cartesian_velocity_trajectory, skill_state_dict['pose_desired'][:,6:9])
                object_velocity_trajectory_finitediff = process_object_velocity(object_trajectory, dt)
                
                visualize_cartesian_trajectories(skill_state_dict['time_since_skill_started'], cartesian_trajectory, title='Position', fig_num=0)
                # visualize_cartesian_trajectories(skill_state_dict['time_since_skill_started'], skill_state_dict['pose_desired'][:,6:9], title='Position', fig_num=1, labels=['goalx','goaly','goalz'])
                visualize_cartesian_trajectories(skill_state_dict['time_since_skill_started'], object_trajectory, title='Position', fig_num=0, labels=['objx','objy','objz'])
                plt.savefig('robot_position.png')

                # visualize_cartesian_trajectories(skill_state_dict['time_since_skill_started'], cartesian_velocity_trajectory, title='Velocity', fig_num=1)
                visualize_cartesian_trajectories(skill_state_dict['time_since_skill_started'], smooth_cartesian_velocity_trajectory, title='Smooth Velocity', fig_num=1)
                visualize_cartesian_trajectories(skill_state_dict['time_since_skill_started'], object_velocity_trajectory, title='Position', fig_num=1, labels=['objxdot','objydot','objzdot'])
                # visualize_cartesian_trajectories(skill_state_dict['time_since_skill_started'], object_velocity_trajectory_finitediff, title='Position', fig_num=1, labels=['objxdotfd','objydotfd','objzdotfd'])
                plt.savefig('robot_velocity.png')

                visualize_cartesian_trajectories(skill_state_dict['time_since_skill_started'], skill_state_dict['pose_desired'][:,:3], title='Force', fig_num=2)
                plt.savefig('robot_force.png')
            
                if True:
                    expert_trajs = np.concatenate((
                        cartesian_trajectory[:,:3],
                        smooth_cartesian_velocity_trajectory[:,:3],
                        object_trajectory[:,:3],
                        object_velocity_trajectory[:,:3],
                        skill_state_dict['pose_desired'][:,:3]
                    ), axis=1)
                    np.savez(f'/home/saumyas/Documents/experiment_data/GNN_planning_with_contact/franka_real_world_data/PickupObject/data_franka_pickup_dt_0_001_T_10_18.npz', 
                        train_expert_trajs=expert_trajs)
            i+=1
            