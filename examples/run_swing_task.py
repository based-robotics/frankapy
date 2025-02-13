import numpy as np
from autolab_core import RigidTransform

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from time import time, sleep

from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

import rospy

if __name__ == "__main__":
    fa = FrankaArm()
    
    # reset franka to its home joints
    fa.reset_joints()
    # fa.close_gripper()
    # sleep(10)
    # fa.close_gripper()

    dt = 0.02

    p0 = fa.get_pose()

    start_pose = p0.copy()
    start_pose.translation = p0.translation + [0.1, 0., -0.1]
    print("Moving down")
    fa.goto_pose(start_pose)

    p1 = p0.copy()
    p1.translation = start_pose.translation + [0., 0.2, 0.2]
    print("Swing up")
    TRANSLATIONAL_STIFFNESSES = [450.0, 450.0, 450.0]
    T1 = 1
    ts1 = np.arange(0, T1, dt)
    traj_swing_up = start_pose.linear_trajectory_to(p1, len(ts1))
    traj_swing_up = [p1]*len(ts1)
    
    p2 = p0.copy()
    p2.translation = start_pose.translation + [0., 0.2, 0.05]
    
    T2 = 1
    ts2 = np.arange(0, T2, dt)
    traj_swing_down = p1.linear_trajectory_to(p2, len(ts2))
    traj_swing_down = [p2]*len(ts2)
    swing_traj = traj_swing_up + traj_swing_down


    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_pose(swing_traj[1], duration=T1+T2, dynamic=True, buffer_time=2,
        cartesian_impedances=TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    )
    init_time = rospy.Time.now().to_time()
    
    swing_fwd = True
    for i in range(2, len(ts1)+len(ts2)):
        timestamp = rospy.Time.now().to_time() - init_time
        
        p_now = fa.get_pose()
        if (np.linalg.norm(p_now.translation - p1.translation) > 0.03) and swing_fwd:
            p_des = p1.copy()
            print("Swing up: ", np.linalg.norm(p_now.translation - p1.translation))
        else:
            print("Swing down")
            swing_fwd = False
            p_des = p2.copy()

        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp, 
            position=p_des.translation, quaternion=p_des.quaternion
        )
        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=i, timestamp=timestamp,
            translational_stiffnesses=TRANSLATIONAL_STIFFNESSES,
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )
        
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)
    fa.stop_skill()

    p3 = p0.copy()
    p3.translation = start_pose.translation + [0., 0.2, -0.0]
    fa.goto_pose(p3, duration=1, cartesian_impedances=[600.0, 600.0, 1500.0] + FC.DEFAULT_ROTATIONAL_STIFFNESSES)

    p4 = p0.copy()
    p4.translation = start_pose.translation + [0., 0.3, -0.0]
    fa.goto_pose(p4, duration=1, cartesian_impedances=[600.0, 600.0, 1500.0] + FC.DEFAULT_ROTATIONAL_STIFFNESSES)

    rospy.loginfo('Done')