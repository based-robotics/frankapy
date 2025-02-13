from ipaddress import ip_address
import numpy as np
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import min_jerk, min_jerk_weight
import rospy
import scipy.stats
import os
from shapes import Rectangle
import cv_bridge
import time
import numpy as np
from autolab_core import RigidTransform, YamlConfig
from perception_utils.apriltags import AprilTagDetector
from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
from perception.camera_intrinsics import CameraIntrinsics

class Perception():
    def __init__(self, visualize=False):
        self.bridge = cv_bridge.CvBridge()
        self.visualize=visualize
        self.rate = 10 #Hz publish rate
        self.id = 0
        # self.init_time = rospy.Time.now().to_time()
        self.setup_perception()
        # self.pose_publisher = rospy.Publisher("/block_pose", PoseStamped, queue_size=10)
        # self.franka_sensor_buffer_pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

    def setup_perception(self):
        self.cfg = YamlConfig("cfg/saumya_april_tag_cfg.yaml") #TODO replace with your yaml file
        self.T_camera_world = RigidTransform.load(self.cfg['T_k4a_franka_path'])
        self.sensor = Kinect2SensorFactory.sensor('bridged', self.cfg)  # Kinect sensor object
        self.sensor.start()
        self.april = AprilTagDetector(self.cfg['april_tag'])
        # {"_frame": "azure_kinect_overhead", "_fx": 975.7285766601562, "_fy": 975.496826171875, "_cx": 1026.7061767578125, "_cy": 776.0191650390625, "_skew": 0.0, "_height": 1536, "_width": 2048, "_K": 0}
        self.intr = CameraIntrinsics('k4a', fx=975.7285766601562,cx=1026.7061767578125, fy=975.496826171875, cy=776.0191650390625, height=1536, width=2048) #fx fy cx cy overhead
        
    def detect_ar_world_pos(self,straighten=True, shape_class = Rectangle, goal=False):
        """
        O, 1, 2, 3 left hand corner. average [0,2] then [1,3]
        @param bool straighten - whether the roll and pitch should be
                    forced to 0 as prior information that the object is flat
        @param shape_class Shape \in {Rectangle, etc} - type of shape the detector should look for
        @param goal whether the detector should look for the object or goal hole
        """
        T_tag_cameras = []
        detections = self.april.detect(self.sensor, self.intr, vis=0)# Set vis=1 for debug

        detected_ids = []
        for new_detection in detections:
            detected_ids.append(int(new_detection.from_frame.split("/")[1])) #won't work for non-int values
            T_tag_cameras.append(new_detection)
        T_tag_camera = shape_class.tforms_to_pose(detected_ids, T_tag_cameras, goal=goal) #as if there were a tag in the center #assumes 1,2,3,4
        if T_tag_camera == None:
            return None
        T_tag_camera.to_frame="azure_kinect_overhead"
        T_tag_world = self.T_camera_world * T_tag_camera

        if straighten:
            T_tag_world  = straighten_transform(T_tag_world)
        return T_tag_world

    def rosmsg_shape_location(self, shape_type, timestamp, p0, stiffness, i, prev_pose):

        T_tag_world = self.detect_ar_world_pos(shape_class=shape_type)
        if T_tag_world == None:
            T_tag_world = prev_pose
        else:
            T_tag_world.translation[2] = p0.translation[2] - 0.1
        # ipdb.set_trace()
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp, 
            position=T_tag_world.translation, quaternion=p0.quaternion
        )

        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=i, timestamp=timestamp,
            translational_stiffnesses=stiffness,
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )

        return traj_gen_proto_msg, ros_msg, T_tag_world

def straighten_transform(rt):
    """
    Straightens object assuming that the roll and pitch are 0. 
    """
    angles = rt.euler_angles
    roll_off = angles[0]
    pitch_off = angles[1]
    roll_fix = RigidTransform(rotation = RigidTransform.x_axis_rotation(np.pi-roll_off),  from_frame = rt.from_frame, to_frame=rt.from_frame)
    pitch_fix = RigidTransform(rotation = RigidTransform.y_axis_rotation(pitch_off), from_frame = rt.from_frame, to_frame=rt.from_frame)
    new_rt = rt*roll_fix*pitch_fix
    return new_rt

if __name__ == "__main__":
    fa = FrankaArm()
    percept = Perception(visualize=True)
    fa.open_gripper()
    fa.reset_joints()

    starting_position = RigidTransform.load('examples/pickup_starting_position.tf')
    fa.goto_pose(starting_position, duration=5, use_impedance=False)
    p0 = fa.get_pose()
    
    object = p0.copy()
    T_tag_world = None
    while T_tag_world is None:
        T_tag_world = percept.detect_ar_world_pos(shape_class=Rectangle)
    object.translation = T_tag_world.translation.copy()
    object.translation[2] += -0.11

    goal = p0.copy()
    goal.translation[0] = p0.translation[0]-0.1
    goal.translation[1] = p0.translation[1]-0.4
    goal.translation[2] = p0.translation[2]+0.1

    T = 5
    dt = 0.02
    ts = np.arange(0, T, dt)

    weights = [min_jerk_weight(t, T) for t in ts]
    pose_traj1 = [p0.interpolate_with(object, w) for w in weights]
    pose_traj2 = [object.interpolate_with(goal, w) for w in weights]
    pose_traj = pose_traj1 + pose_traj2

    T = 10
    ts = np.arange(0, T, dt)

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')

    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_pose(pose_traj[1], duration=T, dynamic=True, buffer_time=2,
        cartesian_impedances=[85,85,85] + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    )
    prev_pose = fa.get_pose()
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        timestamp = rospy.Time.now().to_time() - init_time
        
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp, 
            position=pose_traj[i].translation, quaternion=pose_traj[i].quaternion
        )
        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=i, timestamp=timestamp,
            translational_stiffnesses=[85,85,85],
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )
        
        p_now = fa.get_pose()
        if p_now.translation[1]<object.translation[1]+0.03:
            current_gripper_width = 0.07
        
            fa.goto_gripper(current_gripper_width, block=False)
        
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

    rospy.loginfo('Done')