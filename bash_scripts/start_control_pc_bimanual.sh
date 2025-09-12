#!/bin/bash

die() {
    echo >&2 "$@"
    exit 1
}

usage="$(basename "$0") [-h] [-l LEFT_IP] [-r RIGHT_IP] [-lf LEFT_FT_IP] [-rf RIGHT_FT_IP] [-c CONTROL_PC_IP] [-u USERNAME] -- Initialize two Franka robots

where:
    -h  show this help text
    -l  left robot IP address (required)
    -lf left robot force/torque sensor IP address (required)
    -r  right robot IP address (required)
    -rf right robot force/torque sensor IP address (required)
    
    Example:
    ./init_dual_robots.sh -l 192.168.50.211 -lf 192.168.50.212 -r 192.168.50.221 -rf 192.168.50.222
    "

# Required parameters check flags
left_robot_ip="192.168.50.211"
left_ft_ip="192.168.50.212"
right_robot_ip="192.168.50.221"
right_ft_ip="192.168.50.222"

# Parse command line arguments
while getopts ':hl:lf:r:rf:c:u:p:d:g:' option; do
  case "${option}" in
    h) echo "$usage"
       exit 0
       ;;
    l) left_robot_ip=$OPTARG
       ;;
    lf) left_ft_ip=$OPTARG
        ;;
    r) right_robot_ip=$OPTARG
       ;;
    rf) right_ft_ip=$OPTARG
        ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

echo "Initializing dual Franka robot setup..."
echo "Left robot IP: $left_robot_ip (FT: $left_ft_ip)"
echo "Right robot IP: $right_robot_ip (FT: $right_ft_ip)"
echo "Control PC: $control_pc_uname@$control_pc_ip"

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
START_CONTROL_SCRIPT="$DIR/start_control_pc.sh"

# Check if start_control_pc.sh exists
if [ ! -f "$START_CONTROL_SCRIPT" ]; then
    die "Cannot find start_control_pc.sh at $START_CONTROL_SCRIPT"
fi

# Initialize password parameter
password_param=""
if [ "$control_pc_use_password" -eq 1 ]; then
    password_param="-p $control_pc_password"
fi

# Start the left robot (robot number 1)
echo "Starting left robot (robot number 1)..."
$START_CONTROL_SCRIPT -r 1 -a $left_robot_ip -f $left_ft_ip 

# Wait between robot starts to avoid conflicts
sleep 5

# Start the right robot (robot number 2)
echo "Starting right robot (robot number 2)..."
$START_CONTROL_SCRIPT -r 2 -a $right_robot_ip -f $right_ft_ip 

echo "Both robots have been initialized."