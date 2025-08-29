/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_HPP
#define RL_REAL_HPP

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "loop.hpp"
#include <signal.h>     // for signal handling
#include <termios.h>    // for terminal settings
#include <unistd.h>     // for STDIN_FILENO
#include <stdlib.h>     // for exit
#include <iostream>     // for std::cout
#include <random>   // 用于随机数生成
#include <fstream>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
// xzh
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
#include <csignal>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>  // Add this header for TransformBroadcaster
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::go2;
#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_JOYSTICK "rt/wirelesscontroller"
#define TOPIC_HIGHSTATE "rt/sportmodestate"
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

// 遥控器键值联合体
typedef union
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value;
} xKeySwitchUnion;

class RL_Real : public RL
{
public:
    RL_Real();
    ~RL_Real();
    void publishBBoxVisualization(const ros::TimerEvent&);

private:
    // rl functions
    torch::Tensor Forward() override;
    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command) override;
    void RunModel();
    void RobotControl();

    // history buffer
    ObservationBuffer history_obs_buf;
    torch::Tensor history_obs;

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<double>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // unitree interface
    void InitRobotStateClient();
    void InitLowCmd();
    int QueryServiceStatus(const std::string &serviceName);
    uint32_t Crc32Core(uint32_t *ptr, uint32_t len);
    void LowStateMessageHandler(const void *messages);
    void HighStateMessageHandler(const void *messages);
    void JoystickHandler(const void *message);
    RobotStateClient rsc;
    unitree_go::msg::dds_::LowCmd_ unitree_low_command{};
    unitree_go::msg::dds_::LowState_ unitree_low_state{};
    unitree_go::msg::dds_::SportModeState_ unitree_high_state{};
    unitree_go::msg::dds_::WirelessController_ joystick{};
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> highstate_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;
    xKeySwitchUnion unitree_joy;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer;

    // others
    int motiontime = 0;
    double reaction_time = 3;
    std::vector<double> mapped_joint_positions;
    std::vector<double> mapped_joint_velocities;
    int command_mapping[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
    int state_mapping[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
    
    double timeacc_dt=0.005;


    std::vector<double> base_pos = {0, 0, 0};
    std::vector<double> oobb = {0};
    std::vector<double> zone_active = {0};
    std::vector<double> zone_pre_active = {0};

    // 添加随机数生成相关的成员变量
    std::random_device rd;  // 随机数种子
    std::mt19937 gen;      // Mersenne Twister 随机数生成器
    std::uniform_real_distribution<double> lin_vel_x_dist;  // x方向速度分布
    std::uniform_real_distribution<double> lin_vel_y_dist;  // y方向速度分布
    std::uniform_real_distribution<double> heading_dist;    // 航向角分布

    ros::NodeHandle nh_;
    ros::Timer bbox_timer_;

    ros::Subscriber dog_pose_sub_;
    ros::Subscriber stick_pose_sub_;
    geometry_msgs::PoseStamped dog_pose_;
    geometry_msgs::PoseStamped stick_pose_;

    void dogPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void stickPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
};
#endif // RL_REAL_HPP
