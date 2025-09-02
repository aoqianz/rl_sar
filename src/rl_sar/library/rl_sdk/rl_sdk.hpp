/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SDK_HPP_
#define RL_SDK_HPP_

#include <torch/script.h>
#include <chrono>
#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <unistd.h>
#include <tbb/concurrent_queue.h>
#include <map>
#include <Eigen/Dense>
#include <urdf_parser/urdf_parser.h>
#include <urdf_model/model.h>
#include <ros/ros.h>
#include <urdf/model.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <ros/package.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>

// 在现有的include之后添加
#include "bbox.hpp"

// 定义是否记录观测值到csv
#define CSV_OBS_LOGGER

namespace LOGGER
{
    const char *const INFO = "\033[0;37m[INFO]\033[0m ";
    const char *const WARNING = "\033[0;33m[WARNING]\033[0m ";
    const char *const ERROR = "\033[0;31m[ERROR]\033[0m ";
    const char *const DEBUG = "\033[0;32m[DEBUG]\033[0m ";
}

template <typename T>
struct RobotCommand
{
    struct MotorCommand
    {
        std::vector<int> mode = std::vector<int>(32, 0);
        std::vector<T> q = std::vector<T>(32, 0.0);
        std::vector<T> dq = std::vector<T>(32, 0.0);
        std::vector<T> tau = std::vector<T>(32, 0.0);
        std::vector<T> kp = std::vector<T>(32, 0.0);
        std::vector<T> kd = std::vector<T>(32, 0.0);
    } motor_command;
};

template <typename T>
struct RobotState
{
    struct IMU
    {
        std::vector<T> quaternion = {1.0, 0.0, 0.0, 0.0}; // w, x, y, z
        std::vector<T> gyroscope = {0.0, 0.0, 0.0};
        std::vector<T> accelerometer = {0.0, 0.0, 0.0};
    } imu;

    struct MotorState
    {
        std::vector<T> q = std::vector<T>(32, 0.0);
        std::vector<T> dq = std::vector<T>(32, 0.0);
        std::vector<T> ddq = std::vector<T>(32, 0.0);
        std::vector<T> tau_est = std::vector<T>(32, 0.0);
        std::vector<T> cur = std::vector<T>(32, 0.0);
    } motor_state;

    struct BodyState
    {
        std::vector<T> lin_vel = {0., 0., 0.};
        std::vector<T> ang_vel = {0., 0., 0.};
        std::vector<T> pos = {0.0, 0.0, 0.0};
        std::vector<T> acc = {0.0, 0.0, 0.0};
        
        // 添加滑动窗口相关变量
        static const int WINDOW_SIZE = 10;
        std::array<std::array<T, 3>, WINDOW_SIZE> position_history;
        int current_history_index = 0;
        bool window_filled = false;
    } body_state;

    struct BoundingBox
    {
        double bounding_sphere_radius = 0.0;

        // 修改BoxConfig的定义
        struct BoxConfig {
            std::string link_name;  // 添加link_name成员
            std::vector<T> size = {0.0, 0.0, 0.0};
            std::vector<T> offset = {0.0, 0.0, 0.0};
        };
        std::vector<BoxConfig> manual_box_configs;  // 改为vector而不是map
        
        // 存储URDF中的link信息
        std::vector<std::string> link_names;
        std::map<std::string, std::vector<T>> link_positions;
        std::map<std::string, std::vector<T>> link_orientations;
    } bounding_box;
};

template <typename T>
struct BarrierState
{
    struct BallState
    {
        std::vector<T> zone_vel = {0., 0., 0.}; // FIXME
        std::vector<T> zone_center = {0., 0., 0.};    // FIXME
        std::vector<T> last_zone_center = {0., 0., 0.};
        double zone_radius = 0.0;
        double zone_radius_min = 0.05;
        double zone_radius_max = 0.3;
        double oobb = 0.0; // 球和狗boundingbox的距离
        bool zone_active = false; // 球是否被触发
        bool zone_pre_active = false; // 狗躲避是否被触发
    } ball_state;
};

struct TimeState
{
    std::string START_TIME;
    double start_time;
    double current_time;
    double real_start_time;
    double real_current_time;
    double real_zone_pre_reaction_time;
    double real_zone_activation_time;
};

enum STATE
{
    STATE_WAITING = 0,
    STATE_POS_GETUP,
    STATE_RL_INIT,
    STATE_RL_RUNNING,
    STATE_POS_GETDOWN,
    STATE_RESET_SIMULATION,
    STATE_TOGGLE_SIMULATION,
};

struct Control
{
    STATE control_state;
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double wheel = 0.0;
};

// yaml params
struct ModelParams
{
    std::string model_name;
    std::string framework;
    double dt;
    int decimation;
    int num_observations;
    std::vector<std::string> observations;
    std::vector<int> observations_history;
    double damping;
    double stiffness;
    double action_scale;
    double hip_scale_reduction;
    std::vector<int> hip_scale_reduction_indices;
    double action_scale_wheel;
    std::vector<int> wheel_indices;
    int num_of_dofs;

    // scale
    double lin_vel_scale;
    double acc_body_scale;
    double ang_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    double compliance_scale;

    double clip_obs;
    torch::Tensor clip_actions_upper;
    torch::Tensor clip_actions_lower;
    torch::Tensor torque_limits;
    torch::Tensor rl_kd;
    torch::Tensor rl_kp;
    torch::Tensor fixed_kp;
    torch::Tensor fixed_kd;
    torch::Tensor commands_scale;
    torch::Tensor default_dof_pos;
    std::vector<std::string> joint_controller_names;

    // optitrack
    bool optitrack = false;
    bool stick = false;

    // bounding_sphere
    bool bounding_sphere = true;
    
    // time
    double reaction_time;
    // double oobb;
    // bool zone_active;
    // bool zone_pre_active;
    double zone_activation_time;

    // compliance
    double compliance;

    // oobb
    std::vector<double> zone_center;
    double zone_radius;
    double bounding_sphere_radius;

};

// 观测值，写的很全，但不都用上
struct Observations
{
    torch::Tensor lin_vel;  // 3 机身线速度
    torch::Tensor acc_body; // 3 机身加速度
    torch::Tensor ang_vel; // 3 机身角速度
    torch::Tensor gravity_vec; // 1 重力
    torch::Tensor commands; // 3 命令
    torch::Tensor base_quat; // 4 机身四元数
    torch::Tensor dof_pos; // 12 关节位置
    torch::Tensor dof_vel; // 12 关节速度
    torch::Tensor actions; // 12 关节力矩
    torch::Tensor zone_center; // 3 禁区区域中心
    torch::Tensor zone_radius; // 1 禁区区域半径
    torch::Tensor reaction_time; // 1 狗反应时间
    torch::Tensor ball_positions; // 3 球位置
    torch::Tensor ball_radius; // 1 球半径
    torch::Tensor ball_velocities; // 3 球速度
    torch::Tensor base_pos; // 3 机身位置
    torch::Tensor oobb; // 1 球和狗boundingbox是否碰撞
    torch::Tensor zone_active; // 1 禁区是否激活
    torch::Tensor zone_pre_active; // 1 狗是否激活
    torch::Tensor compliance; // 1 柔顺性
};



class RL
{
public:
    RL() {};
    ~RL() {};

    ModelParams params;
    Observations obs;

    RobotState<double> robot_state;
    RobotCommand<double> robot_command;
    tbb::concurrent_queue<torch::Tensor> output_dof_pos_queue;
    tbb::concurrent_queue<torch::Tensor> output_dof_vel_queue;
    tbb::concurrent_queue<torch::Tensor> output_dof_tau_queue;

    // xzh 障碍物状态
    BarrierState<double> barrier_state;
    TimeState time_state;

    // init
    void InitObservations();
    void InitOutputs();
    void InitControl();
    void InitTimeState();
    void UpdateTimeZoneState();
    void UpdateOOBState();
    
    // rl functions
    virtual torch::Tensor Forward() = 0;
    torch::Tensor ComputeObservation();
    virtual void GetState(RobotState<double> *state) = 0;
    virtual void SetCommand(const RobotCommand<double> *command) = 0;
    void StateController(const RobotState<double> *state, RobotCommand<double> *command);
    void ComputeOutput(const torch::Tensor &actions, torch::Tensor &output_dof_pos, torch::Tensor &output_dof_vel, torch::Tensor &output_dof_tau);
    torch::Tensor QuatRotateInverse(torch::Tensor q, torch::Tensor v, const std::string &framework);

    // yaml params
    void ReadYaml(std::string robot_name);

    // csv logger
    std::string csv_filename;
    void CSVInit(std::string robot_name);
    void CSVLogger(torch::Tensor torque, torch::Tensor tau_est, torch::Tensor joint_pos, torch::Tensor joint_pos_target, torch::Tensor joint_vel);

    // 添加新的csv logger相关变量和函数
    std::string csv_obs_filename;
    void CSVInit_obs(std::string robot_name);
    void CSVLogger_obs(const torch::Tensor& clamped_obs);

    // control
    Control control;
    void KeyboardInterface();

    // others
    std::string robot_name;
    STATE running_state = STATE_RL_RUNNING; // default running_state set to STATE_RL_RUNNING
    bool simulation_running = false;

    // protect func
    void TorqueProtect(torch::Tensor origin_output_dof_tau);
    void AttitudeProtect(const std::vector<double> &quaternion, float pitch_threshold, float roll_threshold);

    // 新增的OOBB相关函数
    bool InitializeOOBB(const std::string& urdf_path);
    std::vector<double> GetRigidBodyOOBB(const std::string& body_name);
    double ComputeOOBBSignedMinDistance(const std::vector<double>& zone_center, const double& zone_radius);
    std::pair<std::vector<double>, double> GetClosestPointsOnOOBBWithSignedDistance(
        const std::vector<double>& point, 
        const std::vector<std::vector<double>>& vertices);

protected:
    // rl module
    torch::jit::script::Module model;
    // output buffer
    torch::Tensor output_dof_tau;
    torch::Tensor output_dof_pos;
    torch::Tensor output_dof_vel;

    // 添加BoundingBoxManager成员
    std::shared_ptr<BoundingBoxManager> bbox_manager_;
};

template <typename T>
T clamp(T value, T min, T max)
{
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

#endif // RL_SDK_HPP_
