/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */


#include "rl_real_go2.hpp"



    
// 定义是否绘制图像
// #define PLOT
// 定义是否记录csv文件
// #define CSV_LOGGER

#define SIM_OR_REAL true

RL_Real::RL_Real() : 
    gen(rd()),  // 在初始化列表中初始化随机数生成器
    lin_vel_x_dist(-1.0, 1.0),
    lin_vel_y_dist(-1.0, 1.0),
    heading_dist(-3.14, 3.14)
{
    // read params from yaml
    this->robot_name = "go2_xzh";
    // this->robot_name = "go2_isaacgym";
    this->ReadYaml(this->robot_name);
    
    // 使用正确的 URDF 路径
    std::string urdf_path;
    if (!nh_.getParam("urdf_path", urdf_path)) {
        // 如果参数服务器中没有，使用默认路径
        urdf_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../robots/go2_description_xzh/urdf/go2.urdf";
    }
    
    // 检查文件是否存在
    std::ifstream f(urdf_path.c_str());
    if (!f.good()) {
        ROS_ERROR("URDF file not found at: %s", urdf_path.c_str());
        return;
    }
    f.close();
    
    // 初始化边界框管理器
    bbox_manager_ = std::make_shared<BoundingBoxManager>(nh_);
    
    // 初始化边界框
    if (!bbox_manager_->initializeFromURDF(urdf_path)) {
        ROS_ERROR("Failed to initialize bounding boxes");
        return;
    }

    // 初始化OOBB，传入urdf_path
    if (!InitializeOOBB(urdf_path)) {
        ROS_ERROR("Failed to initialize OOBB");
        return;
    }

    // 检查是否成功加载了边界框配置
    auto link_names = bbox_manager_->getLinkNames();
    ROS_INFO("Loaded %zu link names", link_names.size());
    for (const auto& name : link_names) {
        ROS_INFO("Loaded link: %s", name.c_str());
    }
    
    // 创建定时器发布可视化信息
    bbox_timer_ = nh_.createTimer(
        ros::Duration(0.1),
        &RL_Real::publishBBoxVisualization,
        this
    );
    
    for (std::string &observation : this->params.observations)
    {
        // In Unitree Go2, the coordinate system for angular velocity is in the body coordinate system.
        if (observation == "ang_vel")
        {
            observation = "ang_vel_body";
        }
    }

    // init robot
    this->InitRobotStateClient();
    while (this->QueryServiceStatus("sport_mode"))
    {
        std::cout << "Try to deactivate the service: " << "sport_mode" << std::endl;
        this->rsc.ServiceSwitch("sport_mode", 0);
        sleep(1);
    }
    this->InitLowCmd();
    // create publisher
    this->lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    this->lowcmd_publisher->InitChannel();
    // create subscriber
    this->lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    this->lowstate_subscriber->InitChannel(std::bind(&RL_Real::LowStateMessageHandler, this, std::placeholders::_1), 1);
    
    this->highstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_HIGHSTATE));
    this->highstate_subscriber->InitChannel(std::bind(&RL_Real::HighStateMessageHandler, this, std::placeholders::_1), 1);

    this->joystick_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    this->joystick_subscriber->InitChannel(std::bind(&RL_Real::JoystickHandler, this, std::placeholders::_1), 1);

    // init rl
    torch::autograd::GradMode::set_enabled(false);
    torch::set_num_threads(4);
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf = ObservationBuffer(1, this->params.num_observations, this->params.observations_history.size());
    }
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();
    running_state = STATE_WAITING;

    this->tf_buffer = std::make_shared<tf2_ros::Buffer>(ros::Duration(10.0));
    this->tf_listener = std::make_shared<tf2_ros::TransformListener>(*(this->tf_buffer));

    // model
    std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + this->robot_name + "/" + this->params.model_name;
    this->model = torch::jit::load(model_path);

    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos)
    {
        vector = std::vector<double>(this->plot_size, 0);
    }
    for (auto &vector : this->plot_target_joint_pos)
    {
        vector = std::vector<double>(this->plot_size, 0);
    }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    // 在构造函数中添加
    if (this->params.optitrack) {
        dog_pose_sub_ = nh_.subscribe("/vrpn_client_node/dog/pose", 1, &RL_Real::dogPoseCallback, this);
        stick_pose_sub_ = nh_.subscribe("/vrpn_client_node/stick/pose", 1, &RL_Real::stickPoseCallback, this);
    }
}

RL_Real::~RL_Real()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::GetState(RobotState<double> *state)
{
    // if ((int)this->unitree_joy.components.R2 == 1)
    // {
    //     this->control.control_state = STATE_POS_GETUP;
    //     std::cout << "control_state change to STATE_POS_GETUP in GetState()" << std::endl;
    // }
    // else if ((int)this->unitree_joy.components.R1 == 1)
    // {
    //     this->control.control_state = STATE_RL_INIT;
    // }
    // else if ((int)this->unitree_joy.components.L2 == 1)
    // {
    //     this->control.control_state = STATE_POS_GETDOWN;
    // }

    if (this->params.framework == "isaacgym")
    {
        state->imu.quaternion[3] = this->unitree_low_state.imu_state().quaternion()[0]; // w
        state->imu.quaternion[0] = this->unitree_low_state.imu_state().quaternion()[1]; // x
        state->imu.quaternion[1] = this->unitree_low_state.imu_state().quaternion()[2]; // y
        state->imu.quaternion[2] = this->unitree_low_state.imu_state().quaternion()[3]; // z
    }
    else if (this->params.framework == "isaacsim")
    {
        state->imu.quaternion[0] = this->unitree_low_state.imu_state().quaternion()[0]; // w
        state->imu.quaternion[1] = this->unitree_low_state.imu_state().quaternion()[1]; // x
        state->imu.quaternion[2] = this->unitree_low_state.imu_state().quaternion()[2]; // y
        state->imu.quaternion[3] = this->unitree_low_state.imu_state().quaternion()[3]; // z
    }

    for (int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->unitree_low_state.imu_state().gyroscope()[i];
    }
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->unitree_low_state.motor_state()[state_mapping[i]].q();
        state->motor_state.dq[i] = this->unitree_low_state.motor_state()[state_mapping[i]].dq();
        state->motor_state.tau_est[i] = this->unitree_low_state.motor_state()[state_mapping[i]].tau_est();
    }

    // xzh
    // useless when turn off Unitree sport_mode
    if (!this->params.optitrack)
    {
        for (int i = 0; i < 3; i++)
        {
            state->body_state.lin_vel[i] = this->unitree_high_state.velocity()[i]; 
            state->body_state.pos[i] = this->unitree_high_state.position()[i];
            // 初始化滑动窗口中的所有位置
            for (int j = 0; j < state->body_state.WINDOW_SIZE; j++) {
                state->body_state.position_history[j][i] = state->body_state.pos[i];
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        state->body_state.acc[i] = this->unitree_low_state.imu_state().accelerometer()[i];
    }
    // std::cout<<"body acc: "<<state->body_state.acc<<std::endl;
    
    for (int i = 0; i < 3; i++)
    {
        state->body_state.ang_vel[i] = this->unitree_low_state.imu_state().gyroscope()[i];
    }
}

void RL_Real::SetCommand(const RobotCommand<double> *command)
{
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->unitree_low_command.motor_cmd()[i].mode() = 0x01;
        this->unitree_low_command.motor_cmd()[i].q() = command->motor_command.q[command_mapping[i]];
        this->unitree_low_command.motor_cmd()[i].dq() = command->motor_command.dq[command_mapping[i]];
        this->unitree_low_command.motor_cmd()[i].kp() = command->motor_command.kp[command_mapping[i]];
        this->unitree_low_command.motor_cmd()[i].kd() = command->motor_command.kd[command_mapping[i]];
        this->unitree_low_command.motor_cmd()[i].tau() = command->motor_command.tau[command_mapping[i]];
    }

    this->unitree_low_command.crc() = Crc32Core((uint32_t *)&unitree_low_command, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(unitree_low_command);
}

void RL_Real::RobotControl()
{
    this->motiontime++;

    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->SetCommand(&this->robot_command);
}

void RL_Real::RunModel()
{
    // OptiTrack
    if (this->params.optitrack)
    {
        geometry_msgs::TransformStamped stick_to_world_transform;
        geometry_msgs::TransformStamped dog_to_world_transform;
        try {
            // 获取 stick 在 world 坐标系中的位置
            stick_to_world_transform = this->tf_buffer->lookupTransform("world", "stick", ros::Time(0));
            // ROS_INFO("Stick position in world frame: x=%.3f, y=%.3f, z=%.3f",
            //     stick_to_world_transform.transform.translation.x,
            //     stick_to_world_transform.transform.translation.y,
            //     stick_to_world_transform.transform.translation.z);

            // 更新 zone_center
            if (this->params.stick) {
                this->barrier_state.ball_state.zone_center = {
                    stick_to_world_transform.transform.translation.x,
                    stick_to_world_transform.transform.translation.y,
                    stick_to_world_transform.transform.translation.z
                };
            }

            
            // 获取 dog 在 world 坐标系中的位置
            dog_to_world_transform = this->tf_buffer->lookupTransform("world", "dog", ros::Time(0));
            // ROS_INFO("Dog position in world frame: x=%.3f, y=%.3f, z=%.3f",
            //     dog_to_world_transform.transform.translation.x,
            //     dog_to_world_transform.transform.translation.y,
            //     dog_to_world_transform.transform.translation.z);

            this->robot_state.body_state.pos = {
                dog_to_world_transform.transform.translation.x,
                dog_to_world_transform.transform.translation.y,
                dog_to_world_transform.transform.translation.z
            };

            // 更新滑动窗口位置历史
            for (int i = 0; i < 3; i++) {
                this->robot_state.body_state.position_history[this->robot_state.body_state.current_history_index][i] = 
                    this->robot_state.body_state.pos[i];
            }
            this->robot_state.body_state.current_history_index = (this->robot_state.body_state.current_history_index + 1) % this->robot_state.body_state.WINDOW_SIZE;
            if (this->robot_state.body_state.current_history_index == 0) {
                this->robot_state.body_state.window_filled = true;
            }

            // 计算平均速度
            std::array<double, 3> avg_vel = {0, 0, 0};
            if (this->robot_state.body_state.window_filled) {
                // 使用整个窗口计算
                for (int i = 0; i < 3; i++) {
                    double total_diff = 0;
                    for (int j = 0; j < this->robot_state.body_state.WINDOW_SIZE - 1; j++) {
                        int curr_idx = (this->robot_state.body_state.current_history_index + j) % this->robot_state.body_state.WINDOW_SIZE;
                        int next_idx = (this->robot_state.body_state.current_history_index + j + 1) % this->robot_state.body_state.WINDOW_SIZE;
                        total_diff += (this->robot_state.body_state.position_history[next_idx][i] - 
                                     this->robot_state.body_state.position_history[curr_idx][i]);
                    }
                    avg_vel[i] = total_diff / ((this->robot_state.body_state.WINDOW_SIZE - 1) * this->params.dt);
                }
            } else {
                // 窗口未填满时使用相邻两帧
                int prev_index = this->robot_state.body_state.current_history_index - 1;
                if (prev_index < 0) prev_index = 0;
                
                for (int i = 0; i < 3; i++) {
                    avg_vel[i] = (this->robot_state.body_state.position_history[this->robot_state.body_state.current_history_index][i] - 
                                this->robot_state.body_state.position_history[prev_index][i]) / this->params.dt;
                }
            }

            this->robot_state.body_state.lin_vel = {avg_vel[0], avg_vel[1], avg_vel[2]};
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN("%s", ex.what());
        }
    }


    if (this->running_state == STATE_RL_RUNNING)
    {
        std::cout << "RunModel: STATE_RL_RUNNING" << std::endl;    
        this->UpdateTimeZoneState();
        // this->UpdateOOBState(); 

        // unitree sdk
        this->obs.acc_body = torch::tensor(this->robot_state.body_state.acc).unsqueeze(0);
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);

        // 生成随机命令
        double rand_vel_x = lin_vel_x_dist(gen);
        double rand_vel_y = lin_vel_y_dist(gen);
        double rand_heading = heading_dist(gen);
        double vel_x = 0;
        double vel_y = 0;
        double ang_z = 0; // placeholder for compliance value
        // this->obs.commands = torch::tensor({{rand_vel_x, rand_vel_y, rand_heading}});
        this->obs.commands = torch::tensor({{vel_x, vel_y, ang_z}});
        

        // this->obs.commands = torch::tensor({{this->joystick.ly(), -this->joystick.rx(), -this->joystick.lx()}});
        // this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        


        // xzh
        // params from yaml
        this->obs.reaction_time = torch::tensor(this->params.reaction_time).unsqueeze(0).unsqueeze(0);
        // barrier state
        this->obs.oobb = torch::tensor(this->barrier_state.ball_state.oobb).unsqueeze(0).unsqueeze(0);
        this->obs.zone_active = torch::tensor(this->barrier_state.ball_state.zone_active).unsqueeze(0).unsqueeze(0);
        this->obs.zone_pre_active = torch::tensor(this->barrier_state.ball_state.zone_pre_active).unsqueeze(0).unsqueeze(0);
        
        // for mode 1
        this->obs.zone_center = torch::tensor(this->barrier_state.ball_state.zone_center).unsqueeze(0);
        this->obs.zone_radius = torch::tensor(this->barrier_state.ball_state.zone_radius).unsqueeze(0).unsqueeze(0);
        // for mode 2
        this->obs.ball_positions = torch::tensor(this->barrier_state.ball_state.zone_center).unsqueeze(0);
        this->obs.ball_velocities = torch::tensor(this->barrier_state.ball_state.zone_vel).unsqueeze(0);
        this->obs.ball_radius = torch::tensor(this->barrier_state.ball_state.zone_radius).unsqueeze(0);
        // OptiTrack
        this->obs.base_pos = torch::tensor(this->robot_state.body_state.pos).unsqueeze(0); 
        // compliance
        this->obs.compliance = torch::tensor(this->params.compliance).unsqueeze(0).unsqueeze(0);




        torch::Tensor clamped_actions = this->Forward();

        this->obs.actions = clamped_actions;

        for (int i : this->params.hip_scale_reduction_indices)
        {
            clamped_actions[0][i] *= this->params.hip_scale_reduction;
        }

        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (this->output_dof_pos.defined() && this->output_dof_pos.numel() > 0)
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (this->output_dof_vel.defined() && this->output_dof_vel.numel() > 0)
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (this->output_dof_tau.defined() && this->output_dof_tau.numel() > 0)
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        this->TorqueProtect(this->output_dof_tau);
        this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tau_est).unsqueeze(0);
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

torch::Tensor RL_Real::Forward()
{
    // 禁用自动求导功能，因为这是推理阶段
    torch::autograd::GradMode::set_enabled(false);

    // 获取并处理观测值
    torch::Tensor clamped_obs = this->ComputeObservation(); 

#ifdef CSV_OBS_LOGGER
    // 记录观测值到csv
    this->CSVLogger_obs(clamped_obs);
#endif

    torch::Tensor actions;
    // 如果使用了观测历史
    if (!this->params.observations_history.empty())
    {
        // 打印观测张量的维度信息（调试用）
        // std::cout << "clamped_obs: " << clamped_obs << std::endl;
        std::cout << "clamped_obs size: " << clamped_obs.sizes() << std::endl;
        
        // 将当前观测插入历史缓冲区
        this->history_obs_buf.insert(clamped_obs);
        // 获取指定长度的观测历史向量
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        std::cout << "history_obs size: " << this->history_obs.sizes() << std::endl;
        // 使用历史观测进行前向传播，获取动作
        actions = this->model.forward({this->history_obs}).toTensor();
    }
    else
    {
        // 不使用历史观测，直接用当前观测进行前向传播
        actions = this->model.forward({clamped_obs}).toTensor();
    }

    // 如果设置了动作的上下限，则对动作进行裁剪
    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        // 否则直接返回原始动作
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

uint32_t RL_Real::Crc32Core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; ++i)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }
            xbit >>= 1;
        }
    }

    return CRC32;
}

void RL_Real::InitLowCmd()
{
    this->unitree_low_command.head()[0] = 0xFE;
    this->unitree_low_command.head()[1] = 0xEF;
    this->unitree_low_command.level_flag() = 0xFF;
    this->unitree_low_command.gpio() = 0;

    for (int i = 0; i < 20; ++i)
    {
        this->unitree_low_command.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
        this->unitree_low_command.motor_cmd()[i].q() = (PosStopF);
        this->unitree_low_command.motor_cmd()[i].kp() = (0);
        this->unitree_low_command.motor_cmd()[i].dq() = (VelStopF);
        this->unitree_low_command.motor_cmd()[i].kd() = (0);
        this->unitree_low_command.motor_cmd()[i].tau() = (0);
    }
}

void RL_Real::InitRobotStateClient()
{
    this->rsc.SetTimeout(10.0f);
    this->rsc.Init();
}

int RL_Real::QueryServiceStatus(const std::string &serviceName)
{
    std::vector<ServiceState> serviceStateList;
    int ret, serviceStatus;
    ret = this->rsc.ServiceList(serviceStateList);
    size_t i, count = serviceStateList.size();
    for (i = 0; i < count; ++i)
    {
        const ServiceState &serviceState = serviceStateList[i];
        if (serviceState.name == serviceName)
        {
            if (serviceState.status == 0)
            {
                std::cout << "name: " << serviceState.name << " is activate" << std::endl;
                serviceStatus = 1;
            }
            else
            {
                std::cout << "name:" << serviceState.name << " is deactivate" << std::endl;
                serviceStatus = 0;
            }
        }
    }
    return serviceStatus;
}

void RL_Real::LowStateMessageHandler(const void *message)
{
    this->unitree_low_state = *(unitree_go::msg::dds_::LowState_ *)message;
}
// xzh
void RL_Real::HighStateMessageHandler(const void *message)
{
    this->unitree_high_state = *(unitree_go::msg::dds_::SportModeState_ *)message;
    std::array<float, 3> &position = this->unitree_high_state.position();
    // std::cout << "HighStateMessageHandler: " << position[0] << ", " << position[1] << ", " << position[0] << ", " << std::endl;
}

void RL_Real::JoystickHandler(const void *message)
{
    joystick = *(unitree_go::msg::dds_::WirelessController_ *)message;
    this->unitree_joy.value = joystick.keys();
}

void signalHandler(int signum)
{
    std::cout << "\nCaught signal " << signum << ", cleaning up..." << std::endl;
    
    // 重置终端设置（因为键盘输入可能改变了终端设置）
    termios term;
    tcgetattr(0, &term);
    term.c_lflag |= ICANON;
    tcsetattr(0, TCSANOW, &term);
    
    // 如果是ROS程序，需要关闭ROS
    if (ros::isInitialized()) {
        ros::shutdown();
    }
    
    // 强制退出
    std::_Exit(0);  // 使用 _Exit 而不是 exit，因为它会立即终止程序
}

void RL_Real::publishBBoxVisualization(const ros::TimerEvent&) {
    if (bbox_manager_) {
        bbox_manager_->publishVisualization();
    }
}

void RL_Real::dogPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    if (!this->params.optitrack) return;
    
    dog_pose_ = *msg;
    
    // 发布 TF
    geometry_msgs::TransformStamped transform;
    transform.header = msg->header;
    transform.child_frame_id = "dog";
    transform.transform.translation.x = msg->pose.position.x;
    transform.transform.translation.y = msg->pose.position.y;
    transform.transform.translation.z = msg->pose.position.z;
    transform.transform.rotation = msg->pose.orientation;
    
    static tf2_ros::TransformBroadcaster br;
    br.sendTransform(transform);
}

void RL_Real::stickPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    if (!this->params.optitrack) return;
    
    stick_pose_ = *msg;
    
    // 发布 TF
    geometry_msgs::TransformStamped transform;
    transform.header = msg->header;
    transform.child_frame_id = "stick";
    transform.transform.translation.x = msg->pose.position.x;
    transform.transform.translation.y = msg->pose.position.y;
    transform.transform.translation.z = msg->pose.position.z;
    transform.transform.rotation = msg->pose.orientation;
    
    static tf2_ros::TransformBroadcaster br;
    br.sendTransform(transform);
}

int main(int argc, char **argv)
{
    // 注册多个信号
    signal(SIGINT, signalHandler);   // Ctrl+C
    signal(SIGTERM, signalHandler);  // 终止信号
    signal(SIGABRT, signalHandler);  // 异常终止
    
    ros::init(argc, argv, "rl_real_go2", ros::init_options::NoSigintHandler);  // 让ROS不处理SIGINT
    
    // if (argc < 2)
    // {
    //     std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    //     exit(-1);
    // }


    if (SIM_OR_REAL) 
    {
        // mujoco
        std::string networkInterface = "lo";
        ChannelFactory::Instance()->Init(1, networkInterface);
        std::cout << "simulation mode!" << std::endl;
    }
    else 
    {
        // real go2
        std::string networkInterface = "enp14s0"; // go2网卡名字，需要go2背部网口通过网线连PC
        ChannelFactory::Instance()->Init(0, networkInterface);
        std::cout << "real world mode!" << std::endl;
    }

    RL_Real rl_sar;

    while (1)
    {
        sleep(10);
    }

    return 0;
}
