/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"


/* You may need to override this Forward() function
torch::Tensor RL_XXX::Forward()
{
    torch::autograd::GradMode::set_enabled(false);
    torch::Tensor clamped_obs = this->ComputeObservation();
    torch::Tensor actions = this->model.forward({clamped_obs}).toTensor();
    torch::Tensor clamped_actions = torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    return clamped_actions;
}
*/

// 计算观测值
torch::Tensor RL::ComputeObservation()
{
    /**
     * 只加狗线速度没有球速度是53维， 只加狗加速度没有球速度是53维， 三个都没有是50维
     * mode2
     * elf.obs_buf = torch.cat((
        self.base_lin_vel * self.obs_scales.lin_vel, # 3 机身线速度
        self.acc_body * self.obs_scales.acc_body, # 3 机身加速度
        self.base_ang_vel * self.obs_scales.ang_vel, # 3 机身角速度
        self.projected_gravity, # 3 重力
        self.commands[:, :3] * self.commands_scale, # 3 命令
        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12 关节位置
        self.dof_vel * self.obs_scales.dof_vel, # 12 关节速度
        self.actions, # 12 关节力矩
        self.ball_positions,  # 3 观测禁区球位置 (x, y, z) 
        self.ball_radii.unsqueeze(-1),  # 1 观测禁区球半径，注意 unsqueeze(-1) 变成 [N, 1]
        self.rl_reaction_time, # 1 狗反应时间
        # self.ball_velocities,  # 3 观测禁区球速度 (vx, vy, vz), 暂时没加
    ), dim=-1) */
    std::vector<torch::Tensor> obs_list;
    // pritn params.observations
    std::cout<<"打印11111111111111111111111111111111111111"<<std::endl;
    std::cout << this->params.observations << std::endl;
    for (const std::string &observation : this->params.observations)
    {

        /*
            The first argument of the QuatRotateInverse function is the quaternion representing the robot's orientation, and the second argument is in the world coordinate system. The function outputs the value of the second argument in the body coordinate system.
            In IsaacGym, the coordinate system for angular velocity is in the world coordinate system. During training, the angular velocity in the observation uses QuatRotateInverse to transform the coordinate system to the body coordinate system.
            In Gazebo, the coordinate system for angular velocity is also in the world coordinate system, so QuatRotateInverse is needed to transform the coordinate system to the body coordinate system.
            In some real robots like Unitree, if the coordinate system for the angular velocity is already in the body coordinate system, no transformation is necessary.
            Forgetting to perform the transformation or performing it multiple times may cause controller crashes when the rotation reaches 180 degrees.
        */
        // if (observation == "lin_vel")
        // {
        //     obs_list.push_back(this->obs.lin_vel * this->params.lin_vel_scale);
        // }
        // else if (observation == "acc_body")
        // {
        //     obs_list.push_back(this->obs.acc_body * this->params.acc_body_scale);
        // }
        // else if (observation == "ang_vel_body")
        // {
        //     obs_list.push_back(this->obs.ang_vel * this->params.ang_vel_scale);
        // }
        // else if (observation == "ang_vel_world") // 没用
        // {
        //     obs_list.push_back(this->QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel, this->params.framework) * this->params.ang_vel_scale);
        // }
        if (observation == "gravity_vec")
        {
            obs_list.push_back(this->QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec, this->params.framework));
        }
        else if (observation == "ang_vel_body")
        {
            obs_list.push_back(this->obs.ang_vel * this->params.ang_vel_scale);
        }
        else if (observation == "dof_pos")
        {
            torch::Tensor dof_pos_rel = this->obs.dof_pos - this->params.default_dof_pos;
            // for (int i : this->params.wheel_indices)
            // {
            //     dof_pos_rel[0][i] = 0.0;
            // }
            obs_list.push_back(dof_pos_rel * this->params.dof_pos_scale);
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.dof_vel_scale);
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        else if (observation == "commands")
        {
            obs_list.push_back(this->obs.commands * this->params.commands_scale);
        }
        else if (observation == "compliance")
        {
            obs_list.push_back(this->obs.compliance * this->params.compliance_scale);
        }
        // else if (observation == "ball_positions")
        // {
        //     obs_list.push_back(this->obs.ball_positions);
        // }
        // else if (observation == "ball_radius")
        // {
        //     obs_list.push_back(this->obs.ball_radius.unsqueeze(-1));
        // }
        // else if (observation == "ball_velocities")
        // {
        //     obs_list.push_back(this->obs.ball_velocities);
        // }
        // else if (observation == "zone_center")
        // {
        //     obs_list.push_back(this->obs.zone_center);
        // }
        // else if (observation == "zone_radius")
        // {
        //     obs_list.push_back(this->obs.zone_radius);
        // }
        // else if (observation == "reaction_time")
        // {
        //     obs_list.push_back(this->obs.reaction_time);
        // }
        // else if (observation == "base_pos")
        // {
        //     obs_list.push_back(this->obs.base_pos);
        // }
        // else if (observation == "oobb")
        // {
        //     obs_list.push_back(this->obs.oobb);
        // }
        // else if (observation == "zone_active")
        // {
        //     obs_list.push_back(this->obs.zone_active);
        // }
        // else if (observation == "zone_pre_active")
        // {
        //     obs_list.push_back(this->obs.zone_pre_active);
        // }
    }

    torch::Tensor obs = torch::cat(obs_list, 1);
    torch::Tensor clamped_obs = torch::clamp(obs, -this->params.clip_obs, this->params.clip_obs);
    
    // output obs_list data
    if (1)
    {
        std::cout << "===== Debugging obs_list =====" << std::endl;
        for (size_t i = 0; i < obs_list.size(); ++i)
        {
            std::cout << this->params.observations[i] << " | " << "shape: " << obs_list[i].sizes()
                      << " | value: " << obs_list[i]
                      << std::endl;
        }
        std::cout << "================================" << std::endl;
        
        // 输出 clamped_obs 的维度
        std::cout << "clamped_obs dimensions: " << clamped_obs.sizes() << std::endl;
    }

    return clamped_obs;
}

// 初始化观测值
void RL::InitObservations()
{
    this->obs.lin_vel = torch::tensor({{0.0, 0.0, 0.0}}); // xzh
    this->obs.acc_body = torch::tensor({{0.0, 0.0, 0.0}}); // xzh
    this->obs.ang_vel = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.gravity_vec = torch::tensor({{0.0, 0.0, -1.0}}); // only gravity direction
    this->obs.commands = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.base_quat = torch::tensor({{0.0, 0.0, 0.0, 1.0}});
    this->obs.dof_pos = this->params.default_dof_pos;
    this->obs.dof_vel = torch::zeros({1, this->params.num_of_dofs});
    this->obs.actions = torch::zeros({1, this->params.num_of_dofs});
    this->obs.reaction_time = torch::tensor({{0.0}});
    this->obs.zone_center = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.zone_radius = torch::tensor({{0.0}});
    this->obs.ball_positions = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.ball_radius = torch::tensor({{0.0}});
    this->obs.ball_velocities = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.base_pos = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.oobb = torch::tensor({{0.0}});
    this->obs.zone_active = torch::tensor({{0.0}});
    this->obs.zone_pre_active = torch::tensor({{0.0}});
    this->obs.compliance = torch::tensor({{0.02}});
}

// 初始化输出
void RL::InitOutputs()
{
    this->output_dof_tau = torch::zeros({1, this->params.num_of_dofs});
    this->output_dof_pos = this->params.default_dof_pos;
    this->output_dof_vel = torch::zeros({1, this->params.num_of_dofs});
}

// 初始化控制
void RL::InitControl()
{
    this->control.control_state = STATE_WAITING;
    this->control.x = 0.0;
    this->control.y = 0.0;
    this->control.yaw = 0.0;
}

// 初始化时间状态
void RL::InitTimeState()
{
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    // 转换为time_t
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    // 使用strftime格式化时间
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now_time));
    this->time_state.START_TIME = std::string(timestamp);

    // 使用同一个now来计算start_time
    this->time_state.start_time = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    this->time_state.current_time = this->time_state.start_time;  // 初始时current_time等于start_time
    this->time_state.real_start_time = 0.0;
    this->time_state.real_current_time = 0.0;  // 初始时real_current_time为0
    // 从零开始的反应时间 6-1=5
    this->time_state.real_zone_pre_reaction_time = this->params.zone_activation_time - this->params.reaction_time;
    // 从零开始的障碍物激活时间 6
    this->time_state.real_zone_activation_time = this->params.zone_activation_time;
}

// 更新时间状态
void RL::UpdateTimeZoneState()
{
    this->time_state.current_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    this->time_state.real_current_time = this->time_state.current_time - this->time_state.start_time;
    // 狗躲避激活
    if (this->barrier_state.ball_state.zone_pre_active == false && this->time_state.real_current_time > this->time_state.real_zone_pre_reaction_time)
    {
        this->barrier_state.ball_state.zone_pre_active = true;
    }
    // 障碍物激活
    if (this->barrier_state.ball_state.zone_active == false && this->time_state.real_current_time > this->time_state.real_zone_activation_time)
    {
        this->barrier_state.ball_state.zone_active = true;
    }
    
}

void RL::UpdateOOBState()
{
    // 使用外接球计算oobb
    if (this->params.bounding_sphere) 
    {
        // 将位置数据转换为tensor以进行计算
        torch::Tensor robot_pos_tensor = torch::tensor({
            this->robot_state.body_state.pos[0],
            this->robot_state.body_state.pos[1],
            this->robot_state.body_state.pos[2]
        });
        ROS_INFO("Robot position: [%f, %f, %f]", robot_pos_tensor[0].item<double>(), robot_pos_tensor[1].item<double>(), robot_pos_tensor[2].item<double>());

        torch::Tensor zone_center_tensor = torch::tensor({
            this->barrier_state.ball_state.zone_center[0],
            this->barrier_state.ball_state.zone_center[1],
            this->barrier_state.ball_state.zone_center[2]
        });
        ROS_INFO("Zone center: [%f, %f, %f]", zone_center_tensor[0].item<double>(), zone_center_tensor[1].item<double>(), zone_center_tensor[2].item<double>());

        this->barrier_state.ball_state.oobb = torch::norm(robot_pos_tensor - zone_center_tensor).item<double>() 
        - this->robot_state.bounding_box.bounding_sphere_radius 
        - this->barrier_state.ball_state.zone_radius;

        ROS_INFO("OOBB: %f", this->barrier_state.ball_state.oobb);
    }
    else
    {
        // 使用boundingbox的OOBB计算方法
        this->barrier_state.ball_state.oobb = this->ComputeOOBBSignedMinDistance(
            this->barrier_state.ball_state.zone_center, 
            this->barrier_state.ball_state.zone_radius
        );
    }
}

// 计算输出
void RL::ComputeOutput(const torch::Tensor &actions, torch::Tensor &output_dof_pos, torch::Tensor &output_dof_vel, torch::Tensor &output_dof_tau)
{
    torch::Tensor joint_actions_scaled = actions * this->params.action_scale;
    torch::Tensor wheel_actions_scaled = torch::zeros({1, this->params.num_of_dofs});
    for (int i : this->params.wheel_indices)
    {
        joint_actions_scaled[0][i] = 0.0;
        wheel_actions_scaled[0][i] = actions[0][i] * this->params.action_scale_wheel;
    }
    torch::Tensor actions_scaled = joint_actions_scaled + wheel_actions_scaled;
    output_dof_pos = joint_actions_scaled + this->params.default_dof_pos;
    output_dof_vel = wheel_actions_scaled;
    output_dof_tau = this->params.rl_kp * (actions_scaled + this->params.default_dof_pos - this->obs.dof_pos) - this->params.rl_kd * this->obs.dof_vel;
    output_dof_tau = torch::clamp(output_dof_tau, -(this->params.torque_limits), this->params.torque_limits);
}

// 四元数旋转
torch::Tensor RL::QuatRotateInverse(torch::Tensor q, torch::Tensor v, const std::string &framework)
{
    // 定义四元数的标量部分和向量部分
    torch::Tensor q_w;
    torch::Tensor q_vec;

    // 根据不同的框架提取四元数的分量
    if (framework == "isaacsim")
    {
        // 在 isaacsim 中，w 是第一个分量，向量部分是后三个分量
        q_w = q.index({torch::indexing::Slice(), 0});
        q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)});
    }
    else if (framework == "isaacgym")
    {
        // 在 isaacgym 中，w 是第四个分量，向量部分是前三个分量
        q_w = q.index({torch::indexing::Slice(), 3});
        q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    }

    // 获取四元数的形状
    c10::IntArrayRef shape = q.sizes();

    // 计算旋转后的向量的三个部分
    torch::Tensor a = v * (2.0 * torch::pow(q_w, 2) - 1.0).unsqueeze(-1); // 计算 a 部分
    torch::Tensor b = torch::cross(q_vec, v, -1) * q_w.unsqueeze(-1) * 2.0; // 计算 b 部分
    torch::Tensor c = q_vec * torch::bmm(q_vec.view({shape[0], 1, 3}), v.view({shape[0], 3, 1})).squeeze(-1) * 2.0; // 计算 c 部分

    // 返回旋转后的向量
    return a - b + c;
}

// 状态机控制器
void RL::StateController(const RobotState<double> *state, RobotCommand<double> *command)
{
    static RobotState<double> start_state;
    static RobotState<double> now_state;
    static float getup_percent = 0.0;
    static float getdown_percent = 0.0;

    // waiting
    if (this->running_state == STATE_WAITING)
    {
        // std::cout << "StateController: STATE_WAITING" << std::endl;
        for (int i = 0; i < this->params.num_of_dofs; ++i)
        {
            command->motor_command.q[i] = state->motor_state.q[i];
        }
        if (this->control.control_state == STATE_POS_GETUP)
        {
            this->control.control_state = STATE_WAITING;
            getup_percent = 0.0;
            for (int i = 0; i < this->params.num_of_dofs; ++i)
            {
                now_state.motor_state.q[i] = state->motor_state.q[i];
                start_state.motor_state.q[i] = now_state.motor_state.q[i];
            }
            this->running_state = STATE_POS_GETUP;
            // std::cout << std::endl
            //           << LOGGER::INFO << "Switching to STATE_POS_GETUP" << std::endl;
        }
    }
    // stand up (position control)
    else if (this->running_state == STATE_POS_GETUP)
    {
        // std::cout << "StateController: STATE_POS_GETUP" << std::endl;
        if (getup_percent < 1.0)
        {
            getup_percent += 1 / 500.0;
            getup_percent = getup_percent > 1.0 ? 1.0 : getup_percent;
            for (int i = 0; i < this->params.num_of_dofs; ++i)
            {
                // 计算并设置电机的目标位置 q[i]
                // 使用插值方法在当前状态和默认位置之间进行过渡
                command->motor_command.q[i] = (1 - getup_percent) * now_state.motor_state.q[i] + getup_percent * this->params.default_dof_pos[0][i].item<double>();
                
                // 将电机的目标速度 dq[i] 设置为 0，表示不希望有速度变化
                command->motor_command.dq[i] = 0;
                
                // 设置电机的比例增益 kp[i]，用于控制电机的响应速度，从yaml文件读取
                command->motor_command.kp[i] = this->params.fixed_kp[0][i].item<double>();
                
                // 设置电机的微分增益 kd[i]，用于控制电机的阻尼效果，从yaml文件读取
                command->motor_command.kd[i] = this->params.fixed_kd[0][i].item<double>();
                
                // 将电机的目标力矩 tau[i] 设置为 0，表示不希望施加额外的力矩
                command->motor_command.tau[i] = 0;
            }
            std::cout << "\r" << std::flush << LOGGER::INFO << "Getting up " << std::fixed << std::setprecision(2) << getup_percent * 100.0 << std::flush;
        }
        else
        {
            // 检查当前控制状态是否为 STATE_RL_INIT
            if (this->control.control_state == STATE_RL_INIT)
            {
                // 如果是，则切换到 STATE_WAITING 状态
                this->control.control_state = STATE_WAITING;
                this->running_state = STATE_RL_INIT; // 更新运行状态
                std::cout << std::endl
                          << LOGGER::INFO << "Switching to STATE_RL_INIT" << std::endl; // 输出状态切换信息
            }
            // 检查当前控制状态是否为 STATE_POS_GETDOWN
            else if (this->control.control_state == STATE_POS_GETDOWN)
            {
                // 如果是，则切换到 STATE_WAITING 状态
                this->control.control_state = STATE_WAITING;
                getdown_percent = 0.0; // 重置 getdown_percent 为 0
                // 更新当前状态的电机位置
                for (int i = 0; i < this->params.num_of_dofs; ++i)
                {
                    now_state.motor_state.q[i] = state->motor_state.q[i]; // 将当前状态的电机位置设置为新的状态
                }
                this->running_state = STATE_POS_GETDOWN; // 更新运行状态
                std::cout << std::endl
                          << LOGGER::INFO << "Switching to STATE_POS_GETDOWN" << std::endl; // 输出状态切换信息
            }
        }
    }
    // init obs and start rl loop
    else if (this->running_state == STATE_RL_INIT)
    {
        std::cout << "StateController: STATE_RL_INIT" << std::endl;

        if (getup_percent == 1)
        {
            this->InitObservations();
            this->InitOutputs();
            this->InitControl();
            this->InitTimeState();
#ifdef CSV_OBS_LOGGER
            this->CSVInit_obs(this->robot_name);
#endif
            this->running_state = STATE_RL_RUNNING;
            std::cout << std::endl
                      << LOGGER::INFO << "Switching to STATE_RL_RUNNING" << std::endl;
        }
    }
    // rl loop
    else if (this->running_state == STATE_RL_RUNNING)
    {
        // std::cout << "\r" << std::flush << LOGGER::INFO << "RL Controller x:" << this->control.x << " y:" << this->control.y << " yaw:" << this->control.yaw << std::flush;

        torch::Tensor _output_dof_pos, _output_dof_vel;
        if (this->output_dof_pos_queue.try_pop(_output_dof_pos) && this->output_dof_vel_queue.try_pop(_output_dof_vel))
        {
            for (int i = 0; i < this->params.num_of_dofs; ++i)
            {
                if (_output_dof_pos.defined() && _output_dof_pos.numel() > 0)
                {
                    command->motor_command.q[i] = this->output_dof_pos[0][i].item<double>();
                }
                if (_output_dof_vel.defined() && _output_dof_vel.numel() > 0)
                {
                    command->motor_command.dq[i] = this->output_dof_vel[0][i].item<double>();
                }
                command->motor_command.kp[i] = this->params.rl_kp[0][i].item<double>();
                command->motor_command.kd[i] = this->params.rl_kd[0][i].item<double>();
                command->motor_command.tau[i] = 0;
            }
        }
        if (this->control.control_state == STATE_POS_GETDOWN)
        {
            this->control.control_state = STATE_WAITING;
            getdown_percent = 0.0;
            for (int i = 0; i < this->params.num_of_dofs; ++i)
            {
                now_state.motor_state.q[i] = state->motor_state.q[i];
            }
            this->running_state = STATE_POS_GETDOWN;
            std::cout << std::endl
                      << LOGGER::INFO << "Switching to STATE_POS_GETDOWN" << std::endl;
        }
        else if (this->control.control_state == STATE_POS_GETUP)
        {
            this->control.control_state = STATE_WAITING;
            getup_percent = 0.0;
            for (int i = 0; i < this->params.num_of_dofs; ++i)
            {
                now_state.motor_state.q[i] = state->motor_state.q[i];
            }
            this->running_state = STATE_POS_GETUP;
            std::cout << std::endl
                      << LOGGER::INFO << "Switching to STATE_POS_GETUP" << std::endl;
        }
    }
    // get down (position control)
    else if (this->running_state == STATE_POS_GETDOWN)
    {
        std::cout << "StateController: STATE_POS_GETDOWN" << std::endl;
        if (getdown_percent < 1.0)
        {
            getdown_percent += 1 / 500.0;
            getdown_percent = getdown_percent > 1.0 ? 1.0 : getdown_percent;
            for (int i = 0; i < this->params.num_of_dofs; ++i)
            {
                command->motor_command.q[i] = (1 - getdown_percent) * now_state.motor_state.q[i] + getdown_percent * start_state.motor_state.q[i];
                command->motor_command.dq[i] = 0;
                command->motor_command.kp[i] = this->params.fixed_kp[0][i].item<double>();
                command->motor_command.kd[i] = this->params.fixed_kd[0][i].item<double>();
                command->motor_command.tau[i] = 0;
            }
            std::cout << "\r" << std::flush << LOGGER::INFO << "Getting down " << std::fixed << std::setprecision(2) << getdown_percent * 100.0 << std::flush;
        }
        if (getdown_percent == 1)
        {
            this->InitObservations();
            this->InitOutputs();
            this->InitControl();
            this->running_state = STATE_WAITING;
            std::cout << std::endl
                      << LOGGER::INFO << "Switching to STATE_WAITING" << std::endl;
        }
    }
}

// 力矩保护
void RL::TorqueProtect(torch::Tensor origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<double> out_of_range_values;
    for (int i = 0; i < origin_output_dof_tau.size(1); ++i)
    {
        double torque_value = origin_output_dof_tau[0][i].item<double>();
        double limit_lower = -this->params.torque_limits[0][i].item<double>();
        double limit_upper = this->params.torque_limits[0][i].item<double>();

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (int i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            double value = out_of_range_values[i];
            double limit_lower = -this->params.torque_limits[0][index].item<double>();
            double limit_upper = this->params.torque_limits[0][index].item<double>();

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.control_state = STATE_POS_GETDOWN;
        // std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

// 姿态保护
void RL::AttitudeProtect(const std::vector<double> &quaternion, float pitch_threshold, float roll_threshold)
{
    float rad2deg = 57.2958;
    float w, x, y, z;

    if (this->params.framework == "isaacgym")
    {
        w = quaternion[3];
        x = quaternion[0];
        y = quaternion[1];
        z = quaternion[2];
    }
    else if (this->params.framework == "isaacsim")
    {
        w = quaternion[0];
        x = quaternion[1];
        y = quaternion[2];
        z = quaternion[3];
    }

    // Calculate roll (rotation around the X-axis)
    float sinr_cosp = 2 * (w * x + y * z);
    float cosr_cosp = 1 - 2 * (x * x + y * y);
    float roll = std::atan2(sinr_cosp, cosr_cosp) * rad2deg;

    // Calculate pitch (rotation around the Y-axis)
    float sinp = 2 * (w * y - z * x);
    float pitch;
    if (std::fabs(sinp) >= 1)
    {
        pitch = std::copysign(90.0, sinp); // Clamp to avoid out-of-range values
    }
    else
    {
        pitch = std::asin(sinp) * rad2deg;
    }

    if (std::fabs(roll) > roll_threshold)
    {
        // this->control.control_state = STATE_POS_GETDOWN;
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        // this->control.control_state = STATE_POS_GETDOWN;
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

// 键盘接口
#include <termios.h>
#include <sys/ioctl.h>
static bool kbhit()
{
    // 定义一个 termios 结构体，用于存储终端属性
    termios term;
    // 获取当前终端的属性
    tcgetattr(0, &term);

    // 创建一个新的 termios 结构体，复制当前属性
    termios term2 = term;
    // 修改 term2 的控制模式，关闭规范模式（ICANON）
    term2.c_lflag &= ~ICANON;
    // 设置新的终端属性，使其立即生效
    tcsetattr(0, TCSANOW, &term2);

    // 定义一个变量，用于存储等待的字节数
    int byteswaiting;
    // 检查标准输入（键盘）上是否有字节可读
    ioctl(0, FIONREAD, &byteswaiting);

    // 恢复原来的终端属性
    tcsetattr(0, TCSANOW, &term);

    // 返回是否有字节可读
    return byteswaiting > 0;
}

// 键盘输入
void RL::KeyboardInterface()
{
    if (kbhit())
    {
        int c = fgetc(stdin);
        switch (c)
        {
        case '0':
            this->control.control_state = STATE_POS_GETUP;
            std::cout << "keyboard 0 pressed, change to STATE_POS_GETUP" << std::endl;
            break;
        case 'p':
            this->control.control_state = STATE_RL_INIT;
            std::cout << "keyboard p pressed, change to STATE_RL_INIT" << std::endl;

            break;
        case '1':
            this->control.control_state = STATE_POS_GETDOWN;
            std::cout << "keyboard 1 pressed, change to STATE_POS_GETDOWN" << std::endl;
            break;
        case 'q':
            break;
        case 'w':
            this->control.x += 0.1;
            break;
        case 's':
            this->control.x -= 0.1;
            break;
        case 'a':
            this->control.yaw += 0.1;
            break;
        case 'd':
            this->control.yaw -= 0.1;
            break;
        case 'i':
            break;
        case 'k':
            break;
        case 'j':
            this->control.y += 0.1;
            break;
        case 'l':
            this->control.y -= 0.1;
            break;
        case ' ':
            this->control.x = 0;
            this->control.y = 0;
            this->control.yaw = 0;
            break;
        case 'r':
            this->control.control_state = STATE_RESET_SIMULATION;
            break;
        case '\n':
            this->control.control_state = STATE_TOGGLE_SIMULATION;
            break;
        default:
            break;
        }
    }
}

// 读取yaml文件
template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    // std::cout << "values: " << values << std::endl;
    return values;
}

// 读取yaml文件
template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node, const std::string &framework, const int &rows, const int &cols)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }

    if (framework == "isaacsim")
    {
        std::vector<T> transposed_values(cols * rows);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                transposed_values[c * rows + r] = values[r * cols + c];
            }
        }
        return transposed_values;
    }
    else if (framework == "isaacgym")
    {
        return values;
    }
    else
    {
        throw std::invalid_argument("Unsupported framework: " + framework);
    }
}

// 读取yaml文件
void RL::ReadYaml(std::string robot_name)
{
    // The config file is located at "rl_sar/src/rl_sar/models/<robot_name>/config.yaml"
    std::string config_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + robot_name + "/config.yaml";
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[robot_name];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    this->params.model_name = config["model_name"].as<std::string>();
    this->params.framework = config["framework"].as<std::string>();
    int rows = config["rows"].as<int>();
    int cols = config["cols"].as<int>();
    this->params.dt = config["dt"].as<double>();
    this->params.decimation = config["decimation"].as<int>();
    this->params.num_observations = config["num_observations"].as<int>();
    this->params.observations = ReadVectorFromYaml<std::string>(config["observations"]);
    if (config["observations_history"].IsNull())
    {
        this->params.observations_history = {};
    }
    else
    {
        this->params.observations_history = ReadVectorFromYaml<int>(config["observations_history"]);
    }
    this->params.clip_obs = config["clip_obs"].as<double>();
    if (config["clip_actions_lower"].IsNull() && config["clip_actions_upper"].IsNull())
    {
        this->params.clip_actions_upper = torch::tensor({}).view({1, -1});
        this->params.clip_actions_lower = torch::tensor({}).view({1, -1});
    }
    else
    {
        this->params.clip_actions_upper = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_upper"], this->params.framework, rows, cols)).view({1, -1});
        this->params.clip_actions_lower = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_lower"], this->params.framework, rows, cols)).view({1, -1});
    }
    this->params.action_scale = config["action_scale"].as<double>();
    this->params.hip_scale_reduction = config["hip_scale_reduction"].as<double>();
    this->params.hip_scale_reduction_indices = ReadVectorFromYaml<int>(config["hip_scale_reduction_indices"]);
    this->params.action_scale_wheel = config["action_scale_wheel"].as<double>();
    this->params.wheel_indices = ReadVectorFromYaml<int>(config["wheel_indices"]);
    this->params.num_of_dofs = config["num_of_dofs"].as<int>();
    this->params.lin_vel_scale = config["lin_vel_scale"].as<double>();
    this->params.ang_vel_scale = config["ang_vel_scale"].as<double>();
    this->params.dof_pos_scale = config["dof_pos_scale"].as<double>();
    this->params.dof_vel_scale = config["dof_vel_scale"].as<double>();

    this->params.commands_scale = torch::tensor(ReadVectorFromYaml<double>(config["commands_scale"])).view({1, -1});
    // this->params.commands_scale = torch::tensor({this->params.lin_vel_scale, this->params.lin_vel_scale, this->params.ang_vel_scale});
    this->params.rl_kp = torch::tensor(ReadVectorFromYaml<double>(config["rl_kp"], this->params.framework, rows, cols)).view({1, -1});
    this->params.rl_kd = torch::tensor(ReadVectorFromYaml<double>(config["rl_kd"], this->params.framework, rows, cols)).view({1, -1});
    this->params.fixed_kp = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kp"], this->params.framework, rows, cols)).view({1, -1});
    this->params.fixed_kd = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kd"], this->params.framework, rows, cols)).view({1, -1});
    this->params.torque_limits = torch::tensor(ReadVectorFromYaml<double>(config["torque_limits"], this->params.framework, rows, cols)).view({1, -1});
    this->params.default_dof_pos = torch::tensor(ReadVectorFromYaml<double>(config["default_dof_pos"], this->params.framework, rows, cols)).view({1, -1});
    this->params.joint_controller_names = ReadVectorFromYaml<std::string>(config["joint_controller_names"], this->params.framework, rows, cols);

    // xzh实验添加的参数
    if (robot_name == "go2_xzh")
    {
        this->params.optitrack = config["optitrack"].as<bool>();
        this->params.stick = config["stick"].as<bool>();

        this->params.acc_body_scale = config["acc_body_scale"].as<double>();

        this->params.zone_radius = config["zone_radius"].as<double>();
        this->barrier_state.ball_state.zone_radius = this->params.zone_radius;

        this->params.zone_center = ReadVectorFromYaml<double>(config["zone_center"], this->params.framework, 1, 3);
        this->barrier_state.ball_state.zone_center = this->params.zone_center;

        this->params.bounding_sphere = config["bounding_sphere"].as<bool>();

        this->params.bounding_sphere_radius = config["bounding_sphere_radius"].as<double>();
        this->robot_state.bounding_box.bounding_sphere_radius = this->params.bounding_sphere_radius;

        this->params.reaction_time = config["reaction_time"].as<double>();

        this->params.zone_activation_time = config["zone_activation_time"].as<double>();

        // 输出测试
        // std::cout << "reaction_time: " << this->params.reaction_time << std::endl;
        // std::cout << "zone_activation_time: " << this->params.zone_activation_time << std::endl;
        // std::cout << "zone_center: " << this->params.zone_center << std::endl;
        // std::cout << "zone_radius: " << this->params.zone_radius << std::endl;
        // std::cout << "bounding_sphere_radius: " << this->params.bounding_sphere_radius << std::endl;
    }
}

// 初始化csv文件
void RL::CSVInit(std::string robot_name)
{
    csv_filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + robot_name + "/motor_" + this->time_state.START_TIME;

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for (int i = 0; i < 12; ++i)
    {
        file << "tau_cal_" << i << ",";
    }
    for (int i = 0; i < 12; ++i)
    {
        file << "tau_est_" << i << ",";
    }
    for (int i = 0; i < 12; ++i)
    {
        file << "joint_pos_" << i << ",";
    }
    for (int i = 0; i < 12; ++i)
    {
        file << "joint_pos_target_" << i << ",";
    }
    for (int i = 0; i < 12; ++i)
    {
        file << "joint_vel_" << i << ",";
    }

    file << std::endl;

    file.close();
}

// 记录csv文件
void RL::CSVLogger(torch::Tensor torque, torch::Tensor tau_est, torch::Tensor joint_pos, torch::Tensor joint_pos_target, torch::Tensor joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for (int i = 0; i < 12; ++i)
    {
        file << torque[0][i].item<double>() << ",";
    }
    for (int i = 0; i < 12; ++i)
    {
        file << tau_est[0][i].item<double>() << ",";
    }
    for (int i = 0; i < 12; ++i)
    {
        file << joint_pos[0][i].item<double>() << ",";
    }
    for (int i = 0; i < 12; ++i)
    {
        file << joint_pos_target[0][i].item<double>() << ",";
    }
    for (int i = 0; i < 12; ++i)
    {
        file << joint_vel[0][i].item<double>() << ",";
    }

    file << std::endl;

    file.close();
}

// 初始化观测值csv文件
void RL::CSVInit_obs(std::string robot_name)
{
    csv_obs_filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + robot_name + "/obs_" + this->time_state.START_TIME + ".csv";
    std::ofstream file(csv_obs_filename.c_str());

    // 写入表头
    file << "current_time,real_current_time,";
    
    // 写入每个观测值的表头
    for (size_t i = 0; i < this->params.observations.size(); ++i)
    {
        const std::string& obs_name = this->params.observations[i];
        
        // 获取当前tensor的维度
        torch::Tensor sample_tensor;
        if (obs_name == "lin_vel") sample_tensor = this->obs.lin_vel;
        else if (obs_name == "acc_body") sample_tensor = this->obs.acc_body;
        else if (obs_name == "ang_vel_body") sample_tensor = this->obs.ang_vel;
        else if (obs_name == "gravity_vec") sample_tensor = this->obs.gravity_vec;
        else if (obs_name == "commands") sample_tensor = this->obs.commands;
        else if (obs_name == "dof_pos") sample_tensor = this->obs.dof_pos;
        else if (obs_name == "dof_vel") sample_tensor = this->obs.dof_vel;
        else if (obs_name == "actions") sample_tensor = this->obs.actions;
        else if (obs_name == "ball_positions") sample_tensor = this->obs.ball_positions;
        else if (obs_name == "ball_radius") sample_tensor = this->obs.ball_radius.unsqueeze(-1);
        else if (obs_name == "ball_velocities") sample_tensor = this->obs.ball_velocities;
        else if (obs_name == "zone_center") sample_tensor = this->obs.zone_center;
        else if (obs_name == "zone_radius") sample_tensor = this->obs.zone_radius;
        else if (obs_name == "reaction_time") sample_tensor = this->obs.reaction_time;
        else if (obs_name == "base_pos") sample_tensor = this->obs.base_pos;
        else if (obs_name == "oobb") sample_tensor = this->obs.oobb;
        else if (obs_name == "zone_active") sample_tensor = this->obs.zone_active;
        else if (obs_name == "zone_pre_active") sample_tensor = this->obs.zone_pre_active;

        // 获取展平后的维度
        int dims = sample_tensor.numel();
        
        // 为每个维度写入表头
        for (int j = 0; j < dims; ++j)
        {
            file << obs_name << "_" << j;
            if (i < this->params.observations.size() - 1 || j < dims - 1)
            {
                file << ",";
            }
        }
    }
    file << std::endl;
    file.close();
}

// 记录观测值到csv文件
void RL::CSVLogger_obs(const torch::Tensor& clamped_obs)
{
    std::ofstream file(csv_obs_filename.c_str(), std::ios_base::app);

    // 写入时间戳
    file << this->time_state.current_time << "," 
         << this->time_state.real_current_time << ",";

    // 确保张量是二维的，形状为 [1, N]
    auto obs = clamped_obs.view({1, -1});  // 重塑张量确保维度正确
    
    // 写入观测值
    for (int64_t j = 0; j < obs.size(1); ++j)
    {
        file << obs[0][j].item<double>();
        if (j < obs.size(1) - 1)
        {
            file << ",";
        }
    }
    file << std::endl;
    file.close();
}

bool RL::InitializeOOBB(const std::string& urdf_path) {
    ROS_INFO("Initializing OOBB...");
    
    // 获取所有link名称
    auto link_names = bbox_manager_->getLinkNames();
    ROS_INFO("Found %zu links for OOBB", link_names.size());
    
    // 设置到robot_state中
    robot_state.bounding_box.link_names = link_names;
    
    ROS_INFO("OOBB initialization completed");
    return true;
}

// 
std::vector<double> RL::GetRigidBodyOOBB(const std::string& body_name) {
    if (!bbox_manager_) {
        ROS_ERROR("BoundingBoxManager not initialized");
        return std::vector<double>();
    }

    // 添加调试输出
    // ROS_INFO("Getting OOBB for link: %s", body_name.c_str());
    // ROS_INFO("Robot position: %f, %f, %f", 
    //     robot_state.body_state.pos[0],
    //     robot_state.body_state.pos[1],
    //     robot_state.body_state.pos[2]);
    // ROS_INFO("Robot orientation: %f, %f, %f, %f",
    //     robot_state.imu.quaternion[0],
    //     robot_state.imu.quaternion[1],
    //     robot_state.imu.quaternion[2],
    //     robot_state.imu.quaternion[3]);

    // 获取link的位置和方向
    Eigen::Vector3d position(
        robot_state.body_state.pos[0],
        robot_state.body_state.pos[1],
        robot_state.body_state.pos[2]
    );

    Eigen::Quaterniond orientation(
        robot_state.imu.quaternion[0], // w
        robot_state.imu.quaternion[1], // x
        robot_state.imu.quaternion[2], // y
        robot_state.imu.quaternion[3]  // z
    );

    // 使用 BoundingBoxManager 计算世界坐标系中的顶点
    auto world_vertices = bbox_manager_->computeWorldVertices(body_name, position, orientation);
    
    // 添加调试输出
    // ROS_INFO("Number of world vertices: %zu", world_vertices.size());
    
    // 转换为std::vector<double>格式返回
    std::vector<double> result;
    result.reserve(24); // 8个顶点 * 3维坐标
    for (const auto& vertex : world_vertices) {
        result.push_back(vertex.x());
        result.push_back(vertex.y());
        result.push_back(vertex.z());
        // ROS_INFO("Vertex: %f, %f, %f", vertex.x(), vertex.y(), vertex.z());
    }
    
    return result;
}

double RL::ComputeOOBBSignedMinDistance(const std::vector<double>& zone_center, const double& zone_radius) {
    double min_distance = std::numeric_limits<double>::infinity();
    
    // ROS_INFO("Computing OOBB distance with zone_radius: %f", zone_radius);
    // ROS_INFO("Zone center: [%f, %f, %f]", zone_center[0], zone_center[1], zone_center[2]);
    // ROS_INFO("Number of links to check: %zu", robot_state.bounding_box.link_names.size());
    
    for (const auto& link_name : robot_state.bounding_box.link_names) {
        // ROS_INFO("Checking link: %s", link_name.c_str());
        auto oobb_vertices = GetRigidBodyOOBB(link_name);
        // ROS_INFO("Got %zu vertices", oobb_vertices.size());
        
        if (!oobb_vertices.empty()) {
            // 将顶点数据重组为需要的格式
            std::vector<std::vector<double>> vertices_grouped;
            for (size_t i = 0; i < oobb_vertices.size(); i += 3) {
                vertices_grouped.push_back({
                    oobb_vertices[i],
                    oobb_vertices[i + 1],
                    oobb_vertices[i + 2]
                });
            }
            
            auto [closest_point, distance] = GetClosestPointsOnOOBBWithSignedDistance(
                zone_center, vertices_grouped);
            distance -= zone_radius;
            // ROS_INFO("Distance: %f", distance);
            min_distance = std::min(min_distance, distance);
            // ROS_INFO("Current min distance for link %s: %f", link_name.c_str(), distance);
        }
    }
    
    // ROS_INFO("Final min distance: %f", min_distance);
    return min_distance;
}

std::pair<std::vector<double>, double> RL::GetClosestPointsOnOOBBWithSignedDistance(
    const std::vector<double>& point,
    const std::vector<std::vector<double>>& vertices) {
    
    Eigen::Vector3d query_point(point[0], point[1], point[2]);
    
    // 计算OOBB的中心点
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const auto& vertex : vertices) {
        center += Eigen::Vector3d(vertex[0], vertex[1], vertex[2]);
    }
    center /= vertices.size();

    // 计算OOBB的主轴
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& vertex : vertices) {
        Eigen::Vector3d v(vertex[0], vertex[1], vertex[2]);
        Eigen::Vector3d centered = v - center;
        covariance += centered * centered.transpose();
    }
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
    Eigen::Matrix3d axes = eigen_solver.eigenvectors();

    // 计算OOBB在各轴上的半长度
    Eigen::Vector3d half_lengths = Eigen::Vector3d::Zero();
    for (const auto& vertex : vertices) {
        Eigen::Vector3d v(vertex[0], vertex[1], vertex[2]);
        Eigen::Vector3d local = axes.transpose() * (v - center);
        half_lengths = half_lengths.cwiseMax(local.cwiseAbs());
    }

    // 将查询点转换到OOBB的局部坐标系
    Eigen::Vector3d local_point = axes.transpose() * (query_point - center);
    
    // 计算最近点（在局部坐标系中）
    Eigen::Vector3d closest_local = local_point.cwiseMin(half_lengths).cwiseMax(-half_lengths);
    
    // 转换回世界坐标系
    Eigen::Vector3d closest_world = axes * closest_local + center;
    
    // 计算带符号距离
    double distance = (query_point - closest_world).norm();
    bool inside = (local_point.cwiseAbs().array() <= half_lengths.array()).all();
    if (inside) {
        distance = -distance;
    }

    // 转换为返回格式
    std::vector<double> closest_point = {
        closest_world.x(),
        closest_world.y(),
        closest_world.z()
    };
    
    return {closest_point, distance};
}
