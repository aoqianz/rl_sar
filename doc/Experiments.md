
# go2 code depository - NUS HcRL
# Main code structure
## training

## deployment 
```bash
sourceh
roscore
rosrun rl_sar rl_real_go2
```
without Optitrack
```bash
roslaunch rl_sar rl_real_go2_with_bbox.launch
```
with Optitrack
主机先通过wifi连接optitrack的路由器上，然后把config.yaml中的optitrack改成true。
当optitrack设置up axis为z时，依然是右手系，但是向上是z
optitrack 红x绿y蓝z
stick 参数选择是否用动捕棍子位置
```bash
roslaunch rl_sar run.launch
```


src/rl_sar/src/rl_real_go2.cpp 
是主文件，负责处理数据读取、模型加载、推理、保存等任务;
#define SIM_OR_REAL true 决定了是mujocu还是实物

src/rl_sar/models/go2_xzh/config.yaml 是
模型配置文件, 包括模型名称、框架、行数、列数、dt、decimation、num_observations、observations、observations_history、clip_obs、clip_actions_lower、clip_actions_upper、rl_kp、rl_kd、fixed_kp、fixed_kd、hip_scale_reduction、hip_scale_reduction_indices、num_of_dofs、wheel_indices、action_scale、action_scale_wheel、acc_body_scale、lin_vel_scale、ang_vel_scale、gravity_scale、dof_pos_scale、dof_vel_scale、zone_center_scale、zone_radius_scale、reaction_time_scale
其中，
- model_name 需要修改成选择的pt
- observations 是观测值，包括线速度、加速度、角速度、重力、命令、关节位置、关节速度、关节力矩、区域中心、区域半径、反应时间、球位置、球半径、球速度，据此拼成observations张量输入网络，因此顺序很关键


src/rl_sar/library/rl_sdk/rl_sdk.cpp
src/rl_sar/library/rl_sdk/rl_sdk.hpp
是rl，包括函数
void RL::ReadYaml(std::string robot_name)，读取yaml文件，并初始化模型参数


## sim2sim mojuco
https://github.com/unitreerobotics/unitree_mujoco
先mojuco:
```bash
cd ～/unitree_mujoco-main/simulate
./build/unitree_mujoco
```

再运行rl_sar:
在vscode打开src/rl_sar/launch/debug.launch，按F5运行调试模式

键盘输入：
0 切换到pos getup状态
p 切换到rl初始化状态
1 切换到pos getdown状态



## vrpn 动捕
```bash
roslaunch vrpn_client_ros ./src/rl_sar/launch/vrpn.launch
```


## 日志
observation保存在src/rl_sar/models/go2_xzh/obs_TIME.csv中



20250411
完成动捕狗+棍位置获取功能

20250414
修正observation的bug，添加timestate，和新观测的计算

20250415
添加oobb和zone时间的计算，添加logger

20250418
policy_0418_1, policy_0418_4 蹦的快但是总顺时针转,  policy_0418_5 会耸肩
