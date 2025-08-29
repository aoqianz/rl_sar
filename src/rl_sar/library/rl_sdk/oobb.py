#手动给各个刚体安装bounding box的尺寸        
self.manual_box_configs_by_name = {
            # Base 主体
            "base":        { "size": [0.32, 0.20, 0.15], "offset": [0.03, 0.0, 0.01] },

            # Front Left（前左腿）
            "FL_hip":      { "size": [0.09, 0.09, 0.09], "offset": [0.0, 0.0, 0.0] },
            "FL_thigh":    { "size": [0.07, 0.07, 0.27], "offset": [0.0, -0.02, -0.09] },   # 向后偏移一半
            "FL_calf":     { "size": [0.06, 0.06, 0.25], "offset": [0.0, 0.0, -0.09] }, # 向下偏移一半
            "FL_foot":     { "size": [0.05, 0.05, 0.05], "offset": [0.0, 0.0, -0.015] }, # 小向下偏移

            # Front Right（前右腿）
            "FR_hip":      { "size": [0.09, 0.09, 0.09], "offset": [0.0, 0.0, 0.0] },
            "FR_thigh":    { "size": [0.07, 0.07, 0.27], "offset": [0.0, 0.02, -0.09] },
            "FR_calf":     { "size": [0.06, 0.06, 0.25], "offset": [0.0, 0.0, -0.09] },
            "FR_foot":     { "size": [0.05, 0.05, 0.05], "offset": [0.0, 0.0, -0.015] },

            # Rear Left（后左腿）
            "RL_hip":      { "size": [0.09, 0.09, 0.09], "offset": [0.0, 0.0, 0.0] },
            "RL_thigh":    { "size": [0.07, 0.07, 0.27], "offset": [0.0, -0.02, -0.09] },
            "RL_calf":     { "size": [0.06, 0.06, 0.25], "offset": [0.0, 0.0, -0.09] },
            "RL_foot":     { "size": [0.05, 0.05, 0.05], "offset": [0.0, 0.0, -0.015] },

            # Rear Right（后右腿）
            "RR_hip":      { "size": [0.09, 0.09, 0.09], "offset": [0.0, 0.0, 0.0] },
            "RR_thigh":    { "size": [0.07, 0.07, 0.27], "offset": [0.0, 0.02, -0.09] },
            "RR_calf":     { "size": [0.06, 0.06, 0.25], "offset": [0.0, 0.0, -0.09] },
            "RR_foot":     { "size": [0.05, 0.05, 0.05], "offset": [0.0, 0.0, -0.015] },

            # Head（头部结构）
            "Head_upper":  { "size": [0.13, 0.1, 0.1], "offset": [-0.02, 0.0, 0.03] },
            "Head_lower":  { "size": [0.08, 0.08, 0.09], "offset": [0.0, 0.0, 0.01] },
        }


#获取所有刚体的bounding box
 def _get_rigid_body_oobb(self, rigid_body_index):
        actor_handle = self.actor_handles[0]
        body_pos = self.rigid_body_states[:, rigid_body_index, :3]
        body_rot = self.rigid_body_states[:, rigid_body_index, 3:7]

        # 获取刚体名
        body_name = self.rigid_body_index_to_name[rigid_body_index]

        # 获取手动配置
        if body_name in self.manual_box_configs_by_name:
            config = self.manual_box_configs_by_name[body_name]
            size = torch.tensor(config["size"], device=self.device)
            offset = torch.tensor(config["offset"], device=self.device)
        else:
            size = torch.tensor([0.05, 0.05, 0.05], device=self.device)
            offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            if self.viewer is not None and self.gym.get_sim_time(self.sim) < 1.0:
                print(f"刚体 \"{body_name}\" 未定义配置，使用默认 size+offset")

        # 构造局部 OOBB 顶点（以中心偏移位置为中心）
        local_vertices = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ], device=self.device, dtype=torch.float32) * (size / 2)

        local_vertices += offset  # 加上手动设置的偏移

        # 应用旋转 + 平移到世界坐标
        local_vertices_expanded = local_vertices.unsqueeze(0).expand(self.num_envs, -1, -1)
        body_rot_expanded = body_rot.unsqueeze(1).expand(-1, 8, -1)
        rotated_vertices = quat_apply(body_rot_expanded, local_vertices_expanded)
        world_vertices = rotated_vertices + body_pos.unsqueeze(1)

        return world_vertices

    
    #计算最短距离值
    def _compute_oobb_signed_min_distance(self, zone_center):
        """
        输入: zone_center [num_envs, 3]
        输出：每个环境中距离所有 OOBB 的最近距离（带符号）
        """
        min_distance = torch.full((self.num_envs,), float('inf'), device=self.device)

        for rigid_body_index in range(self.num_rigid_bodies):
            oobb_vertices = self._get_rigid_body_oobb(rigid_body_index)  # [num_envs, 8, 3]
            _, distances = self._get_closest_points_on_oobb_with_signed_distance(zone_center, oobb_vertices)
            min_distance = torch.min(min_distance, distances)

        return min_distance  # [num_envs]
    
    #找到bounding box到禁区中心最短距离的点
    def _get_closest_points_on_oobb_with_signed_distance(self, point, vertices):
        """
        给定: zone_center [N, 3], OOBB 顶点 [N, 8, 3]
        返回:
            - 最近点 [N, 3]
            - 带符号距离 [N]（负值表示在 OOBB 内部）
        """
        center = torch.mean(vertices, dim=1)  # [N, 3]
        axis_x = vertices[:, 1] - vertices[:, 0]
        axis_y = vertices[:, 2] - vertices[:, 0]
        axis_z = vertices[:, 4] - vertices[:, 0]

        axes = torch.stack([axis_x, axis_y, axis_z], dim=1)  # [N, 3, 3]
        axes = torch.nn.functional.normalize(axes, dim=-1)

        half_sizes = torch.norm(torch.stack([axis_x, axis_y, axis_z], dim=1), dim=-1) / 2  # [N, 3]

        # 转为 OOBB 局部坐标系
        rel = point - center  # [N, 3]
        local = torch.sum(rel.unsqueeze(1) * axes, dim=-1)  # [N, 3]

        # 判断是否在 box 内部
        inside = torch.all((local >= -half_sizes) & (local <= half_sizes), dim=-1)  # [N]

        # 最近点（clamp 到盒子表面）
        clamped_local = torch.clamp(local, -half_sizes, half_sizes)  # [N, 3]
        closest = center + torch.sum(clamped_local.unsqueeze(2) * axes, dim=1)  # [N, 3]

        # 距离计算
        dist = torch.norm(closest - point, dim=-1)  # [N]
        dist[inside] *= -1.0  # 内部为负距离

        return closest, dist