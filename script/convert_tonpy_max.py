import torch
import open3d as o3d
import os
import numpy as np
import pandas as pd
from tqdm import tqdm  # 引入tqdm

# 读取文件夹中的点云文件
folder_path = r'data/DrivAerNetPlusPlus_Processed_Point_Clouds_100k'
output_folder = r'data/Drivernet_execldata_90du_100se_np_max'
# 初始平面法向量为 (0, 1, 0)，平行于XZ平面，绕 x 轴旋转
a, b, c = 0, 1, 0  # 初始法向量，表示 y=0 的平面
d = 0.0            # 平面通过原点 y=0

class RotatingPlane:
    def __init__(self, initial_theta_deg=0):
        self.theta_deg = initial_theta_deg  # 初始旋转角度，单位为度

    def rotate(self, delta_theta_deg=20):
        # 每次旋转增加 delta_theta_deg 角度
        self.theta_deg += delta_theta_deg

    def distances_to_rotated_region(self, points):
        # 将总的旋转角度转换为弧度
        theta = np.radians(self.theta_deg)
        
        # points 是一个 N x 3 的 NumPy 数组，分别代表每个点的 (x0, y0, z0)
        x0, y0, z0 = points[:, 0], points[:, 1], points[:, 2]
        
        # 计算所有点到旋转平面的距离
        distances_to_plane = np.abs(y0 * np.cos(theta) + z0 * np.sin(theta))
        
        # 计算所有点的 z' 值
        z_prime = z0 * np.cos(theta) - y0 * np.sin(theta)
        
        # 根据 z' 判断点是否在区域内
        distances = np.where(z_prime > 0, distances_to_plane, np.sqrt(distances_to_plane**2 + z_prime**2))
        
        return distances

    def calculate_distance_to_x_axis(self, points):
        # 计算点到 x 轴的距离，距离为 sqrt(y^2 + z^2)
        y = points[:, 1]
        z = points[:, 2]
        distances = np.sqrt(y**2 + z**2)
        return distances

# 遍历文件夹中的所有文件，使用 tqdm 显示进度条
for filename in tqdm(os.listdir(folder_path), desc="Processing files", unit="file"):
    file_path = os.path.join(folder_path, filename)
    
    # 读取点云文件
    points = torch.load(file_path)
    points = points.numpy()

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 初始化旋转平面对象
    plane = RotatingPlane()
    z_offset = np.mean(z)
    points[:, 2] -= z_offset

    # x 坐标的范围，用于分段
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    segments = np.linspace(x_min, x_max, 101)  

    # 用于存储所有角度的数据
    all_distances = {}

    dushu = 2
    total_steps = 90 // dushu  # 总的角度步数
    # 使用 tqdm 显示进度条
    for angle in range(0, 90, dushu):
        plane.rotate(dushu)  # 旋转平面

        # 计算当前角度下所有点到旋转平面的距离
        distances = plane.distances_to_rotated_region(points)
        
        # 筛选出距离平面小于阈值的点
        close_points = points[distances < 0.03]

        Distence_list = []
        # num_zero = 0 
        previous_max_distance=0
        for j in range(100):
            segment_min = segments[j]
            segment_max = segments[j + 1]
            
            # 选择位于当前段的点
            points_in_segment = close_points[(close_points[:, 0] >= segment_min) & (close_points[:, 0] < segment_max)]
            if len(points_in_segment) > 0:
                    # 计算每个点到x轴的距离
                    distances = plane.calculate_distance_to_x_axis(points_in_segment)
                    # 选出距离最大的点
                    max_index = np.argmax(distances)  # 获取最大距离的索引
                    max_distance = distances[max_index]

                    if max_distance < 0.2:
                        Distence_list.append(previous_max_distance)
                    else:
                        # 更新上一次的最大距离
                        previous_max_distance = max_distance
                        Distence_list.append(max_distance)  # 记录该段最大距离


            else:
                if previous_max_distance is not None:
                    # print(f"Segment {j} ({segment_min} - {segment_max}) contains no points. Using previous max distance.")
                    Distence_list.append(previous_max_distance)  # 使用上一次的最大距离
                else:
                    print(f"Segment {j} ({segment_min} - {segment_max}) contains no points and no previous max distance. Appending 0.")
                    Distence_list.append(0)  # 如果是第一段没有点，没有上一次的最大值，填充0
        # print(f"文件： {filename} 度数：{angle} ，无点数：{num_zero}.")
        # 将结果展平成单个列表
        distence_list_flat = Distence_list
        
        # 将结果存储到字典中，键为当前角度
        all_distances[f'Angle_{angle}'] = np.array(distence_list_flat)

    # 合并所有角度的数据
    concatenated_distances = np.column_stack([all_distances[f'Angle_{angle}'] for angle in range(0, 90, dushu)])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # 保存 NumPy 数组为 .npy 文件
    output_file_path = os.path.join(output_folder, f'{filename}_distances.npy')
    np.save(output_file_path, concatenated_distances)
    # print(f"保存文件: {output_file_path}")
