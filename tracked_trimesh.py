import numpy as np
import trimesh
from scipy.spatial import KDTree

class Keypoint:
    def __init__(self, idx, pt):
        """
        初始化 Keypoint 类
        
        :param idx: 关键点的索引，数组类型
        :param pt: 关键点的坐标，数组类型
        """
        self.idx = np.array(idx)
        self.pt = np.array(pt)
    
    def update(self, new_idx, new_pt):
        """
        更新关键点的索引和坐标
        
        :param new_idx: 更新后的索引，数组类型
        :param new_pt: 更新后的坐标，数组类型
        """
        self.idx = np.array(new_idx)
        self.pt = np.array(new_pt)


class TrackedTrimesh(trimesh.Trimesh):
    def __init__(self, vertices, faces, initial_keypoints_indices=None, **kwargs):
        """
        初始化 TrackedTrimesh 类，并创建初始的 Keypoint 实例
        
        :param vertices: 网格顶点，形状为 (n, 3) 的 numpy 数组
        :param faces: 网格面信息，形状为 (m, 3) 的 numpy 数组
        :param initial_keypoints_indices: 初始关键点的索引列表 (可选)
        """
        super().__init__(vertices=vertices, faces=faces, **kwargs)
        self.keypoints = []
        
        self.vert_num = len(vertices)

        # 如果提供了初始关键点索引，则创建对应的 Keypoint 实例
        if initial_keypoints_indices is not None:
            for i, idx in enumerate(initial_keypoints_indices):
                pt = self.vertices[idx]  # 获取对应索引的顶点坐标
                self.add_keypoint(f"keypoint_{i}", idx, pt)

    def add_keypoint(self, name, idx=None, pt=None, mode=0):
        """
        新增一个关键点并将其作为类的属性保存
        
        :param name: 关键点的名称，将其作为属性名
        :param idx: 关键点的索引
        :param pt: 关键点的坐标，如果未提供，则根据索引自动获取
        """
        if pt is None:
            if idx is None:
                return
            pt = self.vertices[idx]
        if idx is None:
            idx = self.find_new_points(pt, mode)
        keypoint = Keypoint(idx, pt)
        self.keypoints.append(keypoint)
        
        # 动态将 Keypoint 作为类的属性
        setattr(self, name, keypoint)

    def update_mesh(self, new_vertices=None, new_faces=None):
        """
        更新网格顶点并自动更新所有关键点的信息
        
        :param new_vertices: 更新后的顶点，形状为 (n, 3) 的 numpy 数组
        :param new_faces: 更新后的面，形状为 (m, 3) 的 numpy 数组 (可选)
        """
        if new_vertices is not None:
            new_vertices = np.array(new_vertices)
            if new_faces is not None:
                if len(new_vertices) != self.vert_num:
                    # 重新分配所有关键点
                    self._reassign_keypoints(new_vertices)
                    self.vertices = new_vertices
                    self.vert_num = len(new_vertices)
                    self.faces = np.array(new_faces)
                else:
                    # 顶点数量不变，仅更新关键点的坐标
                    self.vertices = new_vertices
                    self.faces = np.array(new_faces)
                    self._update_keypoints()
            else:
                # 顶点数量不变，仅更新关键点的坐标
                self.vertices = new_vertices
                self._update_keypoints()
        else:
            if len(self.vertices) != self.vert_num:
                # 重新分配所有关键点
                self._reassign_keypoints(self.vertices)
                self.vert_num = len(self.vertices)
            else:
                # 顶点数量不变，仅更新关键点的坐标
                self._update_keypoints()
    
    def find_new_points(self, points, mode=0):
        """
        网格重构后，查找原来的点在新网格中的位置和索引。
        参数:
        mesh (trimesh.Trimesh): 输入的网格。
        points (numpy.ndarray): 网格中原来点的位置。
        mode (int): 选择查找模式。
        如果为0，则查找与原来点距离不超过1e-3的顶点，如果没有则舍弃原来的点。
        如果为1，则查找与原来点距离最近的顶点。
        返回:
        points_ori (numpy.ndarray): 网格中现在点的位置。
        points_id_ori (list): 网格中现在点的ID。
        """
        points_ori = []
        points_id_ori = []
        if mode == 1:
            # points = self.vertices[
            #     self.faces[trimesh.base.proximity.closest_point(self, points)[2]][:, 0]
            # ]
            closest_triangles_id = trimesh.base.proximity.closest_point(self, points)[2]
            closest_triangles_p = self.triangles[closest_triangles_id]
            closest_dis = np.linalg.norm(closest_triangles_p - points.reshape(-1, 1, 3).repeat(3, axis=1), axis=2)
            closest_p_id = np.argmin(closest_dis, axis=1)
            closest_p = self.faces[closest_triangles_id, closest_p_id]
            return closest_p
        for i in range(len(points)):
            n = np.where(np.linalg.norm(self.vertices - points[i], axis=1) < 1e-3)[0]
            if len(n):
                points_id_ori.append(n[0])
                points_ori.append(points[i])
        return points_id_ori
    
    def _update_keypoints(self):
        """
        当顶点数量不变时，更新所有关键点的坐标
        """
        for keypoint in self.keypoints:
            keypoint.pt = self.vertices[keypoint.idx]

    def _reassign_keypoints(self, new_vertices):
        """
        重新为所有关键点找到最近的顶点并更新索引和坐标
        
        :param new_vertices: 更新后的顶点，形状为 (n, 3) 的 numpy 数组
        """
        kdtree = KDTree(new_vertices)

        for keypoint in self.keypoints:
            # 使用旧的关键点坐标找到新网格中的最近顶点
            new_idx = []
            new_pt = []
            for coord in keypoint.pt:
                dist, new_index = kdtree.query(coord)
                new_idx.append(new_index)
                new_pt.append(new_vertices[new_index])
            
            # 更新 Keypoint 的索引和坐标
            keypoint.update(new_idx, new_pt)

    def get_keypoints_info(self):
        """
        获取所有关键点的索引和坐标信息
        
        :return: 返回一个包含关键点索引和坐标信息的列表
        """
        return [(keypoint.idx, keypoint.pt) for keypoint in self.keypoints]


if __name__ == "__main__":
    k = Keypoint(1, [1,2,3])
    mesh = trimesh.load('mesh.stl')
    m = TrackedTrimesh(mesh.vertices, mesh.faces)
    m.add_keypoint('add_points', 1)
    mesh2 = trimesh.Trimesh(m.vertices, m.faces)
    print