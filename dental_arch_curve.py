import os
import json
import copy
import geomdl
import open3d
import DracoPy
import trimesh
import numpy as np


class DAC:
    """
    Class for the Dental Arch Curve (DAC) in 3D. 
    """
    def __init__(self, crowns, kps, prep_id, is_single) -> None:
        if not prep_id or prep_id % 10 not in [1,2,3,4,5,6,7]:
            raise ValueError('miss_id must be an FDI# ends with [1,7]')
        if is_single not in [0,1,2]:
            raise ValueError('is_single must be 0, 1, or 2')
        self.crowns = crowns
        self.kps = kps
        self.prep_id = prep_id
        self.is_single = is_single
        self.up_or_low = 'upper' if prep_id < 30 else 'lower'
        self.teeth_order = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
                       38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]

    def get_dac_nurbs(self, order=4):
        """
        Get the dac from the controls points using NURBS curve.
        
        Reutrn: 
            curve: geomdl NURBS curve object
            control_points: 16 control points on the curve
            sampled_points: 100 sampled points on the curve
            T (4x4 ndarray): transformation matrix to move the scan to the dac
            A (Nx3 ndarray): transformed scan key points
        """
        dac_control_points, T, A = self.get_best_fit_scaled_dac()
        # sort control points
        control_points = np.array([dac_control_points[str(tid)] for tid in self.teeth_order if str(tid) in dac_control_points])
        curve = geomdl.NURBS.Curve() # init curve
        curve.degree = order
        ctrl_pts = [list(p) for p in control_points]
        curve.ctrlpts = ctrl_pts
        curve.knotvector = geomdl.utilities.generate_knot_vector(curve.degree, curve.ctrlpts_size) # knot vector
        curve.delta = 0.01  # number of sampled pts = 1/delta
        curve.sample_size = 100
        return curve, control_points, np.array(curve.evalpts), T, A
    
    def get_best_fit_scaled_dac(self):
        """
        Get the input scan's best fit from many variations of the standard dac.

        Returns:
            best_dac_control_points (Dict[int, List]): 16 control points of the best-fit dac
            best_matrix (4x4 ndarray): transformation matrix to move the scan to the best-fit dac
            best_A_transformed (Nx3 ndarray): transformed scan key points
        """
        dacs_variations = self.get_dac_variations()
        ordered_crown_tids = [tid for tid in self.teeth_order if str(tid) in self.crowns.keys()]
        # register the scanned crowns to all variations of dac and find the best-fitted dac
        # A = np.array([self.tid2control_point(tid) for tid in ordered_crown_tids]) # A contains points on the scan
        A = np.array([read_mesh_bytes(self.crowns[str(tid)]).as_open3d.get_center() for tid in ordered_crown_tids]) # A contains mass centers of the scanned crowns

        min_square_sum_distances = 1e10
        best_matrix, best_A_transformed, best_dac_control_points = None, None, None
        for dac_control_points in dacs_variations:
            B = np.array([dac_control_points[str(tid)] for tid in ordered_crown_tids]) # B is the dac control points
            matrix, A_transformed, square_sum_distances = trimesh.base.registration.procrustes(\
                A, B, reflection=False)
            if square_sum_distances < min_square_sum_distances:
                best_dac_control_points = dac_control_points
                min_square_sum_distances = square_sum_distances
                best_matrix = matrix
                best_A_transformed = A_transformed
        return best_dac_control_points, best_matrix, best_A_transformed

    def tid2control_point(self, tid: int) -> np.array:
        """
        Pick a feature point from the kps of a tid to be the DAC control point of the tooth.

        For anteriors, the control point is the center of the incisal edge, aka, 'occc'
        For pre-molars, the control point is the buccal tip, aka, 'fc'
        For molars, the control point is the center of the buccal tips, aka, center of 'fcm' and 'fcd'
        """
        if tid % 10 in [1,2,3]:
            return np.array(self.kps[str(tid)]['occc'])
        elif tid % 10 in [4,5]:
            return np.array(self.kps[str(tid)]['fc'])
        else:
            return (np.array(self.kps[str(tid)]['fcm']) + np.array(self.kps[str(tid)]['fcd']))/2

    def get_dac_variations(self) -> list:
        """
        Create many variations of the input dac by scaling along the x and z axis

        Args:
            dac_control_points (Dict[int , np.array]): the input original dac control points
        Returns:
            dac_variations (List): a list of scaled versions of the input dac
        """
        with open('dac_control_points.json', 'r') as f:
            dac_control_points = json.load(f)[self.up_or_low]
        # dac_control_points = self.get_std_dac_control_points()[self.up_or_low]
        dac_variations = []
        # scale dac
        for s in np.linspace(0.7, 1.3, 10): # with this range
            for i in [0, 2]: # along x and z axis
                sacle_factor_arr = np.array([1., 1., 1.])
                sacle_factor_arr[i] = s
                dac_control_points_copy = copy.deepcopy(dac_control_points)
                for k in dac_control_points_copy:
                    dac_control_points_copy[k] *= np.array([s, 1., 1.])
                dac_variations.append(dac_control_points_copy)
        return dac_variations
    
    def get_std_dac_control_points(self):
        """
        DEPRECATED !
        
        Get the dict of DAC_control_points

        Returns:
            dac_kps(dict): {'upper': {11: array(x, y, z), 12: array(x, y, z), ...}
                            'lower': {...} }
        """
        stl_dir = '/media/chuanbo/DATA/data/牙冠数据/标准牙列/测试牙列/F0001P/segmentedCrown'
        json_path = '/media/chuanbo/DATA/data/牙冠数据/标准牙列/测试牙列/F0001P/feature_points.json'
        # read stls
        std_dac_crown_fnames = [f for f in os.listdir(stl_dir) if f.endswith('.mq')]
        vis_mesh = open3d.geometry.TriangleMesh()
        vis_pc = open3d.geometry.PointCloud()
        for c in std_dac_crown_fnames:
            with open(os.path.join(stl_dir, c), 'rb') as mq_file:
                mesh = DracoPy.decode(mq_file.read())
                mesh = open3d.geometry.TriangleMesh(\
                    open3d.utility.Vector3dVector(mesh.points), \
                    open3d.utility.Vector3iVector(mesh.faces))
                mesh.compute_vertex_normals()
                vis_mesh += mesh
        # read kps
        std_kps = {}
        raw_kps = json.load(open(json_path, 'r'))
        for d in raw_kps: # rebuild kps dict 
            if d['tid'] not in std_kps:
                std_kps[d['tid']] = {}
            if d['name'] not in std_kps[d['tid']] and 'axis' not in d['name']:
                std_kps[d['tid']][d['name']] = np.asarray([d['x'], d['y'], d['z']])
        # get DAC control points
        dac_control_points = {}
        dac_control_points['upper'] = {}
        dac_control_points['lower'] = {}
        # # using buccal tips as control points
        # for tid in std_kps:
        #     if tid < 30:
        #         if tid % 10 in [6,7,8]:
        #             dac_control_points['upper'][tid] = ((std_kps[tid]['fcm'] + std_kps[tid]['fcd']) / 2).tolist()
        #         elif tid % 10 in [4,5]:
        #             dac_control_points['upper'][tid] = (std_kps[tid]['fc']).tolist()
        #         elif tid % 10 in [1,2,3]:
        #             dac_control_points['upper'][tid] = (std_kps[tid]['occc']).tolist()
        #     else:
        #         if tid % 10 in [6,7,8]:
        #             dac_control_points['lower'][tid] = ((std_kps[tid]['fcm'] + std_kps[tid]['fcd']) / 2).tolist()
        #         elif tid % 10 in [4,5]:
        #             dac_control_points['lower'][tid] = (std_kps[tid]['fc']).tolist()
        #         elif tid % 10 in [1,2,3]:
        #             dac_control_points['lower'][tid] = (std_kps[tid]['occc']).tolist()
        # using crown mess centers as control points
        for c in std_dac_crown_fnames:
            with open(os.path.join(stl_dir, c), 'rb') as mq_file:
                mesh = DracoPy.decode(mq_file.read())
                mesh = open3d.geometry.TriangleMesh(\
                    open3d.utility.Vector3dVector(mesh.points), \
                    open3d.utility.Vector3iVector(mesh.faces))
                uorl = 'upper' if int(c.replace('.mq','')) < 30 else 'lower'
                dac_control_points[uorl][c.replace('.mq','')] = mesh.get_center().tolist()

        # vis
        # for tid in dac_control_points['upper']:
        #     vis_pc += open3d.geometry.PointCloud(open3d.utility.Vector3dVector(dac_control_points['upper'][tid].reshape((1,3))))
        # for tid in dac_control_points['lower']:
        #     vis_pc += open3d.geometry.PointCloud(open3d.utility.Vector3dVector(dac_control_points['lower'][tid].reshape((1,3))))
        # axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        # # vis_mesh.vertices = open3d.utility.Vector3dVector(np.asarray(vis_mesh.vertices)*np.array([1.2, 1., 1.]))
        # # vis_pc.points = open3d.utility.Vector3dVector(np.asarray(vis_pc.points)*np.array([1., 1., 1.]))
        # open3d.visualization.draw_geometries([vis_mesh, vis_pc, axis], mesh_show_back_face=False)
        with open('dac_control_points.json', 'w') as f:
            json.dump(dac_control_points, f)
        return dac_control_points
    

def read_mesh_bytes(buffer):
    import base64
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=V, faces=F)