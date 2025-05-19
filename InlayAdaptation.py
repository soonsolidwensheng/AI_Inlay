import time
import open3d 
import numpy as np
import trimesh
import copy
from utils import read_mesh


def o3d2tri(mesh_open3d):
    vertices = np.asarray(mesh_open3d.vertices)
    faces = np.asarray(mesh_open3d.triangles)
    mesh_trimesh = trimesh.Trimesh(vertices, faces)
    return mesh_trimesh

class InlayAdaptation:
    def __init__(self):
        self.ignoreBoundDis = 0.3
        self.insertExport = 1.5
        self.lastDis = 0.01
        self.alph = 0.1
        self.ad_gap = 0.03

    def setConfig(self, config):
        self.ignoreBoundDis = config["ignoreBoundDis"]
        self.insertExport = config["insertExport"]
        self.lastDis = config["lastDis"]
        self.alph = config["alph"]
        self.fixBoundDis = config["fixBoundDis"]
        self.ad_gap = config["ad_gap"]

        if self.alph < 0.01: self.alph = 0.01
        if self.lastDis < 0 : self.lastDis = 0
        if self.insertExport < 0.1 : self.insertExport = 0.1
        if self.ignoreBoundDis < 0.1 : self.ignoreBoundDis = 0.1

    def read_mesh(self, path: str) -> open3d.geometry.TriangleMesh:
        mesh = open3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        return mesh

    def buildScene(self, mesh: open3d.geometry.TriangleMesh) -> open3d.geometry.TriangleMesh:
        mesh_o3d = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = open3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh_o3d)

    def compute_signed_distances(self, q_points, q_norms=None) -> np.ndarray:
        """_summary_

        Args:
            q_points (_type_): _description_
            q_norms (_type_, optional): _description_. Defaults to None.

        Returns:
            distance: array of distance
            insertId: intersection vertex ids
            flag: whether there is any intersection
        """
        closest_points = self.scene.compute_closest_points(np.asarray(q_points, dtype=np.float32))
        closest_points_array = closest_points["points"].numpy() # closest points on the scene
        distance = np.linalg.norm(q_points - closest_points_array, axis=-1)
        nonzero = distance > 0
        nonzero = np.where(nonzero)[0]
        if q_norms is not None:
            normals = q_norms.astype(np.float32)
        else:
            normals = closest_points["primitive_normals"].numpy()
        sign = np.sign(np.einsum("ij,ij->i", normals[nonzero], closest_points_array[nonzero] - q_points[nonzero]))
        distance[nonzero] *= sign
        # distance = self.scene.compute_signed_distance(open3d.core.Tensor(np.asarray(q_points, dtype=np.float32)))
        insertId = np.where((distance < self.ad_gap) & (distance > -2.))[0].tolist()
        flag = False
        # ########################################
        # # if self.iterMaxNum < 1:
        #     close_pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(closest_points_array))
        #     q_pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(q_points))
        #     close_pc.paint_uniform_color([0,1,0])
        #     q_pc.paint_uniform_color([1,0,0])
        #     q_color = np.asarray(q_pc.colors)
        #     q_color[insertId] -= np.array([1,0,0])
        #     q_color[insertId] += np.array([0,0,1])
        #     q_pc.colors = open3d.utility.Vector3dVector(q_color)
        #     open3d.visualization.draw_geometries([q_pc, close_pc])
        # ########################################
        if len(insertId) > 0: flag = True
        return distance, insertId, flag

    def expMesh(self, subIndex):
        allSet = set(list(range(len(self.U.vertices))))
        searchSet = allSet.difference(set(subIndex))
        searchList = list(searchSet)
        pcd = open3d.geometry.PointCloud()
        vertices = np.asarray(self.U.vertices)
        pcd.points = open3d.utility.Vector3dVector(vertices[subIndex])
        if len(pcd.points) == 0:
            return list(allSet)
        pcd_tree = open3d.geometry.KDTreeFlann(pcd)
        self.fixIds = set()
        for i in searchList:
            p = vertices[i]
            [k, idx, _] = pcd_tree.search_radius_vector_3d(p, self.fixBoundDis)
            if not idx:
                continue
            d = np.linalg.norm(p - np.asarray(pcd.points)[idx[0]])
            if d > self.insertExport: self.fixIds.add(i)
        self.fixIds = self.fixIds.union(self.subBoundaryIndex)

    def getBoundaryPoints(self):
        boundary_index = list(set(np.array(self.U.get_non_manifold_edges(allow_boundary_edges=False)).flatten()))
        return [self.U.vertices[i] for i in boundary_index]
   
    # def draw_crash(self, signed_distances, S, T, subBoundaryIndex, fixIdx = []) -> None:
    #     plus_ids = np.where(signed_distances > 0)[0]
    #     S.paint_uniform_color([0, 0, 1])
    #     colors = np.asarray(S.vertex_colors)
    #     colors[plus_ids] = [1, 0, 0]
    #     colors[fixIdx] = [0, 1, 0]
    #     colors[subBoundaryIndex] = [0, 0, 0]
    #     S.vertex_colors = open3d.utility.Vector3dVector(colors)
    #     open3d.visualization.draw_geometries([S, T], mesh_show_back_face=True)

    def findBoundLocalPointsIndex(self):
        boundaryPoints = self.getBoundaryPoints()
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.asarray(self.U.vertices))
        pcd_tree = open3d.geometry.KDTreeFlann(pcd)
        findIndexSet = set()
        for p in boundaryPoints:
            [k, idx, _] = pcd_tree.search_radius_vector_3d(p, self.ignoreBoundDis)
            outFlag = False
            for j in idx:
                if j not in findIndexSet: 
                    findIndexSet.add(j)
                    if len(findIndexSet) == len(pcd.points):
                        outFlag = True
                        break
            if outFlag: break
        self.subBoundaryIndex = findIndexSet

    def calcuInsertIds(self, sigDis, IndexList):
        d = sigDis[IndexList[0]]
        i_ = IndexList[0]
        for i in IndexList:
            if np.asarray(sigDis)[i] > d:
                d = np.asarray(sigDis)[i]
                i_ = i
        return i_, d

    def calcuFixIds(self, IndexList):
        if IndexList == []: return []
        self.expMesh(IndexList)
    
    def deleteMesh(self, ids):
        U = copy.deepcopy(self.U)
        insertSet = set(ids)
        triangles = np.asarray(U.triangles)
        i = 0
        delIds = []
        for t in triangles:
            flag = True
            for f in t:
                if f in insertSet:
                    flag = False
                    break
            if flag:
                delIds.append(i)
            i += 1
        U.remove_triangles_by_index(delIds)
        return U
    
    def clusterMesh(self, U):
        triangles = np.asarray(U.triangles)
        #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, _ = U.cluster_connected_triangles()
        idxDic  = {}
        for k in range(len(cluster_n_triangles)): idxDic[k] = []
        for i in range(len(triangle_clusters)):
            key = triangle_clusters[i]
            value = triangles[i]
            idxDic[key].append(value)
        return idxDic
    
    def selectIndexByBoundary(self, triangleDic):
        boundarySet = self.subBoundaryIndex
        lastIndexSet = set()
        for k in triangleDic:
            if len(triangleDic[k]) <= 0: continue
            tempSet = set()
            for f in triangleDic[k]:
                for v_ in f:
                    tempSet.add(v_)
            if(len(boundarySet.intersection(tempSet))) == 0:
                lastIndexSet = lastIndexSet.union(tempSet)
        return list(lastIndexSet)
    
    def splitMesh(self, ids):
        U = self.deleteMesh(ids)
        idxDic = self.clusterMesh(U)
        return self.selectIndexByBoundary(idxDic)
    
    def surfaceModeling(self):
        IndexList = []
        insertFlag = False
        U2UDis, InsertIds ,flag = self.compute_signed_distances(np.asarray(self.U.vertices), np.asarray(self.U.vertex_normals)) # signed dist from self.U tp self.scene
        if flag:
            IndexList = self.splitMesh(InsertIds)
        if len(IndexList)>0:
            farestIndex, maxDis = self.calcuInsertIds(U2UDis, IndexList)
            altVector = self.calcuFindPointsNormals(farestIndex, maxDis+self.lastDis)
            nv = np.linalg.norm(altVector)
            if self.alph > 0 and nv > self.alph:
                altVector = self.alph*altVector/nv
            self.verbosity(IndexList, altVector)
            insertFlag = True
        return insertFlag
    
    def lastSurfaceModeling(self):
        try:
            num = 0
            while num < 3:
                num += 1
                centerPointsList, centerMap = self.findCenterPoints()
                Cdis, _, centerCarshflag = self.compute_signed_distances(np.asarray(centerPointsList))
                if not centerCarshflag: return
                #o3d2tri(self.U).export(path+str(teethNum)+"Ud_.ply")
                centerFindIndexSet = set()
                ids = np.where(Cdis > 0)[0]
                if ids.shape[0] == 0: return
                d, farestIndex = 0, centerMap[ids[0]]
                for i_ in ids:
                    if Cdis[i_] > 0 and Cdis[i_] > d:
                        d = Cdis[i_]
                        farestIndex = centerMap[i_][0]
                        centerFindIndexSet = centerFindIndexSet.union(set(centerMap[i_])) 
                i_ = ids[0]
                mult = self.lastDis
                if mult <= 0: mult = 0.01
                altVector  = self.calcuFindPointsNormals(farestIndex, mult)
                self.verbosity(list(centerFindIndexSet), altVector)
        except Exception as e:
            print(e)
            return 
        
    def fstInitlize(self, U, V):
        self.U = U
        self.buildScene(V)
        self.findBoundLocalPointsIndex()
        _, InsertIds, flag = self.compute_signed_distances(np.asarray(self.U.vertices), np.asarray(self.U.vertex_normals))
        IndexList = []
        if flag: 
            IndexList = self.splitMesh(InsertIds)
            if IndexList == []: return False
        self.calcuFixIds(IndexList)
        return True

    def calcuCrash(self, U, V):
        '''
        Arguments
        ---------
            U : open3d mesh
                source mesh, The outer surface of an inlay
            V : open3d mesh
                target mesh, usual conditions is the Maxillary or Mandibular
            return: open3d mesh
                the source mesh, after surface modeling
        '''
        self.iterMaxNum = 100
        if not self.fstInitlize(U, V): # build self.scene from V
            return self.U
        if self.alph > 0:
            self.iterMaxNum = 0.01*self.iterMaxNum/self.alph
        while self.iterMaxNum>0:
            insertFlag = self.surfaceModeling()
            if not insertFlag:
                self.lastSurfaceModeling()
                return self.U
            self.iterMaxNum -= 1
        return self.U

    def verbosity(self, posIds, posVec):
        vertices = np.asarray(self.U.vertices)
        static_ids = list(self.fixIds)
        static_pos = []
        for id in static_ids:
            static_pos.append(vertices[id].tolist())
        handle_ids = posIds
        handle_pos = (vertices[posIds] + posVec).tolist()
        constraint_ids = open3d.utility.IntVector(static_ids + handle_ids)
        constraint_pos = open3d.utility.Vector3dVector(static_pos + handle_pos)
        if not (static_ids + handle_ids):
            print()
        if not all(static_pos + handle_pos):
            print()
        U_vertex_normals = self.U.vertex_normals
        self.U = self.U.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter = 4)
        self.U.vertex_normals = U_vertex_normals

    def findCenterPoints(self):
        boundSet = self.subBoundaryIndex
        faces = np.asarray(self.U.triangles)
        vertices = np.asarray(self.U.vertices)
        centerList = []
        centerMap = {}
        k = 0
        for f in faces:
            center = (vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3
            if len(set(f).intersection(boundSet))<1:
                centerList.append(center)
                centerMap[k] = f.tolist()
                k += 1
        return centerList, centerMap

    def sampleMesh(self, mesh, num):
        mesh.compute_vertex_normals()  
        pcd = mesh.sample_points_uniformly(number_of_points=num)  
        pcd_mesh = open3d.geometry.PointCloud()
        pcd_mesh.points  = open3d.utility.Vector3dVector(mesh.vertices)
        pcd_tree = open3d.geometry.KDTreeFlann(pcd_mesh)
        boundary_index = set()
        for p in pcd.points:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 1)
            boundary_index.add(idx[0])
        return boundary_index

    def getPcdNormalsByMesh(self, pcd, mesh, num = 100, radiu = 0.5):
        mesh.compute_vertex_normals() 
        search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=radiu, max_nn=num)  
        pcd.estimate_normals(search_param=search_param)
        pcd.orient_normals_consistent_tangent_plane(num)
        sampIndex = self.sampleMesh(mesh, 10)
        total = 0
        for i in sampIndex:
            total += np.dot(pcd.normals[i], mesh.vertex_normals[i])
        if total >= 0:
            pcd.normals = open3d.utility.Vector3dVector(-1.0* np.asarray(pcd.normals))
        return pcd.normals

    def calcuFindPointsNormals(self, idx, m):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.asarray(self.U.vertices))
        pcd.normals = self.getPcdNormalsByMesh(pcd, self.U)
        return m * np.array(pcd.normals[idx])

if __name__ == "__main__":
    # 读取模型
    #path, teethNum = "inlays/fullMouth/__16809_170222150709__15966620585__2018-04-10_13-30_尹新芹__/", 27
    path, teethNum = "inlays/fullMouth/__16809_170406102459__13465937836__2018-04-10_10-59_孙剑__/", 46
    #path, teethNum  = "inlays/fullMouth/97728_20220927_1535_王兆杰10902624/", 37
    #path, teethNum  = "inlays/fullMouth/97728_20220927_1535_王兆杰10902624/", 47
    #path, teethNum  = "inlays/fullMouth/cb1b-6637", 47
    #path, teethNum  = "inlays/fullMouth/designer_Zheng_20230202_1158_增奇口腔_袁_10922266__Raw Preparation scan/", 25
    

    #-----read mesh-----
    Maxillary_path  = path + "Maxillary.stl"
    Mandibular_path = path + "Mandibular.stl"
    U_path = path + str(teethNum) + "U.ply"
    
    Maxillary = read_mesh(Maxillary_path)
    Mandibular = read_mesh(Mandibular_path)
    U = read_mesh(U_path)

    # start main class
    config = {                       
    "ignoreBoundDis":0.3,       #Boundary ignore width
    "insertExport":1.5,         #Collision area spread width
    "lastDis":0.01,             #The final collision distance parameters are adjusted <= lastDis
    "alph":0.1                  #Iteration step. The smaller, the better effect, but the slower speed, recommended not less than "lastDis"
     }
    
    t = time.time()
    inlay_adaptation = InlayAdaptation()

    #syntopy.setConfig(config)
    Ut = inlay_adaptation.calcuCrash(U, Maxillary)

    #syntopy.setConfig(config)
    Ud = inlay_adaptation.calcuCrash(Ut, Mandibular)

    print("finishing times: ", time.time()-t)
    o3d2tri(Ud).export(path+str(teethNum)+"Ud.ply")
