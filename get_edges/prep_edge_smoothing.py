# --------------------------------------------------------------------
# Created by: Chuanbo Wang
# Created on: 04/10/2023
# --------------------------------------------------------------------
# This script generates smoothed control points from Yang Yue's prep 
# segmentation results.
# --------------------------------------------------------------------


import os
import copy
import open3d
import numpy as np
import networkx as nx
from geomdl import NURBS, utilities
from scipy.spatial.distance import euclidean

vis = False


def get_cycles_networkx(edge_tup_list):
    """
    Another implementation of cycle detection in a graph based on Networkx
    """
    G = nx.Graph()
    G.add_edges_from(edge_tup_list)
    return nx.cycle_basis(G)


def get_mesh_boundary_edges(mesh):
    """
    This function returns the boundary edges of the input mesh
    Parameters:
        mesh (open3d.geometry.TriangleMesh)
    Reutrns:
        boundary_edges ((M, 2) ndarray): each edge is represented as array([i, j]),
                                        where i, j are vert indices in this edge
    """
    # get triangle_edges (sorted vert ids)
    triangles = np.asarray(mesh.triangles)
    edge_triangles = []
    for i, t in enumerate(triangles):
        edge_triangles.append([])
        edge_triangles[i].append(tuple([t[0], t[1]]) if t[0] < t[1] else tuple([t[1], t[0]]))
        edge_triangles[i].append(tuple([t[0], t[2]]) if t[0] < t[2] else tuple([t[2], t[0]]))
        edge_triangles[i].append(tuple([t[1], t[2]]) if t[1] < t[2] else tuple([t[2], t[1]]))

    # get dict: dege to triangle
    edge_adjacency_dict = {}
    for i, t in enumerate(edge_triangles):
        if t[0] not in edge_adjacency_dict.keys():
            edge_adjacency_dict[t[0]] = []
        edge_adjacency_dict[t[0]].append(i)
        if t[1] not in edge_adjacency_dict.keys():
            edge_adjacency_dict[t[1]] = []
        edge_adjacency_dict[t[1]].append(i)
        if t[2] not in edge_adjacency_dict.keys():
            edge_adjacency_dict[t[2]] = []
        edge_adjacency_dict[t[2]].append(i)

    # get all edges (sorted vert ids)
    edges = []
    mesh.compute_adjacency_list()
    adjacency_list = mesh.adjacency_list
    for i, neighbors in enumerate(adjacency_list):
        for n in neighbors:
            edges.append(tuple([i, n]) if i < n else tuple([n, i]))
            adjacency_list[n].discard(i)

    # get boundary edges
    boundary_edges = [np.array(edge) for edge in edges if len(edge_adjacency_dict[edge]) < 2]
    return np.array(boundary_edges)


def get_ordered_ctrl_pts(prep_mesh, degree, ctrl_pt_spacing=1.):
    """
    Find the ordered boundary verts and return a series of control points sampled from them.
    Args:
        prep_mesh (open3d.geometry.TriangleMesh): the prep mesh
        ctrl_pt_spacing (float, optional): 
                The spacing distance kept between each pair of 
                adjacent control points. Defaults to 1.2.
    Return:
        ([n,3] nd array): The sampled ctrl points
    """
    ########################################################
    # 1. Get the boundary vertices
    ########################################################
    # remove duplicated verts and non-manifold vertices
    prep_mesh.remove_duplicated_vertices()
    prep_mesh.compute_adjacency_list()
    adjacency_list = prep_mesh.adjacency_list
    non_mani_vert_ids = [i for i in range(len(adjacency_list)) if len(adjacency_list[i]) < 3]
    prep_mesh.remove_vertices_by_index(non_mani_vert_ids)
    # remove isolated islands
    triangle_clusters, cluster_n_triangles, cluster_area = (prep_mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    isolated_tri_ids = [i for i, x in enumerate(triangle_clusters) if x != 0]
    prep_mesh.remove_triangles_by_index(isolated_tri_ids)
    prep_mesh.compute_vertex_normals()

    # get the boundary pts
    prep_verts = np.asarray(prep_mesh.vertices)
    boundary_edges = get_mesh_boundary_edges(prep_mesh)
    if vis:
        vis_pts = []
        lines = []
        pt_id_dict = {}
        for id in np.unique(boundary_edges):
            vis_pts.append(np.array(prep_verts[id]))
            pt_id_dict[id] = len(vis_pts) - 1
        for e in boundary_edges:
            lines.append([pt_id_dict[e[0]], pt_id_dict[e[1]]])
        vis_pts_pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(vis_pts))
        vis_pts_pc.paint_uniform_color([1, 0, 0])
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(vis_pts),
            lines=open3d.utility.Vector2iVector(lines),
        )
        line_set.paint_uniform_color([1, 0, 0])
        open3d.visualization.draw_geometries([vis_pts_pc, line_set, prep_mesh], mesh_show_back_face=True,
                                             window_name="all boundary pts")
    ########################################################
    # 2. sort the boundary vertices
    ########################################################
    edge_list = [(e[0], e[1]) for e in boundary_edges]
    boundaries = get_cycles_networkx(edge_list)
    edge_pt_ids_sorted = max(boundaries, key=len)

    if vis:
        vis_pts = []
        for id in edge_pt_ids_sorted:
            vis_pts.append(np.array(prep_verts[id]))
        vis_pts = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(vis_pts))
        vis_pts.paint_uniform_color([0, 0, 1])
        colors = np.asarray(vis_pts.colors)
        for i, c in enumerate(colors):
            x = i / len(colors)
            colors[i] = [0, 0, x]
        vis_pts.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([vis_pts, prep_mesh], mesh_show_back_face=True, window_name="edge pts")
    ########################################################
    # 3. Sample ctrl points from the boundary vertices
    ########################################################
    sampled_boundaries = []
    ctrl_pt_ids = [edge_pt_ids_sorted[0]]
    total_edge_length = 0
    # get total edge length
    for vert_id in range(len(edge_pt_ids_sorted) - 1):
        total_edge_length += euclidean(prep_verts[edge_pt_ids_sorted[vert_id]], \
                                       prep_verts[edge_pt_ids_sorted[vert_id + 1]])
    # start from idx 0, iterate the sorted edge verts and select ctrl pts every ctrl_pt_spacing
    cur_ctrl_pt_idx = edge_pt_ids_sorted[0]
    cur_traveled_dist = 0
    for vert_id in range(len(edge_pt_ids_sorted) - 1):
        if euclidean(prep_verts[cur_ctrl_pt_idx], prep_verts[edge_pt_ids_sorted[vert_id + 1]]) < ctrl_pt_spacing \
                or cur_traveled_dist > total_edge_length:
            cur_traveled_dist += euclidean(prep_verts[edge_pt_ids_sorted[vert_id]], \
                                           prep_verts[edge_pt_ids_sorted[vert_id + 1]])
            continue
        else:
            ctrl_pt_ids.append(edge_pt_ids_sorted[vert_id])
            cur_ctrl_pt_idx = edge_pt_ids_sorted[vert_id]
    if len(ctrl_pt_ids) > degree + 1:
        sampled_boundaries.append(ctrl_pt_ids)

    # return the sampled ctrl points
    ctrl_pt_ids = set(ctrl_pt_ids)
    ret = [np.array([prep_verts[i] for i in ctrl_pt_ids]) for ctrl_pt_ids in sampled_boundaries]
    if vis:
        vis_pts = []
        for p in ret[0]:
            vis_pts.append(p)
        vis_pts = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(vis_pts))
        vis_pts.paint_uniform_color([0, 0, 1])
        colors = np.asarray(vis_pts.colors)
        for i, c in enumerate(colors):
            x = i / len(colors)
            colors[i] = [0, 0, x]
        vis_pts.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([vis_pts, prep_mesh], mesh_show_back_face=True, window_name="ctrl pts")
    return ret


def get_nurbs_curve(ordered_boundary_pts, degree=3):
    """
    Get 100 points on a nurbs curve using the control points.
    Args:
        ordered_boundary_pts ([n,3] nd array): the control points.
    Returns:
        ([n,3] nd array): 100 points on the spline curve.
    """
    # Create a B-Spline curve instance
    curve = NURBS.Curve()
    # Set up curve
    curve.degree = degree
    ctrl_pts = [list(p) for p in ordered_boundary_pts[0]]
    curve.ctrlpts = ctrl_pts
    # Auto-generate knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, curve.ctrlpts_size)
    # # Set evaluation delta
    curve.delta = 0.01  # number of sampled pts = 1/delta
    evalpts = np.array(curve.evalpts)

    return evalpts


def move_outward(curve_pts, dist):
    """Move the curve pts outward, away from the curve center for d millimeters"""
    curve_pts_moved = []
    curve_center = np.average(curve_pts, axis=0)
    for p in curve_pts:
        v = p - curve_center  # vector from p to the curve center
        moved_p = p + (dist / np.linalg.norm(v, 1)) * v
        curve_pts_moved.append(moved_p)
    return curve_pts_moved


def main():
    degree = 4
    data_dir = r'/home/wanglong/PycharmProjects/lambda_crown/data/failed_case'
    # f_paths = [os.path.join(data_dir, dir, '16.stl') for dir in os.listdir(data_dir)]
    f_paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    # data_dir = 'AI牙冠测试报错'
    # f_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    for f in f_paths:
        print(f)
        # read mesh
        mesh = open3d.io.read_triangle_mesh(f)
        # get boundary points
        ordered_ctrl_pts = get_ordered_ctrl_pts(mesh, degree)
        # get curve
        nurbs_curve = get_nurbs_curve(ordered_ctrl_pts, degree=degree)
        # move the curve pts inside the prep mesh
        nurbs_curve_moved = move_outward(nurbs_curve, dist=0.2)
        if vis:
            nurbs_curve = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(nurbs_curve))
            nurbs_curve.paint_uniform_color([0, 0, 0])
            colors = np.asarray(nurbs_curve.colors)
            for i, c in enumerate(colors):
                colors[i] = [i / len(colors), 0, 0]
            nurbs_curve.colors = open3d.utility.Vector3dVector(colors)

            nurbs_curve_moved = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(nurbs_curve_moved))
            nurbs_curve_moved.paint_uniform_color([0, 0, 0])
            colors_moved = np.asarray(nurbs_curve_moved.colors)
            for i, c in enumerate(colors_moved):
                colors_moved[i] = [0, i / len(colors_moved), 0]
            nurbs_curve_moved.colors = open3d.utility.Vector3dVector(colors_moved)

            mesh.compute_vertex_normals()
            open3d.visualization.draw_geometries([nurbs_curve, nurbs_curve_moved, mesh], mesh_show_back_face=True,
                                                 window_name="curve pts")


if __name__ == "__main__":
    main()
