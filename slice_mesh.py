import open3d
import numpy as np
import time


def read_mesh(path: str) -> open3d.geometry.TriangleMesh:
    mesh = open3d.io.read_triangle_mesh(path)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def get_mesh_edges(mesh):
    # Get triangles
    triangles = np.asarray(mesh.triangles)
    
    # Create edges from triangles
    edges = set()
    for triangle in triangles:
        # Add all three edges of each triangle
        # Sort vertex indices to avoid duplicates
        edges.add(tuple(sorted([triangle[0], triangle[1]])))
        edges.add(tuple(sorted([triangle[1], triangle[2]])))
        edges.add(tuple(sorted([triangle[2], triangle[0]])))
    
    return np.array(list(edges))

def find_intersecting_edges(mesh1, mesh2):
    edges = get_mesh_edges(mesh1)
    vertices = np.asarray(mesh1.vertices)
    
    # Create all rays at once
    ray_origins = vertices[edges[:, 0]]
    ray_ends = vertices[edges[:, 1]]
    ray_directions = ray_ends - ray_origins
    ray_lengths = np.linalg.norm(ray_directions, axis=1)
    ray_directions = ray_directions / ray_lengths[:, np.newaxis]
    
    # Stack origins and directions for batch processing
    rays = np.hstack([ray_origins, ray_directions])
    
    # Create ray casting scene
    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(mesh2))
    
    # Batch ray casting
    rays_tensor = open3d.core.Tensor(rays, dtype=open3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_tensor)
    
    # Find intersecting edges
    intersecting_mask = ans['t_hit'].numpy() < ray_lengths
    intersecting_edges = edges[intersecting_mask]
    
    return intersecting_edges


def compute_signed_distance(mesh, q_points):
    mesh_o3d = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)
    closest_points = scene.compute_closest_points(
        np.asarray(q_points, dtype=np.float32)
    )

    distance = np.linalg.norm(q_points - closest_points["points"].numpy(), axis=-1)
    nonzero = distance > 0

    # For closest points that project directly in to the triangle, compute sign from
    # triangle normal Project each point in to the closest triangle plane
    nonzero = np.where(nonzero)[0]
    normals = closest_points["primitive_normals"].numpy()
    projection = (
        q_points[nonzero]
        - (
            normals[nonzero].T
            * np.einsum(
                "ij,ij->i",
                q_points[nonzero] - closest_points["points"].numpy()[nonzero],
                normals[nonzero],
            )
        ).T
    )

    sign = np.sign(
        np.einsum("ij,ij->i", normals[nonzero], q_points[nonzero] - projection)
    )
    distance[nonzero] *= -1.0 * sign
    return distance, closest_points["points"].numpy()


def slice_mesh(mesh, cutter):
    intersecting_edges = find_intersecting_edges(mesh, cutter)
    mesh.remove_vertices_by_index(np.unique(intersecting_edges.flatten()))
    # only keep the largest cluster connected component
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    best_cluster = None
    max_ratio = 0
    for cluster in range(len(cluster_n_triangles)):
        cluster_triangle_ids = np.unique(np.where(triangle_clusters==cluster)[0].flatten())
        cluster_vertex_ids = np.unique(np.asarray(mesh.triangles)[cluster_triangle_ids])
        cluster_distances, _ = compute_signed_distance(cutter, np.asarray(mesh.vertices)[cluster_vertex_ids])
        positive_count = np.sum(cluster_distances < 0)
        ratio = positive_count / len(cluster_distances)
        if ratio > max_ratio:
            max_ratio = ratio
            best_cluster = cluster
    
    triangle_ids = np.where(triangle_clusters != best_cluster)[0]
    mesh.remove_triangles_by_index(triangle_ids)
    mesh.remove_unreferenced_vertices()
    return mesh


if __name__ == "__main__":
    import os, re
    # open3d.t.geometry.TriangleMesh.boolean_difference()
    data_dir = '/media/chuanbo/39DCD6C67C0FD9AF/data/嵌体数据'
    failed_cases = [
        # '2018-04-10_13-30_尹新芹',
        ]
    cases = [c for c in os.listdir(data_dir) if c not in failed_cases]
    # cases = [
    #     '42f42af3-7427-4459-b8e6-4b198ed446e3',
    #     '41edef6d-ba7b-4ee3-97c7-f8b969cf76ce',
        # '060a76aa-818b-4ac9-bac5-d21c33ffa8c4'
        # '2018-04-10_13-30_尹新芹',
        # '8d351eba-b266-48de-bbe5-52a768151a8c',
        # '55ea2904-10c7-4f7b-a95f-d15bb7dc6865',  # submesh 
        # '51aec258-a7a7-4368-bb78-0627473e9f8f',
        # 'designer_Zheng_20230202_1158_增奇口腔_袁_10922266',
        # '97728_20220927_1535_王兆杰10902624',
        # ]
    # cases = [x for _ in range(10) for x in cases]
    # import random
    # random.shuffle(cases)
    for case in cases:
        dir = os.path.join(data_dir, case)
        inlay_fnames = [fname for fname in os.listdir(dir) if re.match(r'^\d{2}[qQ]+.stl$', fname)]
        # inlay_fnames = ['15Q.stl']
        for inlay_fname in inlay_fnames:
            # try:
            tid = int(inlay_fname[:2])
            print(f'tooth {tid} of {case}:')
            mesh = read_mesh(os.path.join(dir, 'test_results_v0.0.6',f'5.5_inflated_outer_{tid}.stl'))
            cutter = read_mesh(os.path.join(dir, 'test_results_v0.0.6',f'5.6_poisson_mesh_{tid}.stl'))
            t = time.time()
            result = slice_mesh(mesh, cutter)
            result.compute_vertex_normals()
            print(f'total time: {time.time() - t}')
            open3d.visualization.draw_geometries([result, cutter])
            # except Exception as e:
            #     print(f'Error in {dir}  {tid}: {e}')
            #     print()