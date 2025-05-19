# -*-coding:utf-8-*-
import numpy as np
import trimesh


class TPS:
    """The thin plate spline deformation warpping.
    """

    def __init__(self,
                 control_points: np.ndarray,
                 target_points: np.ndarray,
                 lambda_: float = 0.,
                 solver: str = 'exact'):
        """Create a instance that preserve the TPS coefficients.

        Arguments
        ---------
            control_points : np.array
                p by d vector of control points
            target_points : np.array
                p by d vector of corresponding target points on the deformed
                surface
            lambda_ : float
                regularization parameter
            solver : str
                the solver to get the coefficients. default is 'exact' for the
                exact solution. Or use 'lstsq' for the least square solution.
        """
        self.control_points = control_points
        self.coefficient = find_coefficients(
            control_points, target_points, lambda_, solver)

    def __call__(self, source_points):
        """Transform the source points form the original surface to the
        destination (deformed) surface.

        Arguments
        ---------
            source_points : np.array
                n by d array of source points to be transformed
        """
        return transform(source_points, self.control_points,
                         self.coefficient)

    transform = __call__


def cdist(K: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distance between K[i, :] and B[j, :].

    Arguments
    ---------
        K : np.array
        B : np.array
    """
    K = np.atleast_2d(K)
    B = np.atleast_2d(B)
    assert K.ndim == 2
    assert B.ndim == 2

    K = np.expand_dims(K, 1)
    B = np.expand_dims(B, 0)
    D = K - B
    return np.linalg.norm(D, axis=2)


def pairwise_radial_basis(K: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the TPS radial basis function phi(r) between every row-pair of K
    and B where r is the Euclidean distance.

    Arguments
    ---------
        K : np.array
            n by d vector containing n d-dimensional points.
        B : np.array
            m by d vector containing m d-dimensional points.

    Return
    ------
        P : np.array
            n by m matrix where.
            P(i, j) = phi( norm( K(i,:) - B(j,:) ) ),
            where phi(r) = r^2*log(r), if r >= 1
                           r*log(r^r), if r <  1
    """
    # r_mat(i, j) is the Euclidean distance between K(i, :) and B(j, :).
    r_mat = cdist(K, B)

    pwise_cond_ind1 = r_mat >= 1
    pwise_cond_ind2 = r_mat < 1
    r_mat_p1 = r_mat[pwise_cond_ind1]
    r_mat_p2 = r_mat[pwise_cond_ind2]

    # P correcponds to the matrix K from [1].
    P = np.empty(r_mat.shape)
    P[pwise_cond_ind1] = (r_mat_p1 ** 2) * np.log(r_mat_p1)
    P[pwise_cond_ind2] = r_mat_p2 * np.log(np.power(r_mat_p2, r_mat_p2))

    return P


def find_coefficients(control_points: np.ndarray,
                      target_points: np.ndarray,
                      lambda_: float = 0.,
                      solver: str = 'exact') -> np.ndarray:
    """Given a set of control points and their corresponding points, compute the
    coefficients of the TPS interpolant deforming surface.

    Arguments
    ---------
        control_points : np.array
            p by d vector of control points
        target_points : np.array
            p by d vector of corresponding target points on the deformed
            surface
        lambda_ : float
            regularization parameter
        solver : str
            the solver to get the coefficients. default is 'exact' for the exact
            solution. Or use 'lstsq' for the least square solution.

    Return
    ------
        coef : np.ndarray
            the coefficients

    .. seealso::

        http://cseweb.ucsd.edu/~sjb/pami_tps.pdf
    """
    # ensure data type and shape
    control_points = np.atleast_2d(control_points)
    target_points = np.atleast_2d(target_points)
    if control_points.shape != target_points.shape:
        raise ValueError(
            'Shape of and control points {cp} and target points {tp} are not the same.'.
                format(cp=control_points.shape, tp=target_points.shape))

    p, d = control_points.shape

    # The matrix
    K = pairwise_radial_basis(control_points, control_points)
    P = np.hstack([np.ones((p, 1)), control_points])

    # Relax the exact interpolation requirement by means of regularization.
    K = K + lambda_ * np.identity(p)

    # Target points
    M = np.vstack([
        np.hstack([K, P]),
        np.hstack([P.T, np.zeros((d + 1, d + 1))])
    ])
    Y = np.vstack([target_points, np.zeros((d + 1, d))])
    for n in range(M.shape[0]):
        M[n, n] += 1e-6
    # solve for M*X = Y.
    # At least d+1 control points should not be in a subspace; e.g. for d=2, at
    # least 3 points are not on a straight line. Otherwise M will be singular.
    solver = solver.lower()
    if solver == 'exact':
        X = np.linalg.solve(M, Y)
    elif solver == 'lstsq':
        X, _, _, _ = np.linalg.lstsq(M, Y, None)
    else:
        raise ValueError('Unknown solver: ' + solver)

    return X


def transform(source_points: np.ndarray, control_points: np.ndarray,
              coefficient: np.ndarray) -> np.ndarray:
    """Transform the source points form the original surface to the destination
    (deformed) surface.

    Arguments
    ---------
        source_points : np.array
            n by d array of source points to be transformed
        control_points : np.array
            the control points used in the function `find_coefficients`
        coefficient : np.array
            the computed coefficients

    Return
    ------
        deformed_points : np.array
            n by d array of the transformed point on the target surface
    """
    source_points = np.atleast_2d(source_points)
    control_points = np.atleast_2d(control_points)
    if source_points.shape[-1] != control_points.shape[-1]:
        raise ValueError(
            'Dimension of source points ({sd}D) and control points ({cd}D) are not the same.'.
                format(sd=source_points.shape[-1], cd=control_points.shape[-1]))

    n = source_points.shape[0]
    A = pairwise_radial_basis(source_points, control_points)
    K = np.hstack([A, np.ones((n, 1)), source_points])

    deformed_points = np.dot(K, coefficient)
    return deformed_points


def tps2(mesh, point_idx, point_dst):
    '''
    Args:
        mesh: 待变形网格模型
        point_idx: 待变形网格顶点索引 shape:(N)
        point_dst: 关键点坐标 shape:(N, 3)

    Returns:
        mesh_out: 变形后的网格模型
    '''
    V_tooth = mesh.vertices
    V_tooth1 = mesh.vertices.copy()
    point_src = V_tooth[point_idx]
    point_num = len(point_idx)
    P = np.ones((point_num, 4))
    K = np.zeros((point_num, point_num))
    L_tmp = np.zeros((point_num + 4, point_num + 4))
    Y = np.zeros((point_num + 4, 3))
    P[:, 1:4] = point_src
    P_T = P.T
    for i in range(point_num):
        for j in range(i, point_num):
            diff_p = point_src[i] - point_src[j]
            K[i, j] = K[j, i] = diff_p[0] ** 2 + diff_p[1] ** 2 + diff_p[2] ** 2
    K = np.where(K == 0, 0, K * np.log(K + 1e-5))
    K = K + 0.5 * np.identity(point_num)
    L_tmp[:point_num, :point_num] = K
    L_tmp[:point_num, point_num:] = P
    L_tmp[point_num:, :point_num] = P_T
    Y[:point_num] = point_dst
    for n in range(L_tmp.shape[0]):
        L_tmp[n, n] += 1e-6
    W = np.linalg.inv(L_tmp) @ Y
    a1_x = W[-4, 0]
    ax_x = W[-3, 0]
    ay_x = W[-2, 0]
    az_x = W[-1, 0]
    a1_y = W[-4, 1]
    ax_y = W[-3, 1]
    ay_y = W[-2, 1]
    az_y = W[-1, 1]
    a1_z = W[-4, 2]
    ax_z = W[-3, 2]
    ay_z = W[-2, 2]
    az_z = W[-1, 2]
    for n in range(len(V_tooth)):
        nonrigid_x, nonrigid_y, nonrigid_z = 0, 0, 0
        affine_x = a1_x + ax_x * V_tooth[n, 0] + ay_x * V_tooth[n, 1] + az_x * V_tooth[n, 2]
        affine_y = a1_y + ax_y * V_tooth[n, 0] + ay_y * V_tooth[n, 1] + az_y * V_tooth[n, 2]
        affine_z = a1_z + ax_z * V_tooth[n, 0] + ay_z * V_tooth[n, 1] + az_z * V_tooth[n, 2]
        for i in range(point_num):
            diff_p = point_src[i] - V_tooth[n]
            diff = diff_p[0] ** 2 + diff_p[1] ** 2 + diff_p[2] ** 2
            tps_diff = np.where(diff == 0, 0, diff * np.log(diff + 1e-5))
            nonrigid_x += W[i, 0] * tps_diff
            nonrigid_y += W[i, 1] * tps_diff
            nonrigid_z += W[i, 2] * tps_diff
        out = np.array([[affine_x + nonrigid_x], [affine_y + nonrigid_y], [affine_z + nonrigid_z]])
        V_tooth1[n] = out[:, 0]
    mesh_out = trimesh.Trimesh(vertices=V_tooth1, faces=mesh.faces, process=False)
    return mesh_out


def tps_runner(mesh, point_idx, point_dst, lambda_=0.5):
    trans = TPS(mesh.vertices[point_idx], point_dst, lambda_=lambda_)
    transformed_xyz = trans(mesh.vertices)
    ret = trimesh.Trimesh(transformed_xyz, mesh.faces)
    ret.vertex_normals = mesh.vertex_normals
    return ret


if __name__ == '__main__':
    """
    A demo of how to use TPS()
    """
    mesh = trimesh.load('cube.obj')
    point_idx = np.array([0, 1, 2])
    point_dst = np.array([[0., 0., 0.],[1., 1., 1.],[2., 2., 2.]])
    trans = TPS(mesh.vertices[point_idx], point_dst)
    transformed_xyz = trans(mesh.vertices)
    transformed_mesh = trimesh.Trimesh(transformed_xyz, mesh.faces)
    pass
