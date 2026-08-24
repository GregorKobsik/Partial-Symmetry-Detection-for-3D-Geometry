from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union
from collections import namedtuple

from tqdm.auto import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

import torch
import pytorch3d.ops.utils as oputil
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.loss import chamfer_distance

from igl import rigid_alignment
import open3d as o3d


if TYPE_CHECKING:
    from pytorch3d.structures.pointclouds import Pointclouds


# named tuples for inputs/outputs
class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor


class ICPSolution(NamedTuple):
    converged: bool
    rmse: Union[torch.Tensor, None]
    Xt: torch.Tensor
    RTs: SimilarityTransform
    t_history: List[SimilarityTransform]


def _apply_similarity_transform(X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X


# [docs]
def iterative_closest_point_to_plane(
    X: Union[torch.Tensor, "Pointclouds"],
    Y: Union[torch.Tensor, "Pointclouds"],
    N: Union[torch.Tensor],
    init_transform: Optional[SimilarityTransform] = None,
    max_iterations: int = 100,
    relative_rmse_thr: float = 1e-6,
    verbose: bool = False,
) -> ICPSolution:
    """
    Executes the iterative closest point (ICP) algorithm [1, 2] in order to find
    a similarity transformation (rotation `R` and translation `T`) between two
    given differently-sized sets of `d`-dimensional points `X` and `Y`, such that:

    `X[i] R[i] + T[i] = Y[NN[i]]`,

    for all batch indices `i` in the least squares sense. Here, Y[NN[i]] stands
    for the indices of nearest neighbors from `Y` to each point in `X`.
    Note, however, that the solution is only a local optimum.

    Args:
        **X**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_X, d)` or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_Y, d)` or a `Pointclouds` object.
        **N**: Batch of `d`-dimensional normal vectors corresponding to Y
            of shape `(minibatch, num_points_Y, d)`.
        **init_transform**: A named-tuple `SimilarityTransform` of tensors
            `R`, `T, `s`, where `R` is a batch of orthonormal matrices of
            shape `(minibatch, d, d)`, `T` is a batch of translations
            of shape `(minibatch, d)` and `s` is a batch of scaling factors
            of shape `(minibatch,)`.
        **max_iterations**: The maximum number of ICP iterations.
        **relative_rmse_thr**: A threshold on the relative root mean squared error
            used to terminate the algorithm.
        **verbose**: If `True`, prints status messages during each ICP iteration.

    Returns:
        A named tuple `ICPSolution` with the following fields:
        **converged**: A boolean flag denoting whether the algorithm converged
            successfully (=`True`) or not (=`False`).
        **rmse**: Attained root mean squared error after termination of ICP.
        **Xt**: The point cloud `X` transformed with the final transformation
            (`R`, `T`, `s`). If `X` is a `Pointclouds` object, returns an
            instance of `Pointclouds`, otherwise returns `torch.Tensor`.
        **RTs**: A named tuple `SimilarityTransform` containing
        a batch of similarity transforms with fields:
            **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
            **T**: Batch of translations of shape `(minibatch, d)`.
            **s**: batch of scaling factors of shape `(minibatch, )`.
        **t_history**: A list of named tuples `SimilarityTransform`
            the transformation parameters after each ICP iteration.

    References:
        [1] TODO: Cite Point to Normal - ICP
        [2] https://en.wikipedia.org/wiki/Iterative_closest_point
    """

    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)
    Nt, num_points_N = oputil.convert_pointclouds_to_tensor(N)

    if Yt.shape[1] != Nt.shape[1]:
        raise ValueError(
            "Point set Y and corresponding normal vectors N " + "have to have the same number of data points."
        )

    b, size_X, dim = Xt.shape

    if (Xt.shape[2] != Yt.shape[2]) or (Xt.shape[0] != Yt.shape[0]):
        raise ValueError("Point sets X and Y have to have the same " + "number of batches and data dimensions.")

    # clone the initial point cloud
    Xt_init = Xt.clone()

    if init_transform is not None:
        # parse the initial transform from the input and apply to Xt
        try:
            R, T, s = init_transform
            assert (
                R.shape == torch.Size((b, dim, dim)) and T.shape == torch.Size((b, dim)) and s.shape == torch.Size((b,))
            )
        except Exception:
            raise ValueError(
                "The initial transformation init_transform has to be "
                "a named tuple SimilarityTransform with elements (R, T, s). "
                "R are dim x dim orthonormal matrices of shape "
                "(minibatch, dim, dim), T is a batch of dim-dimensional "
                "translations of shape (minibatch, dim) and s is a batch "
                "of scalars of shape (minibatch,)."
            ) from None
        # apply the init transform to the input point cloud
        Xt = _apply_similarity_transform(Xt, R, T, s)
    else:
        # initialize the transformation with identity
        R = oputil.eyes(dim, b, device=Xt.device, dtype=Xt.dtype)
        T = Xt.new_zeros((b, dim))
        s = Xt.new_ones(b)

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False

    # initialize the transformation history
    t_history = []

    # the main loop over ICP iterations
    for iteration in range(max_iterations):
        # obtain points and normals by computing the nearest neighbor
        Xt_nn_idx = knn_points(Xt, Yt, lengths1=num_points_X, lengths2=num_points_Y, K=1, return_nn=True).idx
        Xt_nn_points = knn_gather(Yt, Xt_nn_idx, num_points_Y)[:, :, 0]
        Xt_nn_normals = knn_gather(Nt, Xt_nn_idx, num_points_N)[:, :, 0]

        # find the rigid transformation that best aligns the 3D points X to their corresponding points P with associated normals N
        # min ‖(X*R+t-P)’N‖² R∈SO(3) t∈R³
        R = torch.zeros(b, 3, 3, device=Xt_init.device)
        t = torch.zeros(b, 3, device=Xt_init.device)
        for b_idx in range(b):
            _R, _t = rigid_alignment(
                x=Xt_init[b_idx].cpu().numpy(),
                p=Xt_nn_points[b_idx].cpu().numpy(),
                n=Xt_nn_normals[b_idx].cpu().numpy(),
            )
            R[b_idx] = torch.tensor(_R, device=Xt_init.device)
            t[b_idx] = torch.tensor(_t[0], device=Xt_init.device)

        # apply the estimated similarity transform to Xt_init
        Xt = _apply_similarity_transform(Xt_init, R, T, s)

        # add the current transformation to the history
        t_history.append(SimilarityTransform(R, T, s))

        # compute the root mean squared error
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        Xt_sq_diff = ((Xt - Xt_nn_points) ** 2).sum(2)
        rmse = torch.mean(Xt_sq_diff, dim=1).sqrt()

        # compute the relative rmse
        if prev_rmse is None:
            relative_rmse = rmse.new_ones(b)
        else:
            relative_rmse = (prev_rmse - rmse) / prev_rmse

        if verbose:
            rmse_msg = (
                f"ICP iteration {iteration}: mean/max rmse = "
                + f"{rmse.mean():1.2e}/{rmse.max():1.2e} "
                + f"; mean relative rmse = {relative_rmse.mean():1.2e}"
            )
            print(rmse_msg)

        # check for convergence
        if (relative_rmse <= relative_rmse_thr).all():
            converged = True
            break

        # update the previous rmse
        prev_rmse = rmse

    if verbose:
        if converged:
            print(f"ICP has converged in {iteration + 1} iterations.")
        else:
            print(f"ICP has not converged in {max_iterations} iterations.")

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)


def icp_distance_p3d(points, normals=None):

    dev = "cuda"
    points = points.to(dev)
    if normals is not None:
        normals = normals.to(dev)

    # perform ICP for each sample individually
    SimilarityTransform = namedtuple("SimilarityTransform", "R T s")
    rmse_matrix = torch.zeros(points.shape[0], points.shape[0])
    for idx in tqdm(range(points.shape[0]), desc="Samples"):

        length = points.shape[0]
        combinations = torch.tensor(np.array(np.meshgrid([idx], range(length))).T.reshape(-1, 2), device=dev)

        X = torch.index_select(points, dim=0, index=combinations[:, 0])
        Y = torch.index_select(points, dim=0, index=combinations[:, 1])
        if normals is not None:
            N = torch.index_select(normals, dim=0, index=combinations[:, 1])

        distance_matrix = []
        for i in range(10):
            random_transforms = SimilarityTransform(
                R=p3d.transforms.random_rotations(X.shape[0]).to(dev),
                T=torch.zeros(X.shape[0], 3, device=dev),
                s=torch.ones(X.shape[0], device=dev),
            )

            if normals is None:
                converged, rmse, Xt, RTs, t_history = ops.iterative_closest_point(
                    X,
                    Y,
                    estimate_scale=False,
                    allow_reflection=True,
                    init_transform=random_transforms,
                    max_iterations=100,
                )
            else:
                converged, rmse, Xt, RTs, t_history = iterative_closest_point_to_plane(
                    X,
                    Y,
                    N,
                    init_transform=random_transforms,
                    max_iterations=100,
                )

            dist = chamfer_distance(Xt, Y, batch_reduction=None)[0]
            chamfer_distances.append(dist.cpu())

        chamfer_distances = torch.min(torch.stack(chamfer_distances), dim=0)[0]
        distance_matrix[idx] = chamfer_distances.detach().clone()
    return distance_matrix


def icp_distance_o3d(points, normals=None):
    num_samples = points.shape[0]
    rmse_matrix = torch.zeros(num_samples, num_samples)

    if normals is None:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)

    for i in tqdm(range(num_samples), desc="Samples"):
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(points[i])
        if normals is not None:
            target.normals = o3d.utility.Vector3dVector(normals[i])

        for j in range(num_samples):
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(points[j])

            registration_result = None
            for _ in range(10):
                init_transform = np.eye(4, 4)
                init_transform[:3, :3] = Rotation.random().as_matrix()

                registration_icp = o3d.pipelines.registration.registration_icp(
                    source,
                    target,
                    max_correspondence_distance=0.3,
                    estimation_method=estimation_method,
                    criteria=criteria,
                    init=init_transform,
                )

                if registration_result is None:
                    registration_result = registration_icp
                elif registration_icp.inlier_rmse < registration_result.inlier_rmse:
                    registration_result = registration_icp

            rmse_matrix[i][j] = registration_result.inlier_rmse

    return rmse_matrix
