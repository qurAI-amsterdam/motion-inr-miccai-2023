import SimpleITK as sitk
import numpy as np
import torch
from kwatsch.common import make_homegeneous_identity_grid
from torch import inverse as tr_inv


def transform_grid_lax_sax(
    tgt_shape: tuple,
    src_shape: tuple,
    tr_S_tgt: torch.Tensor,
    tr_R_tgt: torch.Tensor,
    tr_T_tgt: torch.Tensor,
    tr_S_src: torch.Tensor,
    tr_R_src: torch.Tensor,
    tr_T_src: torch.Tensor,
    device: str = "cpu",
    lax_target_struc=None,
):
    """
    tgt_shape: target shape [z, y, x]
    src_shape: source shape [z, y, x]
    tr_S_tgt, tr_R_tgt, tr_T_tgt: matrices for scaling, rotation and translation. in homogeneous coordinate
                                    format [4x4]
                                    Extracted from Nifti images (rotation = directions from nifti image)
    (1) convert from source scanner coordinate system to world coordinate system
    (2) from world coordinate system to target scanner coordinate system

    returns: torch tensor with transformed coordinates of source image

    """
    ident_grid = make_homegeneous_identity_grid(tgt_shape, device=device)
    if lax_target_struc is not None:
        lax_coord_mask = (
            torch.from_numpy(lax_target_struc != 0).float().to(device)[None, ..., None]
        )
        ident_grid[..., :3] = ident_grid[..., :3] * lax_coord_mask
    world_grid = (
        ident_grid.reshape(tgt_shape[0], -1, 4) @ tr_S_tgt.T @ tr_R_tgt.T @ tr_T_tgt.T
    )

    # we need to bring world coordinates back to voxel grid
    src_coord = (
        world_grid @ tr_inv(tr_T_src).T @ tr_inv(tr_R_src).T @ tr_inv(tr_S_src).T
    )
    # Todo: yes this is hacky. To normalize coordinates we need to make sure z-dim is not 1
    #  (because we subtract one below)
    if src_shape[0] == 1:
        shape_div = torch.tensor(
            src_shape[1:][::-1]
            + (
                2,
                2,
            ),
            dtype=torch.float32,
            device=tr_S_tgt.device,
        )
    else:
        shape_div = torch.tensor(
            src_shape[::-1] + (2,), dtype=torch.float32, device=tr_S_tgt.device
        )
    print("shape_div ", src_coord.shape, src_shape, shape_div)
    # Normalize coordinates between [-1, 1] for torch grid_sampler
    trans_src_coord = (src_coord / ((shape_div[None, None, None] - 1) / 2)) - 1
    trans_src_coord = trans_src_coord.reshape(tuple(tgt_shape) + (4,))
    # get rid off homogeneous coords
    trans_src_coord = trans_src_coord[..., :3]
    return trans_src_coord


def get_mri_transforms(
    src_image: sitk.Image, tgt_image: sitk.Image, dim=3, device="cpu"
):
    # spacings assumed to be in shape xyz order
    spacing_src, spacing_tgt = src_image.GetSpacing(), tgt_image.GetSpacing()
    R_src = np.zeros((4, 4))
    R_src[:3, :3] = np.reshape(np.asarray(src_image.GetDirection()), (dim, dim)).astype(
        np.float32
    )
    R_src[3, 3] = 1

    T_src = np.eye(4)
    T_src[:3, 3] = np.asarray(src_image.GetOrigin()).astype(np.float32)
    S_src = np.eye(4)
    S_src[:3, :3] = np.diag(np.asarray(spacing_src)).astype(np.float32)
    tr_R_src, tr_S_src, tr_T_src = (
        torch.from_numpy(R_src.astype(np.float32)).to(device),
        torch.from_numpy(S_src.astype(np.float32)).to(device),
        torch.from_numpy(T_src.astype(np.float32)).to(device),
    )

    # same for LAX view
    dim = len(tgt_image.GetSize())
    R_tgt = np.zeros((4, 4))
    R_tgt[:dim, :dim] = np.reshape(
        np.asarray(tgt_image.GetDirection()), (dim, dim)
    ).astype(np.float32)
    R_tgt[dim:, dim:] = 1

    T_tgt = np.eye(4)
    T_tgt[:dim, dim] = np.asarray(tgt_image.GetOrigin()).astype(np.float32)
    S_tgt = np.eye(4)
    S_tgt[:dim, :dim] = np.diag(np.asarray(spacing_tgt)).astype(np.float32)
    tr_R_tgt, tr_S_tgt, tr_T_tgt = (
        torch.from_numpy(R_tgt.astype(np.float32)).to(device),
        torch.from_numpy(S_tgt.astype(np.float32)).to(device),
        torch.from_numpy(T_tgt.astype(np.float32)).to(device),
    )
    return tr_R_src, tr_S_src, tr_T_src, tr_R_tgt, tr_S_tgt, tr_T_tgt
