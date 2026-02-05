import torch
import numpy as np
import SimpleITK as sitk
import yaml
from IPython import get_ipython

KEY_SAX_VIEW = "sax"
KEY_SAX_SEG_VIEW = "sax_seg"
KEY_4CH_VIEW = "lax4ch"
KEY_4CH_SEG_VIEW = "lax4ch_seg"
KEY_2CH_VIEW = "lax2ch"
KEY_2CH_SEG_VIEW = "lax2ch_seg"


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            print("This is a notebook")
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def identity_grid(dims, device="cuda", dtype=torch.float32, do_flip_sequence=False):
    """Make a coordinate tensor."""
    coordinate_tensor = [torch.arange(s, dtype=dtype, device=device) for s in dims]
    coordinate_tensor = torch.meshgrid(coordinate_tensor, indexing="ij")
    if do_flip_sequence:
        coordinate_tensor = torch.stack(coordinate_tensor[::-1], dim=len(dims))
    else:
        coordinate_tensor = torch.stack(coordinate_tensor, dim=len(dims))
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.to(device=device)

    return coordinate_tensor


def make_identity_grid(shape: tuple, stackdim="last", device="cpu"):
    # Assuming shape: [z, y, x]
    if isinstance(stackdim, int):
        dim = stackdim
    elif stackdim == "last":
        dim = len(shape)
    elif stackdim == "first":
        dim = 0
    else:
        Exception("Incorrect stackdim given.")

    coords = [torch.arange(0, s, dtype=torch.float32, device=device) for s in shape]

    grids = torch.meshgrid(coords, indexing="ij")
    # we need to reverse ordering of last dim for coordinates because shape is
    # z, y, x and we want (x, y, z) coords
    return torch.stack(grids[::-1], dim=dim)


def make_homegeneous_identity_grid(target_shape: tuple, device="cpu"):
    # target_shape is [z, y, x]
    grid = make_identity_grid(target_shape, device=device)
    # for homogeneous coordinates we need to add fourth dim with constant value "1"
    h_coords = torch.ones(grid.shape[:-1] + (1,), device=device)
    grid = torch.cat([grid, h_coords], dim=-1)
    return grid


def get_voxel_to_world_transforms(tgt_image: sitk.Image, device="cpu"):
    """
    Generate transformation matrices (homogeneous coordinates) to transform voxel coordinates
    from source image to voxel coordinates of target image (or vice versa)
    For source and target image the matrices describe the transformation from voxel coordinate
    to world coordinates (common patient coordinate system).
    tr_R_tgt: source: rotation matrix from coordinate to world
    tr_S_tgt: source: v

    oxel scaling
    tr_T_tgt: source: set new origin in world coordinate
    """
    # spacings assumed to be in shape xyz order
    spacing_tgt = tgt_image.GetSpacing()

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
    return tr_R_tgt, tr_S_tgt, tr_T_tgt


def execute_resampling(
    src_3d_tensor: torch.Tensor,
    transformed_ident_grid: torch.Tensor,
    mode="bilinear",
    do_detach=True,
) -> torch.Tensor:

    if src_3d_tensor.dim() == 3:
        src_3d_tensor = src_3d_tensor[None, None]
    elif src_3d_tensor.dim() == 2:
        src_3d_tensor = src_3d_tensor[None, None, None]
    if transformed_ident_grid.dim() == 4:
        transformed_ident_grid = transformed_ident_grid[None]
    elif transformed_ident_grid.dim() == 3:
        transformed_ident_grid = transformed_ident_grid[None, None]
    if src_3d_tensor.device != transformed_ident_grid.device:
        src_3d_tensor = src_3d_tensor.to(transformed_ident_grid.device)
    resampled_img = torch.nn.functional.grid_sample(
        src_3d_tensor,
        transformed_ident_grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )  # assuming coords refer to center of coords
    if do_detach:
        return resampled_img.detach().cpu().squeeze()
    else:
        return resampled_img


def loadExperSettings(fname):
    with open(fname, "r") as fp:
        kwargs = yaml.load(fp, Loader=yaml.FullLoader)
    return kwargs


def saveExperimentSettings(args, fname):
    if isinstance(args, dict):
        with open(fname, "w") as fp:
            yaml.dump(args, fp)
    else:
        with open(fname, "w") as fp:
            yaml.dump(vars(args), fp)
