import numpy as np
from CMRI.general import filter_array
from scipy.ndimage import gaussian_filter
from CMRI.general import MMS2MRILabel

# from CMRI.motion.common import determine_aha_part, get_zpart_indices
from CMRI.general import compute_slice_range_cstructure
import math


def get_zpart_indices(z_part_dict, slice_range=None):
    z_part_indices = {"apical": [], "mid": [], "basal": []}
    for slice_id, slice_dict in z_part_dict.items():
        if slice_range is not None and slice_id not in slice_range:
            continue
        cpart = z_part_dict[slice_id].replace("apex", "apical")
        z_part_indices[cpart].extend([int(slice_id)])

    for k, v in z_part_indices.items():
        z_part_indices[k] = np.array(z_part_indices[k]).astype(np.int32)
    return z_part_indices


def determine_aha_part(seg_sax, three_slices=False, label_id=None):
    """Determine the AHA part for each slice."""
    # assume [z, y, x] for seg_sax
    # assume slices are ordered from apex to base (slice with lowest id == APEX)
    if label_id is None:
        zmask = (seg_sax != 0).any((1, 2)).astype(bool)
    else:
        zmask = (seg_sax == label_id).any((1, 2)).astype(bool)
    z_offset = np.where(zmask)[0][0]  # first slice that has cardiac anatomy
    # Divide the slices into three parts: basal, mid-cavity and apical
    n_cardiac_slice = np.count_nonzero(zmask)
    part_z = {}
    if three_slices:
        # Select three slices (basal, mid and apical) for strain analysis, inspired by:
        #
        # [1] Robin J. Taylor, et al. Myocardial strain measurement with
        # feature-tracking cardiovascular magnetic resonance: normal values.
        # European Heart Journal - Cardiovascular Imaging, (2015) 16, 871-881.
        #
        # [2] A. Schuster, et al. Cardiovascular magnetic resonance feature-
        # tracking assessment of myocardial mechanics: Intervendor agreement
        # and considerations regarding reproducibility. Clinical Radiology
        # 70 (2015), 989-998.

        # Use the slice at 25% location from base to apex.
        # Avoid using the first one or two basal slices, as the myocardium
        # will move out of plane at ES due to longitudinal motion, which will
        # be a problem for 2D in-plane motion tracking.
        z = int(round((n_cardiac_slice - 1) * 0.25))
        part_z[z + z_offset] = "apical"

        # Use the central slice.
        z = int(round((n_cardiac_slice - 1) * 0.5))
        part_z[z + z_offset] = "mid"

        # Use the slice at 75% location from base to apex.
        # In the most apical slices, the myocardium looks blurry and
        # may not be suitable for motion tracking.
        z = int(round((n_cardiac_slice - 1) * 0.75))
        part_z[z + z_offset] = "basal"
    else:
        # Use all the slices
        i1 = int(math.ceil(n_cardiac_slice / 3.0))
        i2 = int(math.ceil(2 * n_cardiac_slice / 3.0))
        i3 = n_cardiac_slice
        for i in range(0, i1):
            if i == 0:
                part_z[i + z_offset] = "apex"
            else:
                part_z[i + z_offset] = "apical"

        for i in range(i1, i2):
            part_z[i + z_offset] = "mid"

        for i in range(i2, i3):
            part_z[i + z_offset] = "basal"
    return part_z


def compute_preserve_volume(jacobian: np.ndarray, mask: np.ndarray, label=2):
    arr = jacobian.copy()
    dvf_mask = (mask == label).astype(np.int32)
    arr[dvf_mask != 1] = 1
    if arr.ndim == 2:
        return np.mean(np.abs(arr[dvf_mask == 1] - 1)), arr
    else:
        out = np.zeros(arr.shape[0])
        for i, z in enumerate(arr):
            if np.count_nonzero(z[dvf_mask[i] == 1]) > 0:
                out[i] = np.mean(np.abs(z[dvf_mask[i] == 1] - 1))
            else:
                out[i] = 0
        return out, arr


def compute_ll_strain_4ch(
    F: np.ndarray, strain_mask=None, kernel: tuple = None, do_smooth=False
):
    # NOTE ASSUMING SHAPE OF F -> [1, y, x, 3, 3] deformation gradient. z=1 because we currently only register
    # one slice of the 4- or 2-chamber view...if any. We try to get a better hold of the longitudinal strain.
    # Complication: F was computed in canonical space of SAX view. The strain mask
    if do_smooth and kernel is None:
        kernel = (1, 5, 5, 0, 0)
    if do_smooth:
        F = gaussian_filter(F.copy(), sigma=kernel)
    #
    strain_lax = lagrange_green_strain_tensor(F)
    if strain_mask is not None:
        # assuming binary mask
        strain_mask = strain_mask.astype(np.int32)
        raise NotImplementedError
    return strain_lax


def compute_lv_strain_per_part(
    strain: np.ndarray,
    mask: np.ndarray,
    strain_mask: np.ndarray,
    strain_type: str,
    is_sr=False,
):

    z_part_dict = determine_aha_part(mask, label_id=MMS2MRILabel.LV.value)
    percentile_slices = (5, 90) if is_sr else (0, 100)
    slice_range = compute_slice_range_cstructure(
        mask, MMS2MRILabel.LV, percentile=percentile_slices
    )
    z_part_indices = get_zpart_indices(z_part_dict, slice_range=slice_range)
    strain_parts = {}
    for cpart, indices in z_part_indices.items():
        if len(indices) > 0:
            strain_parts[strain_type + "_lv_" + cpart] = np.mean(
                strain[indices][strain_mask[indices] == 1]
            )
        else:
            strain_parts[strain_type + "_lv_" + cpart] = np.nan

    return strain_parts


def compute_strain_with_normals(
    F: np.ndarray, contour_normal: np.ndarray, strain_mask=None
) -> dict:

    # F is [z, y, x, 3, 3] or [#points, 3, 3] deformation gradient
    # contour_mask and contour_normal are [z, y, x] and [z, y, x, 2] (xy-coordinate normal) resp.
    # so for each voxel that is part of contour we have a normal vector, indicating outward direction
    # with this we can compute radial and circumferential strain e.g. RV myocardium which does not fit polar
    # assumption (RV is not a circle like LV).
    strain = lagrange_green_strain_tensor(F)
    if strain.ndim == 5:
        strain = strain.transpose((2, 1, 0, 3, 4))
        # contour normal is [z, y, x, 2]
        contour_normal_xyz = contour_normal.transpose((2, 1, 0, 3))
    else:
        contour_normal_xyz = contour_normal
    phi = np.arctan2(contour_normal_xyz[..., 1], contour_normal_xyz[..., 0]) + np.pi
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    Q = np.zeros_like(strain).astype(float)
    Q[..., 0, 0], Q[..., 0, 1], Q[..., 1, 0], Q[..., 1, 1] = (
        cos_phi,
        sin_phi,
        -sin_phi,
        cos_phi,
    )
    # we currently bluntly assume that z-axis of coordinates is aligned with apex-base axis of SAX
    Q[..., 2, 2] = 1
    if Q.ndim == 5:
        Q_T = Q.transpose(0, 1, 2, 4, 3)
    else:
        Q_T = Q.transpose(0, 2, 1)
    # again tensor of [z, y, x, 3, 3]
    projected_strain = Q @ strain @ Q_T
    if strain.ndim == 5:
        projected_strain = projected_strain.transpose((2, 1, 0, 3, 4))
        strain = strain.transpose((2, 1, 0, 3, 4))
    if strain_mask is not None:
        projected_strain[strain_mask != 1] = 0
    return {
        "strain": strain,
        "rr": projected_strain[..., 0, 0],
        "cc": projected_strain[..., 1, 1],
        "phi": phi,
    }


def compute_lv_strain(
    F: np.ndarray,
    mask: np.ndarray = None,
    kernel: tuple = None,
    do_smooth=False,
    cmas_xy=None,
    in_xyz_shape=False,
    strain_mask=None,
    conver_to_engineering=False,
):
    # NOTE ASSUMING SHAPE OF F -> [z, y, x, 3, 3] deformation gradient
    # Furthermore, we assume F (in fact jacobian of deformation field) includes already addition of Identity.
    # see function objectives.regularization.compute_jacobian_matrix
    # Furthermore, mask is binary mask and contains only cardiac structure of interest e.g. LV myocardium

    def rollback(arr, cmas_xy, in_xyz_shape=False):
        if in_xyz_shape:
            rx, ry = (np.asarray(arr.shape[:2]) // 2 - cmas_xy).astype(np.int32)
            arr = np.roll(arr, -rx, axis=0)
            arr = np.roll(arr, -ry, axis=1)
        else:
            ry, rx = (np.asarray(arr.shape[1:]) // 2 - cmas_xy[::-1]).astype(np.int32)
            arr = np.roll(arr, -rx, axis=2)
            arr = np.roll(arr, -ry, axis=1)
        return arr

    def convert_to_engineering_strain(Err_sax):
        # convert to engineering strain
        print("DEBUG converting to engineering")
        lam_rr = np.sqrt(1 + 2 * Err_sax)
        eng_rr = lam_rr - 1
        return eng_rr * 100

    if do_smooth and kernel is None:
        # [z, y, x]
        kernel = (2, 3, 3, 0, 0)
    if do_smooth:
        F = gaussian_filter(F.copy(), sigma=kernel)
    #
    strain_sax = lagrange_green_strain_tensor(F)
    if cmas_xy is not None:
        cx, cy = cmas_xy
        strain_sax_roll = roll_to_center(strain_sax, cx, cy, in_xyz_shape=in_xyz_shape)
        mask_roll = roll_to_center(mask, cx, cy, in_xyz_shape=in_xyz_shape)
    else:
        strain_sax_roll = strain_sax
    Err_sax, Ecc_sax, Ell_sax = convert_strain_to_polar(
        strain_sax_roll, in_xyz_shape=in_xyz_shape
    )

    if conver_to_engineering:
        # UPDATED ON 25 TO CONVERT INTO A PERCENTAGE ENGINEERING STRAIN
        Err_sax = convert_to_engineering_strain(Err_sax)
        Ecc_sax = convert_to_engineering_strain(Ecc_sax)
        Ell_sax = convert_to_engineering_strain(Ell_sax)

    Ecc_sax = -Ecc_sax  # TODO: this has been a nightmare but i think it should be here
    Ell_sax = (
        -Ell_sax
    )  # For future generations: no, I am not 100% sure this should be like this

    if cmas_xy is not None:
        Err_sax = rollback(Err_sax, cmas_xy, in_xyz_shape=in_xyz_shape)
        Ecc_sax = rollback(Ecc_sax, cmas_xy, in_xyz_shape=in_xyz_shape)
        Ell_sax = rollback(Ell_sax, cmas_xy, in_xyz_shape=in_xyz_shape)
        mask_rollback = rollback(mask_roll, cmas_xy, in_xyz_shape=in_xyz_shape)
    if strain_mask is not None:
        # assuming binary mask
        strain_mask = strain_mask.astype(np.int32)
        Err_sax_filtered = filter_array(Err_sax, strain_mask, label=1)
        Ecc_sax_filtered = filter_array(Ecc_sax, strain_mask, label=1)
        Ell_sax_filtered = filter_array(Ell_sax, strain_mask, label=1)
        return {
            "strain": strain_sax,
            "rr": Err_sax_filtered,
            "cc": Ecc_sax_filtered,
            "ll": Ell_sax_filtered,
            "mask_roll": mask_roll,
            "mask_rollback": mask_rollback,
        }

    return {
        "strain": strain_sax,
        "rr": Err_sax,
        "cc": Ecc_sax,
        "ll": Ell_sax,
        "mask_roll": mask_roll,
        "mask_rollback": mask_rollback,
    }


def polar_grid(nx=128, ny=128):
    x, y = np.meshgrid(
        np.linspace(-nx // 2, nx // 2, nx), np.linspace(-ny // 2, ny // 2, ny)
    )
    # because x and y have shape [ny, nx], phi has the same shape. but we want to project the x, y components
    # of strain tensor, hence we transpose to [nx, ny] shape...at least, that is the intention...
    phi = (np.arctan2(y, x) + np.pi).T
    r = np.sqrt(x**2 + y**2 + 1e-8)
    return phi, r


def convert_strain_to_polar(E, in_xyz_shape=False):
    # E has shape [x, y, z, 3, 3] then in_xyz_shape = True otherwise we transpose strain tensor
    if not in_xyz_shape:
        # from zyx to xyz
        E = E.transpose((2, 1, 0, 3, 4))
    shape_xyz = E.shape[:3]
    phi, _ = polar_grid(*shape_xyz[:2])
    Err = np.zeros(shape_xyz)
    Ecc = np.zeros(shape_xyz)
    Erc = np.zeros(shape_xyz)
    Ecr = np.zeros(shape_xyz)
    cos = np.cos(phi)
    sin = np.sin(phi)
    num_slices = shape_xyz[-1]
    # (1) original DeepStrain calculation
    # for k in range(num_slices):
    #     # cos = np.cos(np.deg2rad(phi))
    #     # sin = np.sin(np.deg2rad(phi))
    #     Exx, Exy, Eyx, Eyy = E[:, :, k, 0, 0], E[:, :, k, 0, 1], E[:, :, k, 1, 0], E[:, :, k, 1, 1]
    #
    #     Err[:, :, k] = cos * (cos * Exx + sin * Exy) + sin * (cos * Eyx + sin * Eyy)
    #     Ecc[:, :, k] = -sin * (-sin * Exx + cos * Exy) + cos * (-sin * Eyx + cos * Eyy)
    #     Erc[:, :, k] = cos * (-sin * Exx + cos * Exy) + sin * (-sin * Eyx + cos * Eyy)
    #     Ecr[:, :, k] = -sin * (cos * Exx + sin * Exy) + cos * (cos * Eyx + sin * Eyy)
    # We assume original coordinate system is aligned with apex-base long axis hence, e_z = (0, 0, 1).
    # Therefore, we can directly take the (2, 2) position from strain tensor for each voxel
    Q = np.zeros((cos.shape + (2, 2))).astype(np.float32)
    Q[..., 0, 0], Q[..., 0, 1], Q[..., 1, 0], Q[..., 1, 1] = cos, sin, -sin, cos
    Q = Q[:, :, None]
    Q_T = np.moveaxis(Q, 4, 3)
    # sanity check: Q @ Q_T = I
    # I = Q @ Q_T
    # print("check I ", I.shape, I[10, :, :, 1, 1])
    E_tranformed = Q @ E[..., :2, :2] @ Q_T
    Err, Ecc, Ell = E_tranformed[..., 0, 0], E_tranformed[..., 1, 1], E[..., 2, 2]
    if not in_xyz_shape:
        Err, Ecc = Err.transpose((2, 1, 0)), Ecc.transpose((2, 1, 0))
        Ell = Ell.transpose((2, 1, 0))
    return Err, Ecc, Ell


def lagrange_green_strain_tensor(F: np.ndarray, add_identity=False) -> np.ndarray:
    # NOTE: F = deformation gradient.
    # By default we assume that the identity was already added when computing the jacobian matrix of the
    # output displacement to input coordinates (see function compute_jacobian_matrix in this project).
    # Hence, add_indentity is default set to False!
    #
    # deformation gradient: https: // www.continuummechanics.org / deformationgradient.html
    # The deformation gradient F is the derivative of each component of the deformed x vector with respect to each
    # component of the reference X vector.
    #  F = \frac{\partial}{\partial{X}} (X + u) where X is undeformed reference configuration (identity grid)
    #                                                   and u is the deformation matrix (vector per voxel)
    #  F = I +  \frac{\partial{dvf}}{\partial{X}} where X is identity grid (DON'T confuse with IDENTITY MATRIX)
    if add_identity:
        F = F + np.identity(3)
    # Lagrange Green-Strain-Tensor E = 0.5 (F^T F - I)
    #      Hence, we need to transpose last 2 dimensions to take the matrix product
    if F.ndim == 5:
        C = F.transpose((0, 1, 2, 4, 3)) @ F  # result is of shape: (z, y, x, 3, 3)
    elif F.ndim == 3:
        # e.g. for deformation gradient of contour points, shape [#points, 3, 3]
        C = F.transpose((0, 2, 1)) @ F
    else:
        raise ValueError(
            "lagrange_green_strain_tensor: rank of deformation gradient not supported "
            "(rank={})".format(F.ndim)
        )
    return 0.5 * (C - np.identity(3))


def roll(x, rx, ry, in_xyz_shape=False):
    if in_xyz_shape:
        x = np.roll(x, rx, axis=0)
        return np.roll(x, ry, axis=1)
    else:
        x = np.roll(x, rx, axis=2)
        return np.roll(x, ry, axis=1)


def roll_to_center(x, cx, cy, in_xyz_shape=False):
    # assuming x has shape [z,y,x, ...] strain tensor has more dims but irrelevant
    if in_xyz_shape:
        ny, nx = x.shape[1], x.shape[0]
        return roll(x, int(nx // 2 - cx), int(ny // 2 - cy), in_xyz_shape=in_xyz_shape)
    else:
        ny, nx = x.shape[1], x.shape[2]
        return roll(x, int(nx // 2 - cx), int(ny // 2 - cy), in_xyz_shape=in_xyz_shape)
