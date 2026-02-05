import torch
import numpy as np
import SimpleITK as sitk
import cv2 as cv
from math import sin, cos
from scipy import ndimage
from enum import Enum
import cv2

from scipy.ndimage.measurements import label as scipy_label


class MMS2MRILabel(Enum):
    BG = 0
    LVBP = 1
    LV = 2
    RVBP = 3
    SEP = 4
    EPI = 5


def temporal_zoom(
    np_img4d: np.ndarray, zoom_tyx: tuple, order=3, do_blur=True, as_type=np.float32
) -> np.ndarray:
    assert len(zoom_tyx) == 3
    t, z, y, x = np_img4d.shape
    new_img4d = None
    # loop over SLICES! hence, arr3d stack is temporal in z-direction.
    for z_id in np.arange(z):
        arr3d = np_img4d[:, z_id]
        if do_blur:
            # NOTE: we blur in the temporal direction, loop over x (or y)
            for x_id in range(arr3d.shape[-1]):
                sigma = 0.25 / np.asarray(zoom_tyx[:2]).astype(np.float64)
                arr3d[..., x_id] = ndimage.gaussian_filter(arr3d[..., x_id], sigma)

        resized_img = ndimage.interpolation.zoom(arr3d, zoom_tyx, order=order)
        if as_type == np.int:
            # binary/integer labels
            resized_img = np.round(resized_img).astype(as_type)
        new_img4d = (
            np.concatenate((new_img4d, resized_img[:, None]), axis=1)
            if new_img4d is not None
            else resized_img[:, None]
        )
    return new_img4d


def remove_small_cc(binary, thres=10):
    """Remove small connected component in the foreground."""
    cc, n_cc = ndimage.measurements.label(binary)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2


def getLargestCC(segmentation, ndim=3):
    assert ndim in [2, 3]
    if ndim == 3:
        struc = np.ones((3, 3, 3))
    else:
        struc = np.ones((3, 3))
    labels, count = scipy_label(segmentation, structure=struc)
    assert labels.max() != 0  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def sitk_save(
    fname: str,
    arr: np.ndarray,
    spacing_zyx=None,
    dtype=np.float32,
    direction=None,
    origin=None,
    source_image=None,
):
    if arr.ndim == 4:
        volumes = [
            sitk.GetImageFromArray(arr[v].astype(dtype), False)
            for v in range(arr.shape[0])
        ]
        img = sitk.JoinSeries(volumes)
    else:
        img = sitk.GetImageFromArray(arr.astype(dtype))
    if source_image is not None:
        img.CopyInformation(source_image)
    else:
        if spacing_zyx is not None:
            img.SetSpacing(spacing_zyx[::-1])
        if direction is not None:
            img.SetDirection(direction)
        if origin is not None:
            img.SetOrigin(origin)
    sitk.WriteImage(img, fname, True)


def get_center_jorg(
    arr: np.ndarray, label: int = 1, spacing=None, do_flip_sequence=False
):
    # NOTE: returns center of mass coordinates in xyz sequence
    lbl_arr = (arr == int(label)).astype(np.int32)

    center = np.asarray(ndimage.center_of_mass(lbl_arr)).astype(np.float32)
    if do_flip_sequence:
        center = center[::-1]
    if spacing is not None:
        return np.multiply(center, spacing)
    else:
        return center


def get_center(arr: np.ndarray, label: int = 1, spacing=None, do_flip_sequence=False):
    # NOTE: returns center of mass coordinates in xyz sequence
    lbl_arr = arr == int(label)

    arr_masked = arr * lbl_arr

    center = np.asarray(ndimage.center_of_mass(arr_masked)).astype(np.float32)
    if do_flip_sequence:
        center = center[::-1]
    if spacing is not None:
        return np.multiply(center, spacing)
    else:
        return center


def rotation_matrix(rot_x=0, rot_y=0, rot_z=0):
    # https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
    rotmat = np.eye(3)
    if rot_x:
        rotmat = rotmat @ [
            (1, 0, 0),
            (0, cos(rot_x), -sin(rot_x)),
            (0, sin(rot_x), cos(rot_x)),
        ]
    if rot_y:
        rotmat = rotmat @ [
            (cos(rot_y), 0, sin(rot_y)),
            (0, 1, 0),
            (-sin(rot_y), 0, cos(rot_y)),
        ]
    if rot_z:
        rotmat = rotmat @ [
            (cos(rot_z), -sin(rot_z), 0),
            (sin(rot_z), cos(rot_z), 0),
            (0, 0, 1),
        ]

    return rotmat


def check_apex_base_orientation(np_seg, label_value):
    # canonical space: orientation should be lower slice number = APEX
    # higher slice number BASE
    zmask = (np_seg == label_value).any((1, 2))
    num_seg_slices = np.count_nonzero(zmask)
    segmask = (np_seg[zmask] == label_value).astype(np.int32)
    z_cmas = round(get_center(segmask)[0])
    return z_cmas < num_seg_slices / 2


def get_rv_lv_rot_matrix(np_seg, label, device="cpu"):
    lv_com_zyx = get_center(np_seg, label.LVBP.value)
    rv_com_zyx = get_center(np_seg, label.RVBP.value)
    rot_rv_lv_m = torch.eye(4).to(device)
    vec_lv_rv = rv_com_zyx - lv_com_zyx  # NOTE: zyx sequence
    angle = np.arctan2(vec_lv_rv[1], vec_lv_rv[2])
    rotmat = rotation_matrix(rot_z=angle)
    rot_rv_lv_m[:3, :3] = torch.from_numpy(rotmat).float()
    return rot_rv_lv_m


# def bob_zoom(input, output_shape, output=None, order=3, mode='constant', cval=0.0,
#          prefilter=True):
#     if order < 0 or order > 5:
#         raise RuntimeError('spline order not supported')
#     input = np.asarray(input)
#     if np.iscomplexobj(input):
#         raise TypeError('Complex type not supported')
#     if input.ndim < 1:
#         raise RuntimeError('input and output rank must be > 0')
#     mode = _ni_support._extend_mode_to_code(mode)
#     if prefilter and order > 1:
#         filtered = spline_filter(input, order, output=np.float64)
#     else:
#         filtered = input

#     zoom_div = np.array(output_shape, float) - 1
#     # Zooming to infinite values is unpredictable, so just choose
#     # zoom factor 1 instead
#     zoom = np.divide(np.array(input.shape) - 1, zoom_div,
#                         out=np.ones_like(input.shape, dtype=np.float64),
#                         where=zoom_div != 0)

#     output = _ni_support._get_output(output, input,
#                                      shape=output_shape)
#     zoom = np.ascontiguousarray(zoom)
#     # _nd_image.zoom_shift(filtered, zoom, None, output, order, mode, cval)
#     return output


def blur_mask(mask, kernel_shape=(31, 31), num_dilations=2, apply_blur=True):
    mask = mask.astype(np.float32)
    if mask.ndim == 3:
        for idx in range(len(mask)):
            mask[idx] = blur_mask(
                mask[idx],
                kernel_shape,
                num_dilations=num_dilations,
                apply_blur=apply_blur,
            )
    else:
        assert mask.ndim == 2
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_shape)
        mask = cv.dilate(mask, se, iterations=num_dilations)
        if apply_blur:
            mask = cv.GaussianBlur(mask, kernel_shape, sigmaX=0)
    return mask


def check_sitk_volume(sitk_image: sitk.Image) -> sitk.Image:
    if sitk_image.GetSize()[-1] > 1:
        return reduce_volume_to_slice(sitk_image)
    return sitk_image


def reduce_volume_to_slice(sitk_img: sitk.Image, dtype=np.float32) -> sitk.Image:
    np_img = sitk.GetArrayFromImage(sitk_img).astype(dtype)
    original_origin_z = sitk_img.GetOrigin()[-1]
    mid_slice_id = np_img.shape[0] // 2
    new_img = sitk_img[:, :, mid_slice_id : mid_slice_id + 1]
    new_img.SetDirection(sitk_img.GetDirection())
    return new_img


def normalize_4dimage(img4d: np.ndarray, percentile=(0, 100)) -> np.ndarray:
    assert len(img4d.shape) == 4
    new_image = None
    for img3d in img4d:
        new_img3d = normalize_image(img3d, percentile=percentile)
        new_image = (
            np.concatenate([new_image, new_img3d[None]], axis=0)
            if new_image is not None
            else new_img3d[None]
        )
    return new_image


def normalize_image(img3d: np.ndarray, percentile=(0, 100)) -> np.ndarray:
    i_min, i_max = np.percentile(img3d, percentile)
    # div_term = np.array([[[i_max - i_min]]], np.float32)
    div_term = i_max - i_min
    norm_img3d = np.divide(
        (img3d - i_min),
        div_term,
        out=np.ones_like(img3d, dtype=np.float64),
        where=div_term != 0,
    )
    norm_img3d = norm_img3d.clip(0, 1)
    return norm_img3d


def make_bounding_box(mask, min_size=140, verbose=False):
    # mask is [z, y, x] segmentation mask
    def fit_to_minsize(coord0, coord1):
        if verbose:
            print(coord0, coord1, (min_size - (coord1 - coord0)))
        if coord1 - coord0 < min_size:
            pad = (min_size - (coord1 - coord0)) // 2
            if not (min_size - (coord1 - coord0)) % 2:
                coord0 = coord0 - pad
                coord1 = coord1 + pad
            else:
                coord0 = coord0 - pad
                coord1 = coord1 + pad + 1
            if verbose:
                print("New ", coord0, coord1, (coord1 - coord0))
        return coord0, coord1

    slice_mask = mask.any((0)).astype(np.int32)
    bbox = np.zeros(mask.shape).astype(np.int32)
    mask_idcs = np.where(slice_mask > 0)
    y_min, y_max = min(mask_idcs[0]), max(mask_idcs[0])
    x_min, x_max = min(mask_idcs[1]), max(mask_idcs[1])
    x_min, x_max = fit_to_minsize(x_min, x_max)
    y_min, y_max = fit_to_minsize(y_min, y_max)
    bbox[:, y_min : y_max + 1, x_min : x_max + 1] = 1
    return bbox


def generate_rv_myocardium(rv3d, kernel=(3, 3), dilations=2) -> np.ndarray:
    rvmyo = np.zeros_like(rv3d).astype(np.int32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    for i, rvslice in enumerate(rv3d):
        rvmyo[i] = (
            cv2.dilate(rvslice.astype(np.uint8), se, iterations=dilations) - rvslice
        )
    if type(rvmyo) is not np.ndarray:
        rvmyo = rvmyo.numpy()
    return rvmyo


def determine_three_slices(np_mask_zyx: np.ndarray):
    # assuming slices are sorted from apex->base [0,...,Z]
    assert np_mask_zyx.ndim == 3
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
    z_pos = {}
    z_mask = (np_mask_zyx != 0).any((1, 2)).astype(np.int32)
    mask_idcs = np.where(z_mask > 0)
    z_min, z_max = min(mask_idcs[0]), max(mask_idcs[0])
    num_slices = z_max - z_min + 1
    z_pos["apical"] = int(z_min + round((num_slices - 1) * 0.25))
    z_pos["mid"] = int(z_min + round((num_slices - 1) * 0.5))
    z_pos["basal"] = int(z_min + round((num_slices - 1) * 0.75))
    z_idx = np.array([z_pos["apical"], z_pos["mid"], z_pos["basal"]]).astype(np.int32)
    return z_pos, z_idx


def compute_slice_range_cstructure(np_mask, label, percentile=(0, 100)):

    z_mask = (np_mask == label.value).any((1, 2)).astype(np.int32)
    mask_idcs = np.where(z_mask > 0)
    z_min, z_max = min(mask_idcs[0]), max(mask_idcs[0])
    slice_range = np.arange(z_min, z_max + 1)
    z_min, z_max = np.percentile(slice_range, percentile).astype(np.int32)
    return np.arange(z_min, z_max + 1)


def filter_array(arr3d: np.ndarray, mask: np.ndarray, label=2):
    filtered_arr3 = arr3d.copy()
    arr3d_mask = (mask == label).astype(np.int32)
    filtered_arr3[arr3d_mask != 1] = 0
    return filtered_arr3
