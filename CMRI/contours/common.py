import numpy as np
from scipy import interpolate
import cv2
from CMRI.general import MMS2MRILabel
from matplotlib import pyplot as plt
import skimage
from collections import defaultdict
from scipy.ndimage import gaussian_filter


def splinify(contour, s=5, datapoints=512, compute_derivatives=False):
    """
    choose s=5 for a contour that was made from a binary mask
    choose s=0 for a medis contour
    """
    # IMPORTANT: assuming contour is [N, (x, y)]
    x, y = contour.T
    tck, u = interpolate.splprep([x, y], s=s, k=3, per=0)
    spline = interpolate.BSpline(tck[0], tck[1], tck[2], extrapolate=True, axis=1)
    xy = spline(np.linspace(0, 1, datapoints, endpoint=True))
    if compute_derivatives:
        # splev with der=1 returns the TANGENT VECTOR for the spline that we computed above.
        # Hence, for each centerline point we have the tangent vector
        # Actually, we need both, tangent and normal to compute CIRCUMFERENTIAL AND RADIAL strain
        # We compute normal from tangent in Dataset object
        dxdt, dydt = interpolate.splev(
            np.linspace(0, 1, datapoints, endpoint=True), tck, der=1
        )
        # after stacking will be [(dx, dy), N] shape. We will transpose when returning array (see below)
        derivs = np.stack((dxdt.astype(np.float32), dydt.astype(np.float32)))
        return xy.T, derivs.T
    else:
        return xy.T, None


def approximate_contour(
    contour, factor=4, smooth=0.05, periodic=False, compute_derivatives=False
):
    """Approximate a contour.
                            Courtesy to UK Biobank toolkit (Wenjia Bai)
    contour: input contour
    factor: upsampling factor for the contour
    smooth: smoothing factor for controling the number of spline knots.
            Number of knots will be increased until the smoothing
            condition is satisfied:
            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
            which means the larger s is, the fewer knots will be used,
            thus the contour will be smoother but also deviating more
            from the input contour.
    periodic: set to True if this is a closed contour, otherwise False.

    return the upsampled and smoothed contour
    """
    # The input contour
    N = len(contour)
    dt = 1.0 / N
    t = np.arange(N) * dt
    x = contour[:, 0]
    y = contour[:, 1]

    # Pad the contour before approximation to avoid underestimating
    # the values at the end points
    r = int(0.5 * N)
    t_pad = np.concatenate((np.arange(-r, 0) * dt, t, 1 + np.arange(0, r) * dt))
    if periodic:
        x_pad = np.concatenate((x[-r:], x, x[:r]))
        y_pad = np.concatenate((y[-r:], y, y[:r]))
    else:
        x_pad = np.concatenate(
            (np.repeat(x[0], repeats=r), x, np.repeat(x[-1], repeats=r))
        )
        y_pad = np.concatenate(
            (np.repeat(y[0], repeats=r), y, np.repeat(y[-1], repeats=r))
        )

    # Fit the contour with splines with a smoothness constraint
    fx = interpolate.UnivariateSpline(t_pad, x_pad, s=smooth * len(t_pad))
    fy = interpolate.UnivariateSpline(t_pad, y_pad, s=smooth * len(t_pad))

    # Evaluate the new contour
    N2 = N * factor
    dt2 = 1.0 / N2
    t2 = np.arange(N2) * dt2
    x2, y2 = fx(t2), fy(t2)
    contour2 = np.stack((x2, y2), axis=1)
    if compute_derivatives:
        dfx = fx.derivative(1)
        dfy = fy.derivative(1)
        dx2, dy2 = dfx(t2), dfy(t2)
        return contour2, np.stack((dx2, dy2), axis=1)
    return contour2


def contours_from_mask(mask3d, label=MMS2MRILabel):
    endo = (mask3d == label.LVBP.value).astype(np.int32)
    epi = np.logical_or(mask3d == label.LVBP.value, mask3d == label.LV.value).astype(
        np.int32
    )
    endo_contours, epi_contours = [], []
    for z in range(mask3d.shape[0]):
        contours, hierarchy = cv2.findContours(
            cv2.inRange(endo[z], 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        endo_contour = contours[0][:, 0, :]

        # Extract epicardial contour
        contours, hierarchy = cv2.findContours(
            cv2.inRange(epi[z], 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        epi_contour = contours[0][:, 0, :]

        # Smooth the contours
        endo_contours.append(approximate_contour(endo_contour, periodic=True))
        epi_contours.append(approximate_contour(epi_contour, periodic=True))

    return endo_contours, epi_contours


# def get_septum_contour(mask_rv: np.ndarray, lv_epi_contour_xy):
#     # Find the septum, which is the intersection between LV and RV
#     lv_epi_contour_xy_int = lv_epi_contour_xy.copy().astype(np.int32)
#     septum = []
#     dilate_iter = 1
#     while len(septum) == 0:
#         # Dilate the RV till it intersects with LV epicardium.
#         # Normally, this is fulfilled after just one iteration.
#         se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         rv_dilate = cv2.dilate(mask_rv.astype(np.uint8), se, iterations=dilate_iter)
#         dilate_iter += 1
#         for (x_int, y_int), (x, y) in zip(lv_epi_contour_xy_int, lv_epi_contour_xy):
#             if rv_dilate[x_int, y_int] == 1:
#                 septum += [[x, y]]
#         if dilate_iter > 100:
#             print("WARNING - get_septum_contour - BREAKING OUT OF WHILE LOOP...{}".format(len(septum)))
#             break
#     return np.array(septum).astype(float)


def get_septum_contour(
    lvm_mask2d, rv_contour, num_dilations=3, kernel=(3, 3), return_rv=False
):
    """
    rv_contour is [#points, (x, y)] and lvm_mask2d is [y, x] (binary)

    Work around to find the septum contour when we have only segmentation masks for RV bloodpool and LV myocardium


    """
    #
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    lvm_mask2d = cv2.dilate(lvm_mask2d.astype(np.uint8), se, iterations=num_dilations)
    x, y = np.where(lvm_mask2d.transpose(1, 0) != 0)
    # lvm_con = np.concatenate((x[:, None], y[:, None]), axis=-1)
    septum_con = []
    septum_indics = []
    new_rv_contour = []
    new_rv_indices = []
    for idx, (x, y) in enumerate(rv_contour):
        (
            x,
            y,
        ) = np.rint(x).astype(np.int32), np.rint(y).astype(np.int32)
        if lvm_mask2d[y, x] == 1:
            septum_con += [[x, y]]
            septum_indics.append(int(idx))
        else:
            new_rv_contour.append([x, y])
            new_rv_indices.append(int(idx))
    septum_con = np.array(septum_con).astype(float)
    septum_indics = np.array(septum_indics).astype(np.int32)
    if return_rv:
        rv_contour = np.array(new_rv_contour).astype(float)
        return (
            septum_con,
            septum_indics,
            rv_contour,
            np.asarray(new_rv_indices).astype(np.int32),
        )
    return septum_con, septum_indics
    # This is verrrrryyy expensive...
    # for coor_rv in rv_contour:
    #     for coor_lv in lvm_con:
    #         if np.all(coor_rv.round().astype(np.int32) == coor_lv):
    #             septum_con.append(coor_rv[None])
    # if len(septum_con) != 0:
    #     septum_con = np.concatenate(septum_con, axis=0)
    #     return septum_con
    #
    # return None


def create_mask_epi_heart(mask_3d, label=MMS2MRILabel, num_dilations=2, kernel=(2, 2)):
    epi_mask = np.zeros_like(mask_3d).astype(np.int32)
    for s in range(len(mask_3d)):
        rv_mask2d = mask_3d[s] == MMS2MRILabel.RVBP  # label.RVBP.value
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
        rv_mask2d = cv2.dilate(rv_mask2d.astype(np.uint8), se, iterations=num_dilations)
        epi_mask[s, rv_mask2d != 0] = 1
        epi_mask[s, mask_3d[s] != 0] = 1
    return epi_mask


def compute_normals(derivative):
    # assume [#coords, 2] dx,dy
    derivative = np.divide(
        derivative, np.linalg.norm(derivative, axis=-1, keepdims=True)
    )
    # to construct normals we need to flip tangent xy values and negate new x component
    normals = np.flip(derivative, axis=-1)
    normals[:, 0] = -1 * normals[:, 0]
    return normals


def merge_contour_normal_points(
    con_dict, norm_dict, label=MMS2MRILabel.RVBP, slice_range=None, z_part_dict=None
):
    # con_dict and norm_dict are return objects from function convert_mask_to_contour
    # Hence, dict with keys 1) slice_id 2) cardiac structure name (from e.g.  MMS2MRILabel)
    # we return two numpy arrays with contour (1D) and corresponding normals (2D [#points, 2]
    con_3d, norm = None, []
    z_part_indices = {"apical": [], "mid": [], "basal": []}
    for slice_id, slice_dict in con_dict.items():
        if slice_range is not None and slice_id not in slice_range:
            continue

        if label.name in slice_dict.keys():
            if z_part_dict is not None:
                cpart = z_part_dict[slice_id].replace("apex", "apical")

                start_idx = len(con_3d) if con_3d is not None else 0
                z_part_indices[cpart].extend(
                    list(
                        np.arange(
                            start_idx, start_idx + len(slice_dict[label.name])
                        ).astype(np.int32)
                    )
                )
            con = np.concatenate(
                [
                    slice_dict[label.name],
                    np.full((len(slice_dict[label.name]), 1), slice_id),
                ],
                axis=-1,
            )
            con_3d = (
                np.concatenate([con_3d, con], axis=0) if con_3d is not None else con
            )
            norm.append(norm_dict[slice_id][label.name])

    if z_part_dict is not None:
        for k, v in z_part_indices.items():
            z_part_indices[k] = np.array(z_part_indices[k]).astype(np.int32)
        return con_3d, np.concatenate(norm), z_part_indices
    return con_3d, np.concatenate(norm)


def convert_mask_to_contour(
    mask3d,
    label=MMS2MRILabel,
    epi_mask=None,
    tp=0,
    upfactor=None,
    compute_derivatives=False,
):
    """

    :param mask3d: [z, y, x] segmentation mask (corresponding label IDs should be defined in label_dict.
    :param label_dict:
    :param epi_mask:
    :param epi_label_id: Same as for SEPTUM, we derive this structure (whole heart epi contour) and therefore it is not in label_dict

    :param tp:  only used to store the contours. Currently we only use ED timepoint which is stored as "0" but actually this does not
                    mean that this is de actual TP in the time sequence which is defined elsewhere.
                    TODO: if you want to convert other time points than ED this needs proper implementation
    :param septum_label_id: label ID to store SEPTUM contour
    :param lvm_lbl_id: e.g. 2 for MMS2 dataset. We do not want this in label_dict because we iterate over label_dict and we only
                            need LVM mask to find the SEPTUM
    :param samples: number of datapoints for contour
    :return:
    """
    assert tp == 0  # currently only supported
    con_list_out = list()
    # dict to store 2D contour per target structure (this object is not returned)
    contours = defaultdict(dict)
    normals = defaultdict(dict)
    cardiac_structures = [label.LVBP, label.LV, label.RVBP, label.EPI, label.SEP]
    # note +1 for background (not used but easier for indexing)
    contours_as_masks = np.zeros(
        (mask3d.shape[0], len(cardiac_structures) + 1, mask3d.shape[1], mask3d.shape[2])
    ).astype(np.int32)
    normals_as_masks = np.zeros(
        (
            mask3d.shape[0],
            len(cardiac_structures) + 1,
            mask3d.shape[1],
            mask3d.shape[2],
            2,
        )
    ).astype(float)
    for idx, m_slice in enumerate(mask3d):
        con_slice, norm_slice = {}, {}
        for cardiac_struc in [label.LVBP, label.LV, label.RVBP, label.EPI, label.SEP]:
            cls_lbl, cls_idx = cardiac_struc.name, cardiac_struc.value
            # print("class ", cls_lbl, cls_idx, np.count_nonzero(m_slice == cls_idx))
            if np.count_nonzero(m_slice == cls_idx) >= 10:
                c = Contour()
                # NOTE: in case of RVBP we are actually interested in the RV myocardium. Hence, we dilate the RVBP
                # mask, in the hope that we end up with some RV myocardial structure. not ideal
                c.fromMask(
                    (m_slice == cls_idx).astype(np.int32),
                    num_dilations=2 if cls_lbl == "RVBP" else None,
                )
                try:
                    if upfactor is not None:
                        c.increase_resolution(
                            upfactor=upfactor, compute_derivatives=compute_derivatives
                        )
                        c.contour = c.contour.round(decimals=3)
                except ValueError:
                    print("Warning: something went wrong with splinify...retrying!")
                    c = Contour()
                    c.fromMask((m_slice == cls_idx).astype(np.int32), num_dilations=2)
                    if upfactor is not None:
                        c.increase_resolution(
                            upfactor=upfactor, compute_derivatives=compute_derivatives
                        )
                        c.contour = c.contour.round(decimals=3)
                con_slice[cls_lbl] = c.contour
                if compute_derivatives:
                    norm_slice[cls_lbl] = c.normal
                    normals_as_masks[idx, cls_idx] = contour_to_mask(
                        c.contour, m_slice.shape, value=c.normal
                    )
                contours_as_masks[idx, cls_idx] = c.contour_to_mask()
                cntr = np.concatenate(
                    (c.contour, np.full((len(c.contour), 1), idx)), axis=-1
                )
                con_list_out.append(
                    [tp, mask3d.shape, cls_idx, cntr.flatten().tolist()]
                )
                if (
                    label.EPI.name not in con_slice.keys()
                    and epi_mask is not None
                    and np.any(epi_mask[idx] != 0)
                ):
                    c = Contour()
                    c.fromMask(epi_mask[idx], num_dilations=None)
                    if upfactor is not None:
                        c.increase_resolution(
                            upfactor=upfactor, compute_derivatives=compute_derivatives
                        )
                        c.contour = c.contour.round(decimals=3)
                        if compute_derivatives:
                            norm_slice[label.EPI.name] = c.normal
                            normals_as_masks[idx, label.EPI.value] = contour_to_mask(
                                c.contour, m_slice.shape, value=c.normal
                            )
                    contours_as_masks[idx, label.EPI.value] = c.contour_to_mask()
                    cntr = np.concatenate(
                        (c.contour, np.full((len(c.contour), 1), idx)), axis=-1
                    )
                    con_list_out.append(
                        [tp, mask3d.shape, label.EPI.value, cntr.flatten().tolist()]
                    )
                    con_slice[label.EPI.name] = cntr

                # if septum_mask is not None and idx in septum_mask and np.any(septum_mask[idx] != 0):
                if (
                    cls_lbl == "RVBP"
                    and cls_lbl in con_slice
                    and np.any(m_slice == label.LV.value)
                ):
                    # septum_mask[idx] has shape [#pts, 2], extend with third dim for slice ID
                    # We need z-coordinate because this will be used to derive slice ID when loading contours
                    # in annotator/converter.py (load cso function)
                    # cntr = np.concatenate((septum_mask[idx], np.full((len(septum_mask[idx]), 1), idx)), axis=-1)
                    sep_contour, sep_indices, new_rv_contour, new_rv_indices = (
                        get_septum_contour(
                            (m_slice == label.LV.value).astype(np.int32),
                            con_slice["RVBP"],
                            num_dilations=2,
                            return_rv=True,
                        )
                    )
                    norm_x = gaussian_filter(norm_slice[label.RVBP.name][:, 0], sigma=3)
                    norm_y = gaussian_filter(norm_slice[label.RVBP.name][:, 1], sigma=3)
                    norm_slice[label.RVBP.name] = np.concatenate(
                        [norm_x[:, None], norm_y[:, None]], axis=-1
                    )
                    sep_con_mask = contour_to_mask(sep_contour, m_slice.shape)
                    sep_normals = None
                    if np.count_nonzero(sep_con_mask) > 0 and len(new_rv_indices) > 0:
                        sep_normals = norm_slice[label.RVBP.name][sep_indices].copy()
                        new_rv_contour = new_rv_contour[2:-2]
                        new_rv_indices = new_rv_indices[2:-2]
                        norm_slice[label.RVBP.name] = norm_slice[label.RVBP.name][
                            new_rv_indices
                        ]
                        con_slice[label.RVBP.name] = new_rv_contour
                        contours_as_masks[idx, label.RVBP.value] = contour_to_mask(
                            con_slice[label.RVBP.name], m_slice.shape
                        )
                        normals_as_masks[idx, label.RVBP.value] = contour_to_mask(
                            con_slice[label.RVBP.name],
                            m_slice.shape,
                            value=norm_slice[label.RVBP.name],
                        )

                    if (
                        sep_contour is not None
                        and len(new_rv_indices) > 0
                        and np.count_nonzero(sep_con_mask) > 20
                    ):
                        con_slice[label.SEP.name] = sep_contour
                        if compute_derivatives:
                            # flip normals because they point inwards (remember RVBP contour is origin).
                            sep_normals = -1 * sep_normals
                            # septum is an open contour and can be quite wiggly at both ends resulting in weired
                            # normals that we want to avoid, hence, we skip the very last four voxels.
                            norm_slice[label.SEP.name] = sep_normals[4:-4]
                            sep_contour = sep_contour[4:-4]
                            con_slice[label.SEP.name] = con_slice[label.SEP.name][4:-4]
                            normals_as_masks[idx, label.SEP.value] = contour_to_mask(
                                con_slice[label.SEP.name],
                                m_slice.shape,
                                value=norm_slice[label.SEP.name],
                            )
                        contours_as_masks[idx, label.SEP.value] = contour_to_mask(
                            sep_contour, m_slice.shape
                        )
                        sep_contour = np.concatenate(
                            (sep_contour, np.full((len(sep_contour), 1), idx)), axis=-1
                        )
                        con_list_out.append(
                            [
                                tp,
                                mask3d.shape,
                                label.SEP.value,
                                sep_contour.flatten().tolist(),
                            ]
                        )
        contours[idx] = con_slice
        if compute_derivatives:
            normals[idx] = norm_slice

    if compute_derivatives:
        return contours, contours_as_masks, normals, normals_as_masks
    return contours, contours_as_masks


# TODO: could be in principle deleted because not used anymore
def create_septum_mask(
    mask_3d, lbl_id_lvm, lbl_id_rv, num_dilations=3, kernel=(3, 3), datapoints=256
):
    def filter(con, mask):
        # con [#pts, (xy)], mask [y, x]
        dummy = np.zeros_like(mask).astype(np.bool)
        con_coords = np.round(con).astype(np.int32)
        dummy[con_coords[:, 1], con_coords[:, 0]] = True
        common_coords = np.logical_and(mask, dummy)
        y, x = np.where(common_coords)
        return np.vstack((x, y)).T

    sep_mask = np.zeros_like(mask_3d).astype(np.int32)
    contours_list = dict()
    for s in range(len(mask_3d)):
        lvm_mask2d = mask_3d[s] == lbl_id_lvm
        rv_mask2d = mask_3d[s] == lbl_id_rv
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
        # lvm_mask2d = cv2.morphologyEx(lvm_mask2d.astype(np.uint8), cv2.MORPH_DILATE, se)
        lvm_mask2d = cv2.dilate(
            lvm_mask2d.astype(np.uint8), se, iterations=num_dilations
        )
        rv_mask2d = cv2.dilate(rv_mask2d.astype(np.uint8), se, iterations=num_dilations)
        sep_mask2d = np.logical_and(lvm_mask2d, rv_mask2d)
        skeleton = skimage.morphology.skeletonize(sep_mask2d, method="lee")
        sep_mask[s] = skeleton
        y, x = np.where(skeleton != 0)
        if len(x) != 0 and len(y) != 0:
            xy = np.stack((x, y)).T
            # xy_new = splinify(xy, s=0, datapoints=256, compute_derivatives=False)
            # make sure the splinified contour only contains points that are also in the intersection of
            # LVMyo and RV. this is due to the parametric Bspline fitting which closes the open contour (for septum)
            # xy_filtered = filter(xy_new, sep_mask2d)
            # we have to make sure all arrays have same length as specified by datapoints, otherwise we cannot stack
            # rep = int(datapoints - len(xy))
            # xy_filtered = np.concatenate((xy,np.tile(xy[-1], (rep, 1))), axis=0)
            # cntrs, hierarchy = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # cn = np.vstack({tuple(row) for row in np.squeeze(cntrs[0])})
            # xy_new = scipy_bspline(xy, datapoints, 5, periodic=False)

            contours_list[s] = xy

    return {"mask": sep_mask, "contour": contours_list}


def contour_to_mask(contour, shape_yx, value=None):

    coord_xy = np.rint(contour[..., :2]).astype(np.int32)
    if value is None or (value is not None and value.ndim == 1):
        mask = np.zeros(shape_yx).astype(np.uint8 if value is None else float)
    else:
        mask = np.zeros(shape_yx + (value.shape[-1],), float)
    # NOTE: we need to flip coordinates because coord_xy has (x, y) whereas mask2d is (y, x)
    if len(coord_xy) > 0:
        if value is None:
            mask[(coord_xy[:, 1], coord_xy[:, 0])] = 1
        else:
            mask[(coord_xy[:, 1], coord_xy[:, 0])] = value
    return mask


class Contour(object):
    def __init__(self):
        self.contour = None
        self.mask = None
        self.derivative = None  # should be actually tangents to centerline points
        self.normal = None
        self.shape = None
        self.origin = "contour"

    @staticmethod
    def _check_cntr_results(cntrs):
        max_idx = 0
        for idx, con in enumerate(cntrs):
            # con is [#points, 1, 2]
            if len(con) > len(cntrs[max_idx]):
                max_idx = idx
        return max_idx

    def fromContour(self, contour, shape):
        self.contour = contour
        self.shape = shape
        self.origin = "contour"

    def fromMask(self, mask, num_dilations=None):
        if num_dilations is not None:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            mask = cv2.dilate(mask.astype(np.uint8), se, iterations=num_dilations)
            # cv2.RETR_TREE   RETR_CCOMP
        # cntrs, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cntrs, hierarchy = cv2.findContours(
            cv2.inRange(mask.astype(np.uint8), 1, 1),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE,
        )
        idx = self._check_cntr_results(cntrs)
        self.contour = np.squeeze(cntrs[idx])
        self.origin = "mask"
        self.shape = mask.shape
        self.mask = mask

    def contour_to_mask(self, contour=None, shape_yx=None):
        if contour is None:
            contour = self.contour
        if shape_yx is None:
            shape_yx = self.shape
        return contour_to_mask(contour, shape_yx)

    def mask(self, shape=None):
        if not shape:
            shape = self.shape
        img = np.zeros(shape, float)
        pts = self.contour[:, None].round().astype(np.int32)
        return cv2.fillPoly(img, pts=[pts], color=1.0)

    @property
    def grid_indices(
        self,
    ):
        return self.contour.round().astype(np.int32)

    def showMask(self):
        plt.imshow(self.mask(), cmap="gray")

    def showContour(
        self,
        c="b",
        plot_vertices=False,
    ):
        contour = self.contour
        # plt.fill(contour[:, 0], contour[:, 1], fill=False, c=c)
        plt.plot(contour[:, 0], contour[:, 1], c=c)
        if plot_vertices:
            plt.scatter(contour[:, 0], contour[:, 1], c=c)

    def increase_resolution(
        self, compute_derivatives=False, open_contour=True, upfactor=4
    ):
        # open_contour parameter uses different splinify function (see below). e.g. for septum centerline
        # if open_contour:
        #     splinfiy_func = splinify_open_contour
        # else:
        #     splinfiy_func = splinify
        # splinfiy_func = approximate_contour
        splinfiy_func = splinify
        samples = int(len(self.contour) * upfactor)
        if self.origin == "processed":
            pass
        elif self.origin == "mask":
            self.contour = splinfiy_func(
                self.contour, 20, samples, compute_derivatives=compute_derivatives
            )
            # self.contour = splinfiy_func(self.contour, periodic=open_contour, compute_derivatives=compute_derivatives)
            self.origin = "processed"
        elif self.origin == "contour":
            # self.contour = splinfiy_func(self.contour, periodic=open_contour, compute_derivatives=compute_derivatives)
            self.contour = splinfiy_func(
                self.contour, 0, samples, compute_derivatives=compute_derivatives
            )
            self.origin = "processed"
        # if compute_derivatives then return was tuple
        if compute_derivatives:
            # both arrays are [#points, 2] xy coordinates in last dim
            self.contour, self.derivative = self.contour[0], self.contour[1]
            # normalize to unit vector
            self.derivative = np.divide(
                self.derivative, np.linalg.norm(self.derivative, axis=-1, keepdims=True)
            )
            # to construct normals we need to flip tangent xy values and negate new x component
            self.normal = np.flip(self.derivative, axis=-1)
            self.normal[:, 0] = -1 * self.normal[:, 0]

    def equidistant_points(self, segments=8, segments_per_segment=16):
        self.increase_resolution()
        cnt = self.contour
        diff = np.diff(np.vstack((cnt[0], cnt, cnt[0])), axis=0)
        cumsum = np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))
        segment_arclength = cumsum[-1] / (segments * segments_per_segment)
        div, mod = np.modf(cumsum / segment_arclength)
        unique_nums, segment_idcs = np.unique(mod, return_index=True)
        segment_idcs = segment_idcs[:-1]  # remove final, for periodic contours
        all_points = cnt[segment_idcs]
        segment_points = cnt[segment_idcs[::segments_per_segment]]
        return all_points, segment_points


def splinify_open_contour(
    contour, s=0, datapoints=512, compute_derivatives=False, order=2
):
    # s (smoothing) not used. we use same parameters as splinify function for convenience (see above contour obj)

    def filter_unique_values(x, y):
        _, y_idx = np.unique(y, return_index=True)
        y, x = y[y_idx], x[y_idx]
        _, x_idx = np.unique(x, return_index=True)
        y, x = y[x_idx], x[x_idx]
        return x, y

    if isinstance(contour, Contour):
        cnt = contour.contour
    else:
        cnt = contour
    x, y = cnt[:, 0], cnt[:, 1]
    x_idx = np.argsort(x)
    y, x = y[x_idx], x[x_idx]
    z = np.polyfit(x, y, deg=order)
    p = np.poly1d(z)
    xnew = np.linspace(x[0], x[-1], datapoints)
    ynew = p(xnew)
    # x, y = filter_unique_values(x, y)
    # tck = interpolate.splrep(x, y, k=spline_order, )
    # xnew = np.linspace(x[0], x[-1], datapoints, endpoint=True)
    # ynew = interpolate.splev(xnew, tck, der=0)
    # ynew = np.polyfit()
    xy = np.stack((xnew.astype(np.float32), ynew.astype(np.float32)))
    ret = xy
    if compute_derivatives:
        # splev with der=1 returns the TANGENT VECTOR for the spline that we computed above.
        # Hence, for each centerline point we have the tangent vector
        # Actually, we need both, tangent and normal to compute CIRCUMFERENTIAL AND RADIAL strain
        # We compute normal from tangent in Dataset object
        # dydt = interpolate.splev(xnew, tck, der=1)
        dydt = np.gradient(ynew)
        dxdt = np.gradient(xnew)
        # dxdt, dydt
        # after stacking will be [(dx, dy), N] shape. We will transpose when returning array (see below)
        derivs = np.stack((dxdt.astype(np.float32), dydt.astype(np.float32)))
        if isinstance(contour, Contour):
            contour.contour = ret.T
            return contour, derivs.T
        else:
            return ret.T, derivs.T

    return ret.T
