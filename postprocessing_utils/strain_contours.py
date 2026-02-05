import numpy as np
from enum import Enum
from pathlib import Path
from CMRI.evaluation.dvf import (
    compute_lv_strain,
    compute_ll_strain_4ch,
    compute_strain_with_normals,
)
from CMRI.general import MMS2MRILabel
from CMRI.contours.common import merge_contour_normal_points

# from CMRI.motion.common import determine_aha_part


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


def get_strain(
    dvf_jacobian,
    mask,
    cmas_xyz,
    in_xyz_shape=False,
    do_save=False,
    pat_output_dir=None,
    strain_mask=None,
    conver_to_engineering=False,
):
    pat_output_dir = Path(pat_output_dir)
    if strain_mask is None:
        strain_mask = mask.copy()

    # DO not use strain_mask anymore...we can filter later based on cardiac structure of interest.
    strain = compute_lv_strain(
        dvf_jacobian,
        mask=mask,
        strain_mask=None,
        cmas_xy=cmas_xyz[:2],
        in_xyz_shape=in_xyz_shape,
        conver_to_engineering=conver_to_engineering,
    )

    glscc = np.mean(strain["cc"][strain_mask == 1])
    glsrr = np.mean(strain["rr"][strain_mask == 1])
    glsll = np.mean(strain["ll"][strain_mask == 1])

    if do_save:
        strain_arr = np.stack((strain["rr"], strain["cc"], strain["ll"]))
        fname = pat_output_dir / "strain.npz"
        np.savez(fname, strain=strain_arr)
    return {
        "strain": strain["strain"],
        "rr": strain["rr"],
        "cc": strain["cc"],
        "ll": strain["ll"],
        "glsrr": glsrr,
        "glscc": glscc,
        "glsll": glsll,
    }


def get_strain_mask(mask, is_contour=False, omit_base_apex=False, omit_perc=0.1):

    if is_contour:
        assert len(mask.shape) == 4
        # contours is volume with shape [z, #numclasses, y, x]
        # but we want our strain_mask to have [z, y, x]
        strain_mask = np.zeros((mask.shape[0], mask.shape[2], mask.shape[3]))
        strain_mask[mask[:, MMS2MRILabel.LV.value] == 1] = 1
        strain_mask[mask[:, MMS2MRILabel.LVBP.value] == 1] = 1
    else:
        strain_mask = (mask == MMS2MRILabel.LV.value).astype(np.int32)

    if omit_base_apex:
        new_mask = np.zeros_like(strain_mask)
        zmask = (strain_mask == 1).any((1, 2))
        num_slices = np.count_nonzero(zmask)
        offset_z = np.ceil(omit_perc * num_slices).astype(np.int32)
        z_min, z_max = np.where(zmask)[0][0], np.where(zmask)[0][-1]
        # new_mask[z_min+offset_z:z_max-offset_z] = strain_mask[z_min+offset_z:z_max-offset_z]
        # new_mask[4:9] = strain_mask[4:9]
        new_mask[z_min + offset_z : z_max - offset_z] = strain_mask[
            z_min + offset_z : z_max - offset_z
        ]
        strain_mask = new_mask
        print(
            f"Warning - omit {omit_perc * 100}% of slices at apex/base!. S-Range {z_min + offset_z}:{z_max - offset_z}"
        )
    return strain_mask


def get_strain_lax(
    dvf_jacobian,
    strain_mask=None,
    kernel: tuple = None,
    do_smooth=False,
    do_save=False,
    pat_output_dir=None,
):
    pat_output_dir = Path(pat_output_dir)
    if do_save and pat_output_dir is None:
        raise ValueError("Error - output directory must be specified.")
    # other than with sax strain, this function returns complete strain tensor [1, y, x, 3, 3]
    strain_lax = compute_ll_strain_4ch(
        dvf_jacobian, strain_mask=None, kernel=kernel, do_smooth=do_smooth
    )
    strain_lax_ll = strain_lax[..., 2, 2]
    glsll = np.mean(strain_lax_ll[strain_mask == 1])
    if do_save:
        if do_smooth:
            fname = pat_output_dir / "strain_4ch_smoothed.npz"
        else:
            fname = pat_output_dir / "strain_4ch.npz"
        np.savez(fname, strain=strain_lax)
    return {"glsll_4ch": glsll, "strain_4ch": strain_lax_ll}


def get_strain_contour(ImpReg, c_label: Enum, slice_range=None):
    con_dict = ImpReg.cimage_fixed.get_sax_image(image_type="contour")
    norm_dict = ImpReg.cimage_fixed.get_sax_image(image_type="normal")
    label_id = (
        MMS2MRILabel.RVBP.value
        if c_label.name in ["RVBP", "SEP"]
        else MMS2MRILabel.LV.value
    )
    z_part_dict = determine_aha_part(
        ImpReg.cimage_fixed.get_sax_image(image_type="mask"), label_id=label_id
    )

    con3d, norm, z_part_indices = merge_contour_normal_points(
        con_dict,
        norm_dict,
        label=c_label,
        slice_range=slice_range,
        z_part_dict=z_part_dict,
    )

    con_3d_aligned = ImpReg.cimage_fixed.align_coords(con3d)
    warped_con, con_dvf_jac = ImpReg.warp_coords(con_3d_aligned)
    strain_con = compute_strain_with_normals(con_dvf_jac, norm)
    strain_con["cc_apical"] = np.mean(strain_con["cc"][z_part_indices["apical"]])
    strain_con["cc_mid"] = np.mean(strain_con["cc"][z_part_indices["mid"]])
    strain_con["cc_basal"] = np.mean(strain_con["cc"][z_part_indices["basal"]])
    strain_con["rr_apical"] = np.mean(strain_con["rr"][z_part_indices["apical"]])
    strain_con["rr_mid"] = np.mean(strain_con["rr"][z_part_indices["mid"]])
    strain_con["rr_basal"] = np.mean(strain_con["rr"][z_part_indices["basal"]])
    strain_con["cc"] = np.mean(strain_con["cc"])
    strain_con["rr"] = np.mean(strain_con["rr"])
    return strain_con
