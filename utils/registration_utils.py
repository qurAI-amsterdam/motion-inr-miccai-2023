import os
from pathlib import Path
import SimpleITK as sitk
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK
import torch
import math
import pandas as pd
from CMRI.general import MMS2MRILabel
from kwatsch.common import (
    KEY_4CH_SEG_VIEW,
    KEY_4CH_VIEW,
)
from kwatsch.dice_loss import compute_overlap
from kwatsch.canonical_space import CanonicalImage
from CMRI.general import get_center
from CMRI.evaluation.dvf import compute_lv_strain_per_part
from CMRI.general import determine_three_slices, compute_slice_range_cstructure
from CMRI.evaluation.dvf import (
    compute_preserve_volume,
)
from CMRI.general import check_apex_base_orientation, rotation_matrix
from postprocessing_utils.excel_generation import EXCEL_COLUMNS, generate_excel_row
from postprocessing_utils.strain_contours import (
    get_strain_lax,
    get_strain_mask,
    get_strain_contour,
    get_strain,
)

# @DEPRECATED
# def save_experiment_config(description, kwargs, cardiac_views, early_stop, multiple_nets, multi_view, save_net, save, crop, all_timepoints, path_to_experiments):
#     path_to_experiments_file = os.path.join(path_to_experiments, 'experiment_config.txt')

#     # Start with the description
#     config_str = f"Description: {description}\n\nExperiment Configuration:\n"

#     # Add standard configurations
#     config_str += f"cardiac_views: {cardiac_views}\n"
#     config_str += f"early_stop: {early_stop}\n"
#     config_str += f"multiple_nets: {multiple_nets}\n"
#     config_str += f"multi_view: {multi_view}\n"
#     config_str += f"save_net: {save_net}\n"
#     config_str += f"save: {save}\n"
#     config_str += f"crop: {crop}\n"
#     config_str += f"all_timepoints: {all_timepoints}\n"

#     # Append kwargs, each on a new line, indented
#     config_str += "kwargs:\n"
#     for key, value in kwargs.items():
#         config_str += f"  {key}: {value}\n"

#     # Write the formatted string to a text file at the specified path
#     with open(path_to_experiments_file, 'w') as file:
#         file.write(config_str)


def get_es_and_ed_timepoints(patid_basename):
    es = int(patid_basename.split("_")[-3])
    ed = int(patid_basename.split("_")[-1])
    return es, ed


def get_experiments_folder(path_to_data, folder_name=None, addition=None):

    if folder_name is None:
        date_time = time.strftime("%Y%m%d_%H%M")
        folder_name = "Experiment_" + date_time
        if addition is not None:
            folder_name = folder_name + "_" + addition

    path_to_experiments = os.path.join(
        path_to_data, "registration_output_experiments", folder_name
    )

    return path_to_experiments


def save_script_to_experiments_folder(path_to_experiments, script_name):
    script_path = os.path.join(path_to_experiments, script_name)
    os.makedirs(path_to_experiments, exist_ok=True)
    os.system(f"cp {script_name} {script_path}")
    print(f"INFO: Script saved to {script_path}")


# New function to get the bounding box
def convert_to_binary_and_get_bbox(seg):
    # This should assume a stik image as input

    # Get array from sitk image
    np_seg = sitk.GetArrayFromImage(seg)
    # We want a ROI of the whole heart so we convert to binary: all non-zero values are set to 1
    seg_binary = np_seg > 0

    # Cast the binary segmentation to sitkUInt8 for SimpleITK processing
    seg_binary_sitk = sitk.Cast(
        sitk.GetImageFromArray(seg_binary.astype(int)), sitk.sitkUInt8
    )

    # Paste the metadata from the original image to the binary mask
    seg_binary_sitk.CopyInformation(seg)

    # Compute the bounding box of the non-zero region in the binary mask
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(seg_binary_sitk)
    bounding_box = label_shape_filter.GetBoundingBox(
        1
    )  # Assumes the mask has a single label

    return bounding_box


def get_image_objects(
    data_dict_volume,
    device="cpu",
    do_normalize=False,
    sr=False,
    load_lax_view=True,
    sax_aligned=False,
    include_contours=False,
):

    sitk_fixed3d_img_sax = data_dict_volume["sitk_fixed3d_img_sax"]
    sitk_mov3d_img_sax = data_dict_volume["sitk_mov3d_img_sax"]
    sitk_fixed3d_mask_sax = data_dict_volume["sitk_fixed3d_mask_sax"]
    sitk_mov3d_mask_sax = data_dict_volume["sitk_mov3d_mask_sax"]

    do_flip = data_dict_volume["do_flip"]
    rv_lv_rot_matrix = data_dict_volume["rv_lv_rot_matrix"]

    if load_lax_view:
        sitk_fixed3d_img_4ch = data_dict_volume["sitk_fixed3d_img_4ch"]
        sitk_mov3d_img_4ch = data_dict_volume["sitk_mov3d_img_4ch"]
        sitk_fixed3d_mask_4ch = data_dict_volume["sitk_fixed3d_mask_4ch"]
        sitk_mov3d_mask_4ch = data_dict_volume["sitk_mov3d_mask_4ch"]

    # CREATE CANONICAL IMAGE OBJECT
    fixed_cimage = CanonicalImage(
        sitk_fixed3d_img_sax,
        sitk_fixed3d_mask_sax,
        label=MMS2MRILabel,
        device=device,
        normalize=do_normalize if not sr else False,
        z_flip=do_flip,
    )
    # ALIGN IMAGES
    fixed_cimage.align_images(
        rv_lv_rot_matrix=rv_lv_rot_matrix, include_contours=include_contours
    )

    # ADD LAX VIEWS TO FIXED IMAGE OBJECT
    if load_lax_view:
        # origin_offset_4ch, origin_offset_2ch = check_new_lax_offsets(patid, data_source_dir)
        # This comes from some npzs that I dont have any idea where this come from
        # TODO: should check this with Jorg
        fixed_cimage.add_view(
            sitk_fixed3d_img_4ch,
            key=KEY_4CH_VIEW,
            dtype=np.float32,
            normalize=do_normalize,
            keep_3d=False,
            origin_offset=None,
        )
        fixed_cimage.add_view(
            sitk_fixed3d_mask_4ch,
            key=KEY_4CH_SEG_VIEW,
            dtype=np.int32,
            keep_3d=False,
            origin_offset=None,
        )

    # CREATE CANONICAL IMAGE OBJECT
    moving_cimage = CanonicalImage(
        sitk_mov3d_img_sax,
        sitk_mov3d_mask_sax,
        label=MMS2MRILabel,
        device=device,
        normalize=do_normalize,
        source_obj=fixed_cimage,
    )
    # ALIGN IMAGES
    moving_cimage.align_images(
        rv_lv_rot_matrix=rv_lv_rot_matrix, include_contours=include_contours
    )

    # ADD LAX VIEWS TO MOVING OBJECT
    if load_lax_view:
        moving_cimage.add_view(
            sitk_mov3d_img_4ch,
            key=KEY_4CH_VIEW,
            dtype=np.float32,
            normalize=do_normalize,
            keep_3d=False,
            origin_offset=None,
        )
        moving_cimage.add_view(
            sitk_mov3d_mask_4ch, key=KEY_4CH_SEG_VIEW, dtype=np.int32, keep_3d=False
        )

    spacing = sitk_mov3d_img_sax.GetSpacing()

    return {
        "fixed_img": fixed_cimage,  # fixed
        "moving_img": moving_cimage,  # moving
        "spacing": spacing,
    }


def get_images_with_segmentations(
    path_to_sax, path_to_seg_sax, load_lax_view=True, **kwargs
):

    pid = os.path.basename(path_to_sax).replace("SAX", "4CH")
    path_to_4ch = (
        kwargs["path_to_data_lax"]
        if "path_to_data_lax" in kwargs
        else path_to_sax.replace("SAX", "4CH")
    )
    path_to_seg_4ch = (
        kwargs["path_to_segmentation_lax"]
        if "path_to_segmentation_lax" in kwargs
        else path_to_seg_sax.replace("SAX", "4CH")
    )

    path_to_4ch = os.path.join(kwargs["root"], path_to_4ch, pid)
    path_to_seg_4ch = os.path.join(kwargs["root"], path_to_seg_4ch, pid)

    img4d_sax = SimpleITK.ReadImage(path_to_sax)
    seg4d_sax = SimpleITK.ReadImage(path_to_seg_sax)

    if (
        load_lax_view
        and os.path.exists(path_to_4ch)
        and os.path.exists(path_to_seg_4ch)
    ):
        img4d_4ch = SimpleITK.ReadImage(path_to_4ch)
        seg4d_4ch = SimpleITK.ReadImage(path_to_seg_4ch)

        return img4d_sax, seg4d_sax, img4d_4ch, seg4d_4ch

    return img4d_sax, seg4d_sax, None, None


# Load_any_volume from run_registration: NOTE DEPRECATED
def load_any_volume(
    patid: str,
    path_to_data_folder: str = "/home/jorg/",
    device="cpu",
    load_lax_view=True,
) -> dict:

    do_flip, rv_lv_rot_matrix = None, None

    img4d_sax, seg4d_sax, img4d_4ch, seg4d_4ch, img4d_2ch_lv, seg4d_2ch_lv = (
        get_images_with_segmentations(
            patid, path_to_data_folder, load_lax_view=load_lax_view, load_2ch=False
        )
    )

    # previously only if sr is true
    np_seg_sax = SimpleITK.GetArrayFromImage(seg4d_sax).astype(np.int32)
    do_flip = check_apex_base_orientation(np_seg_sax[0], MMS2MRILabel.LVBP.value)
    # NOTE: IMPORTANT TO CHECK WAT THIS IS DOING
    rv_lv_rot_matrix = get_rv_lv_rot_matrix(
        np_seg_sax[0], label=MMS2MRILabel, device=device
    )

    return {
        "do_flip": do_flip,
        "rv_lv_rot_matrix": rv_lv_rot_matrix,
        "img4d_sax": img4d_sax,
        "img4d_4ch": img4d_4ch,
        "seg4d_sax": seg4d_sax,
        "seg4d_4ch": seg4d_4ch,
        "seg4d_2ch_lv": seg4d_2ch_lv,
    }


def post_process(ImpReg, img_fixed, img_moving, pat_out_dir, multi_view, save=False):
    pat_out_dir = Path(pat_out_dir)
    warped_img, displacement_field, dvf_jacobian, dvf_jacobian_det, dvf_physics = (
        ImpReg.warp(return_transformation=True, eval_dvf=True)
    )
    moving_mask = img_moving.get_sax_image(image_type="mask", device="cuda")
    warped_mask = ImpReg.warp(
        moving_image=moving_mask,
        return_transformation=False,
        mode="nearest",
        eval_dvf=False,
    )
    torch.cuda.empty_cache()
    if multi_view:
        (
            resampled_4ch_view,
            displacement_field_4ch,
            dvf_jacobian_4ch,
            dvf_jacobian_det_4ch,
        ) = ImpReg.warp_4ch_view(KEY_4CH_VIEW, eval_dvf=True)
        resampled_4ch_seg_view = ImpReg.warp_4ch_view(KEY_4CH_SEG_VIEW, eval_dvf=False)
        dvf_jacobian_det_4ch = dvf_jacobian_det_4ch.squeeze()
    torch.cuda.empty_cache()
    sitk_warped = sitk.GetImageFromArray(warped_img)
    sitk_warped.SetSpacing(img_fixed.get_sax_meta_data("spacing"))
    sitk_warped.SetDirection(img_fixed.get_sax_meta_data("direction"))
    sitk_warped.SetOrigin(img_fixed.get_sax_meta_data("origin"))
    warped_fn = pat_out_dir / "warped_img.nii.gz"
    if save:
        sitk.WriteImage(sitk_warped, str(warped_fn), True)
    disp_fn = pat_out_dir / "displacement_field.npz"
    if save:
        np.savez(str(disp_fn), array=displacement_field)
    disp_fn = pat_out_dir / "displacement_field_4ch.npz"
    if save and multi_view:
        np.savez(str(disp_fn), array=displacement_field_4ch)
    jacob_fn = pat_out_dir / "jacobian_field.npz"
    if save:
        np.savez(str(jacob_fn), array=dvf_jacobian)

    if multi_view:
        return {
            "warped_sax": warped_img,
            "warped_sax_mask": warped_mask,
            "dvf_sax": displacement_field,
            "dvf_lax": displacement_field_4ch,
            "dvf_jacobian": dvf_jacobian,
            "dvf_jacobian_det": dvf_jacobian_det,
            "warped_4ch": resampled_4ch_view,
            "warped_4ch_mask": resampled_4ch_seg_view,
            "dvf_jacobian_4ch": dvf_jacobian_4ch,
            "dvf_jacobian_det_4ch": dvf_jacobian_det_4ch,
            "dvf_physics": dvf_physics,
        }
    else:
        return {
            "warped_sax": warped_img,
            "warped_sax_mask": warped_mask,
            "dvf_sax": displacement_field,
            "dvf_jacobian": dvf_jacobian,
            "dvf_jacobian_det": dvf_jacobian_det,
            "dvf_physics": dvf_physics,
        }


def plot_grid(save_path, volume):
    """
    Plots a grid of slices from a 3D volume and saves it to a specified path.

    :param save_path: Path where the grid image will be saved.
    :param volume: 3D numpy array representing the volume.
    """
    num_slices = volume.shape[0]
    # Determine grid dimensions
    grid_rows = int(math.sqrt(num_slices))
    grid_cols = int(math.ceil(num_slices / grid_rows))

    # Create a figure with subplots
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(15, 15))

    # Flatten the array of axes for easy indexing
    axs = axs.ravel()

    # Plot each slice in its subplot
    for i in range(num_slices):
        axs[i].imshow(volume[i, :, :], cmap="gray")
        axs[i].axis("off")  # Turn off axis

    # Turn off any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grid_5(save_path, volume):
    """
    Plots 5 slices spread across the 3D volume and saves the grid to a specified path.

    :param save_path: Path where the grid image will be saved.
    :param volume: 3D numpy array representing the volume.
    """
    num_slices = volume.shape[0]
    # Generate 5 evenly spaced indices across the volume
    indices = np.linspace(0, num_slices - 1, 5, dtype=int)

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))  # 1 row, 5 columns

    # Plot each selected slice in its subplot
    for i, ax in enumerate(axs):
        slice_index = indices[i]
        ax.imshow(volume[slice_index, :, :], cmap="gray")
        ax.axis("off")  # Turn off axis

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def post_process_completed(
    ImpReg,
    img_fixed,
    img_moving,
    tp_fixed,
    tp_moving,
    spacing,
    patid,
    save_net,
    multi_view,
    kwargs,
):

    in_xyz_shape = kwargs[
        "in_xyz_shape"
    ]  # Was set to False before, changed bc of other code 20/03. Befre it was followed by an " I think " comment so I am not sure
    result_dict = post_process(
        ImpReg, img_fixed, img_moving, ImpReg.exper_dir, multi_view, save=False
    )
    pat_out_dir = ImpReg.exper_dir
    torch.cuda.empty_cache()
    # print(f"Get image objects time: {time.time() - start_time} seconds")
    np_fixed_mask = img_fixed.get_sax_image(image_type="mask")
    np_moving_mask = img_moving.get_sax_image(image_type="mask")
    strain_mask = get_strain_mask(
        (
            img_fixed.get_sax_image(image_type="contour_as_mask")
            if kwargs["use_contour"]
            else np_fixed_mask
        ),
        is_contour=kwargs["use_contour"],
        omit_base_apex=kwargs["omit_base_apex"],
        omit_perc=kwargs["omit_base_apex_perc"],
    )
    cmas_xyz = get_center(
        np_fixed_mask, label=MMS2MRILabel.LVBP.value, do_flip_sequence=True
    )
    print("DEBUG: computing engineering strain:", kwargs["conver_to_engineering"])
    print(
        "DEBIG: computing the strain with the denormalized jaccobian: ",
        kwargs["compute_physical_dvf"],
    )
    print(type(kwargs["compute_physical_dvf"]))
    strain_dict = get_strain(
        (
            result_dict["dvf_physics"]
            if kwargs["compute_physical_dvf"]
            else result_dict["dvf_jacobian"]
        ),  ########################result_dict["dvf_jacobian"],
        np_fixed_mask,
        cmas_xyz,
        in_xyz_shape=in_xyz_shape,
        pat_output_dir=pat_out_dir,
        strain_mask=strain_mask,
        do_save=True,
        conver_to_engineering=kwargs["conver_to_engineering"],
    )
    strain_dict_lv_per_part = compute_lv_strain_per_part(
        strain_dict["rr"], np_fixed_mask, strain_mask, "rr"
    )
    strain_dict_lv_per_part.update(
        {
            k: v
            for k, v in compute_lv_strain_per_part(
                strain_dict["cc"], np_fixed_mask, strain_mask, "cc"
            ).items()
        }
    )
    strain_dict_lv_per_part.update(
        {
            k: v
            for k, v in compute_lv_strain_per_part(
                strain_dict["ll"], np_fixed_mask, strain_mask, "ll"
            ).items()
        }
    )
    percentile_slices = (5, 90) if kwargs["sr"] else (0, 100)
    slice_range = compute_slice_range_cstructure(
        np_moving_mask, MMS2MRILabel.RVBP, percentile=percentile_slices
    )
    try:
        strain_dict_rv = get_strain_contour(ImpReg, MMS2MRILabel.RVBP, slice_range)
    except:
        strain_dict_rv = None
        print("No RV strain, error while generating strain in post process")
    try:
        strain_dict_sep = get_strain_contour(ImpReg, MMS2MRILabel.SEP, slice_range)
    except:
        strain_dict_sep = None
        print("No SEP strain, error while generating strain in post process")
    # strain_dict {'strain': strain, 'rr': glcc, 'cc': glcc, 'll': glll}
    _, z_idx = determine_three_slices(img_fixed.get_sax_image(image_type="mask"))
    dc_scores_3slices = compute_overlap(
        result_dict["warped_sax_mask"][z_idx], np_fixed_mask[z_idx], classes=[1, 2, 3]
    )
    dc_scores = compute_overlap(
        result_dict["warped_sax_mask"], np_fixed_mask, classes=[1, 2, 3]
    )
    result_dict["dice_3_slices"] = dc_scores_3slices
    result_dict["spacing"] = spacing
    result_dict["dice"] = dc_scores
    result_dict["j_minus_1"], _ = compute_preserve_volume(
        result_dict["dvf_jacobian_det"][z_idx],
        np_fixed_mask[z_idx],
        label=MMS2MRILabel.LV.value,
    )

    if multi_view:
        np_fixed_mask_4ch = img_fixed.get_4ch_image(mask=True)
        strain_dict_4ch = get_strain_lax(
            result_dict["dvf_jacobian_4ch"],
            strain_mask=(np_fixed_mask_4ch == MMS2MRILabel.LV.value).astype(np.int32),
            do_save=True,
            pat_output_dir=pat_out_dir,
        )
        strain_dict["glsll_4ch"], strain_dict["strain_4ch"] = (
            strain_dict_4ch["glsll_4ch"],
            strain_dict_4ch["strain_4ch"],
        )
        dc_scores_4ch = compute_overlap(
            result_dict["warped_4ch_mask"], np_fixed_mask_4ch, classes=[1, 2, 3]
        )
        result_dict["dice_4ch"] = dc_scores_4ch

    excel_row = generate_excel_row(
        patid,
        tp_fixed,
        tp_moving,
        result_dict,
        strain_dict,
        strain_dict_rv,
        strain_dict_sep,
        strain_dict_lv_per_part=strain_dict_lv_per_part,
        multi_view=multi_view,
    )

    excel_row = excel_row + [ImpReg.stopped_at_epoch]
    csv_frame = pd.DataFrame([excel_row], columns=EXCEL_COLUMNS)
    csv_frame.to_excel(pat_out_dir / "results.xlsx", index=False)

    if save_net:
        # Save model
        torch.save(ImpReg, pat_out_dir / "model_network.pth")
        in_and_out_dict = ImpReg.get_input_output_inr(
            multi_view
        )  # This returns the final input and output of the model for this patient\
        np.savez(pat_out_dir / "in_and_out_dict.npz", **in_and_out_dict)
    #     ImpReg.savenets(f'{str_patid}_alpha{str(kwargs["alpha_jacobian"]).replace(".", "_")}.pth')

    result_dict["fixed_img"] = img_fixed.get_sax_image()
    result_dict["fixed_mask"] = img_fixed.get_sax_image(image_type="mask")
    result_dict["moving_img"] = img_moving.get_sax_image()
    result_dict["moving_mask"] = img_moving.get_sax_image(image_type="mask")
    result_dict["warped_img"] = result_dict["warped_sax"]  # just because
    warped_mask = result_dict["warped_sax_mask"]

    fname = pat_out_dir / "result_dict.npz"

    np.savez(fname, result_dict=result_dict)

    fname = pat_out_dir / "img_fixed.npz"
    plot_grid_5(pat_out_dir / "img_fixed.png", result_dict["fixed_img"])
    np.savez(fname, img=result_dict["fixed_img"], mask=result_dict["fixed_mask"])

    fname = pat_out_dir / "img_moving.npz"
    plot_grid_5(pat_out_dir / "img_moving.png", result_dict["moving_img"])
    np.savez(fname, img=result_dict["moving_img"], mask=result_dict["moving_mask"])

    fname = pat_out_dir / "warped_img.npz"
    plot_grid_5(pat_out_dir / "warped_img.png", result_dict["warped_img"])
    np.savez(fname, img=result_dict["warped_img"], mask=warped_mask)

    # plot dvf
    plot_dvf_grid([result_dict], pat_out_dir / "dvf_img.png")

    # save kwargs dic as json in pat_out_dir
    kwargs_path = pat_out_dir / "kwargs.json"
    with open(kwargs_path, "w") as f:
        json.dump(kwargs, f, indent=4)


def get_rv_lv_rot_matrix(
    np_seg, label, device="cpu", angle=None, do_flip=None, load_dict=None
):

    lv_com_zyx = get_center(np_seg, label.LVBP.value)
    rv_com_zyx = get_center(np_seg, label.RVBP.value)

    rot_rv_lv_m = torch.eye(4)
    vec_lv_rv = rv_com_zyx - lv_com_zyx  # NOTE: zyx sequence
    angle = angle if angle is not None else np.arctan2(vec_lv_rv[1], vec_lv_rv[2])
    rotmat = rotation_matrix(rot_z=angle)
    rot_rv_lv_m[:3, :3] = torch.from_numpy(rotmat).float()
    # Until here is previous function implementation from jorg

    # First we get the output image from jorgs model
    load_dict["rv_lv_rot_matrix"] = rot_rv_lv_m.to(device)
    load_dict["do_flip"] = do_flip

    data_dict = get_image_objects(
        load_dict,
        do_normalize=True,
        device="cuda",
        sr=True,
        include_contours=True,
        sax_aligned=True,
        load_lax_view=False,
    )

    img_fixed = data_dict["fixed_img"]
    rotated_seg = img_fixed.get_sax_image(image_type="mask")  # This is the moved mask

    # Based on this segmentation we will define if the image needs to be flipped
    y_index = 1  # I think this will always be 1 but just in case
    need_flip = False

    # Get new centers
    lv_com_zyx = get_center(rotated_seg, label.LVBP.value)
    rv_com_zyx = get_center(rotated_seg, label.RVBP.value)

    # We calculate the percentage of pixels above and below the RV
    porc_above_rv = np.sum(
        rotated_seg[:, : int(lv_com_zyx[y_index]), :] == label.RVBP.value
    )
    porc_below_rv = np.sum(
        rotated_seg[:, int(lv_com_zyx[y_index]) :, :] == label.RVBP.value
    )

    # If there is more pixels above the RV than below we need to flip the image
    if porc_above_rv > porc_below_rv:
        need_flip = True
        rot_rv_lv_m = torch.matmul(
            rot_rv_lv_m.to(device),
            torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            .float()
            .to(device),
        )

    # This can return the new matrix with the flip applied
    return rot_rv_lv_m.to(device), need_flip, rotated_seg


def get_canonical_image_aligned(
    img,
    mask,
    lax_img=None,
    lax_seg=None,
    tp_fixed=None,
    tp_moving=None,
    swap_labels=False,
    crop_ROI=False,
    device="cuda",
):

    sitk_fixed_3d_img_sax = img[:, :, :, tp_fixed]
    sitk_mov_3d_img_sax = img[:, :, :, tp_moving]
    sitk_fixed_3d_mask_sax = mask[:, :, :, tp_fixed]
    sitk_mov_3d_mask_sax = mask[:, :, :, tp_moving]

    if lax_img is not None and lax_seg is not None:
        sitk_fixed3d_img_4ch = lax_img[:, :, :, tp_fixed]
        sitk_mov3d_img_4ch = lax_img[:, :, :, tp_moving]
        sitk_fixed3d_mask_4ch = lax_seg[:, :, :, tp_fixed]
        sitk_mov3d_mask_4ch = lax_seg[:, :, :, tp_moving]
    else:
        sitk_fixed3d_img_4ch = None
        sitk_mov3d_img_4ch = None
        sitk_fixed3d_mask_4ch = None
        sitk_mov3d_mask_4ch = None

    if swap_labels:
        # New functionality added, if ACDC, we have to swap the labels of the segmentation, the LVBP is the RVBP and viceversa
        np_sax_seg = sitk.GetArrayFromImage(sitk_fixed_3d_mask_sax)
        np_sax_seg_temporal = np.where(
            np_sax_seg == MMS2MRILabel.LVBP.value, MMS2MRILabel.RVBP.value, np_sax_seg
        )
        sitk_fixed3d_mask_sax_new = np.where(
            np_sax_seg == MMS2MRILabel.RVBP.value,
            MMS2MRILabel.LVBP.value,
            np_sax_seg_temporal,
        )
        sitk_fixed3d_mask_sax_new = sitk.GetImageFromArray(sitk_fixed3d_mask_sax_new)
        sitk_fixed3d_mask_sax_new.CopyInformation(sitk_fixed_3d_mask_sax)
        sitk_fixed_3d_mask_sax = sitk_fixed3d_mask_sax_new
        # We dont have to do this for the LAX because this is an issue only with ACDC and they dont have LAX views

    if crop_ROI:
        padding = [15, 10, 5]  # Experimentally found, do not ask questions
        bounding_box = convert_to_binary_and_get_bbox(sitk_fixed_3d_mask_sax)

        start_x, start_y, start_z, size_x, size_y, size_z = bounding_box
        start_x = max(0, start_x - padding[0])
        start_y = max(0, start_y - padding[1])
        start_z = max(0, start_z - padding[2])
        size_x = min(
            sitk_fixed_3d_img_sax.GetSize()[0] - start_x, size_x + 2 * padding[0]
        )
        size_y = min(
            sitk_fixed_3d_img_sax.GetSize()[1] - start_y, size_y + 2 * padding[1]
        )
        size_z = min(
            sitk_fixed_3d_img_sax.GetSize()[2] - start_z, size_z + 2 * padding[2]
        )

        sitk_fixed_3d_img_sax = sitk_fixed_3d_img_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_fixed_3d_mask_sax = sitk_fixed_3d_mask_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_mov_3d_img_sax = sitk_mov_3d_img_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_mov_3d_mask_sax = sitk_mov_3d_mask_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]

    # if images not between 0 and 1, normalize them
    if (
        max(sitk.GetArrayFromImage(sitk_fixed_3d_img_sax).flatten()) > 1.0
        or min(sitk.GetArrayFromImage(sitk_fixed_3d_img_sax).flatten()) < 0.0
    ):
        print("Normalizing images to be between 0 and 1")
        sitk_fixed_3d_img_sax = sitk.RescaleIntensity(sitk_fixed_3d_img_sax, 0, 1)
        sitk_mov_3d_img_sax = sitk.RescaleIntensity(sitk_mov_3d_img_sax, 0, 1)
        if lax_img is not None and lax_seg is not None:
            sitk_fixed3d_img_4ch = sitk.RescaleIntensity(sitk_fixed3d_img_4ch, 0, 1)
            sitk_mov3d_img_4ch = sitk.RescaleIntensity(sitk_mov3d_img_4ch, 0, 1)
    print(
        "Images in sitk shape (flipped when converted to arr): ",
        sitk_fixed_3d_img_sax.GetSize(),
    )
    load_dict = {
        "sitk_fixed3d_img_sax": sitk_fixed_3d_img_sax,
        "sitk_mov3d_img_sax": sitk_mov_3d_img_sax,
        "sitk_fixed3d_mask_sax": sitk_fixed_3d_mask_sax,
        "sitk_mov3d_mask_sax": sitk_mov_3d_mask_sax,
        "sitk_fixed3d_img_4ch": sitk_fixed3d_img_4ch,
        "sitk_mov3d_img_4ch": sitk_mov3d_img_4ch,
        "sitk_fixed3d_mask_4ch": sitk_fixed3d_mask_4ch,
        "sitk_mov3d_mask_4ch": sitk_mov3d_mask_4ch,
        "seg4d_2ch_lv": None,
    }
    np_seg_sax = SimpleITK.GetArrayFromImage(sitk_fixed_3d_mask_sax).astype(np.int32)
    print("Images in array shape : ", np_seg_sax.shape)
    do_flip = check_apex_base_orientation(np_seg_sax, MMS2MRILabel.LVBP.value)
    rv_lv_rot_matrix, y_flip, _ = get_rv_lv_rot_matrix(
        np_seg_sax,
        label=MMS2MRILabel,
        device=device,
        do_flip=do_flip,
        load_dict=load_dict,
    )  # We need to pass it here cause we also load here...

    load_dict["do_flip"] = do_flip
    load_dict["rv_lv_rot_matrix"] = rv_lv_rot_matrix

    # Shape of images (array) here is z,y,x
    data_dict = get_image_objects(
        load_dict,
        do_normalize=True,
        device="cuda",
        sr=True,
        include_contours=True,
        sax_aligned=True,
        load_lax_view=True if lax_img is not None else False,
    )

    # Leaving this here in case I want to debug later
    # spacing = data_dict['spacing']
    # img_fixed = data_dict['fixed_img']
    # img_moving = data_dict['moving_img']

    # sax_img_fixed = img_fixed.get_sax_image()
    # sax_img_moving = img_moving.get_sax_image()
    # sax_seg_fixed = img_fixed.get_sax_image(image_type='mask')
    # sax_seg_moving = img_moving.get_sax_image(image_type='mask')

    return data_dict


from matplotlib.colors import Normalize


def plot_dvf_grid(
    result_dicts,
    save_path,
    ids=None,
    mask=True,
    invert=False,
    mask_id=0,
    slice="mid",
    alpha=0.1,
):
    """
    Plots a grid showing the DVF overlay and heatmaps for a set of results and saves the plot.

    Parameters:
      result_dicts : list of dicts
          Each dictionary must contain keys 'fixed_img', 'fixed_mask', 'moving_img',
          'moving_mask', 'warped_img', and 'dvf_sax'. If 'dvf_sax' is None it is replaced by zeros.
      save_path : str or Path
          The path where the figure will be saved.
      ids : list, optional
          Identifiers for each result (used for titles).
      mask : bool, optional
          Whether to apply a binary threshold to the masks.
      invert : bool, optional
          (Not used in this implementation.)
      mask_id : int, optional
          If nonzero, the mask is thresholded to that value.
      slice : {'mid', 'base', or other}
          Determines which slice to visualize.
      alpha : float, optional
          Alpha transparency for overlays.
    """
    org_len = len(result_dicts)
    # Remove any empty dictionaries from the list
    result_dicts = [
        result_dict for result_dict in result_dicts if result_dict is not None
    ]
    if len(result_dicts) != org_len:
        print(f"{org_len - len(result_dicts)} empty dicts removed")

    n_items = len(result_dicts)
    # Create a grid with 5 columns (one row per result)
    fig, axs = plt.subplots(n_items, 5, figsize=(25, 5 * n_items))

    # Compute global minimum and maximum displacement magnitudes for color normalization
    all_magnitudes = []
    for result_dict in result_dicts:
        dvf_sax = result_dict["dvf_sax"]
        if dvf_sax is None:
            dvf_sax = np.zeros_like(result_dict["fixed_img"])
        displacement_magnitudes = np.sqrt(
            dvf_sax[:, :, :, 0] ** 2 + dvf_sax[:, :, :, 1] ** 2
        )
        all_magnitudes.append(displacement_magnitudes)
    global_min = min([m.min() for m in all_magnitudes])
    global_max = max([m.max() for m in all_magnitudes])
    norm = Normalize(vmin=global_min, vmax=global_max)
    cmap = plt.cm.jet

    # Loop over each result dictionary
    for i, result_dict in enumerate(result_dicts):
        fixed_img = result_dict["fixed_img"]
        fixed_mask = result_dict["fixed_mask"]
        moving_img = result_dict["moving_img"]
        moving_mask = result_dict["moving_mask"]
        warped_img = result_dict["warped_img"]
        dvf_sax = result_dict["dvf_sax"]
        name = ids[i] if ids is not None else None

        if dvf_sax is None:
            dvf_sax = np.zeros_like(result_dict["fixed_img"])

        # Choose a slice index for 2D visualization based on the parameter 'slice'
        if slice == "mid":
            slice_index = fixed_img.shape[0] // 2
        elif slice == "base":
            slice_index = fixed_img.shape[0] - int(fixed_img.shape[0] * 0.2)
        else:
            slice_index = int(fixed_img.shape[0] * 0.2)

        # Extract corresponding 2D slices
        moving_slice = moving_img[slice_index, :, :]
        warped_slice = warped_img[slice_index, :, :]
        fixed_slice = fixed_img[slice_index, :, :]
        mask_slice_fixed = fixed_mask[slice_index, :, :]
        mask_slice_moving = moving_mask[slice_index, :, :]

        # Apply mask thresholding if required
        if mask:
            if mask_id == 0:
                mask_slice_fixed = mask_slice_fixed != 0
                mask_slice_moving = mask_slice_moving != 0
            else:
                mask_slice_fixed = mask_slice_fixed == mask_id
                mask_slice_moving = mask_slice_moving == mask_id

        # Get the DVF for the corresponding slice
        dvf_slice = dvf_sax[slice_index, :, :, :]
        displacement_magnitudes = np.sqrt(
            dvf_slice[:, :, 0] ** 2 + dvf_slice[:, :, 1] ** 2
        )
        local_min = displacement_magnitudes.min()
        local_max = displacement_magnitudes.max()
        norm_local = Normalize(vmin=local_min, vmax=local_max)

        rgba_heatmap_local = cmap(norm_local(displacement_magnitudes))
        rgba_heatmap_local[..., 3] = np.where(
            mask_slice_moving, rgba_heatmap_local[..., 3], 0
        )
        rgba_heatmap_moving = cmap(norm(displacement_magnitudes))
        rgba_heatmap_moving[..., 3] = np.where(
            mask_slice_moving, rgba_heatmap_moving[..., 3], 0
        )

        # Downsample DVF for visualization
        step = 3
        Y, X = np.mgrid[
            0 : fixed_slice.shape[0] : step, 0 : fixed_slice.shape[1] : step
        ]
        U = dvf_slice[::step, ::step, 0] * mask_slice_fixed[::step, ::step]
        V = dvf_slice[::step, ::step, 1] * mask_slice_fixed[::step, ::step]

        # Select the proper axis for the subplot (if only one row, axs is not a 2D array)
        if n_items > 1:
            ax = axs[i]
        else:
            ax = axs

        # Plot Moving Image
        ax[0].imshow(moving_slice, cmap="gray")
        ax[0].set_title(name if name is not None else "Moving Image")
        ax[0].axis("off")

        # Plot Warped Image
        ax[1].imshow(warped_slice, cmap="gray")
        ax[1].set_title("Warped")
        ax[1].axis("off")

        # Plot Fixed Image with DVF Overlay (quiver)
        ax[2].imshow(fixed_slice, cmap="gray")
        ax[2].quiver(
            X[mask_slice_fixed[::step, ::step]],
            Y[mask_slice_fixed[::step, ::step]],
            U[mask_slice_fixed[::step, ::step]],
            V[mask_slice_fixed[::step, ::step]],
            color="red",
            angles="xy",
            scale_units="xy",
            scale=1.5,
            headwidth=2,
            headlength=2,
            width=0.005,
        )
        ax[2].set_title("Fixed with DVF Overlay")
        ax[2].axis("off")

        # Plot Moving Image with Global Heatmap Overlay
        ax[3].imshow(moving_slice, cmap="gray")
        ax[3].imshow(rgba_heatmap_moving, interpolation="nearest", cmap=cmap, norm=norm)
        ax[3].set_title("Moving with Global Heatmap")
        ax[3].axis("off")

        # Plot Moving Image with Local Heatmap Overlay
        ax[4].imshow(moving_slice, cmap="gray")
        ax[4].imshow(
            rgba_heatmap_local, interpolation="nearest", cmap=cmap, norm=norm_local
        )
        ax[4].set_title("Moving with Local Heatmap")
        ax[4].axis("off")

    # Add a colorbar to the right side of the figure
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, cax=cbar_ax)

    # Save the figure to the specified file
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
