from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from CMRI.ARVC.common import (
    get_patient_data_arvc,
    get_arvc_ed_es,
)
from CMRI.align.common import (
    get_aligner,
    get_aligner_input,
    run_aligner,
    get_filename,
    data_dir,
)
from CMRI.align.align_long_sax import AlignLaxSax
import SimpleITK as sitk
import numpy as np
from CMRI.plots.align_plots import plot_alignment
from CMRI.SEG_.process import CMRISegContainer
from CMRI.align.simple_aligner import warp_3d_image
from CMRI.general import sitk_save
from CMRI.SR.process import CMRISRContainer
from kwatsch.dice_loss import BinaryDiceLoss
from CMRI.MMS2.common import MMS2MRILabel
from CMRI.align.simple_aligner import Image
import shutil

output_dir = Path("/home/jorg/expers/cmri_motion/aligner/ARVC")
outdir_sax_aligned = data_dir / "nifti/sax_aligned"
outdir_sax_aligned_seg = data_dir / "auto_annotations/sax_aligned"
output_dir_sr = data_dir / "nifti/sax_sr"
output_dir_sr_seg = data_dir / "auto_annotations/sax_sr"


def plot_check(patid, target_img_lax, aligned_mask_in_tgt, orig_mask_in_tgt):
    fig, fig_axis = plt.subplots(1, 5, figsize=(15, 6))
    dcl = BinaryDiceLoss()
    # new_lv_mask_in_tgt = (aligned_mask_in_tgt1 == 1).astype(np.int32)
    new_lv_mask_in_tgt = aligned_mask_in_tgt
    orig_lv_2d_mask = (
        target_img_lax.torch_mask.detach().cpu().numpy().squeeze() == 2
    ).astype(np.int32)
    orig_lv_mask_in_tgt = orig_mask_in_tgt

    dc = dcl(orig_lv_mask_in_tgt[None], orig_lv_2d_mask[None])
    dc_str = "{}: \n Before {:.3f}".format(patid, dc)
    if new_lv_mask_in_tgt is not None:
        dc_after = dcl(new_lv_mask_in_tgt[None], orig_lv_2d_mask[None])
        dc_str += "\n after {:.3f}".format(dc_after)
    (
        fig_axis[0].imshow(orig_lv_mask_in_tgt, cmap=cm.gray),
        fig_axis[0].set_title(dc_str),
        fig_axis[0].axis("off"),
    )
    if new_lv_mask_in_tgt is not None:
        fig_axis[1].imshow(new_lv_mask_in_tgt, cmap=cm.gray), fig_axis[1].axis("off")
    (
        fig_axis[2].imshow(orig_lv_2d_mask, cmap=cm.gray),
        fig_axis[2].axis("off"),
        fig_axis[2].set_title("aligned"),
    )
    (
        fig_axis[3].imshow(orig_lv_2d_mask - orig_lv_mask_in_tgt, cmap=cm.bwr),
        fig_axis[3].axis("off"),
    )
    (
        fig_axis[4].imshow(orig_lv_2d_mask - new_lv_mask_in_tgt, cmap=cm.bwr),
        fig_axis[4].axis("off"),
        plt.show(),
    )
    return {"dc_before": abs(dc.item()), "dc_after": abs(dc_after.item())}


def plot_align_result(
    patid,
    tp,
    path_to_data_folder_plots,
    original_tgt_image,
    original_source_to_tgt,
    aligned_source_to_tgt,
    loss_mask=None,
    do_save=True,
    image_type=None,
):

    output_dir_pat = path_to_data_folder_plots
    file_name = "{}_sax_{}_tp{:02d}.png".format(patid, image_type, tp)
    fname_plot = os.path.join(output_dir_pat, file_name)

    fig = plot_alignment(
        original_source_to_tgt,
        aligned_source_to_tgt,
        original_tgt_image,
        loss_mask=loss_mask,
        patid=patid,
    )

    if do_save:
        fig.savefig(fname_plot, dpi=400)
        # print("INFO saved to {}".format(fname_plot))


def execute_sr(patid, do_save=True):
    f_sax = data_dir / "nifti/sax_aligned/{}.nii.gz".format(patid)
    f_sax_out = output_dir_sr / "{}.nii.gz".format(patid)
    if not output_dir_sr.is_dir():
        output_dir_sr.mkdir(parents=True)
    aligned_4d = sitk.ReadImage(str(f_sax))
    resolver = CMRISRContainer()
    _ = resolver(aligned_4d, fname_out=str(f_sax_out) if do_save else None)


def execute_segmentation(patid, do_save=True):

    f_sax = output_dir_sr / "{}.nii.gz".format(patid)
    f_sax_out = output_dir_sr_seg / "{}.nii.gz".format(patid)
    if not output_dir_sr_seg.is_dir():
        output_dir_sr_seg.mkdir(parents=True)
    aligned_4d = sitk.ReadImage(str(f_sax))
    segmenter = CMRISegContainer()
    _ = segmenter(aligned_4d, fname_out=str(f_sax_out) if do_save else None)


def excute_alignment(patid, tp, do_save=True):
    source_dir = output_dir

    if not outdir_sax_aligned.is_dir():
        outdir_sax_aligned.mkdir(parents=True)
    if not outdir_sax_aligned_seg.is_dir():
        outdir_sax_aligned_seg.mkdir(parents=True)
    param_file = (
        str(source_dir) + os.sep + patid + os.sep + "params_{:02d}.npz".format(tp)
    )
    params = np.load(param_file)["trans_params"]
    print("INFO - Aligning 4d sax for {} & tp {}".format(patid, tp))
    f_sax = data_dir / "nifti/sax/{}.nii.gz".format(patid)
    f_sax_seg = data_dir / "auto_annotations/sax/{}.nii.gz".format(patid)
    sax_4d_img = sitk.ReadImage(str(f_sax))
    sax_4d_seg = sitk.ReadImage(str(f_sax_seg))
    np_sax_4d_img = sitk.GetArrayFromImage(sax_4d_img).astype(np.float32)
    np_sax_4d_seg = sitk.GetArrayFromImage(sax_4d_seg).astype(np.int32)
    np_new_4d_img, np_new_4d_seg = None, None
    for i, np_array in enumerate(np_sax_4d_img):
        new_3d_img = warp_3d_image(np_array, params, mode="bilinear")
        new_3d_seg = warp_3d_image(np_sax_4d_seg[i], params, mode="nearest")
        np_new_4d_img = (
            np.concatenate([np_new_4d_img, new_3d_img[None]], axis=0)
            if np_new_4d_img is not None
            else new_3d_img[None]
        )
        np_new_4d_seg = (
            np.concatenate([np_new_4d_seg, new_3d_seg[None]], axis=0)
            if np_new_4d_seg is not None
            else new_3d_seg[None]
        )

    if do_save:
        fname_out_sax = outdir_sax_aligned / "{}.nii.gz".format(patid)
        fname_out_sax_seg = outdir_sax_aligned_seg / "{}.nii.gz".format(patid)
        sitk_save(str(fname_out_sax), np_new_4d_img, source_image=sax_4d_img)
        sitk_save(str(fname_out_sax_seg), np_new_4d_seg, source_image=sax_4d_seg)
        print("Saved to {}".format(str(fname_out_sax)))


def excute_alignment_any_dataset(
    sax_4d_img, sax_4d_seg, params, fname_out_sax, fname_out_sax_seg, do_save=True
):

    np_sax_4d_img = sitk.GetArrayFromImage(sax_4d_img).astype(np.float32)
    np_sax_4d_seg = sitk.GetArrayFromImage(sax_4d_seg).astype(np.int32)
    np_new_4d_img, np_new_4d_seg = None, None

    for i, np_array in enumerate(np_sax_4d_img):
        new_3d_img = warp_3d_image(np_array, params, mode="bilinear")
        new_3d_seg = warp_3d_image(np_sax_4d_seg[i], params, mode="nearest")
        np_new_4d_img = (
            np.concatenate([np_new_4d_img, new_3d_img[None]], axis=0)
            if np_new_4d_img is not None
            else new_3d_img[None]
        )
        np_new_4d_seg = (
            np.concatenate([np_new_4d_seg, new_3d_seg[None]], axis=0)
            if np_new_4d_seg is not None
            else new_3d_seg[None]
        )

    if do_save:
        sitk_save(str(fname_out_sax), np_new_4d_img, source_image=sax_4d_img)
        sitk_save(str(fname_out_sax_seg), np_new_4d_seg, source_image=sax_4d_seg)
        print("Saved to {}".format(str(fname_out_sax)))


def align_lax_to_sax(
    sax_image: sitk.Image,
    sax_mask: sitk.Image,
    target_img_lax: Image,
    normalize=True,
    do_save=False,
    early_stopping=False,
):
    iters, lr, use_masked_loss = 1500, 0.01, True
    source_img = Image(sax_image, sax_mask, normalize=normalize, image_type="sax")
    aligner_lax = AlignLaxSax(
        source_img,
        target_img_lax,
        use_masked_loss=use_masked_loss,
        ndim=2,
        loss_func="NCC",
        loss_weight=1,
        combined_loss=True,
    )
    res_dict = aligner_lax.optimize(
        iters, lr=lr, loss_lax=True, loss_dice_lax=True, early_stopping=early_stopping
    )
    # print("Aligner sax/lax params: ", aligner_lax.param_trans_tgt1.detach().cpu().numpy())
    return res_dict


def create_aligned_scan_any_data(
    patid,
    source_img,
    target_img_ch4,
    target_img_ch2,
    tp,
    do_save=True,
    early_stopping=True,
    optimizer_params=["trans"],
    iters=1600,
    align_lax=False,
    plot=False,
):

    excel_results = {"patid": patid}
    use_masked_loss = True
    aligner = get_aligner(
        source_img,
        target_img_ch4,
        target_img_ch2,
        use_masked_loss=use_masked_loss,
    )
    do_align = True if aligner.valid_2ch_mask or aligner.valid_4ch_mask else False
    lr, use_masked_loss = 0.01, True
    res_dict = run_aligner(
        aligner,
        iters,
        lr,
        loss_2ch=True,
        loss_4ch=True,
        loss_dice_4ch=True,
        loss_dice_2ch=True,
        early_stopping=early_stopping,
        optimizer_params=optimizer_params,
    )

    original_source_to_tgt1 = res_dict["warped_src_to_tgt1"]
    original_source_to_tgt2 = res_dict["warped_src_to_tgt2"]
    aligned_source_to_tgt1 = res_dict["warped_aligned_src_to_tgt1"]
    aligned_source_to_tgt2 = res_dict["warped_aligned_src_to_tgt2"]
    orig_mask_in_tgt1 = res_dict["orig_mask_in_tgt1"]
    orig_mask_in_tgt2 = res_dict["orig_mask_in_tgt2"]
    aligned_mask_in_tgt1 = res_dict["aligned_mask_in_tgt1"]
    aligned_mask_in_tgt2 = res_dict["aligned_mask_in_tgt2"]
    aligned_sax_image = res_dict["sitk_aligned_image"]
    aligned_sax_mask = res_dict["sitk_aligned_mask"]

    dc_4ch = plot_check(
        patid,
        target_img_ch4,
        aligned_mask_in_tgt1,
        orig_mask_in_tgt1 == MMS2MRILabel.LV.value,
    )
    excel_results["4ch_original"], excel_results["4ch_after_sax"] = (
        dc_4ch["dc_before"],
        dc_4ch["dc_after"],
    )
    if target_img_ch2 is not None:
        dc_2ch = plot_check(
            patid,
            target_img_ch2,
            aligned_mask_in_tgt2,
            orig_mask_in_tgt2 == MMS2MRILabel.LV.value,
        )
    if (
        target_img_ch2 is not None
        and (dc_4ch["dc_after"] - dc_4ch["dc_before"]) < -0.01
        or (dc_2ch["dc_after"] - dc_2ch["dc_before"]) < -0.01
    ):
        print(
            "!!! Warning - sax alignment resulted in poor overlap "
            "performance 4ch/2ch {:.3f}/{:.3f}".format(
                (dc_4ch["dc_after"] - dc_4ch["dc_before"]),
                (dc_2ch["dc_after"] - dc_2ch["dc_before"]),
            )
        )
        do_align = False
    else:
        if (dc_4ch["dc_after"] - dc_4ch["dc_before"]) < -0.01:
            print(
                "!!! Warning - sax alignment resulted in poor overlap "
                "performance 4ch/2ch {:.3f}".format(
                    (dc_4ch["dc_after"] - dc_4ch["dc_before"])
                )
            )
            do_align = False
    lax4ch_sax_res_dict, lax2ch_sax_res_dict = None, None
    # I am not going to do this so ignore that 2ch is not in None
    if do_align and align_lax:
        _, target_img_ch4, target_img_ch2 = get_aligner_input(
            patid, t0=tp, mask_dilations=2
        )

        # 4-chamber
        lax4ch_sax_res_dict = align_lax_to_sax(
            patid,
            aligned_sax_image,
            aligned_sax_mask,
            target_img_ch4,
            normalize=False,
            early_stopping=early_stopping,
        )
        dc = plot_check(
            patid,
            target_img_ch4,
            lax4ch_sax_res_dict["warped_aligned_mask_to_tgt"],
            aligned_mask_in_tgt1,
        )
        excel_results["4ch_after_lax"] = dc["dc_after"]
        if excel_results["4ch_after_lax"] - excel_results["4ch_after_sax"] >= 0.005:
            excel_results["4ch_origin_offset"] = tuple(lax4ch_sax_res_dict["params"])
            f_4ch = str(get_filename(patid, "image", "4ch"))
            np.savez(
                f_4ch.replace(".nii.gz", ".npz"),
                origin_offset=lax4ch_sax_res_dict["params"].astype(np.float32),
            )
        else:
            excel_results["4ch_new_origin"] = np.nan

        # 2-chamber
        excel_results["2ch_original"], excel_results["2ch_after_sax"] = (
            dc_2ch["dc_before"],
            dc_2ch["dc_after"],
        )
        lax2ch_sax_res_dict = align_lax_to_sax(
            patid,
            aligned_sax_image,
            aligned_sax_mask,
            target_img_ch2,
            normalize=False,
            early_stopping=early_stopping,
        )
        dc = plot_check(
            patid,
            target_img_ch2,
            lax2ch_sax_res_dict["warped_aligned_mask_to_tgt"],
            aligned_mask_in_tgt2,
        )

        # we overwrite the results from the sax->lax alignment here. Images will be used in plot_align_result (below)
        aligned_source_to_tgt1 = lax4ch_sax_res_dict["warped_aligned_src_to_tgt"]
        aligned_source_to_tgt2 = lax2ch_sax_res_dict["warped_aligned_src_to_tgt"]
        excel_results["2ch_after_lax"] = dc["dc_after"]
        if excel_results["2ch_after_lax"] - excel_results["2ch_after_sax"] >= 0.005:
            excel_results["2ch_origin_offset"] = tuple(lax2ch_sax_res_dict["params"])
            f_2ch = str(get_filename(patid, "image", "2ch"))
            np.savez(
                f_2ch.replace(".nii.gz", ".npz"),
                origin_offset=lax2ch_sax_res_dict["params"].astype(np.float32),
            )
        else:
            excel_results["2ch_new_origin"] = np.nan
    else:
        if target_img_ch2 is not None:
            excel_results["2ch_original"], excel_results["2ch_after_sax"] = (
                dc_2ch["dc_before"],
                dc_2ch["dc_after"],
            )

    output_dir_pat = output_dir / patid
    if not output_dir_pat.is_dir():
        output_dir_pat.mkdir(parents=True)
    fname_params = output_dir_pat / "params_{:02d}.npz".format(tp)

    if do_save:
        np.savez(fname_params, trans_params=res_dict["trans_params"])
        # print("INFO - saved alignment parameters to {}".format(str(fname_params)))
        fname_aligned = output_dir_pat / "sax_aligned_{:02d}.nii.gz".format(tp)
        sitk.WriteImage(aligned_sax_image, str(fname_aligned), True)
        fname_params = output_dir / patid / "params_{:02d}.npz".format(tp)
        np.savez(fname_params, trans_params=res_dict["trans_params"])

    loss_mask = target_img_ch4.loss_mask.detach().cpu().squeeze().numpy()
    original_tgt1_image = target_img_ch4.torch_image.detach().cpu().squeeze()

    if plot:
        plot_align_result(
            patid,
            tp,
            path_to_data_folder_plots,
            original_tgt1_image,
            original_source_to_tgt1,
            aligned_source_to_tgt1,
            do_save=do_save,
            image_type="4ch",
            loss_mask=loss_mask if use_masked_loss else None,
        )

    if target_img_ch2 is not None:
        loss_mask = target_img_ch2.loss_mask.detach().cpu().squeeze().numpy()
        original_tgt2_image = target_img_ch2.torch_image.detach().cpu().squeeze()
        if plot and original_tgt2_image is not None:
            plot_align_result(
                patid,
                tp,
                original_tgt2_image,
                original_source_to_tgt2,
                aligned_source_to_tgt2,
                loss_mask=loss_mask if use_masked_loss else None,
                do_save=do_save,
                image_type="2ch",
            )
    if do_save:
        if do_align:
            excute_alignment(patid, tp, do_save=do_save)
        else:
            f_sax, f_sax_seg = (
                get_filename(patid, "image", "sax"),
                get_filename(patid, "mask", "sax"),
            )
            print(
                "WARNING -> did not execute alignment. Saved original sax img/seg to align "
                "output dir! {}".format(str(outdir_sax_aligned))
            )
            # write dummy file not aligned
            with open(str(output_dir_pat / "not_aligned.txt"), "w") as fp:
                pass
            if not outdir_sax_aligned.is_dir():
                outdir_sax_aligned.mkdir(parents=True)
            if not outdir_sax_aligned_seg.is_dir():
                outdir_sax_aligned_seg.mkdir(parents=True)
            fname_out_sax = outdir_sax_aligned / "{}.nii.gz".format(patid)
            fname_out_sax_seg = outdir_sax_aligned_seg / "{}.nii.gz".format(patid)
            try:
                shutil.copyfile(str(f_sax), str(fname_out_sax))
                shutil.copyfile(str(f_sax_seg), str(fname_out_sax_seg))
            except FileExistsError:
                print(
                    "Warning - could not copy files, they already seem to exists {}".format(
                        patid
                    )
                )

        execute_sr(patid, do_save=do_save)
        execute_segmentation(patid, do_save=do_save)

    res_dict["do_align"] = do_align
    excel_results["aligned"] = do_align
    return res_dict, excel_results


def create_aligned_scan_any_data_general(
    patid,
    path_to_data_folder_plots,
    source_img,
    target_img_ch4,
    target_img_ch2,
    tp,
    do_save=True,
    early_stopping=True,
    optimizer_params=["trans"],
    iters=1600,
    align_lax=False,
    plot=False,
):

    excel_results = {"patid": patid}
    use_masked_loss = True
    aligner = get_aligner(
        source_img,
        target_img_ch4,
        target_img_ch2,
        use_masked_loss=use_masked_loss,
    )
    do_align = True if aligner.valid_2ch_mask or aligner.valid_4ch_mask else False
    lr, use_masked_loss = 0.01, True
    res_dict = run_aligner(
        aligner,
        iters,
        lr,
        loss_2ch=True if target_img_ch2 is not None else False,
        loss_4ch=True,
        loss_dice_4ch=True,
        loss_dice_2ch=True if target_img_ch2 is not None else False,
        early_stopping=early_stopping,
        optimizer_params=optimizer_params,
    )

    original_source_to_tgt1 = res_dict["warped_src_to_tgt1"]
    original_source_to_tgt2 = res_dict["warped_src_to_tgt2"]
    aligned_source_to_tgt1 = res_dict["warped_aligned_src_to_tgt1"]
    aligned_source_to_tgt2 = res_dict["warped_aligned_src_to_tgt2"]
    orig_mask_in_tgt1 = res_dict["orig_mask_in_tgt1"]
    orig_mask_in_tgt2 = res_dict["orig_mask_in_tgt2"]
    aligned_mask_in_tgt1 = res_dict["aligned_mask_in_tgt1"]
    aligned_mask_in_tgt2 = res_dict["aligned_mask_in_tgt2"]
    aligned_sax_image = res_dict["sitk_aligned_image"]
    aligned_sax_mask = res_dict["sitk_aligned_mask"]

    dc_4ch = plot_check(
        patid,
        target_img_ch4,
        aligned_mask_in_tgt1,
        orig_mask_in_tgt1 == MMS2MRILabel.LV.value,
    )
    excel_results["4ch_original"], excel_results["4ch_after_sax"] = (
        dc_4ch["dc_before"],
        dc_4ch["dc_after"],
    )

    if target_img_ch2 is not None:
        dc_2ch = plot_check(
            patid,
            target_img_ch2,
            aligned_mask_in_tgt2,
            orig_mask_in_tgt2 == MMS2MRILabel.LV.value,
        )
    if target_img_ch2 is not None:
        if (dc_4ch["dc_after"] - dc_4ch["dc_before"]) < -0.01 or (
            dc_2ch["dc_after"] - dc_2ch["dc_before"]
        ) < -0.01:
            print(
                "!!! Warning - sax alignment resulted in poor overlap "
                "performance 4ch/2ch {:.3f}/{:.3f}".format(
                    (dc_4ch["dc_after"] - dc_4ch["dc_before"]),
                    (dc_2ch["dc_after"] - dc_2ch["dc_before"]),
                )
            )
        do_align = False
    else:
        if (dc_4ch["dc_after"] - dc_4ch["dc_before"]) < -0.01:
            print(
                "!!! Warning - sax alignment resulted in poor overlap "
                "performance 4ch/2ch {:.3f}".format(
                    (dc_4ch["dc_after"] - dc_4ch["dc_before"])
                )
            )
            do_align = False

    lax4ch_sax_res_dict, lax2ch_sax_res_dict = None, None
    # I am not going to do this so ignore that 2ch is not in None
    if do_align and align_lax:
        _, target_img_ch4, target_img_ch2 = get_aligner_input(
            patid, t0=tp, mask_dilations=2
        )

        # 4-chamber
        lax4ch_sax_res_dict = align_lax_to_sax(
            patid,
            aligned_sax_image,
            aligned_sax_mask,
            target_img_ch4,
            normalize=False,
            early_stopping=early_stopping,
        )
        dc = plot_check(
            patid,
            target_img_ch4,
            lax4ch_sax_res_dict["warped_aligned_mask_to_tgt"],
            aligned_mask_in_tgt1,
        )
        excel_results["4ch_after_lax"] = dc["dc_after"]
        if excel_results["4ch_after_lax"] - excel_results["4ch_after_sax"] >= 0.005:
            excel_results["4ch_origin_offset"] = tuple(lax4ch_sax_res_dict["params"])
            f_4ch = str(get_filename(patid, "image", "4ch"))
            np.savez(
                f_4ch.replace(".nii.gz", ".npz"),
                origin_offset=lax4ch_sax_res_dict["params"].astype(np.float32),
            )
        else:
            excel_results["4ch_new_origin"] = np.nan

        # 2-chamber
        excel_results["2ch_original"], excel_results["2ch_after_sax"] = (
            dc_2ch["dc_before"],
            dc_2ch["dc_after"],
        )
        lax2ch_sax_res_dict = align_lax_to_sax(
            patid,
            aligned_sax_image,
            aligned_sax_mask,
            target_img_ch2,
            normalize=False,
            early_stopping=early_stopping,
        )
        dc = plot_check(
            patid,
            target_img_ch2,
            lax2ch_sax_res_dict["warped_aligned_mask_to_tgt"],
            aligned_mask_in_tgt2,
        )

        # we overwrite the results from the sax->lax alignment here. Images will be used in plot_align_result (below)
        aligned_source_to_tgt1 = lax4ch_sax_res_dict["warped_aligned_src_to_tgt"]
        aligned_source_to_tgt2 = lax2ch_sax_res_dict["warped_aligned_src_to_tgt"]
        excel_results["2ch_after_lax"] = dc["dc_after"]
        if excel_results["2ch_after_lax"] - excel_results["2ch_after_sax"] >= 0.005:
            excel_results["2ch_origin_offset"] = tuple(lax2ch_sax_res_dict["params"])
            f_2ch = str(get_filename(patid, "image", "2ch"))
            np.savez(
                f_2ch.replace(".nii.gz", ".npz"),
                origin_offset=lax2ch_sax_res_dict["params"].astype(np.float32),
            )
        else:
            excel_results["2ch_new_origin"] = np.nan
    else:
        if target_img_ch2 is not None:
            excel_results["2ch_original"], excel_results["2ch_after_sax"] = (
                dc_2ch["dc_before"],
                dc_2ch["dc_after"],
            )

    loss_mask = target_img_ch4.loss_mask.detach().cpu().squeeze().numpy()
    original_tgt1_image = target_img_ch4.torch_image.detach().cpu().squeeze()

    if plot:
        plot_align_result(
            patid,
            tp,
            path_to_data_folder_plots,
            original_tgt1_image,
            original_source_to_tgt1,
            aligned_source_to_tgt1,
            do_save=do_save,
            image_type="4ch",
            loss_mask=loss_mask if use_masked_loss else None,
        )

    if target_img_ch2 is not None:
        loss_mask = target_img_ch2.loss_mask.detach().cpu().squeeze().numpy()
        original_tgt2_image = target_img_ch2.torch_image.detach().cpu().squeeze()
        if plot and original_tgt2_image is not None:
            plot_align_result(
                patid,
                tp,
                path_to_data_folder_plots,
                original_tgt2_image,
                original_source_to_tgt2,
                aligned_source_to_tgt2,
                loss_mask=loss_mask if use_masked_loss else None,
                do_save=do_save,
                image_type="2ch",
            )

    if not do_align:
        with open(
            os.path.join(path_to_data_folder_plots, f"not_aligned_{patid}.txt"), "w"
        ) as fp:
            pass

    res_dict["do_align"] = do_align
    excel_results["aligned"] = do_align

    return res_dict, excel_results


def create_aligned_scan(
    patid,
    do_save=True,
    early_stopping=True,
    optimizer_params=["trans"],
    iters=1600,
    align_lax=False,
):

    excel_results = {"patid": patid}
    pat_info = get_patient_data_arvc(
        "/mnt/laura_amc_storage/QIA/Users/Laura/", patid=patid
    )
    use_masked_loss = True
    tp_ED, tp_ES = get_arvc_ed_es(pat_info, "LV", patid=patid, is_new=True)
    tp = tp_ES
    print("INFO - Timepoints ED/ES {}/{} - Using TP {}".format(tp_ED, tp_ES, tp))
    excel_results["tp_align"] = tp
    source_img, target_img_ch4, target_img_ch2 = get_aligner_input(
        patid, t0=tp, mask_dilations=1
    )
    aligner = get_aligner(
        source_img,
        target_img_ch4,
        target_img_ch2,
        use_masked_loss=use_masked_loss,
    )
    do_align = True if aligner.valid_2ch_mask or aligner.valid_4ch_mask else False
    lr, use_masked_loss = 0.01, True
    res_dict = run_aligner(
        aligner,
        iters,
        lr,
        loss_2ch=True,
        loss_4ch=True,
        loss_dice_4ch=True,
        loss_dice_2ch=True,
        early_stopping=early_stopping,
        optimizer_params=optimizer_params,
    )

    original_source_to_tgt1 = res_dict["warped_src_to_tgt1"]
    original_source_to_tgt2 = res_dict["warped_src_to_tgt2"]
    aligned_source_to_tgt1 = res_dict["warped_aligned_src_to_tgt1"]
    aligned_source_to_tgt2 = res_dict["warped_aligned_src_to_tgt2"]
    orig_mask_in_tgt1 = res_dict["orig_mask_in_tgt1"]
    orig_mask_in_tgt2 = res_dict["orig_mask_in_tgt2"]
    aligned_mask_in_tgt1 = res_dict["aligned_mask_in_tgt1"]
    aligned_mask_in_tgt2 = res_dict["aligned_mask_in_tgt2"]
    aligned_sax_image = res_dict["sitk_aligned_image"]
    aligned_sax_mask = res_dict["sitk_aligned_mask"]

    dc_4ch = plot_check(
        patid,
        target_img_ch4,
        aligned_mask_in_tgt1,
        orig_mask_in_tgt1 == MMS2MRILabel.LV.value,
    )
    excel_results["4ch_original"], excel_results["4ch_after_sax"] = (
        dc_4ch["dc_before"],
        dc_4ch["dc_after"],
    )
    dc_2ch = plot_check(
        patid,
        target_img_ch2,
        aligned_mask_in_tgt2,
        orig_mask_in_tgt2 == MMS2MRILabel.LV.value,
    )
    if (dc_4ch["dc_after"] - dc_4ch["dc_before"]) < -0.01 or (
        dc_2ch["dc_after"] - dc_2ch["dc_before"]
    ) < -0.01:
        print(
            "!!! Warning - sax alignment resulted in poor overlap "
            "performance 4ch/2ch {:.3f}/{:.3f}".format(
                (dc_4ch["dc_after"] - dc_4ch["dc_before"]),
                (dc_2ch["dc_after"] - dc_2ch["dc_before"]),
            )
        )
        do_align = False
    lax4ch_sax_res_dict, lax2ch_sax_res_dict = None, None
    if do_align and align_lax:
        _, target_img_ch4, target_img_ch2 = get_aligner_input(
            patid, t0=tp, mask_dilations=2
        )
        # 4-chamber
        lax4ch_sax_res_dict = align_lax_to_sax(
            patid,
            aligned_sax_image,
            aligned_sax_mask,
            target_img_ch4,
            normalize=False,
            early_stopping=early_stopping,
        )
        dc = plot_check(
            patid,
            target_img_ch4,
            lax4ch_sax_res_dict["warped_aligned_mask_to_tgt"],
            aligned_mask_in_tgt1,
        )
        excel_results["4ch_after_lax"] = dc["dc_after"]
        if excel_results["4ch_after_lax"] - excel_results["4ch_after_sax"] >= 0.005:
            excel_results["4ch_origin_offset"] = tuple(lax4ch_sax_res_dict["params"])
            f_4ch = str(get_filename(patid, "image", "4ch"))
            np.savez(
                f_4ch.replace(".nii.gz", ".npz"),
                origin_offset=lax4ch_sax_res_dict["params"].astype(np.float32),
            )
        else:
            excel_results["4ch_new_origin"] = np.nan
        # 2-chamber
        excel_results["2ch_original"], excel_results["2ch_after_sax"] = (
            dc_2ch["dc_before"],
            dc_2ch["dc_after"],
        )
        lax2ch_sax_res_dict = align_lax_to_sax(
            patid,
            aligned_sax_image,
            aligned_sax_mask,
            target_img_ch2,
            normalize=False,
            early_stopping=early_stopping,
        )
        dc = plot_check(
            patid,
            target_img_ch2,
            lax2ch_sax_res_dict["warped_aligned_mask_to_tgt"],
            aligned_mask_in_tgt2,
        )

        # we overwrite the results from the sax->lax alignment here. Images will be used in plot_align_result (below)
        aligned_source_to_tgt1 = lax4ch_sax_res_dict["warped_aligned_src_to_tgt"]
        aligned_source_to_tgt2 = lax2ch_sax_res_dict["warped_aligned_src_to_tgt"]
        excel_results["2ch_after_lax"] = dc["dc_after"]
        if excel_results["2ch_after_lax"] - excel_results["2ch_after_sax"] >= 0.005:
            excel_results["2ch_origin_offset"] = tuple(lax2ch_sax_res_dict["params"])
            f_2ch = str(get_filename(patid, "image", "2ch"))
            np.savez(
                f_2ch.replace(".nii.gz", ".npz"),
                origin_offset=lax2ch_sax_res_dict["params"].astype(np.float32),
            )
        else:
            excel_results["2ch_new_origin"] = np.nan
    else:
        excel_results["2ch_original"], excel_results["2ch_after_sax"] = (
            dc_2ch["dc_before"],
            dc_2ch["dc_after"],
        )
    output_dir_pat = output_dir / patid
    if not output_dir_pat.is_dir():
        output_dir_pat.mkdir(parents=True)
    fname_params = output_dir_pat / "params_{:02d}.npz".format(tp)
    if do_save:
        np.savez(fname_params, trans_params=res_dict["trans_params"])
        # print("INFO - saved alignment parameters to {}".format(str(fname_params)))
        fname_aligned = output_dir_pat / "sax_aligned_{:02d}.nii.gz".format(tp)
        sitk.WriteImage(aligned_sax_image, str(fname_aligned), True)
        fname_params = output_dir / patid / "params_{:02d}.npz".format(tp)
        np.savez(fname_params, trans_params=res_dict["trans_params"])

    loss_mask = target_img_ch4.loss_mask.detach().cpu().squeeze().numpy()
    original_tgt1_image = target_img_ch4.torch_image.detach().cpu().squeeze()
    plot_align_result(
        patid,
        tp,
        original_tgt1_image,
        original_source_to_tgt1,
        aligned_source_to_tgt1,
        do_save=do_save,
        image_type="4ch",
        loss_mask=loss_mask if use_masked_loss else None,
    )
    loss_mask = target_img_ch2.loss_mask.detach().cpu().squeeze().numpy()
    original_tgt2_image = target_img_ch2.torch_image.detach().cpu().squeeze()
    plot_align_result(
        patid,
        tp,
        original_tgt2_image,
        original_source_to_tgt2,
        aligned_source_to_tgt2,
        loss_mask=loss_mask if use_masked_loss else None,
        do_save=do_save,
        image_type="2ch",
    )
    if do_save:
        if do_align:
            excute_alignment(patid, tp, do_save=do_save)
        else:
            f_sax, f_sax_seg = (
                get_filename(patid, "image", "sax"),
                get_filename(patid, "mask", "sax"),
            )
            print(
                "WARNING -> did not execute alignment. Saved original sax img/seg to align "
                "output dir! {}".format(str(outdir_sax_aligned))
            )
            # write dummy file not aligned
            with open(str(output_dir_pat / "not_aligned.txt"), "w") as fp:
                pass
            if not outdir_sax_aligned.is_dir():
                outdir_sax_aligned.mkdir(parents=True)
            if not outdir_sax_aligned_seg.is_dir():
                outdir_sax_aligned_seg.mkdir(parents=True)
            fname_out_sax = outdir_sax_aligned / "{}.nii.gz".format(patid)
            fname_out_sax_seg = outdir_sax_aligned_seg / "{}.nii.gz".format(patid)
            try:
                shutil.copyfile(str(f_sax), str(fname_out_sax))
                shutil.copyfile(str(f_sax_seg), str(fname_out_sax_seg))
            except FileExistsError:
                print(
                    "Warning - could not copy files, they already seem to exists {}".format(
                        patid
                    )
                )

        execute_sr(patid, do_save=do_save)
        execute_segmentation(patid, do_save=do_save)

    res_dict["do_align"] = do_align
    excel_results["aligned"] = do_align
    return res_dict, excel_results
