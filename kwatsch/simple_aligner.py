import torch.nn as nn
import torch
import numpy as np
from torch import inverse as tr_inv
from kwatsch.common import execute_resampling, make_homegeneous_identity_grid
from kwatsch.common import get_voxel_to_world_transforms
from kwatsch.bob_metrics import NMI, NCC
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from kwatsch.common import isnotebook
import torch.nn.functional as F
from kwatsch.dice_loss import BinaryDiceLoss

# from CMRI.MMS2.common import MMS2MRILabel
import SimpleITK as sitk
from CMRI.general import normalize_image
from CMRI.general import check_sitk_volume, blur_mask


def warp_3d_image(
    np_array: np.ndarray, params: np.ndarray, mode="bilinear", device="cpu"
):
    t_params = torch.from_numpy(params).float().to(device)
    t_array = torch.from_numpy(np_array).float().to(device)
    shape_zyx = t_array.shape
    ident_grid = make_homegeneous_identity_grid(shape_zyx, device=device)
    ident_grid = ident_grid.reshape(shape_zyx[0], -1, 4)
    # ident_grid[..., :3] = ident_grid[..., :3] - 0.5
    new_grid = ident_grid @ t_params.transpose(1, 2)
    new_grid = new_grid.reshape(tuple(shape_zyx) + (4,))
    shape_div = torch.tensor(shape_zyx[::-1] + (2,), device=device)
    new_grid = (new_grid / ((shape_div[None, None, None] - 1) / 2)) - 1
    new_grid = new_grid[..., :3]
    array_warped = execute_resampling(
        t_array, new_grid, mode=mode, do_detach=False
    ).squeeze()
    return array_warped.detach().cpu().numpy()


class TimePointSearch(object):
    def __int__(self, sax4d_sitk, lax4d_sitk, label=MMS2MRILabel, device="cpu"):
        # both 4d masks
        self.device = device
        self.sax4d_sitk = sax4d_sitk
        self.lax4d_sitk = lax4d_sitk
        self.sax3d_sitk = self.sax4d_sitk[:, :, :, 0]
        self.lax3d_sitk = self.lax4d_sitk[:, :, :, 0]
        self.label = label
        self.lax3d_sitk = check_sitk_volume(self.lax3d_sitk)
        self.lax3d_shape_zyx = self.lax3d_sitk.GetSize()[::-1]
        self.sax_rotate_m, self.sax_scale_m, self.sax_translate_m = (
            get_voxel_to_world_transforms(self.sax3d_sitk, device=self.device)
        )
        self.lax_rotate_m, self.lax_scale_m, self.lax_translate_m = (
            get_voxel_to_world_transforms(self.lax3d_sitk, device=self.device)
        )
        self.coords_tgt_in_src = self._get_tgt_to_src_grid()

    def _search(self):
        pass

    def _get_tgt_to_src_grid(self):
        ident_grid = make_homegeneous_identity_grid(
            self.lax3d_shape_zyx, device=self.device
        )
        world_grid = (
            ident_grid.reshape(self.lax3d_shape_zyx[0], -1, 4)
            @ self.lax_scale_m.T
            @ self.lax_rotate_m.T
            @ self.lax_translate_mm.T
        )
        # from world coordinates to source coordinates. where to
        # interpolate in the source image (SAX), to fill in the interpolated value in the target coordinate system (LA)
        return (
            world_grid
            @ tr_inv(self.sax_translate_m).T
            @ tr_inv(self.sax_rotate_m).T
            @ tr_inv(self.sax_scale_m).T
        )


class Image(object):
    def __init__(
        self,
        sitk_image: sitk.Image,
        sitk_mask: sitk.Image,
        image_type: str,
        device="cuda",
        label=MMS2MRILabel,
        normalize=True,
        generate_loss_mask=False,
        mask_dilations=1,
    ):
        assert image_type in ["sax", "lax2ch", "lax4ch"]
        self.label = label
        self.num_mask_dilations = mask_dilations
        self.image_type = image_type
        self.sitk_image = sitk_image
        self.sitk_mask = sitk_mask
        if image_type in ["lax2ch", "lax4ch"]:
            self.sitk_image = check_sitk_volume(self.sitk_image)

        np_image = sitk.GetArrayFromImage(self.sitk_image).astype(np.float32)
        if normalize:
            np_image = normalize_image(np_image, (1, 99))
        if image_type in ["lax2ch", "lax4ch"]:
            self.sitk_mask = check_sitk_volume(self.sitk_mask)
        np_mask = sitk.GetArrayFromImage(self.sitk_mask).astype(np.int32)
        if image_type == "lax2ch":
            # np_mask[np_mask == self.label.RVBP.value] = 0
            np_mask = self._check_segmentations(np_mask)
        self.torch_image = torch.from_numpy(np_image).float().to(device)
        self.torch_mask = torch.from_numpy(np_mask).float().to(device)
        self.shape_zyx = self.torch_image.shape
        self.loss_mask = None
        if generate_loss_mask:
            self.loss_mask = blur_mask(
                np_mask, apply_blur=False, num_dilations=mask_dilations
            ).squeeze()
            self.loss_mask = torch.from_numpy(self.loss_mask).bool().to(device)
        self.rotate_m, self.scale_m, self.translate_m = get_voxel_to_world_transforms(
            self.sitk_image, device=device
        )

    def _check_segmentations(self, np_mask):
        np_mask[np_mask == self.label.RVBP.value] = 0
        # filtered_mask = np.zeros_like(np_mask).astype(np.int32)
        # for lbl in self.label:
        #     if np.count_nonzero(np_mask == lbl.value) > 0:
        #         filtered_mask[getLargestCC(np_mask == lbl.value, ndim=np_mask.ndim) == 1] = lbl.value
        return np_mask


class SimpleAligner(nn.Module):
    def __init__(
        self,
        source_img: Image,
        target_img1: Image,
        target_img2: Image = None,
        ndim=2,
        sigma=0.5,
        use_masked_loss=False,
        device="cuda",
        loss_func="NMI",
        do_regularize=False,
        loss_weight=1000,
        lambda_reg=0.0001,
        combined_loss=False,
    ):
        super().__init__()
        # In this formulation tgt=target=fixed image (e.g. LAX, if we want to resample SAX to LAX view)
        #                     src=source=moving image (e.g. SAX)
        # We work with 4D homogeneous coordinates. Matrices are extracted from NIFTI images of e.g. SAX (src) and
        # LAX (tgt) views. These transformations are defined from image coordinate system to world coordinate system.
        # Hence, we need to apply inverse and reversed order (normally Scale, Rotate, Translate) to get from
        # world coordinates to image coordinate system.
        # tgt_shape and src_shape expected to be in [z, y, x]
        self.device = device
        self.dim = ndim
        self.do_regularize = do_regularize
        self.combined_loss = combined_loss
        self.loss_weight = loss_weight
        self.lambda_reg = lambda_reg
        self.source_img = source_img
        self.target_img1 = target_img1
        self.target_img2 = target_img2
        self.loss_4ch = True
        self.loss_2ch = True
        self.use_masked_loss = use_masked_loss
        self.src_num_slices, self.src_y, self.src_x = source_img.shape_zyx
        if loss_func == "NMI":
            self.criterion = NMI(sigma=sigma, use_mask=use_masked_loss)  #
            self.extra_criterion = NCC(use_mask=use_masked_loss)
        else:
            self.criterion = NCC(use_mask=use_masked_loss)  #
            self.extra_criterion = NMI(sigma=sigma, use_mask=use_masked_loss)
        self.dice_loss = BinaryDiceLoss(reduction="mean")
        self.param_theta = None
        self.param_trans = None
        self.param_trans_z = None
        self.optimize_params = None
        self.saved_params = {}
        self._init_params()

        self.src_origin = torch.eye(4).to(device)
        # origin of source image as 4d homogeneous coordinate matrix. We need this to translate to origin before
        # rotating
        self.src_origin[:3, 3] = (
            -(
                torch.tensor(
                    self.source_img.shape_zyx[::-1], dtype=torch.float32, device=device
                )
                / 2
            )
            - 0.5
        )
        # we will expand src_origin and therefore insert a dim [1, 4, 4]
        self.src_origin = self.src_origin.unsqueeze(0)
        # from target to world coordinates
        ident_grid = make_homegeneous_identity_grid(
            self.target_img1.shape_zyx, device=device
        )
        # ident_grid[..., :3] = ident_grid[..., :3] - 0.5
        self.world_grid = (
            ident_grid.reshape(self.target_img1.shape_zyx[0], -1, 4)
            @ self.target_img1.scale_m.T
            @ self.target_img1.rotate_m.T
            @ self.target_img1.translate_m.T
        )
        # from world coordinates to source coordinates. where to
        # interpolate in the source image (SAX), to fill in the interpolated value in the target coordinate system (LA)
        self.coords_tgt1_in_src = (
            self.world_grid
            @ tr_inv(self.source_img.translate_m).T
            @ tr_inv(self.source_img.rotate_m).T
            @ tr_inv(self.source_img.scale_m).T
        )
        self.img_src_in_tgt = (
            self.src_to_tgt(
                self.source_img.torch_image,
                self.coords_tgt1_in_src,
                self.target_img1.shape_zyx,
                mode="bilinear",
            )
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        self.coords_tgt2_in_src = None
        if self.target_img2 is not None:
            ident_grid = make_homegeneous_identity_grid(
                self.target_img2.shape_zyx, device=device
            )
            # ident_grid[..., :3] = ident_grid[..., :3] - 0.5
            self.world_grid = (
                ident_grid.reshape(self.target_img2.shape_zyx[0], -1, 4)
                @ self.target_img2.scale_m.T
                @ self.target_img2.rotate_m.T
                @ self.target_img2.translate_m.T
            )
            self.coords_tgt2_in_src = (
                self.world_grid
                @ tr_inv(self.source_img.translate_m).T
                @ tr_inv(self.source_img.rotate_m).T
                @ tr_inv(self.source_img.scale_m).T
            )
        self.orig_source_in_tgt1 = self.src_to_tgt(
            self.source_img.torch_image.clone(),
            self.coords_tgt1_in_src,
            self.target_img1.shape_zyx,
            mode="bilinear",
        )
        self.orig_mask_in_tgt1 = self.src_to_tgt(
            self.source_img.torch_mask,
            self.coords_tgt1_in_src,
            self.target_img1.shape_zyx,
            mode="nearest",
        )
        self.orig_mask_in_tgt2 = self.src_to_tgt(
            self.source_img.torch_mask,
            self.coords_tgt2_in_src,
            self.target_img2.shape_zyx,
            mode="nearest",
        )
        if self.target_img2 is not None:
            self.orig_source_in_tgt2 = self.src_to_tgt(
                self.source_img.torch_image.clone(),
                self.coords_tgt2_in_src,
                self.target_img2.shape_zyx,
                mode="bilinear",
            )
        self.mask_tgt1 = (self.target_img1.torch_mask == MMS2MRILabel.LV.value).float()
        self.mask_tgt2 = (self.target_img2.torch_mask == MMS2MRILabel.LV.value).float()
        dc_4ch = self.dice_loss(
            self.orig_mask_in_tgt1[None] == MMS2MRILabel.LV.value, self.mask_tgt1
        )
        dc_2ch = self.dice_loss(
            self.orig_mask_in_tgt2[None] == MMS2MRILabel.LV.value, self.mask_tgt2
        )
        self.valid_2ch_mask = True if dc_2ch < -0.5 else False
        self.valid_4ch_mask = True if dc_4ch < -0.5 else False
        if not self.valid_2ch_mask:
            # we do not use 2ch loss mask because of crappy quality
            print(
                "*** Warning *** 2ch mask is of low quality. Using SAX resampled mask for masked image loss! ({:.3f})".format(
                    dc_2ch
                )
            )
            loss_mask_tgt2 = blur_mask(
                self.orig_mask_in_tgt2.detach().cpu().numpy(),
                apply_blur=False,
                num_dilations=self.target_img2.num_mask_dilations,
            ).squeeze()
            self.target_img2.loss_mask = (
                torch.from_numpy(loss_mask_tgt2).bool().to(self.device)
            )
        if not self.valid_4ch_mask:
            # we do not use 4ch loss mask because of crappy quality
            print(
                "*** Warning *** 4ch mask is of low quality. Cannot use masked loss! ({:.3f}".format(
                    dc_4ch
                )
            )
            loss_mask_tgt1 = blur_mask(
                self.orig_mask_in_tgt1.detach().cpu().numpy(),
                apply_blur=False,
                num_dilations=self.target_img1.num_mask_dilations,
            ).squeeze()
            self.target_img1.loss_mask = (
                torch.from_numpy(loss_mask_tgt1).bool().to(self.device)
            )
        self.align_src_image = None
        self.losses = []
        self.lrs = []

    def optimize(
        self,
        n_iters,
        misaligned_img=None,
        lr=0.001,
        optimize_params=("rot", "trans"),
        loss_4ch=True,
        loss_2ch=True,
        loss_dice_4ch=False,
        loss_dice_2ch=False,
    ):
        self.loss_4ch, self.loss_2ch, self.loss_dice_4ch, self.loss_dice_2ch = (
            loss_4ch,
            loss_2ch,
            loss_dice_4ch,
            loss_dice_2ch,
        )
        if not self.valid_2ch_mask and self.loss_dice_2ch:
            self.loss_dice_2ch = False
            print(
                "*** Warning *** cannot use dice loss for 2ch view because it is of low quality!"
            )
        if not self.valid_4ch_mask and self.loss_dice_4ch:
            self.loss_dice_4ch = False
            print(
                "*** Warning *** cannot use dice loss for 4ch view because it is of low quality!"
            )
        aligned_mask_in_tgt1, aligned_mask_in_tgt2 = None, None
        self.optimize_params = optimize_params
        if misaligned_img is None:
            misaligned_img = self.source_img.torch_image.clone()
        else:
            if misaligned_img.device != self.device:
                misaligned_img = misaligned_img.to(self.device)
        self.reset_src_image()
        self._init_params()
        if "trans_z" in self.optimize_params:
            optimizer_trans = torch.optim.Adam([self.param_trans, self.trans_z], lr=lr)
        else:
            optimizer_trans = torch.optim.Adam([self.param_trans], lr=lr)
        optimizer_rot = torch.optim.Adam([self.param_theta], lr=lr)
        # optimizer_trans_z = torch.optim.Adam([self.param_trans_z], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_trans, step_size=n_iters - (0.1 * n_iters)
        )
        if isnotebook():
            pbar = tqdm_notebook(np.arange(n_iters), desc="Optimizing", total=n_iters)
        else:
            pbar = tqdm(np.arange(n_iters), desc="Optimizing", total=n_iters)
        aligned_src_image, warped_aligned_src_image = None, None
        source_mask = (self.source_img.torch_mask == MMS2MRILabel.LV.value).float()
        i = 0
        for _ in pbar:
            optimizer_trans.zero_grad()
            optimizer_rot.zero_grad()
            # optimizer_trans_z.zero_grad()
            aligned_src_image1 = self(misaligned_img, mode="bilinear", z_idx=0)
            warped_src_to_tgt1 = self.src_to_tgt(
                aligned_src_image1,
                self.coords_tgt1_in_src,
                self.target_img1.shape_zyx,
                mode="bilinear",
            )
            if self.target_img2 is not None:
                aligned_src_image2 = self(misaligned_img, mode="bilinear", z_idx=1)
                warped_src_to_tgt2 = self.src_to_tgt(
                    aligned_src_image2,
                    self.coords_tgt2_in_src,
                    self.target_img2.shape_zyx,
                    mode="bilinear",
                )
            if self.loss_dice_4ch:
                # NOTE: to make this loss differentiable (smooth) we use bilinear interpolation instead of
                # nearest
                aligned_mask_in_tgt1 = self.align_mask(
                    source_mask,
                    self.coords_tgt1_in_src,
                    self.target_img1.shape_zyx,
                    mode="bilinear",
                )
            if self.loss_dice_2ch:
                aligned_mask_in_tgt2 = self.align_mask(
                    source_mask,
                    self.coords_tgt2_in_src,
                    self.target_img2.shape_zyx,
                    mode="bilinear",
                )
            if self.loss_4ch:
                loss_img = self._get_loss(
                    warped_src_to_tgt1,
                    self.target_img1.torch_image,
                    loss_mask=self.target_img1.loss_mask,
                )
            if self.target_img2 is not None and self.loss_2ch:
                loss2img = self._get_loss(
                    warped_src_to_tgt2,
                    self.target_img2.torch_image,
                    loss_mask=self.target_img2.loss_mask,
                )
                if self.loss_4ch:
                    loss_img = loss_img + loss2img
                else:
                    loss_img = loss2img
            if self.loss_dice_4ch:
                pred = aligned_mask_in_tgt1
                loss_dc = 10 * self._get_dice_loss(pred[None], self.mask_tgt1)
                if self.loss_4ch or self.loss_2ch:
                    loss_img = loss_img + loss_dc
                else:
                    loss_img = loss_dc
            if self.loss_dice_2ch:
                pred = aligned_mask_in_tgt2
                loss_dc = 10 * self._get_dice_loss(pred[None], self.mask_tgt2)
                if self.loss_4ch or self.loss_2ch or self.loss_dice_4ch:
                    loss_img = loss_img + loss_dc
                else:
                    loss_img = loss_dc
            loss_img.backward()
            if "trans" in optimize_params:
                optimizer_trans.step()

            if "rot" in optimize_params:
                optimizer_rot.step()
            # if 'trans_z' in optimize_params:
            #    optimizer_trans_z.step()
            scheduler.step()
            self.losses.append(loss_img.item())
            if "rot" in optimize_params:
                pbar.set_description(
                    "Optimizing loss {:.6f} - sum-t {:.2f} sum-r {:.2f}".format(
                        loss_img.item(),
                        torch.sum(torch.abs(self.param_trans)),
                        torch.sum(torch.abs(self.param_theta)),
                    )
                )
            else:
                pbar.set_description(
                    "Optimizing loss {:.6f} - sum-t {:.2f}".format(
                        loss_img.item(), torch.sum(torch.abs(self.param_trans))
                    )
                )
            self.eval()
            with torch.no_grad():
                aligned_src_image = self(misaligned_img, mode="bilinear")
            self._set_rotation_matrix()
            self._set_translation_matrix()
            self._set_translation_z_matrix()
        if "rot" in optimize_params:
            self.save_rot()
        if "trans" in optimize_params:
            self.save_trans()
        aligned_mask_in_tgt1 = self.align_mask(
            source_mask,
            self.coords_tgt1_in_src,
            self.target_img1.shape_zyx,
            mode="nearest",
        )
        aligned_mask_in_tgt2 = self.align_mask(
            source_mask,
            self.coords_tgt2_in_src,
            self.target_img2.shape_zyx,
            mode="nearest",
        )
        return {
            "aligned_src_image": aligned_src_image.detach().cpu().squeeze(),
            "warped_src_to_tgt1": self.orig_source_in_tgt1.detach().cpu().squeeze(),
            "warped_src_to_tgt2": self.orig_source_in_tgt2.detach().cpu().squeeze(),
            "warped_aligned_src_to_tgt1": warped_src_to_tgt1.detach().cpu().squeeze(),
            "warped_aligned_src_to_tgt2": warped_src_to_tgt2.detach().cpu().squeeze(),
            "orig_mask_in_tgt1": self.orig_mask_in_tgt1.detach()
            .cpu()
            .squeeze()
            .numpy()
            .astype(np.int32),
            "aligned_mask_in_tgt1": (
                None
                if not self.loss_dice_4ch
                else aligned_mask_in_tgt1.detach()
                .cpu()
                .squeeze()
                .numpy()
                .astype(np.int32)
            ),
            "orig_mask_in_tgt2": self.orig_mask_in_tgt2.detach()
            .cpu()
            .squeeze()
            .numpy()
            .astype(np.int32),
            "aligned_mask_in_tgt2": (
                None
                if not self.loss_dice_2ch
                else aligned_mask_in_tgt2.detach()
                .cpu()
                .squeeze()
                .numpy()
                .astype(np.int32)
            ),
            "original_tgt1_image": self.target_img1.torch_image.detach()
            .cpu()
            .squeeze(),
            "original_tgt2_image": self.target_img2.torch_image.detach()
            .cpu()
            .squeeze(),
            "sitk_aligned_image": self._make_sitk_image(aligned_src_image),
            "trans_params": self.trans.detach().cpu().numpy(),
        }

    def _get_loss(self, warped_img, target_img, loss_mask=None):
        if self.use_masked_loss:
            loss = self.criterion(target_img.squeeze(), warped_img, loss_mask)
            if self.combined_loss:
                loss_extra = self.extra_criterion(
                    target_img.squeeze(), warped_img, loss_mask
                )
                # assuming here that loss_extra is NMI loss, which is much smaller than NCC
                loss = loss + 2100 * loss_extra
        else:
            loss = self.criterion(target_img.squeeze(), warped_img)
        loss = self.loss_weight * loss
        if self.do_regularize:
            # print(loss.item(), self.lambda_reg * torch.abs(torch.sum(self.trans)))
            loss = loss + self.lambda_reg * torch.mean(torch.abs(self.trans))
        return loss

    def _get_dice_loss(self, pred, target):
        if pred.dim() == 2:
            pred = pred[None]
        if target.dim() == 2:
            target = target[None]
        dc = self.dice_loss(pred, target)
        return dc

    def _get_segmentation_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def forward(self, src_image, mode="bilinear", z_idx=0):
        # src_origin is [1, 4, 4] and rot is [#slices, 4, 4]
        # ident_grid is [#slices, w, h, 4]
        ident_grid = make_homegeneous_identity_grid(src_image.shape, device=self.device)
        trans_grid = ident_grid.reshape(src_image.shape[0], -1, 4)
        # trans_grid = trans_grid - torch.FloatTensor([[0.5, 0.5, 0.5, 0]]).to(self.device)
        if "trans_z" in self.optimize_params:
            trans_grid[..., 2] = trans_grid[..., 2] + self.trans_z[..., z_idx]
        if "trans" in self.optimize_params:
            trans_grid = trans_grid @ self.trans.transpose(1, 2)
        # Second combine translation to origin and rotation matrix, self.src_origin is negative
        if "rot" in self.optimize_params:
            m_trans = (
                self.src_origin.transpose(1, 2)
                @ self.rot.transpose(1, 2)
                @ tr_inv(self.src_origin.transpose(1, 2))
            )
            trans_grid = trans_grid @ m_trans

        trans_grid = trans_grid.reshape(tuple(src_image.shape) + (4,))
        shape_div = torch.tensor(
            src_image.shape[::-1] + (2,), device=self.device
        )  # (2,) is for homogeneous coord, kind of dummy
        # expand shape_div to fit trans_grid with shape [#slices, w x h, 4]
        # rescale coordinates x,y,z to [-1,1] for pytorch resample_grid function
        trans_grid = (trans_grid / ((shape_div[None, None, None] - 1) / 2)) - 1
        trans_grid = trans_grid[..., :3]  # get rid off homegeneous coordinate
        return execute_resampling(
            src_image, trans_grid, mode=mode, do_detach=False
        ).squeeze()

    def align_mask(self, source_mask, coords_src_in_tgt, tgt_shape_zyx, mode="nearest"):
        aligned_src_mask = self(source_mask, mode=mode)
        return self.src_to_tgt(
            aligned_src_mask, coords_src_in_tgt, tgt_shape_zyx, mode="nearest"
        )

    def src_to_tgt(self, src_image, coords_tgt_in_src, tgt_shape_zyx, mode="bilinear"):
        # Normalize coordinates between [-1, 1] for torch grid_sampler
        shape_div = torch.tensor(
            self.source_img.shape_zyx[::-1] + (2,),
            dtype=torch.float32,
            device=self.device,
        )
        trans_src_coord = (
            coords_tgt_in_src / ((shape_div[None, None, None] - 1) / 2)
        ) - 1
        trans_src_coord = trans_src_coord.reshape(tuple(tgt_shape_zyx) + (4,))[..., :3]
        warped = execute_resampling(
            src_image, trans_src_coord, mode=mode, do_detach=False
        ).squeeze()
        return warped

    def reset_src_image(self):
        self.align_src_image = None

    def _init_params(self):
        # homogeneous coordinates:
        # for rotation matrix require following shape
        # [ cos     -sin    0   0 ]
        # [ sin     cos     0   0 ]
        # [ 0       0       1   0 ]
        # [ 0       0       0   1 ]
        self._init_rot()
        self._init_trans()
        self._init_trans_z()

    def _init_trans_z(self):
        self.param_trans_z = torch.zeros((1, 2), device=self.device, requires_grad=True)
        self._set_translation_z_matrix()

    def _init_trans(self):
        self.param_trans = torch.zeros(
            (self.src_num_slices, 2), device=self.device, requires_grad=True
        )
        self._set_translation_matrix()

    def _init_rot(self):
        self.param_theta = torch.zeros(
            self.src_num_slices, device=self.device, requires_grad=True
        )
        self._set_rotation_matrix()

    def set_rot_params(self, theta: torch.Tensor):
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        self.param_theta = theta.to(self.device)
        self._set_rotation_matrix()

    def set_trans_params(self, trans: torch.tensor):
        if trans.dim() == 1:
            trans = trans.unsqueeze(0)
        self.param_trans = trans.to(self.device)
        self._set_translation_matrix()

    def set_trans_z_params(self, trans: torch.tensor):
        if trans.dim() == 1:
            trans = trans.unsqueeze(0)
        self.param_trans_z = trans.to(self.device)
        self._set_translation_z_matrix()

    def _set_rotation_matrix(self):
        cos_a, sin_a = torch.cos(self.param_theta), torch.sin(self.param_theta)
        self.rot = torch.zeros((self.src_num_slices, 4, 4), device=self.device)
        self.rot[:, 3, 3] = 1
        self.rot[:, 2, 2] = 1
        self.rot[:, 0, 0] = cos_a
        self.rot[:, 0, 1] = -sin_a
        self.rot[:, 1, 0] = sin_a
        self.rot[:, 1, 1] = cos_a

    def _set_translation_matrix(self):
        self.trans = torch.eye(4, device=self.device)[None]
        self.trans = torch.repeat_interleave(self.trans, self.src_num_slices, dim=0)
        self.trans[:, : self.dim, 3] = self.param_trans

    def _set_translation_z_matrix(self):
        # self.trans_z = torch.eye(4, device=self.device)[None]
        # self.trans_z = torch.repeat_interleave(self.trans_z, self.src_num_slices, dim=0)
        # self.trans_z[:, 2, 3] = self.param_trans_z
        self.trans_z = self.param_trans_z

    def _make_sitk_image(self, aligned_image: torch.FloatTensor):
        sitk_image = sitk.GetImageFromArray(
            aligned_image.detach().cpu().squeeze().numpy().astype(np.float32)
        )
        sitk_image.CopyInformation(self.source_img.sitk_image)
        return sitk_image

    def _get_mask_loss(self, warped_mask, target_mask, label=None, n_classes=2):
        return F.mse_loss(warped_mask.unsqueeze(0), target_mask)

    def apply_transform(self, image3d, mode="bilinear", eval=True):
        if not isinstance(image3d, torch.Tensor):
            image3d = torch.from_numpy(image3d)
        if image3d.dtype != torch.float32:
            image3d = image3d.float()
        if image3d.device != self.device:
            image3d = image3d.to(self.device)
        if eval:
            with torch.no_grad():
                return self(image3d, mode=mode)
        else:
            return self(image3d, mode=mode)

    def save_trans(self):
        self.saved_params["trans"] = self.param_trans.detach().clone()

    def save_rot(self):
        self.saved_params["theta"] = self.param_theta.detach().clone()

    def set_saved_params(self):
        if "theta" in self.saved_params:
            self.param_theta = self.saved_params["theta"].clone()
            self._set_rotation_matrix()
        self.param_trans = self.saved_params["trans"].clone()
        self._set_translation_matrix()
