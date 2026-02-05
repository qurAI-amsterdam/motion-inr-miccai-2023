import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import tqdm
import numpy as np
from utils import general
from networks import networks
from objectives import ncc
from objectives import regularizers
from pathlib import Path
from objectives.regularizers import compute_jacobian_matrix
from kwatsch.canonical_space import CanonicalImage
from collections import defaultdict
from kwatsch.common import (
    KEY_SAX_VIEW,
    KEY_4CH_SEG_VIEW,
    KEY_4CH_VIEW,
    KEY_SAX_SEG_VIEW,
)
from kwatsch.common import execute_resampling
from kwatsch.dice_loss import BinaryDiceLoss
from CMRI.general import blur_mask, MMS2MRILabel
from CMRI.general import generate_rv_myocardium
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __init__(
        self,
        fixed_image: CanonicalImage = None,
        moving_image: CanonicalImage = None,
        cardiac_views=(KEY_SAX_VIEW, KEY_4CH_VIEW),
        **kwargs,
    ):

        self.cimage_fixed = fixed_image
        self.cimage_moving = moving_image
        self.early_stopping = kwargs.get("early_stopping", False)
        self.stopped_at_epoch = 0
        self.device = "cuda"

        if isinstance(cardiac_views, tuple):
            cardiac_views = list(cardiac_views)
        self.cardiac_views = cardiac_views
        kwargs["cardiac_views"] = "_".join([c for c in cardiac_views])

        self.reg_with_mask = kwargs.get("reg_with_mask", True)
        self.use_shape_loss = kwargs.get("use_shape_loss", False)
        self.use_mask_loss = kwargs.get("use_mask_loss", False)
        self.warpinn_init = kwargs.get("warpinn_init", False)
        self.displacement_loss = kwargs.get("displacement_loss", False)

        self.exper_dir = Path(kwargs["exper_dir"] if "exper_dir" in kwargs else "/")
        self.model_dir = self.exper_dir / "models"
        kwargs["model_dir"] = str(self.model_dir)
        self.set_default_arguments()
        self._registration_init(**kwargs)

        self.xyz_sequence = kwargs[
            "xyz_sequence"
        ]  # This should be false, changing it TODO check on jorgs original code and images, changing again to True as it was casue stik
        self.normalize_coords = False

        self.multiview = False if len(cardiac_views) == 1 else True

        if fixed_image and moving_image:
            self.set_images(fixed_image, moving_image)

        # added new properties
        # coords_scale_factor & min_coord_offset are needed for interpolation purposes to get coords back to
        # voxel/world coordinates.
        # We make sure that network input is somehow in the range [-1, 1] but we input space
        # is more flexible and not limited as in [-1, 1] case.

        self.coords_scale_factor = None
        self.min_coord_offset = None

        self.dice_loss = BinaryDiceLoss(reduction="mean")
        if not self.exper_dir.is_dir():
            self.exper_dir.mkdir(parents=True)

    def set_images(self, fixed_image: CanonicalImage, moving_image: CanonicalImage):
        self.cimage_fixed = fixed_image
        self.cimage_moving = moving_image
        self._init_images(fixed_image, moving_image)
        self.coords_scale_factor = None
        self.min_coord_offset = None
        self._init_coords()

    def _init_coords(self):
        # Note: we use coords_scale_factor to bring coordinates close to [-1, 1] range for SIREN network
        self.possible_coordinate_tensor = defaultdict(dict)
        self.min_coord_offset, self.max_coord_offset = {}, {}
        self._collect_all_coords()
        self._collect_possible_coords()
        self.scale_coords()

    def _init_images(self, fixed_image, moving_image):
        self.moving_image = moving_image.get_sax_image(device=self.device)
        self.fixed_image = fixed_image.get_sax_image(device=self.device)
        self.moving_mask = moving_image.get_sax_image(
            image_type="mask", device=self.device
        )

        # Get mask for RV MYO by dilating the RV BP mask
        moving_rv_mask = (self.moving_mask == MMS2MRILabel.RVBP.value).long()
        rv_myo_mask = generate_rv_myocardium(
            moving_rv_mask.detach().cpu().numpy(), dilations=2
        )
        rv_myo_mask = torch.from_numpy(rv_myo_mask).long().to(self.device)

        # Define binary mask only based on myocardium for moving image
        self.moving_mask[self.moving_mask != MMS2MRILabel.LV.value] = 0
        self.moving_mask[self.moving_mask == MMS2MRILabel.LV.value] = 1
        self.moving_mask[rv_myo_mask == 1] = 1

        # Same for fixed image
        self.fixed_mask = fixed_image.get_sax_image(
            image_type="mask", device=self.device
        )
        fixed_rv_mask = (self.fixed_mask == MMS2MRILabel.RVBP.value).long()
        rv_myo_mask = generate_rv_myocardium(
            fixed_rv_mask.detach().cpu().numpy(), dilations=2
        )
        rv_myo_mask = torch.from_numpy(rv_myo_mask).long().to(self.device)
        self.fixed_mask[self.fixed_mask != MMS2MRILabel.LV.value] = 0
        self.fixed_mask[self.fixed_mask == MMS2MRILabel.LV.value] = 1
        self.fixed_mask[rv_myo_mask == 1] = 1
        self.voxel_size_xyz = fixed_image.get_spacing(
            device=self.device, key=KEY_SAX_VIEW
        )

        # Same masking for 4ch view
        if self.multiview:
            self.moving_image_4ch = moving_image.get_4ch_image(device=self.device)
            self.fixed_image_4ch = fixed_image.get_4ch_image(device=self.device)
            self.moving_mask_4ch = moving_image.get_4ch_image(
                mask=True, device=self.device
            )
            self.moving_mask_4ch[self.moving_mask_4ch != MMS2MRILabel.LV.value] = 0
            self.moving_mask_4ch[self.moving_mask_4ch == MMS2MRILabel.LV.value] = 1
            self.fixed_mask_4ch = fixed_image.get_4ch_image(
                mask=True, device=self.device
            )
            self.fixed_mask_4ch[self.fixed_mask_4ch != MMS2MRILabel.LV.value] = 0
            self.fixed_mask_4ch[self.fixed_mask_4ch == MMS2MRILabel.LV.value] = 1
            self.voxel_size_4ch_xyz = fixed_image.get_spacing(
                device=self.device, key=KEY_4CH_VIEW
            )

    def _collect_possible_coords(self):
        self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed"] = (
            self.cimage_fixed.get_sax_coords(device=self.device)
        )
        self.possible_coordinate_tensor[KEY_SAX_VIEW]["move"] = (
            self.cimage_moving.get_sax_coords(device=self.device)
        )

        coords_sax_seg = self.cimage_fixed.get_sax_image(image_type="mask")
        coords_sax_mask = coords_sax_seg == MMS2MRILabel.LV.value
        fixed_sax_seg_dilated = blur_mask(
            coords_sax_mask, kernel_shape=(2, 2), num_dilations=1, apply_blur=False
        )
        self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW]["fixed"] = (
            self.fixed_mask.flatten().bool()
        )

        coords_sax_seg = self.cimage_moving.get_sax_image(image_type="mask")
        coords_sax_mask = coords_sax_seg == MMS2MRILabel.LV.value
        moving_sax_seg_dilated = blur_mask(
            coords_sax_mask, kernel_shape=(2, 2), num_dilations=1, apply_blur=False
        )
        dilated_loss_mask = (
            torch.from_numpy(
                np.logical_or(fixed_sax_seg_dilated, moving_sax_seg_dilated)
            )
            .bool()
            .to(self.device)
        )
        self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW]["move"] = (
            self.moving_mask.flatten().bool()
        )
        self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed_roi_mask"] = (
            dilated_loss_mask
        )

        if self.multiview:
            self.possible_coordinate_tensor[KEY_4CH_VIEW]["fixed"] = (
                self.cimage_fixed.get_lax_4ch_coords_in_canon(
                    device=self.device
                ).squeeze()[..., :3]
            )
            self.possible_coordinate_tensor[KEY_4CH_VIEW]["move"] = (
                self.cimage_moving.get_lax_4ch_coords_in_canon(
                    device=self.device
                ).squeeze()[..., :3]
            )
            self.possible_coordinate_tensor[KEY_4CH_SEG_VIEW]["fixed"] = (
                self.fixed_mask_4ch.flatten().bool()
            )
            self.possible_coordinate_tensor[KEY_4CH_SEG_VIEW]["move"] = (
                self.moving_mask_4ch.flatten().bool()
            )

    def _collect_all_coords(self):
        # we need the full possible range of coordinates for scaling later, this variable is not used during training
        if self.multiview:
            self.all_coords = torch.cat(
                [
                    self.cimage_fixed.get_sax_coords(device=self.device),
                    self.cimage_moving.get_sax_coords(device=self.device),
                    self.cimage_fixed.get_lax_4ch_coords_in_canon(
                        device=self.device
                    ).squeeze()[..., :3],
                    self.cimage_moving.get_lax_4ch_coords_in_canon(
                        device=self.device
                    ).squeeze()[..., :3],
                ],
                dim=0,
            )
        else:
            # The small adaptation here to also support 1 view is to reducing the dimension of the tensor
            # In theory both coordinates live now in the same space so this should be fine
            self.all_coords = torch.cat(
                [
                    self.cimage_fixed.get_sax_coords(device=self.device),
                    self.cimage_moving.get_sax_coords(device=self.device),
                ],
                dim=0,
            )

    def scale_coords(self):
        cardiac_views = self.cardiac_views
        max_coords, _ = torch.max(self.all_coords, dim=0)
        min_coords, _ = torch.min(self.all_coords, dim=0)
        self.min_coord_offset["ALL"] = min_coords
        self.max_coord_offset["ALL"] = max_coords
        self.coords_scale_factor = (
            max_coords - min_coords
        )  # this is our total range for x, y, z dimensions

        for view_type in cardiac_views:
            for move_fixed in ["move", "fixed"]:
                self.possible_coordinate_tensor[view_type][move_fixed] = (
                    2
                    * (
                        (
                            self.possible_coordinate_tensor[view_type][move_fixed]
                            - min_coords
                        )
                        / (self.max_coord_offset["ALL"] - self.min_coord_offset["ALL"])
                    )
                    - 1
                )
                self.min_coord_offset[view_type], _ = torch.min(
                    self.possible_coordinate_tensor[view_type][move_fixed], dim=0
                )
                self.max_coord_offset[view_type], _ = torch.max(
                    self.possible_coordinate_tensor[view_type][move_fixed], dim=0
                )
                if self.verbose:
                    print(
                        "INFO - {}-{}: min_coord_offsets ".format(
                            view_type, move_fixed
                        ),
                        self.min_coord_offset[view_type].detach().cpu().numpy(),
                    )
                    print(
                        "INFO - {}-{}: max_coord_offsets ".format(
                            view_type, move_fixed
                        ),
                        self.max_coord_offset[view_type].detach().cpu().numpy(),
                    )

        self.all_coords = (
            2 * ((self.all_coords - min_coords) / (max_coords - min_coords)) - 1
        )
        self.coords_scale_factor = torch.abs(self.coords_scale_factor)

    def warp_coords(self, coords, spacing_xyz=None, eval_dvf=True, do_scale=True):
        if spacing_xyz is None:
            spacing_xyz = self.voxel_size_xyz

        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()

        if coords.device != self.device:
            coords = coords.to(self.device)

        if do_scale:
            coords = (
                2
                * (
                    (coords - self.min_coord_offset["ALL"])
                    / (self.max_coord_offset["ALL"] - self.min_coord_offset["ALL"])
                )
                - 1
            )

        transform_rel, dvf_jacobian = self._forward_chunked(
            coords, chunk_size=10000, eval_dvf=eval_dvf
        )

        if transform_rel.device != self.device:
            transform_rel = transform_rel.to(self.device)

        forward_estimate = torch.add(transform_rel, coords)

        x_indices, y_indices, z_indices = self._model_to_image_voxel_coords(
            forward_estimate, "offset", spacing_xyz
        )

        forward_estimate = torch.cat(
            [x_indices[:, None], y_indices[:, None], z_indices[:, None]], dim=-1
        )[None]
        forward_estimate = forward_estimate.detach().cpu().numpy().squeeze()

        if eval_dvf:
            return forward_estimate, dvf_jacobian.detach().cpu().numpy()
        else:
            return forward_estimate

    def warp(
        self,
        moving_image=None,
        spacing_xyz=None,
        return_transformation=False,
        eval_dvf=False,
        mode="bilinear",
    ):
        self.optimizer.zero_grad()

        if spacing_xyz is None:
            spacing_xyz = self.voxel_size_xyz

        dvf_jacobian = None

        if moving_image is None:
            moving_image = self.moving_image

        coordinates = self.possible_coordinate_tensor[KEY_SAX_VIEW][
            "fixed"
        ]  # tensor([[-1.0000, -1.0000, -1.0000],

        if eval_dvf:
            # we need to chunk to fit 3D image coordinates on gpu
            # Note this chuck is different from the batch size, and is only used to fit the coordinates on the gpu, so it is not a parameter to tune
            transform_rel, dvf_jacobian, dvf_jacobian_phys = self._forward_chunked(
                coordinates,
                chunk_size=10000,
                eval_dvf=eval_dvf,
                spacing_xyz=spacing_xyz,
                img_shape=moving_image.shape,
                compute_physical_dvf=True,
            )
            dvf_jacobian_det = (
                torch.det(dvf_jacobian).reshape(moving_image.shape).numpy()
            )
            dvf_jacobian = dvf_jacobian.reshape(moving_image.shape + (3, 3)).numpy()
            dvf_jacobian_phys = dvf_jacobian_phys.reshape(
                moving_image.shape + (3, 3)
            ).numpy()
            dvf_jacobian_phys = dvf_jacobian_phys + np.identity(3)
        else:
            # dvf_jacobian is returned as None in this case
            transform_rel, dvf_jacobian = self._forward_chunked(
                coordinates, chunk_size=10000
            )

        if transform_rel.device != self.device:
            transform_rel = transform_rel.to(self.device)

        forward_estimate = torch.add(transform_rel, coordinates)

        x_indices, y_indices, z_indices = self._model_to_image_voxel_coords(
            forward_estimate, "offset", spacing_xyz
        )
        forward_estimate = torch.cat(
            [x_indices[:, None], y_indices[:, None], z_indices[:, None]], dim=-1
        )[None]

        warped_img = self._torch_grid_sampling(
            moving_image, forward_estimate, moving_image.shape, mode=mode
        )
        warped_img = warped_img.detach().cpu().numpy()
        warped_img = warped_img.astype(np.int32) if mode == "nearest" else warped_img

        if return_transformation:
            # Convert form model coordinates to img coordinates
            x_indices, y_indices, z_indices = self._model_to_image_voxel_coords(
                coordinates, "offset", spacing_xyz
            )
            coordinates = torch.cat(
                [x_indices[:, None], y_indices[:, None], z_indices[:, None]], dim=-1
            )[None]
            dvf = forward_estimate - coordinates
            del coordinates
            return (
                warped_img,
                dvf.reshape(moving_image.shape + (3,)).detach().cpu().numpy(),
                dvf_jacobian,
                dvf_jacobian_det,
                dvf_jacobian_phys,
            )

        else:
            del coordinates
            return warped_img

    def warp_4ch_view(self, view_key, eval_dvf=False):
        img_type = "seg" if "seg" in view_key else "img"
        mode = "bilinear" if img_type == "img" else "nearest"

        # NOTE: 2ch views images and masks are stored with separate view_key (different for SAX images/masks)
        torch_moving = (
            torch.from_numpy(self.cimage_moving.views[view_key]["np_img"])
            .float()
            .to(self.device)
        )
        tgt_shape_zyx = torch_moving.shape
        coords_4ch_in_sax = self.possible_coordinate_tensor[KEY_4CH_VIEW][
            "fixed"
        ].clone()  # also for mask take image view key
        coords_4ch_in_sax = coords_4ch_in_sax.to(self.device)

        if eval_dvf:
            coords_4ch_in_sax = coords_4ch_in_sax.requires_grad_(True)
            transform_rel, dvf_jacobian = self._forward_chunked(
                coords_4ch_in_sax, eval_dvf=eval_dvf
            )
            dvf_jacobian_det = torch.det(dvf_jacobian).reshape(tgt_shape_zyx).numpy()
            dvf_jacobian = dvf_jacobian.reshape(tgt_shape_zyx + (3, 3)).numpy()
        else:
            transform_rel, dvf_jacobian = self._forward_chunked(coords_4ch_in_sax)

        if transform_rel.device != self.device:
            transform_rel = transform_rel.to(self.device)
        forward_estimate = torch.add(transform_rel, coords_4ch_in_sax)

        forward_estimate = self._model_to_image_voxel_coords(
            forward_estimate, "backward_2dview", None, None, key_of_view=KEY_4CH_VIEW
        )
        forward_estimate = torch.cat(
            [
                forward_estimate[0][:, None],
                forward_estimate[1][:, None],
                forward_estimate[2][:, None],
            ],
            dim=-1,
        )

        if tgt_shape_zyx[0] == 1:
            shape_div = torch.tensor(
                tgt_shape_zyx[1:][::-1] + (2,), dtype=torch.float32, device=self.device
            )
        else:
            shape_div = torch.tensor(
                tgt_shape_zyx[::-1], dtype=torch.float32, device=self.device
            )

        t_coords_normed = (forward_estimate / ((shape_div[None, None] - 1) / 2)) - 1
        t_coords_normed = t_coords_normed.reshape(tuple(tgt_shape_zyx) + (3,))
        t_coords_normed = t_coords_normed[..., :3]
        resampled_view = execute_resampling(torch_moving, t_coords_normed, mode=mode)
        resampled_view = resampled_view.detach().cpu().numpy()
        resampled_view = (
            resampled_view.astype(np.int32) if img_type == "seg" else resampled_view
        )

        if eval_dvf:
            return (
                resampled_view,
                transform_rel.reshape(torch_moving.shape + (3,)).detach().cpu().numpy(),
                dvf_jacobian,
                dvf_jacobian_det,
            )

        return resampled_view

    def get_input_output_inr(self, multi_view):

        coordinates = self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed"]
        coordinates = coordinates.to(self.device)

        if multi_view:
            coords_4ch_in_sax = self.possible_coordinate_tensor[KEY_4CH_VIEW][
                "fixed"
            ].clone()
            coords_4ch_in_sax = coords_4ch_in_sax.to(self.device)

        transform_rel, _ = self._forward_chunked(coordinates, chunk_size=10000)
        if multi_view:
            transform_rel_4ch, _ = self._forward_chunked(coords_4ch_in_sax)

        if transform_rel.device != self.device:
            transform_rel = transform_rel.to(self.device)

        if multi_view:
            if transform_rel_4ch.device != self.device:
                transform_rel_4ch = transform_rel_4ch.to(self.device)

        forward_estimate = torch.add(
            transform_rel, coordinates
        )  # Add the relative displacement to the coordinates
        if multi_view:
            forward_estimate_4ch = torch.add(transform_rel_4ch, coords_4ch_in_sax)

        moving_mask = self.moving_mask
        fixed_mask = self.fixed_mask
        spacing_xyz = self.voxel_size_xyz

        if multi_view:
            moving_mask_4ch = self.moving_mask_4ch
            fixed_mask_4ch = self.fixed_mask_4ch
            return_dict = {
                "SAX": {
                    "coordinates": coordinates,
                    "transform_rel": transform_rel,
                    "forward_estimate": forward_estimate,
                    "fixed_mask": fixed_mask,
                    "moving_mask": moving_mask,
                    "spacing_xyz": spacing_xyz,
                },
                "4CH": {
                    "coordinates": coords_4ch_in_sax,
                    "transform_rel": transform_rel_4ch,
                    "forward_estimate": forward_estimate_4ch,
                    "fixed_mask": fixed_mask_4ch,
                    "moving_mask": moving_mask_4ch,
                },
            }
        else:
            return_dict = {
                "SAX": {
                    "coordinates": coordinates,
                    "transform_rel": transform_rel,
                    "forward_estimate": forward_estimate,
                    "fixed_mask": fixed_mask,
                    "moving_mask": moving_mask,
                    "spacing_xyz": spacing_xyz,
                }
            }

        return return_dict

    def _torch_grid_sampling(self, image, coords, shape_zyx, mode="bilinear"):
        # NOTE: ideally, coords is [1, #coords, 3]
        if coords.shape[-1] == 4:
            coords = coords[..., :3]

        if len(coords.shape) == 2:
            coords = coords[None]

        if shape_zyx[0] == 1:
            shape_div = torch.tensor(
                shape_zyx[1:][::-1] + (2,), dtype=torch.float32, device=self.device
            )
        else:
            shape_div = torch.tensor(
                shape_zyx[::-1], dtype=torch.float32, device=self.device
            )

        t_coords_normed = (coords / ((shape_div[None, None] - 1) / 2)) - 1
        t_coords_normed = t_coords_normed.reshape(tuple(shape_zyx) + (3,))
        t_coords_normed = t_coords_normed[..., :3]

        return execute_resampling(image, t_coords_normed, mode=mode)

    def scale_jacobian(
        self,
        J_norm: torch.Tensor,
        image_shape: tuple[int, int, int],
        spacing_xyz: tuple[float, float, float],
    ) -> torch.Tensor:
        """
        Convert a batch of Jacobians from normalized coords to physical (mm) coords.

        Args:
        J_norm: [..., 3, 3] Jacobian w.r.t. normalized inputs.
        image_shape: (nx, ny, nz) number of voxels in each dim.
        spacing_xyz: (sx, sy, sz) physical spacing in mm per voxel.

        Returns:
        J_phys: [..., 3, 3] Jacobian in mm-space.
        """
        nx, ny, nz = image_shape
        sx, sy, sz = spacing_xyz

        # half-widths in mm along each axis:
        a_x = (nx * sx) / 2.0
        a_y = (ny * sy) / 2.0
        a_z = (nz * sz) / 2.0

        # build the scale matrices:
        S_in = torch.diag(
            torch.tensor([1 / a_x, 1 / a_y, 1 / a_z], device=J_norm.device)
        )
        S_out = torch.diag(torch.tensor([a_x, a_y, a_z], device=J_norm.device))

        # broadcast and multiply: S_out @ J_norm @ S_in
        # if J_norm shape is [N,3,3], this yields [N,3,3]
        J_phys = S_out.unsqueeze(0) @ J_norm @ S_in.unsqueeze(0)
        return J_phys

    def _forward_chunked(
        self,
        coordinates,
        chunk_size=10000,
        eval_dvf=False,
        spacing_xyz=None,
        img_shape=None,
        compute_physical_dvf=False,
    ):

        num_coords = coordinates.shape[0]  # coordinate shape is [#coords, 3]

        if num_coords % chunk_size == 0:
            num_chunks = num_coords // chunk_size
        else:
            num_chunks = (num_coords // chunk_size) + 1

        transform_rel, dvf_jacobian, dvf_jaccobian_no_dentity = None, None, None

        for chunk_coords in torch.chunk(coordinates, chunks=num_chunks, dim=0):
            if eval_dvf:
                chunk_coords = chunk_coords.requires_grad_(
                    True
                )  # torch.Size([9603, 3])
                output_rel = self.network(chunk_coords)  # torch.Size([9603, 3])
                dvf_jacobian_chunk = compute_jacobian_matrix(
                    chunk_coords, output_rel, add_identity=True
                )
                dvf_jacobian_chunk_no_identity = compute_jacobian_matrix(
                    chunk_coords, output_rel, add_identity=False
                )
                # x_indices, y_indices, z_indices = self._model_to_image_voxel_coords(chunk_coords, 'offset', spacing_xyz)
                # #  TODO: NOT SURE IF IT IS XYZ OR ZYX LOL
                # chunk_coords_scaled = torch.cat([x_indices[:, None], y_indices[:, None], z_indices[:, None]], dim=-1)
                # x_indices, y_indices, z_indices = self._model_to_image_voxel_coords(output_rel, 'offset', spacing_xyz)
                # # TODO
                # output_rel_scaled = torch.cat([x_indices[:, None], y_indices[:, None], z_indices[:, None]], dim=-1)
                # dvf_jaccobian_chunk_scaled = compute_jacobian_matrix(chunk_coords_scaled, output_rel_scaled, add_identity=True)
                dvf_jacobian = (
                    torch.cat([dvf_jacobian, dvf_jacobian_chunk.detach().cpu()], dim=0)
                    if dvf_jacobian is not None
                    else dvf_jacobian_chunk.detach().cpu()
                )
                dvf_jaccobian_no_dentity = (
                    torch.cat(
                        [
                            dvf_jaccobian_no_dentity,
                            dvf_jacobian_chunk_no_identity.detach().cpu(),
                        ],
                        dim=0,
                    )
                    if dvf_jaccobian_no_dentity is not None
                    else dvf_jacobian_chunk_no_identity.detach().cpu()
                )
            else:
                self.network.eval()
                with torch.no_grad():
                    output_rel = self.network(chunk_coords)

            output_rel = output_rel.detach().cpu()
            torch.cuda.empty_cache()
            transform_rel = (
                torch.cat([transform_rel, output_rel], dim=0)
                if transform_rel is not None
                else output_rel
            )

            del chunk_coords

        if (
            compute_physical_dvf
        ):  # Note the dvf scaled comes with the identity but the physical one doesnt
            dvf_jacobian_phys = self.scale_jacobian(
                dvf_jaccobian_no_dentity,
                image_shape=np.flip(img_shape),  # e.g. (nx,ny,nz)
                spacing_xyz=spacing_xyz,  # your (sx,sy,sz)
            )
            return transform_rel, dvf_jacobian, dvf_jacobian_phys
        else:
            return transform_rel, dvf_jacobian

    def _select_indices(self, possible_coordinate_tensor, batch_size=10000):
        indices = torch.randperm(possible_coordinate_tensor.shape[0], device="cuda")[
            :batch_size
        ]
        indices_rev = torch.randperm(
            possible_coordinate_tensor.shape[0], device="cuda"
        )[:batch_size]
        coordinate_tensor = possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)
        coordinate_tensor_rev = possible_coordinate_tensor[indices_rev, :]
        coordinate_tensor_rev = coordinate_tensor_rev.requires_grad_(True)
        return indices, indices_rev, coordinate_tensor, coordinate_tensor_rev

    def _model_to_image_voxel_coords(
        self,
        coordinate_tensor,
        input_scaling,
        spacing_xyz=None,
        array_shape=None,
        key_of_view=None,
    ):
        # the registration model uses centered (center of mass lv) world coordinates (mm).
        # furthermore, to adjust to the requirements of SIREN network (approx [-1, 1] scaling) we
        # need to re-scale to the non-centered (offset) "VOXEL" coordinates (1/spacing).
        if input_scaling == "offset":
            # IMPORTANT:  We should end up here only for SAX coordinates
            new_coords = (
                (coordinate_tensor + 1)
                * (self.max_coord_offset["ALL"] - self.min_coord_offset["ALL"])
                * 0.5
            ) + self.min_coord_offset["ALL"]

            # we need to de-center and de-scale the SAX coordinates
            new_coords = self.cimage_fixed.de_scale_aligned_voxel_coords(
                new_coords, scale_m=spacing_xyz
            )
            x_indices, y_indices, z_indices = (
                new_coords[..., 0],
                new_coords[..., 1],
                new_coords[..., 2],
            )

        elif input_scaling == "backward_2dview":
            # IMPORTANT: We should end up here only for LAX coordinates
            assert key_of_view is not None

            # scale back to scaled voxel coordinates in SAX space (canon-space)
            coords = (
                (coordinate_tensor + 1)
                * (self.max_coord_offset["ALL"] - self.min_coord_offset["ALL"])
                * 0.5
            ) + self.min_coord_offset["ALL"]

            coords = self.cimage_fixed.from_canon_to_original_view(
                key_of_view, coords
            ).squeeze()[..., :3]
            x_indices, y_indices, z_indices = (
                coords[..., 0],
                coords[..., 1],
                coords[..., 2],
            )

        else:
            assert array_shape is not None
            x_indices, y_indices, z_indices = general.de_normalize(
                array_shape,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )

            # multiply with inverse spacing
            x_indices = x_indices / spacing_xyz[0]
            y_indices = y_indices / spacing_xyz[1]
            z_indices = z_indices / spacing_xyz[2]

        return x_indices, y_indices, z_indices

    def _interpolate(
        self,
        image,
        coordinate_tensor,
        spacing_xyz,
        input_scaling="offset",
        key_of_view=None,
        move_fixed="move",
    ):

        image = image if not self.xyz_sequence else torch.permute(image, (2, 1, 0))

        x_indices, y_indices, z_indices = self._model_to_image_voxel_coords(
            coordinate_tensor,
            input_scaling,
            spacing_xyz,
            array_shape=image.shape,
            key_of_view=key_of_view,
        )

        image_coord_samples = general.fast_trilinear_interpolation(
            image, x_indices, y_indices, z_indices
        )

        return image_coord_samples

    def fit_new(self, epochs=None, fixed_image=None, moving_image=None):

        if fixed_image and moving_image:
            self.set_images(fixed_image, moving_image)

        if epochs is None:
            epochs = self.epochs

        torch.manual_seed(self.seed)
        self.loss_list, self.data_loss_list, self.cycle_loss_list = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )

        if not len(self.loss_list) == epochs:
            for type_of_view in self.cardiac_views:
                self.loss_list[type_of_view] = [0 for _ in range(epochs)]
                self.data_loss_list[type_of_view] = [0 for _ in range(epochs)]
                self.cycle_loss_list[type_of_view] = [0 for _ in range(epochs)]

        pbar = tqdm.tqdm(
            range(epochs),
            desc="fit with {}".format(",".join(self.cardiac_views)),
            total=epochs,
        )
        loss_trigger, saved_loss = 0, 0
        loss_patience = epochs // 10

        for i in pbar:
            type_of_view = KEY_SAX_VIEW

            if self.multiview:
                torch_fixed_4ch = (
                    torch.from_numpy(self.cimage_fixed.views[KEY_4CH_VIEW]["np_img"])
                    .float()
                    .to(self.device)
                )
                torch_moving_4ch = (
                    torch.from_numpy(self.cimage_moving.views[KEY_4CH_VIEW]["np_img"])
                    .float()
                    .to(self.device)
                )
                possible_coordinate_tensor_4ch = self.possible_coordinate_tensor[
                    "lax4ch"
                ]["fixed"]
                spacing_xyz_4ch = self.voxel_size_4ch_xyz

            else:
                (
                    torch_fixed_4ch,
                    torch_moving_4ch,
                    possible_coordinate_tensor_4ch,
                    spacing_xyz_4ch,
                ) = (None, None, None, None)

            self.training_iteration_new_combined(
                i,
                fixed_image_sax=self.fixed_image,
                moving_image_sax=self.moving_image,
                fixed_image_4ch=torch_fixed_4ch,
                moving_image_4ch=torch_moving_4ch,
                possible_coordinate_tensor_sax=self.possible_coordinate_tensor["sax"][
                    "fixed"
                ],
                possible_coordinate_tensor_4ch=possible_coordinate_tensor_4ch,
                spacing_xyz_sax=self.voxel_size_xyz,
                spacing_xyz_4ch=spacing_xyz_4ch,
            )

            pbar.set_description("Loss: {:.3f}".format(self.loss_list[type_of_view][i]))

            try:
                if i % (epochs // 50) == 0:
                    np.savetxt(
                        str(self.exper_dir / "loss_log.txt"),
                        self.loss_list[type_of_view],
                    )
                    np.savetxt(
                        str(self.exper_dir / "data_loss_log.txt"),
                        self.data_loss_list[type_of_view],
                    )
                    np.savetxt(
                        str(self.exper_dir / "cycle_loss_log.txt"),
                        self.cycle_loss_list[type_of_view],
                    )
            except:
                continue

            if self.early_stopping:
                if (
                    abs(self.loss_list[type_of_view][i] - saved_loss) < 0.001
                    or (self.loss_list[type_of_view][i] - saved_loss) > 0
                ):
                    loss_trigger += 1
                    if loss_trigger > loss_patience:
                        print("Warning - early optimizer stop @{}".format(i))
                        break
                else:
                    loss_trigger = 0
                saved_loss = self.loss_list[type_of_view][i]

        self.stopped_at_epoch = i + 1

        print(
            self.loss_list[type_of_view][0],
            self.loss_list[type_of_view][epochs // 2],
            self.loss_list[type_of_view][-1],
        )

        np.savetxt(str(self.exper_dir / "loss_log.txt"), self.loss_list[type_of_view])
        np.savetxt(
            str(self.exper_dir / "data_loss_log.txt"), self.data_loss_list[type_of_view]
        )
        np.savetxt(
            str(self.exper_dir / "cycle_loss_log.txt"),
            self.cycle_loss_list[type_of_view],
        )

        return saved_loss

    def training_iteration_new_combined(
        self,
        epoch,
        fixed_image_sax,
        moving_image_sax,
        fixed_image_4ch,
        moving_image_4ch,
        possible_coordinate_tensor_sax,
        possible_coordinate_tensor_4ch,
        spacing_xyz_sax,
        spacing_xyz_4ch,
    ):

        cycle_training = self.cycle_alpha > 0
        self.network.train()

        (
            indices_sax,
            indices_rev_sax,
            coordinate_tensor_sax,
            coordinate_tensor_rev_sax,
        ) = self._select_indices(possible_coordinate_tensor_sax, self.batch_size)
        if self.multiview:
            (
                indices_4ch,
                indices_rev_4ch,
                coordinate_tensor_4ch,
                coordinate_tensor_rev_4ch,
            ) = self._select_indices(possible_coordinate_tensor_4ch, self.batch_size)

        coordinate_mask_sax = self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW][
            "fixed"
        ][indices_sax]
        coordinate_mask_rev_sax = self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW][
            "move"
        ][indices_rev_sax]

        if self.multiview:
            coordinate_mask_4ch = self.possible_coordinate_tensor[KEY_4CH_SEG_VIEW][
                "fixed"
            ][indices_4ch]
            coordinate_mask_rev_4ch = self.possible_coordinate_tensor[KEY_4CH_SEG_VIEW][
                "move"
            ][indices_rev_4ch]

        fixed_coord_samples_sax = self._interpolate(
            fixed_image_sax,
            coordinate_tensor_sax,
            spacing_xyz_sax,
            input_scaling="offset",
            key_of_view="sax",
        )
        moving_coord_samples_sax = self._interpolate(
            moving_image_sax,
            coordinate_tensor_rev_sax,
            spacing_xyz_sax,
            input_scaling="offset",
            key_of_view="sax",
        )

        if self.multiview:
            fixed_coord_samples_4ch = self._interpolate(
                fixed_image_4ch,
                coordinate_tensor_4ch,
                spacing_xyz_4ch,
                input_scaling="backward_2dview",
                key_of_view="lax4ch",
            )

        forward_estimate_rel_sax = self.network(coordinate_tensor_sax)
        backward_estimate_rel_sax = self.network_rev(coordinate_tensor_rev_sax)

        if self.multiview:
            forward_estimate_rel_4ch = self.network(coordinate_tensor_4ch)
            backward_estimate_rel_4ch = self.network_rev(coordinate_tensor_rev_4ch)

        forward_estimate_sax = torch.add(
            forward_estimate_rel_sax, coordinate_tensor_sax
        )
        backward_estimate_sax = torch.add(
            backward_estimate_rel_sax, coordinate_tensor_rev_sax
        )

        if self.multiview:
            forward_estimate_4ch = torch.add(
                forward_estimate_rel_4ch, coordinate_tensor_4ch
            )
            backward_estimate_4ch = torch.add(
                backward_estimate_rel_4ch, coordinate_tensor_rev_4ch
            )

        transformed_samples_fw_sax = self.transform_no_add(
            forward_estimate_sax,
            spacing_xyz_sax,
            moving_image_sax,
            input_scaling="offset",
            key_of_view="sax",
        )
        transformed_samples_bw_sax = self.transform_no_add(
            backward_estimate_sax,
            spacing_xyz_sax,
            fixed_image_sax,
            input_scaling="offset",
            key_of_view="sax",
        )

        if self.multiview:
            transformed_samples_fw_4ch = self.transform_no_add(
                forward_estimate_4ch,
                spacing_xyz_4ch,
                moving_image_4ch,
                input_scaling="backward_2dview",
                key_of_view="lax4ch",
            )
            transformed_samples_bw_4ch = self.transform_no_add(
                backward_estimate_4ch,
                spacing_xyz_4ch,
                fixed_image_4ch,
                input_scaling="backward_2dview",
                key_of_view="lax4ch",
            )

        loss_sax = self.criterion(transformed_samples_fw_sax, fixed_coord_samples_sax)
        loss = loss_sax

        if self.multiview:
            loss_4ch = self.criterion(
                transformed_samples_fw_4ch, fixed_coord_samples_4ch
            )
            loss = (loss_sax + loss_4ch) / 2

        if cycle_training:
            cycled_output = self.network_rev(forward_estimate_sax)
            cycled_output_B = self.network(backward_estimate_sax)

            # if -rev is a true inverse, these should cancel
            cycle_error_A = cycled_output + forward_estimate_rel_sax
            cycle_error_B = cycled_output_B + backward_estimate_rel_sax

        if cycle_training:
            cycle_loss = self.criterion(
                transformed_samples_bw_sax, moving_coord_samples_sax
            )
            loss = loss + cycle_loss

        self.data_loss_list["sax"][epoch] = loss.detach().cpu().numpy()

        if cycle_training:
            if self.cycle_l1:
                cycle_loss = torch.mean(
                    torch.linalg.vector_norm(cycle_error_A, axis=1)
                ) + torch.mean(torch.linalg.vector_norm(cycle_error_B, axis=1))
            else:
                cycle_loss = torch.mean(torch.square(cycle_error_A)) + torch.mean(
                    torch.square(cycle_error_B)
                )
            if epoch > self.cycle_loss_delay:
                if self.cycle_loss_schedule:
                    # Jorg: this is not used and probably wasn't used by louis either because it does not do anything
                    cycle_alpha = (
                        (epoch - self.cycle_loss_delay)
                        / (self.epochs - self.cycle_loss_delay)
                        * self.cycle_alpha
                    )
                else:
                    cycle_alpha = self.cycle_alpha
                loss = loss + cycle_alpha * cycle_loss

            self.cycle_loss_list["sax"][epoch] = cycle_loss.detach().cpu().numpy()

        output_rel_sax = torch.subtract(forward_estimate_sax, coordinate_tensor_sax)
        output_rel_rev_sax = torch.subtract(
            backward_estimate_sax, coordinate_tensor_rev_sax
        )

        if self.multiview:
            output_rel_4ch = torch.subtract(forward_estimate_4ch, coordinate_tensor_4ch)

        if not self.reg_with_mask:
            self.reg_with_mask = True
            print(
                "INFO - reg_with_mask overrided to True, for future runs change kwargs in model init"
            )

        if (
            self.jacobian_regularization
            and self.alpha_jacobian > 0
            and self.reg_with_mask
        ):
            reg_loss_in = regularizers.compute_balanced_jacobian_loss(
                coordinate_tensor_sax, output_rel_sax, loss_mask=coordinate_mask_sax
            )

            reg_loss_bg = regularizers.compute_balanced_jacobian_loss(
                coordinate_tensor_sax, output_rel_sax, loss_mask=~coordinate_mask_sax
            )

            loss = (
                loss
                + self.alpha_jacobian * reg_loss_in
                + self.alpha_jacobian / 1000 * reg_loss_bg
            )

            if self.multiview:
                reg_loss_in = regularizers.compute_balanced_jacobian_loss(
                    coordinate_tensor_4ch, output_rel_4ch, loss_mask=coordinate_mask_4ch
                )

                reg_loss_bg = regularizers.compute_balanced_jacobian_loss(
                    coordinate_tensor_4ch,
                    output_rel_4ch,
                    loss_mask=~coordinate_mask_4ch,
                )

                loss = (
                    loss
                    + self.alpha_jacobian * reg_loss_in
                    + self.alpha_jacobian / 1000 * reg_loss_bg
                )

            if cycle_training:
                loss += self.alpha_jacobian * regularizers.compute_jacobian_loss(
                    coordinate_tensor_rev_sax,
                    output_rel_rev_sax,
                    batch_size=self.batch_size,
                )

        self.optimizer.zero_grad()

        loss.backward()

        if self.onecycle_policy:
            if epoch < self.epochs / 4:
                cur_lr = self.lr / 1000 + (4 * epoch / self.epochs * self.lr)
            elif epoch < self.epochs * 0.8:
                cur_lr = (
                    self.lr
                    - (1 / 0.8)
                    * 4
                    / 3
                    * (epoch - self.epochs / 4)
                    / self.epochs
                    * self.lr
                )
            else:
                cur_lr = self.lr / 200
            for g in self.optimizer.param_groups:
                g["lr"] = cur_lr

        self.optimizer.step()
        if self.scheduler_arg:
            self.scheduler.step(loss)

        if self.wandb:
            wandb.log({"loss": loss.detach().cpu().numpy()})
        if self.wandb:
            wandb.log({"loss_sax": loss_sax.detach().cpu().numpy()})
        if self.multiview and self.wandb:
            wandb.log({"loss_4ch": loss_4ch.detach().cpu().numpy()})
        if cycle_training and self.wandb:
            wandb.log({"cycle_loss": cycle_loss.detach().cpu().numpy()})

        # TODO: Not sure how to best manage this, both should be saved separetely for comparison
        if self.multiview:
            self.loss_list["lax4ch"][epoch] = loss.detach().cpu().numpy()
        self.loss_list["sax"][epoch] = loss.detach().cpu().numpy()

    def transform(
        self,
        transformation,
        spacing_xyz,
        coordinate_tensor,
        moving_image,
        input_scaling="offset",
        key_of_view=None,
    ):
        """Transform moving image given a transformation."""
        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return self._interpolate(
            moving_image,
            transformation,
            spacing_xyz,
            input_scaling=input_scaling,
            key_of_view=key_of_view,
        )

    def transform_no_add(
        self,
        transformation,
        spacing_xyz,
        moving_image,
        input_scaling="offset",
        key_of_view=None,
    ):
        """Transform moving image given a transformation."""
        return self._interpolate(
            moving_image,
            transformation,
            spacing_xyz,
            input_scaling=input_scaling,
            key_of_view=key_of_view,
        )

    def savenets(self, fname="default.pth"):
        if not self.model_dir.is_dir():
            self.model_dir.mkdir(parents=True)
        f_fname = str(self.model_dir / f"F_{fname}")
        print("INFO - Save models to {}".format(f_fname))
        torch.save(self.network.state_dict(), f_fname)

    def _registration_init(self, **kwargs):
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.scheduler_arg = kwargs["scheduler"] if "scheduler" in kwargs else False
        self.seed = kwargs["seed"] if "seed" in kwargs else self.args["seed"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.onecycle_policy = (
            kwargs["onecycle_policy"]
            if "onecycle_policy" in kwargs
            else self.args["onecycle_policy"]
        )
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )
        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )
        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]
        self.cycle_loss_list = [0 for _ in range(self.epochs)]

        # Set seed
        torch.manual_seed(self.seed)

        # Load network
        self.network_from_file = (
            self.model_dir / kwargs["network"]
            if "network" in kwargs
            else self.args["network"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )
        if self.network_type == "MLP":
            self.network = networks.MLP(self.layers)
            self.network_rev = networks.MLP(self.layers)
        else:
            self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            self.network_rev = networks.Siren(self.layers, self.weight_init, self.omega)
        if self.network_from_file is not None:
            self.network.load_state_dict(torch.load(self.network_from_file))
            self.network.eval()
            if self.gpu:
                self.network.cuda()

        # Choose the optimizer
        m_params = list(self.network.parameters()) + list(self.network_rev.parameters())
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(m_params, lr=self.lr, momentum=self.momentum)

        elif self.optimizer_arg.lower() == "adamw":
            self.optimizer = optim.AdamW(m_params, lr=self.lr)

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(m_params, lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(m_params, lr=self.lr)

        else:
            self.optimizer = optim.SGD(m_params, lr=self.lr, momentum=self.momentum)
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            verbose=True,
            min_lr=1e10,
        )

        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()
            self.network_rev.cuda()

        # Parse arguments from kwargs
        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]
        self.mask_2 = kwargs["mask_2"] if "mask_2" in kwargs else self.args["mask_2"]
        self.cycle_l1 = (
            kwargs["cycle_l1"] if "cycle_l1" in kwargs else self.args["cycle_l1"]
        )
        self.cycle_loss_schedule = (
            kwargs["cycle_loss_schedule"]
            if "cycle_loss_schedule" in kwargs
            else self.args["cycle_loss_schedule"]
        )
        self.cycle_loss_delay = (
            kwargs["cycle_loss_delay"]
            if "cycle_loss_delay" in kwargs
            else self.args["cycle_loss_delay"]
        )

        # Parse regularization kwargs
        self.raw_jacobian_regularization = (
            kwargs["raw_jacobian_regularization"]
            if "raw_jacobian_regularization" in kwargs
            else self.args["raw_jacobian_regularization"]
        )
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.background_weight = (
            kwargs["background_weight"]
            if "background_weight" in kwargs
            else self.args["background_weight"]
        )

        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bendreg_paperversion = (
            kwargs["bendreg_paperversion"]
            if "bendreg_paperversion" in kwargs
            else self.args["bendreg_paperversion"]
        )
        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        # Set seed
        torch.manual_seed(self.seed)

        self.wandb = kwargs["wandb"] if "wandb" in kwargs else self.args["wandb"]

        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )
        self.cycle_alpha = (
            kwargs["cycle_alpha"]
            if "cycle_alpha" in kwargs
            else self.args["cycle_alpha"]
        )
        self.cycle_in_mm = (
            kwargs["cycle_in_mm"]
            if "cycle_in_mm" in kwargs
            else self.args["cycle_in_mm"]
        )

    def cuda(self):
        """Move the model to the GPU."""

        # Standard variables
        self.network.cuda()
        self.network_rev.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # jorg begin
        self.args["exper_dir"] = "/home/jorg/expers/cmri_motion"
        self.args["normalize_coords"] = True
        self.args["xyz_sequence"] = False
        # jorg end
        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None
        self.args["wandb"] = False

        self.args["cycle_alpha"] = 0.1
        self.args["cycle_in_mm"] = False
        self.args["voxel_size"] = (1, 1, 1)

        self.args["method"] = 1

        self.args["lr"] = 0.00001
        self.args["onecycle_policy"] = False
        self.args["batch_size"] = 10000
        self.args["layers"] = [
            3,
            256,
            256,
            256,
            3,
        ]  # If I change here to 2 then this should do the trick 3 TODO as a kwarg
        self.args["velocity_steps"] = 1

        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1

        self.args["raw_jacobian_regularization"] = False
        self.args["jacobian_regularization"] = False
        self.args["alpha_jacobian"] = 0.05
        self.args["background_weight"] = 0.001

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bendreg_paperversion"] = False
        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (200, 200)

        self.args["network"] = None

        self.args["epochs"] = 2500
        self.args["cycle_loss_delay"] = 1250
        self.args["cycle_loss_schedule"] = False
        self.args["cycle_l1"] = False
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"

        self.args["network_type"] = "MLP"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1
