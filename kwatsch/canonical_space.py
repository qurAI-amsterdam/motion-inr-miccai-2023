import numpy as np
import torch
import SimpleITK as sitk
from kwatsch.common import (
    identity_grid,
    execute_resampling,
    get_voxel_to_world_transforms,
)
from kwatsch.common import make_homegeneous_identity_grid
import scipy
from enum import Enum
from torch import inverse as tr_inv
from collections import defaultdict
from CMRI.general import (
    normalize_image,
    get_center,
    check_apex_base_orientation,
    rotation_matrix,
)
from kwatsch.common import (
    KEY_SAX_VIEW,
    KEY_2CH_VIEW,
    KEY_2CH_SEG_VIEW,
    KEY_4CH_SEG_VIEW,
    KEY_4CH_VIEW,
)
from CMRI.contours.common import convert_mask_to_contour, create_mask_epi_heart


class MRILabel(Enum):
    BG = 0
    LVBP = 1
    LV = 2
    RVBP = 3


class CanonicalImage(object):
    def __init__(
        self,
        sitk_image: sitk.Image,
        sitk_seg: sitk.Image,
        label=MRILabel,
        key=KEY_SAX_VIEW,
        device="cuda",
        normalize=False,
        xyz_sequence=True,
        dtype=torch.float32,
        source_obj=None,
        z_flip=None,
    ):
        self.views = defaultdict(dict)
        self.meshes = defaultdict(dict)
        self.main_key = key
        self.dtype = dtype
        self.device = device
        self.z_flip = z_flip
        self.views[key]["sitk_img"] = sitk_image
        (
            self.views[self.main_key]["rotate"],
            self.views[self.main_key]["scale"],
            self.views[self.main_key]["translate"],
        ) = get_voxel_to_world_transforms(sitk_image, device=device)
        self.views[self.main_key]["origin_xyz"] = (
            torch.from_numpy(np.asarray(sitk_image.GetOrigin()).astype(np.float32))
            .float()
            .to(device)
        )
        self.views[key]["sitk_seg"] = sitk_seg
        self.views[key]["spacing_xyz"] = (
            torch.from_numpy(
                np.asarray(self.views[key]["sitk_img"].GetSpacing()).astype(np.float32)
            )
            .float()
            .to(self.device)
        )
        self.xyz_sequence = xyz_sequence
        self.label = label
        self.shape_zyx = sitk_image.GetSize()[::-1]
        self.t_possible_coords = defaultdict(dict)
        self.t_coords_aligned_xyz = None
        self.views[key]["np_img"] = sitk.GetArrayFromImage(sitk_image).astype(
            np.float32
        )
        self.views[key]["np_seg"] = sitk.GetArrayFromImage(sitk_seg).astype(np.int32)
        self.source_obj = source_obj
        if normalize:
            self.views[key]["np_img"] = normalize_image(
                self.views[key]["np_img"], percentile=(1, 99)
            )
        if source_obj is None:
            self._prepare_transformation()
        else:
            self._copyInformation(source_obj)
            lv_com_zyx = get_center(
                self.views[self.main_key]["np_seg"], self.label.LVBP.value
            )
            self.views[self.main_key]["original_lv_com_xyz"] = torch.multiply(
                torch.from_numpy(lv_com_zyx[::-1].copy()).float().to(self.device),
                self.views[self.main_key]["spacing_xyz"],
            )

    def _generate_canonical_direction(self):
        # this matrix should be the new Direction matrix for the new sitk image (aligned to canonical view)
        # TODO: this is not correct !!! BUT we only use this to create the new SITK image which is not used further on
        self.views[self.main_key]["canon_rotate"] = self.views[self.main_key]["rotate"]

    def _copyInformation(self, source):
        # We use this to copy information from fixed image to moving image
        self.z_flip = source.z_flip
        self.flip_y_m = source.flip_y_m.clone()
        self.flip_z_m = source.flip_z_m.clone()
        self.rot_rv_lv_m = source.rot_rv_lv_m.clone()
        self.views[self.main_key]["lv_com_xyz"] = source.views[self.main_key][
            "lv_com_xyz"
        ].clone()
        self.views[self.main_key]["coords_scaled"] = source.views[self.main_key][
            "coords_scaled"
        ].clone()
        self.views[self.main_key]["canon_rotate"] = source.views[self.main_key][
            "canon_rotate"
        ].clone()
        # in method "align_images" we set the aligned center of mass. But if we copied this info from a pre-
        # existing object, setting this property in align_images will be skipped (see below).
        self.views[self.main_key]["lv_com_aligned_xyz"] = source.views[self.main_key][
            "lv_com_aligned_xyz"
        ].clone()
        self.views[self.main_key]["coords_scaled_centered"] = source.views[
            self.main_key
        ]["coords_scaled_centered"].clone()
        (
            self.views[self.main_key]["rotate"],
            self.views[self.main_key]["scale"],
            self.views[self.main_key]["translate"],
        ) = (
            source.views[self.main_key]["rotate"].clone(),
            source.views[self.main_key]["scale"].clone(),
            source.views[self.main_key]["translate"].clone(),
        )

    def _prepare_transformation(self):
        self.flip_y_m, self.flip_z_m, self.rotmat = (
            torch.eye(4, device=self.device),
            torch.eye(4, device=self.device),
            torch.eye(4, device=self.device),
        )
        # NOTE: sometimes we need to pass z_flip as argument to object (it's != None then) because of erroneous
        # automatic segmentations
        if self.z_flip is None:
            self._check_apex_base_orientation(self.views[self.main_key]["np_seg"])
        lv_com_zyx = get_center(
            self.views[self.main_key]["np_seg"], self.label.LVBP.value
        )
        rv_com_zyx = get_center(
            self.views[self.main_key]["np_seg"], self.label.RVBP.value
        )
        self.rot_rv_lv_m = torch.eye(4).to(self.device)
        self.vec_lv_rv = rv_com_zyx - lv_com_zyx  # NOTE: zyx sequence
        self.rot_rv_lv_m[:3, :3] = torch.from_numpy(
            CanonicalImage.get_orientation(y=self.vec_lv_rv[1], x=self.vec_lv_rv[2])
        ).float()
        self.prepare_y_z_flip()
        # IMPORTANT: we use the center of mass of the left ventricle for centering the scaled coordinates.
        #            we will do this again for the segmentations when aligned to the canonical view.
        self.views[self.main_key]["lv_com_xyz"] = torch.multiply(
            torch.from_numpy(lv_com_zyx[::-1].copy()).float().to(self.device),
            self.views[self.main_key]["spacing_xyz"],
        )
        coords = self._init_scale_coords(
            self.shape_zyx, self.views[self.main_key]["scale"]
        )
        self.views[self.main_key]["coords_scaled"] = coords  # [#coords, 3]
        # generate sitk Direction for image in canonical view
        self._generate_canonical_direction()

    def _init_scale_coords(
        self,
        shape_zyx: tuple,
        m_scale: torch.FloatTensor,
        filter_on_indices=None,
        return_homogenous=False,
    ):
        # returns [#coords, 3]
        coords = identity_grid(
            shape_zyx, device=self.device, do_flip_sequence=self.xyz_sequence
        )
        # add homogenous coordinate
        coords = torch.cat(
            [coords, torch.ones(coords.shape[:-1] + (1,), device=self.device)], dim=-1
        )
        # scale with spacing and translate with image origin
        coords = coords @ m_scale
        # set grid coordinate to voxel center (half spacing in all directions)
        voxel_origin = -torch.diag(m_scale) / 2
        coords = coords + voxel_origin

        if filter_on_indices is not None:
            coords = coords[
                filter_on_indices
            ].squeeze()  # ToDO: filter_on_indices has shape [#, 1] get rid off last dim
        if return_homogenous:
            return coords

        return coords[..., :3]

    def align_images(self, rv_lv_rot_matrix=None, include_contours=False):
        if rv_lv_rot_matrix is not None:
            self.rot_rv_lv_m = rv_lv_rot_matrix
        # NOTE: we call this method ONLY for the SAX images to initialize the object!!! This is not used
        # when we add additional 2D views that first need to be transformed to 3D SAX volume
        t_img = torch.from_numpy(self.views[self.main_key]["np_img"]).float()
        self.views[self.main_key]["np_img_aligned"] = (
            self.align(
                t_img,
                mode="bilinear",
                src_shape_zyx=t_img.shape,
                tgt_shape_zyx=t_img.shape,
            )
            .detach()
            .cpu()
            .numpy()
        )
        direction = (
            self.views[self.main_key]["rotate"][:3, :3].detach().cpu().numpy().flatten()
        )
        spacing = (
            torch.diag(self.views[self.main_key]["scale"])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        origin = self.views[self.main_key]["translate"][:3, 3].detach().cpu().numpy()
        self.views[self.main_key]["sitk_img_aligned"] = self.create_sitk_image(
            self.views[self.main_key]["np_img_aligned"],
            spacing_xyz=spacing,
            origin_xyz=origin,
            direction=direction,
            dtype=np.float32,
        )
        t_seg = torch.from_numpy(self.views[self.main_key]["np_seg"]).float()
        self.views[self.main_key]["np_seg_aligned"] = (
            self.align(
                t_seg,
                mode="nearest",
                src_shape_zyx=t_seg.shape,
                tgt_shape_zyx=t_seg.shape,
            )
            .detach()
            .cpu()
            .numpy()
        )

        if include_contours:
            epi_mask = create_mask_epi_heart(
                self.views[self.main_key]["np_seg_aligned"],
                num_dilations=2,
                kernel=(2, 2),
            )
            contours, contour_as_mask, normals, normal_as_mask = (
                convert_mask_to_contour(
                    self.views[self.main_key]["np_seg_aligned"],
                    epi_mask=epi_mask,
                    upfactor=8,
                    compute_derivatives=True,
                )
            )
            self.views[self.main_key]["np_contour_as_mask"] = contour_as_mask
            self.views[self.main_key]["np_normal_as_mask"] = normal_as_mask
            self.views[self.main_key]["np_contour"] = contours
            self.views[self.main_key]["np_normal"] = normals
        self.views[self.main_key]["origin_shift"] = np.zeros(3).astype(np.float32)
        zmask = (self.views[self.main_key]["np_seg_aligned"] == 1).any((1, 2))

        if ~zmask[0]:
            last_empty_slice = int(
                np.min(np.where(zmask)[0])
            )  # actually the first with mask but due to 0-indexing this works
            new_origin = self.views[self.main_key]["sitk_img"][
                :, :, last_empty_slice:
            ].GetOrigin()
            self.views[self.main_key]["origin_shift"] = np.asarray(new_origin) - origin

        self.views[self.main_key]["sitk_seg_aligned"] = self.create_sitk_image(
            self.views[self.main_key]["np_seg_aligned"],
            spacing_xyz=spacing,
            origin_xyz=origin,
            direction=direction,
            dtype=np.int32,
        )

        # Only if we did not copy this information from the source CanonicalImage (e.g. source->FixedImage)
        if self.source_obj is None:
            lv_com_zyx = get_center(
                self.views[self.main_key]["np_seg_aligned"], label=self.label.LVBP.value
            )
            self.views[self.main_key]["lv_com_aligned_xyz"] = torch.multiply(
                torch.from_numpy(lv_com_zyx[::-1].copy()).float().to(self.device),
                self.views[self.main_key]["spacing_xyz"],
            )
            coords = self._init_scale_coords(
                self.shape_zyx, self.views[self.main_key]["scale"]
            )
            self.views[self.main_key]["coords_scaled"] = coords  # [#coords, 3]
            # jorg changed 25-11: not centered
            self.views[self.main_key]["coords_scaled_centered"] = (
                coords - self.views[self.main_key]["lv_com_aligned_xyz"]
            )
        # print("INFO - center of mass aligned sax view ", self.views[self.main_key]['lv_com_aligned_xyz'])

    def align_coords(self, coords):
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        if coords.device != self.device:
            coords = coords.to(self.device)
        coords = torch.cat(
            [coords, torch.ones(coords.shape[:-1] + (1,), device=self.device)], dim=-1
        )
        # scale with spacing and translate with image origin
        coords = coords @ self.views[self.main_key]["scale"]
        # set grid coordinate to voxel center (half spacing in all directions)
        voxel_origin = -torch.diag(self.views[self.main_key]["scale"]) / 2
        coords = coords + voxel_origin
        coords = coords[..., :3]
        coords = coords - self.views[self.main_key]["lv_com_aligned_xyz"]
        return coords

    def _remove_rv_mask(self, np_array_seg: np.ndarray) -> np.ndarray:
        np_array_seg[np_array_seg == self.label.RVBP.value] = 0
        return np_array_seg

    def _get_sitk_slice(self, key: str, sitk_img) -> sitk.Image:

        mid_slice_id = sitk_img.GetSize()[-1] // 2
        sitk_new = sitk_img[:, :, mid_slice_id : mid_slice_id + 1]
        self.views[key]["slice_id"] = mid_slice_id
        return sitk_new

    def add_mesh(self, key, mesh, latent=None):
        # latent: optional, numpy vector of latents used to create the mesh with shape model
        if np.any(self.views[self.main_key]["origin_shift"] != 0):
            # mesh.points[:, 2] = mesh.points[:, 2] + self.views[self.main_key]['spacing_xyz'].detach().cpu().numpy()[-1] / 2
            # mesh.points = mesh.points + self.views[self.main_key]['origin_shift']
            # print("Canonical image - add mesh - adding origin-shit !!! ", self.views[self.main_key]['origin_shift'])
            pass
        com_mesh_xyz = mesh.points.mean(0)
        mesh.points = mesh.points - com_mesh_xyz
        self.meshes[key]["mesh"] = mesh
        self.meshes[key]["latent"] = None
        if latent is not None:
            self.meshes[key]["latent"] = latent

    def add_view(
        self,
        sitk_img: sitk.Image,
        key,
        dtype=np.float32,
        normalize=False,
        do_align=True,
        keep_3d=False,
        origin_offset=None,
    ):
        # try-out: origin_offset -> determined by alignment between LAX and SAX view. Should be added to
        # original LAX origin from NifTi image, see below
        mid_slice_id = None
        if (
            not keep_3d
            and key in [KEY_4CH_VIEW, KEY_4CH_SEG_VIEW, KEY_2CH_VIEW, KEY_2CH_SEG_VIEW]
            and (len(sitk_img.GetSize()) == 3 and sitk_img.GetSize()[-1] > 1)
        ):
            sitk_img = self._get_sitk_slice(key, sitk_img)
        self.views[key]["sitk_img"] = sitk_img
        self.views[key]["np_img"] = sitk.GetArrayFromImage(sitk_img).astype(dtype)
        self.views[key]["shape_zyx"] = self.views[key]["np_img"].shape
        self.views[key]["spacing_xyz"] = (
            torch.from_numpy(
                np.asarray(self.views[key]["sitk_img"].GetSpacing()).astype(np.float32)
            )
            .float()
            .to(self.device)
        )
        # Hacky but: ARVC LAX volumes can have more than one slice (normally cine LAX is one slice over timepoints.
        # in this case we select the middle slice of the volume

        if normalize:
            self.views[key]["np_img"] = normalize_image(
                self.views[key]["np_img"], percentile=(1, 99)
            )
        if key == KEY_2CH_SEG_VIEW:
            # the 2CH left ventricle segmentations contain invalid RV bloodpool masks (TODO: due to auto method)
            # THERE SHOULD not be RV in 2-chamber left ventricle view!
            self.views[key]["np_img"] = self._remove_rv_mask(self.views[key]["np_img"])
        R_tgt, S_tgt, T_tgt = get_voxel_to_world_transforms(
            sitk_img, device=self.device
        )
        if origin_offset is not None:
            T_tgt[:3, 3] = T_tgt[:3, 3] + torch.from_numpy(origin_offset).float().to(
                self.device
            )
        if mid_slice_id is not None:
            S_tgt[2, 2] = (
                1.0  # force slice thickness to 1mm for LAX views that do actually have whole volume...for now
            )
        (
            self.views[key]["rotate"],
            self.views[key]["scale"],
            self.views[key]["translate"],
        ) = (R_tgt, S_tgt, T_tgt)
        mode = "nearest" if dtype == np.int32 else "bilinear"
        if do_align:
            self._align_added_view(key, mode, keep_3d=keep_3d)
        self.views[key]["coords_in_main_scaled_centered"] = (
            self._get_coords_2dview_in_sax(key, keep_3d=keep_3d)
        )

    def _get_coords_2dview_in_sax(self, key, keep_3d=False):
        # find SAX coordinates of 2D view coordinates (e.g. target 4ch (key_to) source SAX (key_from)
        # Voxel coords need to be aligned with canoncial view
        view_coords_in_main_coords, _ = self._coords_to_view(
            key, self.main_key
        )  # key_to, key_from
        # align_voxel_coords was designed for 3D volumes. but here we have 2d view with coordinates in 3D
        # HACKY: we set z_dim used for reshape to 1
        view_coords_in_main_coords = self._align_voxel_coords(
            view_coords_in_main_coords, self.shape_zyx, z_dim=1
        )
        self.views[key]["bogus"] = view_coords_in_main_coords
        # scale coordinates with SAX spacing and its ORIGIN
        view_coords_in_main_coords = (
            view_coords_in_main_coords @ self.views[self.main_key]["scale"]
        )
        view_coords_in_main_coords[..., :3] = (
            view_coords_in_main_coords[..., :3]
            - self.views[self.main_key]["lv_com_aligned_xyz"]
        )
        # view_coords_in_main_coords[..., 2] = view_coords_in_main_coords[..., 2] - self.views[KEY_SAX_VIEW]['spacing_xyz'][-1]
        return view_coords_in_main_coords

    def _align_added_view(self, key, mode, keep_3d=False):
        # aligned_coords: These are the possible coordinates of the lax view in the SAX canonical space
        # with shape [#coords, 4] <- still homogenous
        # aligned_view, filtered_coords = self.resample_to_canonical_view(key, mode=mode)
        aligned_view, filtered_coords_scaled, filter_indices = (
            self.resample_to_canonical_view(key, mode=mode, keep_3d=keep_3d)
        )
        self.views[key]["canon_spacing_xyz"] = torch.diag(
            self.views[key]["canon_scale"]
        )[:3]
        # print("INFO - add_view {} - aligned 3D spacing ".format(key), self.views[key]['canon_spacing_xyz'])
        self.views[key]["np_img_aligned"] = aligned_view.numpy()
        filter_indices = filter_indices.detach().cpu().numpy()
        # np_img_aligned_filtered: contains only voxels of 2D cardiac view in 3D sax view [#coords]
        # values of coordinates coincides with values of segmentation mask e.g. [0, 1, 2, 3] for MMS2 dataset.
        self.views[key]["np_img_aligned_filtered"] = np.squeeze(
            self.views[key]["np_img_aligned"].flatten()[filter_indices]
        )
        # (a 2D cross section in this space)
        # coords live in canonical space, hence, we center them with the LV com in that space
        self.views[key]["filtered_coords_scaled"] = filtered_coords_scaled[..., :3]
        filtered_coords_scaled[..., :3] = (
            filtered_coords_scaled[..., :3]
            - self.views[self.main_key]["lv_com_aligned_xyz"]
        )
        filtered_coords_centered = filtered_coords_scaled[..., :3].clone()
        self.views[key]["filtered_coords_centered"] = filtered_coords_centered
        direction = (
            self.views[self.main_key]["canon_rotate"][:3, :3]
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )
        spacing = (
            self.views[key]["canon_spacing_xyz"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        origin = self.views[key]["canon_translate"][:3, 3].detach().cpu().numpy()
        self.views[key]["sitk_img_aligned"] = self.create_sitk_image(
            aligned_view.detach().cpu().numpy(),
            spacing_xyz=spacing,
            origin_xyz=origin,
            direction=direction,
            dtype=np.float32,
        )

    @staticmethod
    def _blur(np_array, zoom, sigma=None):
        if sigma is None:
            sigma = 0.25 / zoom
        # print("!!! Canonical Image - Blur !!! zoom {:.3f} & sigma {:.3f}".format(zoom, sigma))
        for z in range(np_array.shape[0]):
            np_array[z, :, :] = scipy.ndimage.gaussian_filter(np_array[z, :, :], sigma)
        return np_array

    def resample_to_canonical_view(
        self, key, mode="bilinear", keep_3d=False
    ) -> (torch.Tensor, torch.Tensor):
        # NOTE: this method is only used when we call add_view method, hence, for the additional long-axis
        # views. SAX images are aligned to canonical view using align() method.
        np_array = self.views[key][
            "np_img"
        ].copy()  # NOTE confusion: for 2D views np_img can also be a Mask (mode...)

        # new_spacing_xyz: currently, we increase resolution of SAX view in z-direction, hence spacing of resampled
        # volume is not the same as original sax volume
        vox_to_canon, grid_indices, tgt_shape_zyx, zoom = (
            self._coords_to_canonical_view(key, keep_3d=keep_3d)
        )
        if zoom is not None:
            np_array = CanonicalImage._blur(np_array, zoom)
        # we need to scale and translate these coordinates (defined in method _coords_to_canonical_view)
        # NOTE, _init_scale_coords also CENTERS the coordinate grid to the middle of the voxel (same for SAX coords)
        # filtered == although, the canonical space is a 3D volume, we here filter the coords to only contain the
        # slice coords of the original 2D long-axis view.
        torch_array = torch.from_numpy(np_array).float().to(self.device)
        src_shape_zyx = torch_array.shape
        filtered_coords_scaled = self._init_scale_coords(
            tgt_shape_zyx,
            self.views[key]["canon_scale"],
            grid_indices,
            return_homogenous=True,
        )
        resampled_array = self._resample(
            torch_array, vox_to_canon, src_shape_zyx, tgt_shape_zyx, mode=mode
        )
        np_resampled_array = resampled_array.detach().cpu().squeeze()
        return np_resampled_array, filtered_coords_scaled, grid_indices

    def _determine_new_shape_and_scale(self, key):
        zoom_z = int(
            torch.ceil(
                self.views[self.main_key]["scale"][2, 2]
                / self.views[key]["scale"][0, 0]
            ).item()
        )
        new_scale = self.views[self.main_key]["scale"].clone()
        new_scale[2, 2] = self.views[self.main_key]["scale"][2, 2] / zoom_z
        # x and y
        zoom_x = int(
            torch.ceil(
                self.views[self.main_key]["scale"][0, 0]
                / self.views[key]["scale"][0, 0]
            ).item()
        )
        new_scale[0, 0] = self.views[self.main_key]["scale"][0, 0] / zoom_x
        zoom_y = int(
            torch.ceil(
                self.views[self.main_key]["scale"][1, 1]
                / self.views[key]["scale"][1, 1]
            ).item()
        )
        new_scale[1, 1] = self.views[self.main_key]["scale"][1, 1] / zoom_y
        tgt_shape_zyx = np.asarray(
            (
                self.shape_zyx[0] * zoom_z,
                self.shape_zyx[1] * zoom_y,
                self.shape_zyx[2] * zoom_x,
            )
        )
        return tgt_shape_zyx, new_scale

    def _coords_to_canonical_view(
        self, key, keep_3d=False
    ) -> (torch.Tensor, tuple, torch.Tensor):
        # We use this function when ADDING 2D VIEWs. Find its coordinates in the ALIGNED 3D SAX volume
        # returns torch tensor with [#slices, #voxels, 4]
        # we assume here that sax view has larger slice thickness than in-plane resolution of long-axis view.
        # Hence, we increase the SAX grid resolution in z-axis by a factor
        #                    new_scale:   sax-slice-thickness / lax-in-plane resolution
        zoom = int(
            torch.ceil(
                self.views[self.main_key]["scale"][2, 2]
                / self.views[key]["scale"][0, 0]
            ).item()
        )
        num_z_slices = int(self.shape_zyx[0] * zoom)
        tgt_shape_zyx = np.asarray(
            (num_z_slices,) + self.shape_zyx[1:]
        )  # target is SAX view
        new_scale = self.views[self.main_key]["scale"].clone()
        new_scale[2, 2] = self.views[self.main_key]["scale"][2, 2] / zoom
        # tgt_shape_zyx, new_scale = self._determine_new_shape_and_scale(key)
        self.views[key]["canon_scale"] = new_scale
        new_trans = self.views[self.main_key]["translate"].clone()
        self.views[key]["canon_translate"] = (
            new_trans  # currently new_trans == original translate, hence, redundant
        )
        ident_grid = make_homegeneous_identity_grid(tgt_shape_zyx, device=self.device)
        # First align to canonical view
        voxel_grid_aligned = self._align_voxel_coords(ident_grid, tgt_shape_zyx)
        # Second, transform to world coordinate space
        world_grid = (
            voxel_grid_aligned
            @ new_scale.T
            @ self.views[self.main_key]["rotate"].T
            @ new_trans.T
        )
        # we need to bring world coordinates back to voxel coordinates (in lax volume)
        voxel_grid_in_tgt = (
            world_grid
            @ tr_inv(self.views[key]["translate"]).T
            @ tr_inv(self.views[key]["rotate"]).T
            @ tr_inv(self.views[key]["scale"]).T
        )
        voxel_grid_in_tgt = voxel_grid_in_tgt.reshape(-1, 4)
        # - _coords_to_canonical_view - centering voxel coords with -0.5 !!! for grid_resampler (align_corners=True)
        voxel_grid_in_tgt[:, :3] = voxel_grid_in_tgt[:, :3] - 0.5
        # IMPORTANT: we filter the SAX voxel coordinate grid pointing to LAX to only contain LAX slice 1
        #               still not clear to me why this otherwise won't work
        # unique_z_values = torch.unique(torch.round(voxel_grid_in_tgt[..., 2]))
        # z_bound = unique_z_values[int(len(unique_z_values) // 2)]
        if not keep_3d:
            z_bound_min, z_bound_max = -1.5, 1.5
            indices = torch.nonzero(
                (voxel_grid_in_tgt[..., 2] > z_bound_max)
                | (voxel_grid_in_tgt[..., 2] < z_bound_min)
            )
            coords_indices_2dview_only = torch.nonzero(
                (voxel_grid_in_tgt[..., 2] <= z_bound_max)
                & (voxel_grid_in_tgt[..., 2] >= z_bound_min)
            )
            voxel_grid_in_tgt[indices, :] = 0
        else:
            z_bound_min, z_bound_max = (
                torch.min(voxel_grid_in_tgt[..., 2]),
                torch.max(voxel_grid_in_tgt[..., 2]),
            )
            coords_indices_2dview_only = torch.nonzero(
                (voxel_grid_in_tgt[..., 2] <= z_bound_max)
                & (voxel_grid_in_tgt[..., 2] >= z_bound_min)
            )
        # filtered_voxel_grid_in_tgt: these are the actual grid coordinates that describe the additional (long-axis)
        # view in the original SAX space (it is a sparse array because additional views are 2D only).
        # We set all voxels in 3D volume to 0 when voxels ARE NOT IN original 2D view.

        return voxel_grid_in_tgt, coords_indices_2dview_only, tuple(tgt_shape_zyx), zoom

    def resample_to_view(
        self,
        input_image=None,
        key_to=KEY_4CH_VIEW,
        img_type="img",
        key_from=None,
        de_align=False,
    ) -> torch.Tensor:

        assert img_type in ["img", "seg"]
        mode = "nearest" if img_type == "seg" else "bilinear"
        if key_from is not None:
            np_array = (
                self.views[key_from]["np_seg"]
                if img_type == "seg"
                else self.views[key_from]["np_img"]
            )
        else:
            np_array = input_image
        torch_array = torch.from_numpy(np_array).float().to(self.device)
        src_shape_zyx = np_array.shape
        vox_to_2dview, tgt_shape_zyx = self._coords_to_view(
            key_to, key_from
        )  # transformation LAX->SAX
        if de_align:
            vox_to_2dview = self._de_align_voxel_coords(
                vox_to_2dview, src_shape_zyx, z_dim=1
            )

        resampled_array = self._resample(
            torch_array, vox_to_2dview, src_shape_zyx, tgt_shape_zyx, mode=mode
        )

        return resampled_array.detach().cpu().squeeze()

    def _align_voxel_coords(self, voxel_grid, shape_zyx, z_dim=None):
        if isinstance(shape_zyx, tuple):
            shape_zyx = np.asarray(shape_zyx).astype(np.int32)
        trans_origin_xyz = torch.eye(4).to(self.device)
        trans_origin_xyz[:3, 3] = (
            torch.from_numpy(shape_zyx[::-1].copy()).float() / 2
        ) - 0.5
        trans_m = (
            tr_inv(trans_origin_xyz).T
            @ self.flip_z_m.T
            @ self.flip_y_m.T
            @ tr_inv(self.rot_rv_lv_m)
            @ trans_origin_xyz.T
        )
        if z_dim is None:
            z_dim = shape_zyx[0]
        return voxel_grid.reshape(tuple((z_dim, -1, 4))) @ trans_m

    def _de_align_voxel_coords(self, voxel_grid, shape_zyx=None, z_dim=None):
        if shape_zyx is None:
            shape_zyx = self.shape_zyx

        # from aligned voxel coordinates in SAX to de-aligned (original) voxel coordinates
        # actually shape_zyx should be equal to original SAX shape
        # we use z-dim when de aligning 2D view coordinates
        trans_origin_xyz = torch.eye(4).to(self.device)
        trans_origin_xyz[:3, 3] = (torch.tensor(shape_zyx[::-1]) / 2) - 0.5
        trans_m = (
            tr_inv(trans_origin_xyz).T
            @ self.rot_rv_lv_m
            @ self.flip_y_m.T
            @ self.flip_z_m.T
            @ trans_origin_xyz.T
        )
        if z_dim is None:
            z_dim = shape_zyx[0]
        voxel_grid = voxel_grid.reshape(z_dim, -1, 4) @ trans_m

        return voxel_grid

    def de_scale_aligned_voxel_coords(self, voxel_grid, scale_m=None):
        # from aligned, scaled, centered SAX coordinates to VOXEL coordinates of original SAX volume
        # jorg changed 25-11: removed centering
        voxel_grid[..., :3] = (
            voxel_grid[..., :3] + self.views[self.main_key]["lv_com_aligned_xyz"]
        )
        # voxel_grid[..., 2] = voxel_grid[..., 2] + self.views[KEY_SAX_VIEW]['spacing_xyz'][-1] / 2
        if scale_m is None:
            scale_m = self.views[self.main_key]["scale"]
        # in some circumstances (e.g. during registration of volume) we use method to pass non-homogenous voxel grid
        # in that case we use vector division instead of matrix multiplication.
        if voxel_grid.shape[-1] == 3:
            if scale_m.dim() == 1:
                voxel_grid = torch.divide(voxel_grid, scale_m)
            else:
                voxel_grid = voxel_grid @ tr_inv(scale_m[:3, :3])
        else:
            voxel_grid = voxel_grid @ tr_inv(scale_m)
        return voxel_grid

    def _de_align_view(self, torch_array, mode="bilinear"):
        # NOTE: we only use this for 2D views that were first resampled to aligned 3D sax view.
        # print("_de_align_view...", shape_zyx)
        # print("_de_align_view...",
        #      torch.min(voxel_grid[..., 2]), torch.max(voxel_grid[..., 2]))
        shape_zyx = torch_array.shape
        ident_grid = make_homegeneous_identity_grid(shape_zyx, device=self.device)
        ident_grid = ident_grid.reshape(shape_zyx[0], -1, 4)
        trans_origin_xyz = torch.eye(4).to(self.device)
        trans_origin_xyz[:3, 3] = (torch.tensor(shape_zyx[::-1]) / 2) - 0.5
        # when aligning we first flip_z, then flip_y and then rotate.
        # the reverse: first rotate, then flip y, flip z (we do not have to take inverse of flips because
        # inverse of diagonal matrix is equal to original matrix!
        trans_m = (
            tr_inv(trans_origin_xyz).T
            @ self.rot_rv_lv_m
            @ self.flip_y_m.T
            @ self.flip_z_m.T
            @ trans_origin_xyz.T
        )
        transformed_grid = ident_grid @ trans_m
        return self._resample(
            torch_array, transformed_grid, shape_zyx, shape_zyx, mode=mode
        )

    def _coords_to_view(self, key_to, key_from) -> (torch.Tensor, tuple):
        # returns torch tensor with [#slices, #voxels, 4]
        tgt_shape_zyx = self.views[key_to]["np_img"].shape
        ident_grid = make_homegeneous_identity_grid(tgt_shape_zyx, device=self.device)
        ident_grid = ident_grid.reshape(tgt_shape_zyx[0], -1, 4)
        # transform to world coordinates
        world_grid = (
            ident_grid
            @ self.views[key_to]["scale"].T
            @ self.views[key_to]["rotate"].T
            @ self.views[key_to]["translate"].T
        )
        # we need to bring world coordinates back to voxel coordinates (in sax volume)
        target_grid = (
            world_grid
            @ tr_inv(self.views[key_from]["translate"]).T
            @ tr_inv(self.views[key_from]["rotate"]).T
            @ tr_inv(self.views[key_from]["scale"]).T
        )

        return target_grid, tgt_shape_zyx

    def from_canon_to_original_view(
        self,
        key_to: str,
        coords: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # coords: these are scaled, centered coordinates (as in our shape model) of LAX view in canonical SAX view (aligned SAX)
        # e.g. as returned by self.get_lax_4ch_coords_in_canon()
        # method returns VOXEL coordinates of original LAX view
        assert key_to in [
            KEY_4CH_VIEW,
            KEY_4CH_SEG_VIEW,
            KEY_2CH_VIEW,
            KEY_2CH_SEG_VIEW,
        ]
        if len(coords.shape) == 3:
            coords = coords.squeeze()
        if coords.shape[-1] == 3:
            # add homogenous coordinate
            coords = torch.cat(
                [coords, torch.ones((len(coords), 1), device=coords.device)], dim=-1
            )
        # make sure we have tensor of dim 3
        coords = coords[None]
        # This volume contains 2D view in 3D canonical space. Compared to original SAX volume, these have higher through-plane
        # resolution -> canon_scale, in order to preserve 2D HR in-plane
        src_shape_zyx = self.views[key_to]["np_img_aligned"].shape
        # _de_scale_aligned_voxel_coords
        # RETURNS VOXEL COORDINATES, REMEMBER THESE ARE VOXELS OF SAX WITH HIGH THROUGH-PLANE
        # which we created for lax in sax coordinates to retain high LAX resolution
        # therefore, we rescale with fixed_cimage.views['lax4ch']['canon_scale']
        coords = self.de_scale_aligned_voxel_coords(
            coords, scale_m=self.views[key_to]["canon_scale"]
        )
        # _de_align_voxel_coords: returns de-aligned SAX VOXEL coordinates (but HR through plane!)
        # NOTE: _de_align_voxel_coords expects coords to be voxels!
        coords = self._de_align_voxel_coords(coords, src_shape_zyx, z_dim=1)
        # to world coordinates
        trans_m = (
            self.views[key_to]["canon_scale"].T
            @ self.views[self.main_key]["rotate"].T
            @ self.views[self.main_key]["translate"].T
        )
        world_coords = coords @ trans_m
        # from World to LAX
        trans_m = (
            tr_inv(self.views[key_to]["translate"]).T
            @ tr_inv(self.views[key_to]["rotate"]).T
            @ tr_inv(self.views[key_to]["scale"]).T
        )
        return world_coords @ trans_m

    def _resample(
        self, torch_array, new_grid, src_shape_zyx, tgt_shape_zyx, mode="bilinear"
    ) -> torch.Tensor:
        if src_shape_zyx[0] == 1:
            shape_div = torch.tensor(
                src_shape_zyx[1:][::-1]
                + (
                    2,
                    2,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            shape_div = torch.tensor(
                src_shape_zyx[::-1] + (2,), dtype=torch.float32, device=self.device
            )
        # this I found always confusing, but:
        # new_grid are the coordinates in the source image for which we need the intensity values
        # and we fill in those int-values at the xyz location in the target image
        # furthermore: 1st -1 = because when we made the grid in the first place the coord range per dimension is
        # size_of_dim - 1. 2nd -1: because we scale between [-1, 1]
        t_coords_normed = (new_grid / ((shape_div[None, None, None] - 1) / 2)) - 1
        t_coords_normed = t_coords_normed.reshape(tuple(tgt_shape_zyx) + (4,))
        t_coords_normed = t_coords_normed[..., :3]
        return execute_resampling(torch_array, t_coords_normed, mode=mode)

    def create_sitk_image(
        self,
        np_array: np.ndarray,
        spacing_xyz=None,
        origin_xyz=None,
        direction=None,
        dtype=np.float32,
    ):
        sitk_img = sitk.GetImageFromArray(np_array.astype(dtype))
        sitk_img.SetOrigin(origin_xyz.astype(np.float64))
        sitk_img.SetSpacing(spacing_xyz.astype(np.float64))
        sitk_img.SetDirection(direction.astype(np.float64))
        return sitk_img

    @property
    def shape(self):
        return self.shape_zyx

    def get_sax_image(self, device=None, image_type="image"):
        assert image_type in [
            "image",
            "mask",
            "contour",
            "normal",
            "contour_as_mask",
            "normal_as_mask",
        ]
        if image_type == "mask":
            obj = self.views[self.main_key]["np_seg_aligned"]
        elif image_type == "image":
            obj = self.views[self.main_key]["np_img_aligned"]
        elif image_type == "contour_as_mask":
            obj = self.views[self.main_key]["np_contour_as_mask"]
        elif image_type == "normal_as_mask":
            obj = self.views[self.main_key]["np_normal_as_mask"]
        elif image_type == "contour":
            obj = self.views[self.main_key]["np_contour"]
        elif image_type == "normal":
            obj = self.views[self.main_key]["np_normal"]
        else:
            raise ValueError("*** Not a valid image type {}".format(image_type))
        if device is not None:
            if type(obj) is not np.ndarray:
                obj = obj.numpy()
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_4ch_image(self, device=None, mask=False):
        # returns original 2D 4CH aligned in 3D sax view [z, y, x] where z is higher than in SAX view
        # due to high in-plane resolution in LAX view. Array contains intensity values of voxels
        if mask:
            obj = self.views[KEY_4CH_SEG_VIEW]["np_img"]
        else:
            obj = self.views[KEY_4CH_VIEW]["np_img"]
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_4ch_image_in_sax(self, device=None, mask=False):
        # returns original 2D 4CH aligned in 3D sax view [z, y, x] where z is higher than in SAX view
        # due to high in-plane resolution in LAX view. Array contains intensity values of voxels
        if mask:
            obj = self.views[KEY_4CH_SEG_VIEW]["np_img_aligned"]
        else:
            obj = self.views[KEY_4CH_VIEW]["np_img_aligned"]
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_4ch_slice_in_sax(self, device=None, mask=False):
        # returns original 2D 4CH aligned in 3D sax view, BUT compared to method get_4ch_image() this one only
        # returns the intensity values of the voxels from the original LAX view. [#coords_in_lax]
        if mask:
            obj = self.views[KEY_4CH_SEG_VIEW]["np_img_aligned_filtered"]
        else:
            obj = self.views[KEY_4CH_VIEW]["np_img_aligned_filtered"]
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_2ch_image(self, device=None, mask=False):
        # returns original 2D 2CH aligned in 3D sax view [z, y, x] where z is higher than in SAX view
        # due to high in-plane resolution in LAX view. Array contains intensity values of voxels
        if mask:
            obj = self.views[KEY_2CH_SEG_VIEW]["np_img"]
        else:
            obj = self.views[KEY_2CH_VIEW]["np_img"]
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_2ch_slice_in_sax(self, device=None, mask=False):
        # returns original 2D 2CH aligned in 3D sax view, BUT compared to method get_4ch_image() this one only
        # returns the intensity values of the voxels from the original LAX view. [#coords_in_lax]
        if mask:
            obj = self.views[KEY_2CH_SEG_VIEW]["np_img_aligned_filtered"]
        else:
            obj = self.views[KEY_2CH_VIEW]["np_img_aligned_filtered"]
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_sax_coords(self, device=None) -> torch.FloatTensor:
        obj = self.views[self.main_key]["coords_scaled_centered"]
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_lax_4ch_coords(self, device=None) -> torch.FloatTensor:
        # NOTE: THIS METHOD SHOULD NOT BE USED ANYMORE. USE get_lax_4ch_coords_in_canon() INSTEAD
        # COULD BE USED TO VISUALIZE LONG-AXIS VIEW AS 3D VOLUME ALIGNED WITH SAX VIEW.
        # LATER, DEVELOPED get_lax_4ch_coords_in_canon() THAT SHOULD BE USED TO OBTAIN LONG-AXIS COORDINATES IN (ALIGNED) SAX VOLUME
        obj = self.views[KEY_4CH_VIEW]["filtered_coords_centered"]
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_lax_4ch_coords_in_canon(self, device=None) -> torch.FloatTensor:
        # so what is the difference between get_lax_4ch_coords() and get_lax_4ch_coords_in_canon():
        # get_lax_4ch_coords: was the first attempt to obtain long-axis view coordinates in SAX view
        obj = self.views["lax4ch"][
            "coords_in_main_scaled_centered"
        ].clone()  # [..., :3].squeeze()
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_lax_2ch_coords(self, device=None) -> torch.FloatTensor:
        obj = self.views[KEY_2CH_VIEW]["filtered_coords_centered"]
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_lax_2ch_coords_in_canon(self, device=None) -> torch.FloatTensor:
        # so what is the difference between get_lax_4ch_coords() and get_lax_4ch_coords_in_canon():
        # get_lax_4ch_coords: was the first attempt to obtain long-axis view coordinates in SAX view
        obj = self.views["lax2ch"][
            "coords_in_main_scaled_centered"
        ].clone()  # [..., :3].squeeze()
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_mesh(self, key):
        return self.meshes[key]["mesh"]

    def get_mesh_coords(self, key, device=None, filter_base=True):
        points = np.array(self.meshes[key]["mesh"].points).astype(np.float32)
        # Filter at the base of the heart because sometimes we do not want the whole LV outflow tract
        if filter_base:
            coords_max_z = np.max(points, axis=0)[-1]
            coords_max_z = coords_max_z - 2.5 * float(
                self.views[self.main_key]["spacing_xyz"][-1]
            )
            points = points[points[:, 2] <= coords_max_z]
        if device is None:
            return points
        else:
            return torch.from_numpy(points).float().to(device)

    def get_mesh_latent(self, key, device=None):
        if device is None:
            return self.meshes[key]["latent"]
        else:
            return torch.from_numpy(self.meshes[key]["latent"]).float().to(device)

    def get_spacing(self, key, device=None):
        obj = self.views[key]["spacing_xyz"]
        if isinstance(obj, tuple):
            obj = torch.as_tensor(obj).float()
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_4ch_spacing(self, device=None):
        obj = self.views[KEY_4CH_VIEW]["canon_spacing_xyz"]
        if isinstance(obj, tuple):
            obj = torch.as_tensor(obj).float()
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_sax_meta_data(self, info_type):
        if info_type == "direction":
            return self.views[KEY_SAX_VIEW]["sitk_img"].GetDirection()
        elif info_type == "origin":
            return self.views[KEY_SAX_VIEW]["sitk_img"].GetOrigin()
        elif info_type == "spacing":
            return self.views[KEY_SAX_VIEW]["sitk_img"].GetSpacing()
        else:
            raise ValueError()

    @staticmethod
    def get_orientation(y, x):
        angle = np.arctan2(y, x)
        rotmat = rotation_matrix(rot_z=angle)
        return rotmat

    def _check_apex_base_orientation(self, np_seg):
        # canonical space: orientation should be lower slice number = APEX
        # higher slice number BASE
        self.z_flip = check_apex_base_orientation(np_seg, self.label.LVBP.value)
        # zmask = (np_seg == self.label.LVBP.value).any((1, 2))
        # num_seg_slices = np.count_nonzero(zmask)
        # segmask = (np_seg[zmask] == self.label.LVBP.value).astype(np.int32)
        # z_cmas = round(get_center(segmask)[0])
        # self.z_flip = z_cmas < num_seg_slices / 2
        print("INFO - CanonicalImage flip z-axis orientation? {}".format(self.z_flip))

    def align(
        self, torch_array, tgt_shape_zyx=None, src_shape_zyx=None, mode="bilinear"
    ):
        t_coords_aligned_xyz = identity_grid(
            tgt_shape_zyx, device=self.device, do_flip_sequence=self.xyz_sequence
        )
        # add homogeneous coordinate
        t_coords_aligned_xyz = torch.cat(
            [
                t_coords_aligned_xyz,
                torch.ones(t_coords_aligned_xyz.shape[:-1] + (1,), device=self.device),
            ],
            dim=-1,
        )
        # self.shape_zyx is zyx shape of MRI segmentation mask (original)
        trans_origin_xyz = torch.eye(4).to(self.device)
        trans_origin_xyz[:3, 3] = (
            torch.tensor(tgt_shape_zyx[::-1], device=self.device) / 2
        ) - 0.5
        trans_m = (
            tr_inv(trans_origin_xyz).T
            @ self.flip_z_m.T
            @ self.flip_y_m.T
            @ tr_inv(self.rot_rv_lv_m)
            @ trans_origin_xyz.T
        )
        t_coords_aligned_xyz = (
            t_coords_aligned_xyz.reshape(tgt_shape_zyx[0], -1, 4) @ trans_m
        )
        warped_arr = self._resample(
            torch_array, t_coords_aligned_xyz, src_shape_zyx, tgt_shape_zyx, mode=mode
        )
        return warped_arr.squeeze()

    def prepare_y_z_flip(self):
        self.flip_y_m = torch.eye(4).to(self.device)
        self.flip_y_m[0, 0] = 1
        self.flip_y_m[1, 1] = -1
        self.flip_z_m = torch.eye(4).to(self.device)
        if self.z_flip:
            self.flip_z_m[2, 2] = -1
