import torch
import torch.nn.functional as F
from CMRI.DINF.process import CMRILVShapeContainer


class ShapeLoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super(ShapeLoss, self).__init__()
        print("INFO - ShapeLoss - Loading Container...")
        self.shape_model = CMRILVShapeContainer()

    def forward(
        self, coords_fixed, latent_fixed, coords_moving, latent_moving, verbose=False
    ):
        if coords_fixed.dim() == 2:
            coords_fixed = coords_fixed[None]
        if coords_fixed.shape[-1] == 3:
            coords_fixed = torch.permute(coords_fixed, (0, 2, 1))
        if coords_moving.dim() == 2:
            coords_moving = coords_moving[None]
        if coords_moving.shape[-1] == 3:
            coords_moving = torch.permute(coords_moving, (0, 2, 1))
        if latent_fixed.dim() == 1:
            latent_fixed = latent_fixed[None]
        if latent_moving.dim() == 1:
            latent_moving = latent_moving[None]
        sdf_fixed = self.shape_model.predict_sdf(coords_fixed, latent_fixed)
        sdf_moving = self.shape_model.predict_sdf(coords_moving, latent_moving)
        if verbose:
            pass
        return F.l1_loss(sdf_moving, torch.abs(sdf_fixed))
