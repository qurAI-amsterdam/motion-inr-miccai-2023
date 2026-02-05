import numpy as np
import os
import torch
import torch.nn.functional as F
from scipy import ndimage
import SimpleITK as sitk
import h5py


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 4)
    stds = np.round(stds, 4)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def compute_landmarks(network, landmarks_pre, image_size):
    # Jorg: because coords are scaled between [-1, 1] take 0.5 of image shape and subtract 1 below
    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

    output = network(coordinate_tensor.cuda())

    delta = output.cpu().detach().numpy() * (scale_of_axes)

    return landmarks_pre + delta, delta


def compute_unified_landmarks(network, dirsel, landmarks_pre, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0
    dirsel_tensor = dirsel * torch.ones((landmarks_pre.shape[0], 1))
    coordinate_tensor = torch.cat([coordinate_tensor, dirsel_tensor], axis=1)

    output = network(coordinate_tensor.cuda())

    delta = output.cpu().detach().numpy() * (scale_of_axes)

    return landmarks_pre + delta, delta


def landmarks_to_coordmarks(landmarks, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]
    coordmarks = torch.FloatTensor(landmarks / (scale_of_axes)) - 1.0
    return coordmarks


def coordmarks_to_landmarks(coordmarks, image_size):
    scale_of_axes = torch.FloatTensor([(0.5 * s) for s in image_size])
    if "cuda" in str(coordmarks.device):
        scale_of_axes = scale_of_axes.cuda()
    landmarks = (coordmarks + 1.0) * (scale_of_axes)
    return landmarks


def scale_coorddiffs_to_mm(coorddiffs, image_size):
    scale_of_axes = torch.FloatTensor([(0.5 * s) for s in image_size])
    if "cuda" in str(coorddiffs.device):
        scale_of_axes = scale_of_axes.cuda()
    coorddiffs_mm = (coorddiffs) * (scale_of_axes)
    return coorddiffs_mm


def load_image_MOTAC(
    im_index,
    t0=2,
    t1=3,
    folder="/home/louis/Data/motility_MOT3D_multi_tslice_ordered_media_seg_and_cl_corrections/",
    segs_as_dict=False,
    isopad=False,
    fill_mask=False,
):
    im_numbers = [
        "04",
        "05",
        "07",
        "10",
        "13",
        "14",
        "15",
        "16",
        "18",
        "19",
        "22",
        "23",
        "25",
        "26",
    ]
    imnum = im_numbers[im_index]

    setfile = h5py.File(folder + f"MOT3D_multi_tslice_MII{imnum}a.hdf5", "r")
    data = torch.Tensor(np.array(setfile["MOT3DBH"])).cuda()
    interp_shape = [x for x in data.shape]
    interp_shape[-1] = interp_shape[-1] * 2 - 1
    interp_data = torch.zeros(interp_shape).cuda()
    for i in range(14):
        interp_data[:, :, :, i * 2] = data[:, :, :, i]
    for i in range(13):
        interp_data[:, :, :, i * 2 + 1] = (data[:, :, :, i] + data[:, :, :, i + 1]) / 2

    if not fill_mask:
        mask = 1 * (interp_data > 50)
        mask[:, :, :, :4] = 0
        mask[:, :, :, -4:] = 0
    else:
        mask = 1 * (interp_data > setfile["MOT3DBH"].attrs["median"] / 4)
        for t in [t0, t1]:
            dilslice = mask[t, :, :, :].cpu()
            dilslice = ndimage.binary_dilation(
                dilslice, iterations=2, structure=np.ones([5, 5, 5])
            )

            dilslice[:, :4, 4:-4] = 1
            dilslice[:, -4:, 4:-4] = 1
            dilslice = ndimage.binary_fill_holes(dilslice)
            dilslice[:, :, :] = ndimage.binary_erosion(
                dilslice, iterations=2, structure=np.ones([5, 5, 5])
            )
            dilslice[:, :, :4] = 0
            dilslice[:, :, -4:] = 0

            mask[t, :, :, :] = torch.Tensor(dilslice).cuda()

    interp_data = interp_data / setfile["MOT3DBH"].attrs["median"]
    interp_data = torch.clamp(interp_data, 0, 4)
    vox_size = setfile["MOT3DBH"].attrs["spacing"]
    vox_size[-1] *= 0.5

    image_t0 = interp_data[t0, :]
    image_t1 = interp_data[t1, :]

    if isopad:  # pad sides so coords are isotropic (for regularizers)
        mspacing = vox_size
        padn = (
            mspacing[-2] / mspacing[-1] * (interp_shape[-2] - interp_shape[-1])
        ) // 2
        padding = (int(padn), int(padn))
        refpadding = (interp_shape[-1] - 1, interp_shape[-1] - 1)
        otherpadding = (
            int(padn) - interp_shape[-1] + 1,
            int(padn) - interp_shape[-1] + 1,
        )
        image_t0 = F.pad(image_t0, refpadding, "reflect")
        image_t0 = F.pad(image_t0, otherpadding, "constant", 0)
        image_t1 = F.pad(image_t1, refpadding, "reflect")
        image_t1 = F.pad(image_t1, otherpadding, "constant", 0)
        mask = F.pad(mask, padding, "constant", 0)

    sint_segs_dense_vox = {}
    all_sint_segs = []
    for key in setfile["sint_segs_dense_vox"].keys():
        voxseg = np.array(setfile["sint_segs_dense_vox"][key])
        voxseg[:, 2] *= 2
        all_sint_segs.append(voxseg)
        sint_segs_dense_vox[key] = voxseg

    if segs_as_dict:
        retsegs = sint_segs_dense_vox
    else:
        retsegs = np.concatenate(all_sint_segs, axis=0)

    # just load the same ones for now
    landmarks_insp = retsegs
    landmarks_exp = retsegs

    return (image_t0, image_t1, landmarks_insp, landmarks_exp, mask, vox_size)


def load_image_DIRLab(variation=1, folder=r"D:\Data\DIRLAB\Case"):
    # Size of data, per image pair
    image_sizes = [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    # Scale of data, per image pair
    voxel_sizes = [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    shape = image_sizes[variation]

    folder = folder + str(variation) + r"Pack" + os.path.sep

    # Images
    dtype = np.dtype(np.int16)

    # with open(folder + r"Images/case" + str(variation) + "_T00.nii.gz", "rb") as f:
    #    data = np.fromfile(f, dtype)
    # image_insp = data.reshape(shape)
    fname_insp = folder + r"Images/case" + str(variation) + "_T00.nii.gz"
    image_insp = sitk.GetArrayFromImage(sitk.ReadImage(fname_insp))

    # with open(folder + r"Images/case" + str(variation) + "_T50.nii.gz", "rb") as f:
    #    data = np.fromfile(f, dtype)
    # image_exp = data.reshape(shape)
    fname_exp = folder + r"Images/case" + str(variation) + "_T50.nii.gz"
    image_exp = sitk.GetArrayFromImage(sitk.ReadImage(fname_exp))

    imgsitk_in = sitk.ReadImage(folder + r"Masks/case" + str(variation) + "_T00.nii.gz")

    mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)

    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Landmarks
    with open(
        folder + r"extremePhases/Case" + str(variation) + "_300_T00_xyz.txt"
    ) as f:
        landmarks_insp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    with open(
        folder + r"extremePhases/Case" + str(variation) + "_300_T50_xyz.txt"
    ) as f:
        landmarks_exp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    landmarks_insp[:, [0, 2]] = landmarks_insp[:, [2, 0]]
    landmarks_exp[:, [0, 2]] = landmarks_exp[:, [2, 0]]

    return (
        image_insp,
        image_exp,
        landmarks_insp,
        landmarks_exp,
        mask,
        voxel_sizes[variation],
    )


def de_normalize(array_shape, x_indices, y_indices, z_indices, min_offset):
    x_indices = (x_indices + min_offset) * (array_shape[0] - 1) * 0.5
    y_indices = (y_indices + min_offset) * (array_shape[1] - 1) * 0.5
    z_indices = (z_indices + min_offset) * (array_shape[2] - 1) * 0.5
    return x_indices, y_indices, z_indices


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )

    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_slice(
    dims=(28, 28),
    dimension=0,
    slice_pos=0,
    device="cuda",
    dtype=torch.float32,
    normalize=True,
):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims.insert(dimension, 1)

    if normalize:
        coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(2)]
    else:
        coordinate_tensor = (torch.arange(s, dtype=dtype, device=device) for s in dims)
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.to(device=device)

    return coordinate_tensor


def make_coordinate_tensor(
    dims=(28, 28, 28),
    device="cuda",
    dtype=torch.float32,
    normalize=True,
    do_flip_sequence=False,
):
    """Make a coordinate tensor."""

    if normalize:
        coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    else:
        coordinate_tensor = (torch.arange(s, dtype=dtype, device=device) for s in dims)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    if do_flip_sequence:
        coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    else:
        coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.to(device=device)

    return coordinate_tensor


def make_masked_coordinate_tensor(
    mask, dims=(28, 28, 28), device="cuda", dtype=torch.float32, normalize=True
):
    """Make a coordinate tensor."""

    if normalize:
        coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    else:
        coordinate_tensor = (torch.arange(s, dtype=dtype, device=device) for s in dims)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.to(device=device)

    return coordinate_tensor
