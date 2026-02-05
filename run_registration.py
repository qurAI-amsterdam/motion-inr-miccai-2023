import os

# set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
from pathlib import Path
import torch
import subprocess
from kwatsch.common import (
    KEY_SAX_VIEW,
    KEY_4CH_VIEW,
)
from models import models_v4
from utils.registration_utils import (
    post_process_completed,
    get_canonical_image_aligned,
    get_images_with_segmentations,
    save_script_to_experiments_folder,
    get_experiments_folder,
    get_es_and_ed_timepoints,
)

import sys
import multiprocessing
from functools import partial
import configparser
import logging
import ast

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("registration.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


def load_all_kwargs(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    out = {}
    for section in cfg.sections():
        for key, raw in cfg.items(section):
            out[key] = parse_value(raw)
    return out, cfg


def parse_value(val):
    """
    Try to coerce "True"/"1"/"1.0"/"[...]" into Python types;
    fall back to string if it won't literal_eval.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def load_kwargs_from_cfg(cfg_path, *sections):
    """
    Reads `cfg_path`, and for each section in `sections`,
    parses all key=values into a single dict.
    """
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    out = {}
    for section in sections:
        # configparser lower-cases keys by default; if you need case preserved,
        # pass `optionxform=str` to ConfigParser().
        for key, raw in cfg.items(section):
            out[key] = parse_value(raw)
    return out


def check_cuda_enabled():
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        if "NVIDIA-SMI" in output and "CUDA" in output:
            if torch.cuda.is_available():
                logging.info("CUDA is available.")
                return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")

    logging.info("CUDA is not available or nvidia-smi is not found.")
    return False


def run_registration(
    ImpReg,
    data_dict,
    tp_fixed,
    tp_moving,
    patid,
    path_to_pat_experiments,
    save_net,
    multi_view,
    kwargs,
):

    ImpReg.exper_dir = Path(path_to_pat_experiments)
    ImpReg.model_dir = ImpReg.exper_dir / "models"
    ImpReg.save_folder = ImpReg.exper_dir

    spacing = data_dict["spacing"]
    img_fixed = data_dict["fixed_img"]
    img_moving = data_dict["moving_img"]

    registration_start = time.time()
    ImpReg.fit_new(fixed_image=img_fixed, moving_image=img_moving)
    print(f"Registration time: {time.time() - registration_start}")

    post_process_start = time.time()
    post_process_completed(
        ImpReg,
        img_fixed,
        img_moving,
        tp_fixed,
        tp_moving,
        spacing,
        patid,
        save_net=save_net,
        multi_view=multi_view,
        kwargs=kwargs,
    )
    print(f"Post-processing time: {time.time() - post_process_start}")


def process_patients(
    patid,
    path_to_experiments,
    path_to_images_sax,
    path_to_segmentations_sax,
    cardiac_views,
    kwargs,
):
    start = time.time()
    print(f"Running registration for patient {patid}")

    patid_basename = patid.split(".")[0]
    center = patid_basename.split("_")[0]

    path_to_pat_experiments = Path(os.path.join(path_to_experiments, patid_basename))

    if not os.path.exists(path_to_pat_experiments):
        ImpReg = models_v4.ImplicitRegistrator(
            img_fixed=None, img_moving=None, cardiac_views=cardiac_views, **kwargs
        )

        try:
            os.makedirs(path_to_pat_experiments, exist_ok=True)
            path_to_pat_sax = os.path.join(path_to_images_sax, patid)
            path_to_segmentations_sax = os.path.join(path_to_segmentations_sax, patid)
            img4d_sax, seg4d_sax, img4d_4ch, seg4d_4ch = get_images_with_segmentations(
                path_to_pat_sax,
                path_to_segmentations_sax,
                load_lax_view=kwargs["multi_view"],
                **kwargs,
            )

            if kwargs["all_timepoints"] == False:
                tp_moving, tp_fixed = get_es_and_ed_timepoints(
                    patid_basename
                )  # returns es, ed
                data_dict = get_canonical_image_aligned(
                    img4d_sax,
                    seg4d_sax,
                    lax_img=img4d_4ch,
                    lax_seg=seg4d_4ch,
                    tp_fixed=tp_fixed,
                    tp_moving=tp_moving,
                    crop_ROI=kwargs["crop"],
                )
                print(f"Loading time: {time.time() - start}")
                run_registration(
                    ImpReg,
                    data_dict,
                    tp_fixed,
                    tp_moving,
                    patid,
                    path_to_pat_experiments,
                    kwargs["save_net"],
                    kwargs["multi_view"],
                    kwargs,
                )

            else:
                end = img4d_sax.GetSize()[-1] - 2

                for i in range(0, end):
                    tp_moving = i
                    tp_fixed = end
                    print(
                        f"Fixed timepoint: {tp_fixed}, Moving timepoint: {tp_moving}, original shape was {img4d_sax.GetSize()}"
                    )
                    path_to_pat_experiments_tp = os.path.join(
                        path_to_pat_experiments, f"tp_{tp_fixed}_{tp_moving}"
                    )
                    os.makedirs(path_to_pat_experiments_tp, exist_ok=True)
                    if i == 0:
                        print(f"Creating folder {path_to_pat_experiments_tp}")

                    data_dict = get_canonical_image_aligned(
                        img4d_sax,
                        seg4d_sax,
                        lax_img=img4d_4ch,
                        lax_seg=seg4d_4ch,
                        tp_fixed=tp_fixed,
                        tp_moving=tp_moving,
                        crop_ROI=kwargs["crop"],
                    )
                    try:
                        run_registration(
                            ImpReg,
                            data_dict,
                            tp_fixed,
                            tp_moving,
                            patid,
                            path_to_pat_experiments_tp,
                            kwargs["save_net"],
                            kwargs["multi_view"],
                            kwargs,
                        )
                    except Exception as e:
                        print(f"Error with patient {patid}: {e}")
                        with open(
                            os.path.join(path_to_pat_experiments, "error_test.txt"), "w"
                        ) as f:
                            f.write(
                                f"Error {e} with patient {patid} with size {img4d_sax.GetSize()}"
                            )

                        break

        except Exception as e:
            print(f"Error with patient {patid}: {e}")
            with open(os.path.join(path_to_experiments, "error.txt"), "w") as f:
                f.write(f"Error with patient {patid}")
                f.write(f"Error: {e}")


if __name__ == "__main__":
    check_cuda_enabled()

    cfg_name = "registration_config.cfg"
    kwargs, config = load_all_kwargs(cfg_name)

    print(f"Loaded kwargs: {kwargs}")

    cardiac_views = (
        [KEY_SAX_VIEW, KEY_4CH_VIEW] if kwargs["multi_view"] else [KEY_SAX_VIEW]
    )
    kwargs["cardiac_views_parameter"] = cardiac_views

    root = Path(kwargs["root"])
    data_folder = Path(kwargs["path_to_data"])
    seg_folder = Path(kwargs["path_to_segmentation"])
    path_to_images_sax = root / data_folder
    path_to_segmentations_sax = root / seg_folder

    experiment_folder_name = kwargs["experiment_folder_name"]
    path_to_experiments = get_experiments_folder(
        root, experiment_folder_name, addition=""
    )

    if not os.path.exists(path_to_experiments):
        os.makedirs(path_to_experiments, exist_ok=True)
        save_script_to_experiments_folder(
            path_to_experiments, os.path.basename(__file__)
        )
        config_file_path = os.path.join(path_to_experiments, cfg_name)
        with open(config_file_path, "w") as config_file:
            config.write(config_file)

    patid_list = [
        file for file in os.listdir(path_to_images_sax) if file.endswith(".nii.gz")
    ]

    if kwargs["debug"]:
        print(patid_list)
        patid_list = [patid_list[-7]]
        kwargs["multiprocessing"] = False

    logging.info(
        f"Running registration for {len(patid_list)} from {kwargs['dataset']} patients: {patid_list}"
    )

    if kwargs["multiprocessing"]:
        logging.info("Using multiprocessing for registration.")
        multiprocessing.set_start_method("spawn")

        with multiprocessing.Pool(processes=4) as pool:
            partial_func = partial(
                process_patients,
                path_to_experiments=path_to_experiments,
                path_to_images_sax=path_to_images_sax,
                path_to_segmentations_sax=path_to_segmentations_sax,
                cardiac_views=cardiac_views,
                kwargs=kwargs,
            )
            pool.map(partial_func, patid_list)

    else:
        for patid in patid_list:
            process_patients(
                patid,
                path_to_experiments,
                path_to_images_sax,
                path_to_segmentations_sax,
                cardiac_views,
                kwargs=kwargs,
            )

    print("finished")
