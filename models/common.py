from CMRI.ARVC.common import get_patient_data_arvc, get_arvc_ed_es
from models.models_v4 import ImplicitRegistrator
from kwatsch.common import loadExperSettings


def get_model(
    patid,
    img_fixed,
    img_moving,
    cardiac_views: str,
    exper_dir="/home/jorg/expers/cmri_motion/reg/ARVC/bogus/",
):

    str_patid = "{:03d}".format(patid) if isinstance(patid, int) else patid
    pat_info = get_patient_data_arvc(patid=patid)
    tp_fixed, tp_moving = get_arvc_ed_es(pat_info, "LV", patid=None, is_new=True)
    pat_out_dir = (
        exper_dir
        / str_patid
        / "{}_{}_tp{:02d}_to_tp{:02d}".format(
            patid, "_".join(cardiac_views), tp_fixed, tp_moving
        )
    )
    fname = pat_out_dir / "kwargs.yml"
    kwargs = loadExperSettings(fname)
    kwargs["network"] = (
        f"F_{str_patid}_alpha{str(kwargs['alpha_jacobian']).replace('.', '_')}.pth"
    )
    del kwargs["cardiac_views"]
    return ImplicitRegistrator(
        img_fixed, img_moving, cardiac_views=cardiac_views, **kwargs
    )
