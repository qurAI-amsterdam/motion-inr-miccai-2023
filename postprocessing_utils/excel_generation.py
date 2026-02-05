EXCEL_COLUMNS = [
    "Patid",
    "tp_fixed",
    "tp_moving",
    "glsrr_raw",
    "glscc_raw",
    "glsll_raw",
    "glsrr_lv_apical",
    "glsrr_lv_mid",
    "glsrr_lv_basal",
    "glscc_lv_apical",
    "glscc_lv_mid",
    "glscc_lv_basal",
    "glsll_lv_apical",
    "glsll_lv_mid",
    "glsll_lv_basal",
    "glsll_raw_4ch",
    "glsrr_raw_rv",
    "glscc_raw_rv",
    "glsrr_raw_sep",
    "glscc_raw_sep",
    "glsrr_rv_apical",
    "glsrr_rv_mid",
    "glsrr_rv_basal",
    "glscc_rv_apical",
    "glscc_rv_mid",
    "glscc_rv_basal",
    "glsrr_sep_apical",
    "glsrr_sep_mid",
    "glsrr_sep_basal",
    "glscc_sep_apical",
    "glscc_sep_mid",
    "glscc_sep_basal",
    "dice_lvpb",
    "dice_lv",
    "dice_rv",
    "J-1 (1)",
    "J-1 (2)",
    "J-1 (3)",
    "dice_4ch_lvpb",
    "dice_4ch_lv",
    "dice_4ch_rv",
    "epoch_stopped",
]


def generate_excel_row(
    patid,
    tp_fixed,
    tp_moving,
    result_dict,
    strain_dict,
    strain_dict_rv,
    strain_dict_sep,
    strain_dict_lv_per_part=None,
    multi_view=True,
):

    if multi_view:
        excel_row = (
            [
                patid,
                tp_fixed,
                tp_moving,
                strain_dict["glsrr"],
                strain_dict["glscc"],
                strain_dict["glsll"],
            ]
            + [
                strain_dict_lv_per_part["rr_lv_apical"],
                strain_dict_lv_per_part["rr_lv_mid"],
                strain_dict_lv_per_part["rr_lv_basal"],
            ]
            + [
                strain_dict_lv_per_part["cc_lv_apical"],
                strain_dict_lv_per_part["cc_lv_mid"],
                strain_dict_lv_per_part["cc_lv_basal"],
            ]
            + [
                strain_dict_lv_per_part["ll_lv_apical"],
                strain_dict_lv_per_part["ll_lv_mid"],
                strain_dict_lv_per_part["ll_lv_basal"],
            ]
            + [strain_dict["glsll_4ch"]]
            + [strain_dict_rv["rr"], strain_dict_rv["cc"]]
            + [strain_dict_sep["rr"], strain_dict_sep["cc"]]
            + [
                strain_dict_rv["rr_apical"],
                strain_dict_rv["rr_mid"],
                strain_dict_rv["rr_basal"],
            ]
            + [
                strain_dict_rv["cc_apical"],
                strain_dict_rv["cc_mid"],
                strain_dict_rv["cc_basal"],
            ]
            + [
                strain_dict_sep["rr_apical"],
                strain_dict_sep["rr_mid"],
                strain_dict_sep["rr_basal"],
            ]
            + [
                strain_dict_sep["cc_apical"],
                strain_dict_sep["cc_mid"],
                strain_dict_sep["cc_basal"],
            ]
            + [
                result_dict["dice"][0],
                result_dict["dice"][1],
                result_dict["dice"][2],
                result_dict["j_minus_1"][0],
                result_dict["j_minus_1"][1],
                result_dict["j_minus_1"][2],
                result_dict["dice_4ch"][0],
                result_dict["dice_4ch"][1],
                result_dict["dice_4ch"][2],
            ]
        )
    else:
        excel_row = (
            [
                patid,
                tp_fixed,
                tp_moving,
                strain_dict["glsrr"],
                strain_dict["glscc"],
                strain_dict["glsll"],
            ]
            + [
                strain_dict_lv_per_part["rr_lv_apical"],
                strain_dict_lv_per_part["rr_lv_mid"],
                strain_dict_lv_per_part["rr_lv_basal"],
            ]
            + [
                strain_dict_lv_per_part["cc_lv_apical"],
                strain_dict_lv_per_part["cc_lv_mid"],
                strain_dict_lv_per_part["cc_lv_basal"],
            ]
            + [
                strain_dict_lv_per_part["ll_lv_apical"],
                strain_dict_lv_per_part["ll_lv_mid"],
                strain_dict_lv_per_part["ll_lv_basal"],
            ]
            + [0]
            + [0, 0]
            + [0, 0]
            + [0, 0, 0]
            + [0, 0, 0]
            + [0, 0, 0]
            + [0, 0, 0]
            + [
                result_dict["dice"][0],
                result_dict["dice"][1],
                result_dict["dice"][2],
                result_dict["j_minus_1"][0],
                result_dict["j_minus_1"][1],
                result_dict["j_minus_1"][2],
                0,
                0,
                0,
            ]
        )

    return excel_row
