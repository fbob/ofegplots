import multiprocessing as mp

import cantera as ct
import numpy as np
import pandas as pd
import tensorflow as tf
from molmass import Formula


def read_of(xy, gas):
    pts = xy.points
    fields = xy.point_arrays
    fields["rho"] = fields["p"] * fields["thermo:psi"]
    df = pd.DataFrame()

    # for sp in species:
    for sp in gas.species_names:
        df[sp + "_Y"] = fields[sp]
        df[sp] = fields[sp] * fields["rho"] / Formula(sp).mass
        # df[sp + "_RR"] = fields["RR." + sp] / Formula(sp).mass
        df[sp + "_RR"] = fields["RR." + sp] / Formula(sp).mass

    df["Hs"] = fields["hs"]
    df["Temp"] = fields["T"]
    df["rho"] = fields["rho"]
    df["pd"] = fields["pd"]
    df["p"] = fields["p"]

    df["dt"] = 1e-6

    df["thermo:Df"] = fields["thermo:Df"]

    df["x"] = pts[:, 0]
    df["y"] = pts[:, 1]
    return df


class ct_chem:
    gas = "gas"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def ct_calc(cls, test):
        gas = cls.gas
        gas.transport_model = "UnityLewis"

        Y = [test[sp + "_Y"] for sp in gas.species_names]
        gas.Y = Y
        # gas.TP = test["Temp"], ct.one_atm
        gas.TP = test["Temp"], test["pd"]

        Hs_dot = np.dot(gas.partial_molar_enthalpies, -gas.net_production_rates)
        T_dot = Hs_dot / (gas.density * gas.cp)

        df = np.hstack(
            [
                # gas.net_production_rates / gas.molecular_weights,
                gas.net_production_rates,
                Hs_dot,
                T_dot,
                # test.Temp,
                gas.mix_diff_coeffs[0],
                gas.density,
                test["x"],
                test["y"],
            ]
        )

        return df

    @classmethod
    def column_names(cls):
        column_names = [
            *cls.gas.species_names,
            "Hs",
            "Temp",
            "thermo:Df",
            "rho",
            "x",
            "y",
        ]
        return column_names

    @classmethod
    def wdot(cls, df_plane):
        with mp.Pool() as pool:
            rows = [row for _, row in df_plane.iterrows()]
            raw = pool.map(cls.ct_calc, rows)
        df_ct = pd.DataFrame(np.vstack(raw), columns=cls.column_names())
        df_ct = df_ct.sort_values(["y", "x"], axis=0, ascending=[True, False])

        return df_ct


def euler_pred(df, gas, model_file=""):

    # input_species = gas.species_names
    # input_features = input_species + ["Hs", "Temp", "dt"]
    *input_species, _ = gas.species_names
    input_features = input_species + ["Temp", "dt"]

    model = tf.keras.models.load_model(model_file)

    pred = model.predict(df[input_features], batch_size=1024 * 8)

    # df_dnn = pd.DataFrame(pred, columns=input_species + ["Hs", "Temp"])
    df_dnn = pd.DataFrame(pred, columns=input_species + ["Temp"])

    df_dnn[["x", "y"]] = df[["x", "y"]]
    return df_dnn
