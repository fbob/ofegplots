import multiprocessing as mp

import numpy as np
import pandas as pd
import tensorflow as tf
from molmass import Formula
import cantera as ct


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

    df["dt"] = 1e-6

    df["thermo:Df"] = fields["thermo:Df"]

    df["x"] = pts[:, 0]
    df["y"] = pts[:, 1]
    return df


def calc_ct(df, gas):
    def ct_calc(test):
        # *test, gas = data_in
        reaction_mechanism = "/home/edison/OpenFOAM/flameD/data/smooke.cti"
        gas = ct.Solution(reaction_mechanism)
        Y = []
        # for sp in species:
        for sp in gas.species_names:
            Y_sp = test[sp + "_Y"]
            Y.append(Y_sp)
        Y = np.asarray(Y).reshape(1, -1)

        gas.Y = Y
        gas.TP = test.Temp, ct.one_atm
        gas.transport_model = "UnityLewis"

        Hs_dot = np.dot(gas.partial_molar_enthalpies, -gas.net_production_rates)
        T_dot = Hs_dot / (gas.density * gas.cp)

        df = pd.DataFrame()
        # for sp in species:
        for sp in gas.species_names:
            df[sp] = gas[sp].net_production_rates

        df["Hs"] = Hs_dot
        df["Temp"] = T_dot

        df["thermo:Df"] = gas.mix_diff_coeffs[0]

        df["x"] = test["x"]
        df["y"] = test["y"]
        return df

    with mp.Pool() as pool:
        rows = [row for _, row in df.iterrows()]
        raw = pool.map(ct_calc, rows)

    df_ct = pd.concat(raw)
    df_ct = df_ct.sort_values(["y", "x"], axis=0, ascending=[True, False])
    return df_ct


def euler_pred(df, gas):
    *input_species, _ = gas.species_names
    # input_species = gas.species_names
    # input_features = input_species + ["Hs", "Temp", "dt"]
    # model = tf.keras.models.load_model("eulerModel_wHs.h5")
    input_features = input_species + ["Temp", "dt"]
    # model = tf.keras.models.load_model("eulerModel_sk_04.h5")
    # model = tf.keras.models.load_model("eulerModel_sk_02.h5")
    model = tf.keras.models.load_model("eulerModel_sk_02new.h5")

    pred = model.predict(df[input_features], batch_size=1024 * 8)

    # df_dnn = pd.DataFrame(pred, columns=input_species + ["Hs", "Temp"])
    df_dnn = pd.DataFrame(pred, columns=input_species + ["Temp"])

    df_dnn["x"] = df["x"]
    df_dnn["y"] = df["y"]
    return df_dnn
