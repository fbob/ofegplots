import multiprocessing as mp

import cantera as ct
import numpy as np
import pandas as pd
import tensorflow as tf
from molmass import Formula
from collections import namedtuple


def read_of(xy, gas, plane="xy"):
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
    df["p"] = fields["p"]
    # df["pd"] = fields["pd"]

    if "f_Bilger" in xy.scalar_names:
        df["f"] = fields["f_Bilger"]

    df["dt"] = 1e-6
    df["thermo:Df"] = fields["thermo:Df"]

    if "Dft" in xy.scalar_names:
        df["Dft"] = fields["Dft"]

    if plane == "xy":
        df["x"] = np.around(pts[:, 0], decimals=5)
        df["y"] = np.around(pts[:, 1], decimals=5)
    if plane == "yz":
        df["x"] = np.around(pts[:, 1], decimals=5)
        df["y"] = np.around(pts[:, 2], decimals=5)

    return df


class ct_chem:
    gas = "gas"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def ct_calc(cls, test):
        gas = cls.gas
        gas.transport_model = "UnityLewis"

        if test["Temp"] > 800:
            Y = [test[sp + "_Y"] for sp in gas.species_names]
            gas.Y = Y
            # gas.TP = test["Temp"], ct.one_atm
            gas.TP = test["Temp"], test["pd"]

            w_dot = gas.net_production_rates
            Hs_dot = np.dot(gas.partial_molar_enthalpies, -gas.net_production_rates)
            T_dot = Hs_dot / (gas.density * gas.cp)
        else:
            w_dot = np.zeros(len(gas.species_names))
            Hs_dot = 0
            T_dot = 0

        df = np.hstack(
            [
                # gas.net_production_rates / gas.molecular_weights,
                w_dot,
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

    input_species = gas.species_names
    # input_features = input_species + ["Hs", "Temp", "dt"]
    # *input_species, _ = gas.species_names
    # input_species = gas.species_names
    input_features = input_species + ["Temp", "dt"]

    model = tf.keras.models.load_model(model_file)

    pred = model.predict(df[input_features], batch_size=1024 * 8)

    # df_dnn = pd.DataFrame(pred, columns=input_species + ["Hs", "Temp"])
    df_dnn = pd.DataFrame(pred, columns=input_species + ["Temp"])

    df_dnn[["x", "y"]] = df[["x", "y"]]
    return df_dnn


def euler_pred_grid(grid, gas, model_file=""):
    input_species = gas.species_names
    # *input_species, _ = gas.species_names
    input_features = input_species + ["Temp", "dt"]
    dt = 1e-8
    grid_out = grid.copy()

    Gca = namedtuple("Gca", ["arr", "n"])

    for gca in [
        Gca(grid_out.cell_arrays, grid.n_cells),
        Gca(grid_out.point_arrays, grid.n_points),
    ]:
        print(gca.n)
        gca.arr["rho"] = gca.arr["p"] * gca.arr["thermo:psi"]
        gca.arr["Temp"] = gca.arr["T"]
        gca.arr["dt"] = np.ones([gca.n]) * dt

        input_arr = np.empty((gca.n, 0), float)
        for sp in input_species:
            gca.arr[sp + "_c"] = gca.arr[sp] * gca.arr["rho"] / Formula(sp).mass
            input_arr = np.hstack([input_arr, gca.arr[sp + "_c"].reshape(-1, 1)])

        input_arr = np.hstack(
            [input_arr, gca.arr["Temp"].reshape(-1, 1), gca.arr["dt"].reshape(-1, 1)]
        )

        model = tf.keras.models.load_model(model_file)

        pred = model.predict(input_arr, batch_size=1024 * 8)

        df_dnn = pd.DataFrame(pred, columns=input_species + ["Temp"])

        for sp in input_species:
            gca.arr["RR." + sp] = np.array(df_dnn[sp].values) * Formula(sp).mass
        # for idx, sp in enumerate(input_species):
        #     print(sp)
        #     a = np.array(pred[:, idx])
        #     print(a)
        #     gca.arr["RR." + sp] = a

    # df_dnn[["x", "y"]] = df[["x", "y"]]
    return grid_out
