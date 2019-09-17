import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_plane = pd.DataFrame()


def plot_compare(
    sp="CH4", d=7.2e-3, df_u=df_plane, df_b=df_plane, case="up:xx|bottom:yy"
):
    fig, axes = plt.subplots(2, 1)
    sp_range = df_u[sp].max() - df_u[sp].min()
    cb = axes[0].contourf(
        df_u["x"].values.reshape(len(set(df_u["y"])), len(set(df_u["x"]))) / d,
        df_u["y"].values.reshape(len(set(df_u["y"])), len(set(df_u["x"]))) / d,
        df_u[sp].values.reshape(len(set(df_u["y"])), len(set(df_u["x"]))),
        levels=np.linspace(
            df_u[sp].min() - 0.1 * sp_range, df_u[sp].min() + 1.1 * sp_range, 100
        ),
    )
    axes[0].set_aspect("equal")
    axes[1].set_aspect("equal")
    axes[0].set_title(f"{sp} ({case})")
    axes[1].contourf(
        df_b["x"].values.reshape(len(set(df_b["y"])), len(set(df_b["x"]))) / d,
        df_b["y"].values.reshape(len(set(df_b["y"])), len(set(df_b["x"]))) / d,
        df_b[sp].values.reshape(len(set(df_b["y"])), len(set(df_b["x"]))),
        levels=cb.levels,
    )
    fig.colorbar(cb, ax=axes.flat)
    plt.show()


def plot_single(sp="CH4", df=df_plane, d=7.2e-3, case="defaut"):
    # case = [name for name in globals() if globals()[name] is df]
    sp_range = df[sp].max() - df[sp].min()
    plt.contourf(
        df["x"].values.reshape(len(set(df["y"])), len(set(df["x"]))) / d,
        df["y"].values.reshape(len(set(df["y"])), len(set(df["x"]))) / d,
        df[sp].values.reshape(len(set(df["y"])), len(set(df["x"]))),
        levels=np.linspace(
            df[sp].min() - 0.1 * sp_range, df[sp].min() + 1.1 * sp_range, 100
        ),
    )
    plt.colorbar()
    plt.axes().set_aspect("equal")
    plt.title(f"{sp}:{case}")
    plt.show()

