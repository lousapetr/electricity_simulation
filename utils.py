import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


def read_parameters(
    excel_path: str, nrows_table_1: int, sheet_name: str = "Sheet1"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read parameters from an Excel file.

    Expects a single sheet with two tables separated by a blank line.

    Args:
        excel_path (str): path to Excel file with parameters
        nrows_table_1 (int): number of rows of first table
        sheet_name (str, optional): name of sheet with parameters. Defaults to "Sheet1".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: two pandas dfs - each corresponds to one table in the Excel file
    """
    xl = pd.ExcelFile(excel_path)
    yearly_params: pd.DataFrame = xl.parse(sheet_name, nrows=nrows_table_1)  # type: ignore
    sources: pd.DataFrame = xl.parse(sheet_name, skiprows=(nrows_table_1 + 1))  # type: ignore
    return yearly_params, sources


def prepare_table_blocks(sources_definition: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a table with definitions of individual blocks

    Args:
        sources_definition (pd.DataFrame): definition per source type

    Returns:
        pd.DataFrame: definition per block
    """
    block_counts = []
    for _, row in sources_definition.iterrows():
        block_counts += [{"type": row["Zdroj"], "block_no": i} for i in range(row["Pocet bloku"])]
    sources_long = pd.DataFrame(block_counts)

    sources_definition["power_per_block"] = (
        sources_definition["Instalovany vykon [MWe]"] / sources_definition["Pocet bloku"]
    )
    df_sources = sources_long.merge(
        sources_definition[["Zdroj", "power_per_block", "Priorita", "Pravdepodobnost odstavky [%]"]].set_axis(
            ["type", "power", "priority", "shutdown_prob"], axis=1
        ),
        on="type",
    )

    order = list(sources_definition.sort_values(by="Priorita")["Zdroj"])
    df_sources["type"] = pd.Categorical(df_sources["type"], order)
    return df_sources


def _custom_cosine_wave(index: pd.Index, january: float, july: float, column_name: str = "wave") -> pd.Series:
    """
    Generate custom cosine wave that has predefined extremes in January and July.

    Args:
        january (float): maximum/minimum value, achieved in January
        july (float): minimum/maximum value, achieved in July
        column_name (str): name of value column in the final series

    Returns:
        pd.Series: series with calendar days as index, smooth wave in values
    """
    cosine_wave = np.cos(np.linspace(0, 2 * np.pi, len(index), endpoint=False))
    wave = (cosine_wave + 1) / 2 * (january - july) + july
    return pd.Series(wave, index=index, name=column_name)


def adjust_fve_power(fve_power: pd.Series, january_utilization: float, july_utilization: float) -> pd.Series:
    """
    Adjust FVE power to account for weather conditions.
    Approximates weather using a cosine wave with minimum in January and maximum in July.

    Args:
        fve_power (pd.Series): series containing unadjusted (nominal) FVE power
        january_utilization (float): lowest utilization in percent, corresponds to January 1st
        july_utilization (float): highest utilization in percent, corresponds to July 1st

    Returns:
        pd.Series: FVE power adjusted to weather
    """
    wave = _custom_cosine_wave(fve_power.index, january_utilization / 100.0, july_utilization / 100.0)
    return fve_power * wave


def prepare_demand(days: pd.DataFrame, january_demand_monthly: float, july_demand_monthly: float) -> pd.DataFrame:
    """
    Fill the df with mean hourly demand in MW.
    The seasonal changes are approximated by a cosine wave with extremes in January and July.

    Args:
        days (pd.DataFrame): time dimension, daily granularity
        january_demand_monthly (float): monthly demand in January in MW, extreme value (usually maximum)
        july_demand_monthly (float): monthly demand in July in MW, extreme value (usually minimum)

    Returns:
        pd.DataFrame: mean hourly demand in MW with daily granularity
    """
    demand_hourly_january_MW = january_demand_monthly / 30 / 24
    demand_hourly_july_MW = july_demand_monthly / 30 / 24
    result = days.copy()
    result["demand"] = _custom_cosine_wave(
        index=days.index,
        january=demand_hourly_january_MW,
        july=demand_hourly_july_MW,
        column_name="demand",
    )
    return result


def apply_shutdowns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use column `shutdown_prob` that specifies a probability in percent that given block
    will not be operatable on a given day.

    Add column `available_power` that contains either 0 (when block is shut down), or value of `power`
    (when the block is operable).

    Args:
        df (pd.DataFrame): df with granularity (day x block) containing `shutdown_prob` in percent

    Returns:
        pd.DataFrame: df with additional column `available_power`
    """
    df["_random"] = np.random.default_rng().random(size=len(df))
    df["_available"] = df["_random"] > (df["shutdown_prob"] / 100.0)
    df["available_power"] = df["power"] * df["_available"]
    return df.drop(columns=["_random", "_available"])


def plot_power(df: pd.DataFrame, value_column: str) -> go.Figure:
    """
    Plots barplot of available power per source type and time.

    Args:
        df (pd.DataFrame): long df with sources and power with (time x block) granularity
        value_column (str): column name to plot on the y-axis
        palette (dict[str, str]): ordered dict specifying colors for each type of source and their respective ordering,
                                  first is lowest in plot
    """
    plot_data_supply = df.groupby(["date", "type"])[value_column].sum().reset_index()
    plot_data_demand = df.groupby(["date"])["demand"].max().reset_index()

    fig = px.histogram(plot_data_supply, x="date", y=value_column, color="type", nbins=len(plot_data_demand))
    fig.add_trace(
        go.Scatter(
            x=plot_data_demand["date"],
            y=plot_data_demand["demand"],
            name="Demand",
            mode="lines",
            line=dict(color="black", width=3),
        )
    )
    fig.update_layout(
        title_text="Final System",  # title of plot
        xaxis_title_text="Date",  # xaxis label
        yaxis_title_text="Total supply/demand [MW]",  # yaxis label
        bargap=0,  # gap between bars of adjacent location coordinates
    )
    return fig


def plot_utilization(df: pd.DataFrame) -> go.Figure:
    """
    Plot a barplot of utilization per source type
    """
    fig = px.bar(
        df.reset_index(),
        x="type",
        y=["utilization_avail", "utilization_final"],
        text_auto=".1f",
        barmode="group",
    )
    newnames = {"utilization_avail": "Shutdowns only", "utilization_final": "After demand fitting"}
    fig.for_each_trace(
        lambda t: t.update(
            name=newnames[t.name],
            legendgroup=newnames[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]),
        )
    )
    fig.update_layout(
        title_text="Source utilization",  # title of plot
        xaxis_title_text="Source type",  # xaxis label
        yaxis_title_text="Utilization [%]",  # yaxis label
        legend_title="",
    )
    return fig


def plot_total_energy(df: pd.DataFrame) -> go.Figure:
    """
    Plot a barplot of total energy generated per source type
    """
    fig = px.bar(df, x="final_energy_GWh", y="type", color="type", text="mix_percent")
    fig.update_traces(texttemplate="%{text:.1f} %", textposition="auto")
    fig.update_layout(
        title_text="Total energy by source type",  # title of plot
        yaxis_title_text="Source type",  # xaxis label
        xaxis_title_text="Total energy [GWh]",  # yaxis label
        legend_title="",
    )
    fig.update_layout(showlegend=False)
    fig.show()
    return fig


def run_monte_carlo(parameters_filename: str, n_reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read parameters from the excel file, run full simulation for `n_reps` times, report results
    of utilization and total power generated.

    Args:
        parameters_filename (str): excel file with input parameters

    Returns:
        pd.DataFrame: total energy and utilization per source type and replica number
    """
    # read params
    yearly_params, sources_definition = read_parameters(parameters_filename, 3)
    yearly_params = yearly_params.set_index("Rocni cyklus")

    # prepare generic table
    days = pd.DataFrame(pd.date_range(start="2022-01-01", end="2022-12-31", freq="D"), columns=["date"])
    blocks = prepare_table_blocks(sources_definition)
    df_system = days.merge(blocks, how="cross")

    # adjust fve
    fve_utilization = yearly_params.loc["Utilizace FVE [%]"]
    mask = df_system["type"] == "FVE"
    df_system.loc[mask, "power"] = adjust_fve_power(
        df_system.loc[mask, "power"],
        january_utilization=fve_utilization["leden"],
        july_utilization=fve_utilization["cervenec"],
    )

    # add demand
    demand_params_MWh = yearly_params.loc["Spotreba [GWh]"] * 1000
    demand = prepare_demand(days, demand_params_MWh["leden"], demand_params_MWh["cervenec"])
    df_system = df_system.merge(demand, on="date")

    list_results_missed = []
    list_results_power = []
    # run Monte Carlo simulation
    for i in tqdm(range(n_reps)):
        # add random shutdowns
        df_system = apply_shutdowns(df_system)

        # turn off unnecessary generators
        df_system.sort_values(by=["date", "priority", "available_power", "block_no"], inplace=True)
        df_system["total_power"] = df_system.groupby("date")["available_power"].cumsum()
        df_system["turn_off"] = (df_system["total_power"] - df_system["power"]) > df_system["demand"]
        df_system = df_system.assign(final_power=lambda x: (~x.turn_off) * x.available_power)

        # calculate Energy not served
        df_missed = df_system.groupby("date")[["total_power", "demand"]].max()
        df_missed = df_missed.assign(missed=lambda x: np.maximum(x.demand - x.total_power, 0))
        df_missed.reset_index(inplace=True)
        df_missed["replica"] = i
        df_missed = df_missed.loc[(df_missed["missed"] > 0), ["replica", "date", "missed"]]
        list_results_missed.append(df_missed)

        # total power and utilization
        df_total = (
            df_system.groupby("type")[["power", "available_power", "final_power"]]
            .sum()
            .assign(
                utilization_avail=lambda x: x.available_power / x.power * 100,
                utilization_final=lambda x: x.final_power / x.power * 100,
            )
            .reset_index()
        )
        df_total["final_energy_GWh"] = df_total["final_power"] * 24 / 1000
        df_total["mix_percent"] = df_total["final_power"] / df_total["final_power"].sum() * 100
        df_total["replica"] = i
        list_results_power.append(
            df_total[["replica", "type", "final_energy_GWh", "mix_percent", "utilization_avail", "utilization_final"]]
        )

    # gather results
    df_results_missed = pd.concat(list_results_missed)
    df_results_power = pd.concat(list_results_power)

    return df_results_missed, df_results_power
