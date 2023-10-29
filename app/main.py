"""
# DSM 2023 Workshop
This app helps the participants of the DSM Industry Sprint Workshop.
"""

###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import datetime
import json
import pandas as pd
import numpy as np
import streamlit as st
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit import session_state as ss
import plotly.express as px
from google.cloud import firestore
from google.oauth2 import service_account
import seaborn as sns
from ragraph.graph import Graph
from ragraph.node import Node
from ragraph.edge import Edge
from ragraph import plot
from ragraph.colors import (
    get_diverging_redblue,
)

###############################################################################
# Formatting
###############################################################################

# Set wide display, if not done before
try:
    st.set_page_config(
        layout="wide",
        page_title="Industry Sprint Workshop 2023 - The 25th International DSM Conference",
        page_icon="assets/favicon.jpg",
    )
except:
    pass

# Hide the top decoration menu, footer, and top padding
hide_streamlit_style = """
                <style>
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                background: rgba(250,250,250, 0) !important;
                }
                .stActionButton {
                display: none;
                visibility: hidden;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
                .modebar-group {
                background-color: rgba(0, 0, 0, 0.1) !important;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

###############################################################################
# Setup
###############################################################################


# Authenticate to Firestore
@st.cache_resource
def authenticate_to_firestore():
    """Authenticates to Firestore and returns a client."""
    key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds, project="dsm2023isw")
    return db


db = authenticate_to_firestore()


# Disable SettingWithCopyWarning from pandas
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None  # default='warn'


# Colors
systems_colors = ["#264653", "#E9C46A", "#E76F51"]  # , "#2A9D8F", "#F4A261", "#E63946"]
markets_colors = ["#3A86FF", "#FF006E", "#8338EC"]

# dataframe colors
cm_g2r = sns.diverging_palette(130, 12, as_cmap=True)
cm_r2g = sns.diverging_palette(12, 130, as_cmap=True)


###############################################################################
# Classes
###############################################################################


# System
class System:
    """A class to represent a system."""

    def __init__(
        self,
        name: str,
        description: str,
        min_R: float,
        reliability: float,
        price: float,
        cost: float,
    ):
        """Initialize the system."""
        self.name = name
        self.description = description
        self.min_R = min_R
        self.reliability = reliability
        self.price = price
        self.cost = cost

    def __repr__(self):
        """Return a string representation of the system."""
        return f"{self.name} ({self.description})"


###############################################################################
# Functions
###############################################################################


# Timestamp string
def get_timestamp():
    """Returns a timestamp string"""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


def on_system_selection():
    """Callback function when system is selected."""
    # print(f"{ss.system} selected")
    for item in [
        "risks_selected_s1",
        "risks_selected_s2",
        "risks_selected_s3",
    ]:
        ss[item] = ss[item]
    pass


def on_risks_selection(selection):
    """Callback function when risk is selected."""
    # print(f"Risks selected for {ss.system}: {selection}")
    pass

def on_mitigations_selection(selection):
    """Callback function when mitigation is selected."""
    # print(f"Mitigations selected for {ss.system}: {selection}")
    calculate_ms()
    pass


def get_session_id() -> str:
    """Get the Session ID for the current session."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None
        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None
    return session_info._session_id


def calculate_ms(new_df: pd.DataFrame | None = None):
    """Calculate market shares and update the dataframe."""
    if new_df is not None:
        if new_df.equals(ss["df_systems"]):
            return
        ss["df_systems"] = new_df

    df_systems = ss["df_systems"]
    df_systems["market_share_1"] = df_systems["price"] * 2
    df_systems["market_share_2"] = df_systems["price"] * 3
    df_systems["market_share_3"] = df_systems["price"] * 4
    df_systems["market_units_1"] = df_systems["market_share_1"] * ss.market_sizes[0]
    df_systems["market_units_2"] = df_systems["market_share_2"] * ss.market_sizes[1]
    df_systems["market_units_3"] = df_systems["market_share_3"] * ss.market_sizes[2]
    df_systems["market_revenue_1"] = df_systems["price"] * df_systems["market_units_1"]
    df_systems["market_revenue_2"] = df_systems["price"] * df_systems["market_units_2"]
    df_systems["market_revenue_3"] = df_systems["price"] * df_systems["market_units_3"]
    df_systems["market_profit_1"] = (
        df_systems["price"] - df_systems["cost"] + ss.cost_mitigations[0]
    ) * df_systems["market_units_1"]
    df_systems["market_profit_2"] = (
        df_systems["price"] - df_systems["cost"] + ss.cost_mitigations[1]
    ) * df_systems["market_units_2"]
    df_systems["market_profit_3"] = (
        df_systems["price"] - df_systems["cost"] + ss.cost_mitigations[2]
    ) * df_systems["market_units_3"]
    df_systems["total_units"] = (
        df_systems["market_units_1"]
        + df_systems["market_units_2"]
        + df_systems["market_units_3"]
    )
    df_systems["total_revenue"] = (
        df_systems["market_revenue_1"]
        + df_systems["market_revenue_2"]
        + df_systems["market_revenue_3"]
    )
    df_systems["total_profit"] = (
        df_systems["market_profit_1"]
        + df_systems["market_profit_2"]
        + df_systems["market_profit_3"]
    )
    ss["df_systems"] = df_systems
    # st.rerun()


###############################################################################
# Import data
###############################################################################


# Market shares
if "market_sizes" not in ss:
    ss.market_sizes = [10000, 20000, 40000]

# Selected system
if "system" not in ss:
    ss.system = None


@st.cache_data
def get_data(csv_file):
    return pd.read_csv(csv_file, sep=";", decimal=",")


# Import data from data/Components.csv into dataframe
# df_components = pd.read_csv("data/Components.csv", sep=";", decimal=",")
df_components = get_data("data/Components.csv")
# Components per system
# df_components_s1 = df_components[df_components["s1"] == True]
# df_components_s2 = df_components[df_components["s2"] == True]
# df_components_s3 = df_components[df_components["s3"] == True]

# DSMs
# df_dsm = pd.read_csv("data/dsm.csv", sep=";", header=None, decimal=",").fillna(0)
df_dsm = get_data("data/dsm.csv").fillna(0)

# Distances
# df_distances = pd.read_csv("data/distances.csv", sep=";", header=None, decimal=",").fillna(0)
df_distances = pd.read_csv(
    "data/distances.csv", index_col=False, header=None, sep=";", decimal=","
).fillna(0)


# Combined risks from CPM
df_risk_s1 = pd.read_csv(
    "data/risk/s1_risk.csv", index_col=False, header=None, sep=";", decimal=","
).fillna(0)
# df_risk_s1.columns = df_components[df_components["s1"] == True]['name']
df_risk_s1.insert(0, "id", df_components[df_components["s1"] == True]["name"])

df_risk_s2 = pd.read_csv(
    "data/risk/s2_risk.csv", index_col=False, header=None, sep=";", decimal=","
).fillna(0)
# df_risk_s2.columns = df_components[df_components["s2"] == True]['name']
df_risk_s2.insert(0, "id", df_components[df_components["s2"] == True]["name"])

df_risk_s3 = pd.read_csv(
    "data/risk/s3_risk.csv", index_col=False, header=None, sep=";", decimal=","
).fillna(0)
# df_risk_s3.columns = df_components[df_components["s3"] == True]['name']
df_risk_s3.insert(0, "id", df_components[df_components["s3"] == True]["name"])

# Kinds of interfaces
kinds = {
    "M": "mechanical",
    "E": "electrical",
    "I": "information",
    "H": "hydraulic",
}

# Import data from data/Risks.csv into dataframe
df_risks = get_data("data/TechRisks.csv")

if (
    "risks_selected_s1" not in ss
    and "risks_selected_s2" not in ss
    and "risks_selected_s3" not in ss
):
    ss.risks_selected_s1 = []
    ss.risks_selected_s2 = []
    ss.risks_selected_s3 = []

# Matrices: fig, g, matrix
if "fig" not in ss:
    ss.fig = None

if "g" not in ss:
    ss.g = Graph()

    for component in df_components.iterrows():
        # print(f'id: {component[1]["id"]} name:{component[1]["name"]}')
        systems_full_names = {
            "s1": "System 1",
            "s2": "System 2",
            "s3": "System 3",
        }
        labels = [
            systems_full_names[s] for s in ["s1", "s2", "s3"] if component[1][s] == True
        ]
        fancy_node = Node(
            name=component[1]["name"],
            kind="component",
            labels=labels,
            weights={
                "x": component[1]["x"],
                "y": component[1]["y"],
                "z": component[1]["z"],
                "force_e": component[1]["force_e"],
                "force_t": component[1]["force_t"],
                "force_r": component[1]["force_r"],
                "electro_e": component[1]["electro_e"],
                "electro_t": component[1]["electro_t"],
                "electro_r": component[1]["electro_r"],
                "thermo_e": component[1]["thermo_e"],
                "thermo_t": component[1]["thermo_t"],
                "thermo_r": component[1]["thermo_r"],
            },
            annotations={
                "id": component[1]["id"],
            },
        )
        ss.g.add_node(fancy_node)

    for i, row in df_dsm.iterrows():
        for j, value in enumerate(row):
            # print(i, j, value)
            # print()
            if i == j:
                continue
            if value in kinds.keys():
                kind = kinds[value]
            else:
                kind = None
            ss.g.add_edge(
                Edge(
                    source=ss.g.nodes[i],
                    target=ss.g.nodes[j],
                    name=f'{ss.g.nodes[i].annotations["id"]}_{ss.g.nodes[j].annotations["id"]}',
                    kind=kind,
                    labels=[],
                    weights={
                        "distance": df_distances.iloc[i, j],
                    },
                    annotations={},
                )
            )

if "matrix" not in ss:
    ss.matrix = "Interfaces DSM"
    # on_matrix_selection(ss.matrix)

# Import data from data/Mitigations.csv into dataframe
df_mitigations = get_data("data/Mitigations.csv").fillna(False)

if (
    "mitigations_selected_s1" not in ss
    and "mitigations_selected_s2" not in ss
    and "mitigations_selected_s3" not in ss
):
    ss.mitigations_selected_s1 = []
    ss.mitigations_selected_s2 = []
    ss.mitigations_selected_s3 = []

if "mitigations_selected" not in ss:
    ss.mitigations_selected = [[], [], []]

if "cost_mitigations" not in ss and "reliability_mitigations" not in ss:
    ss.cost_mitigations = [0, 0, 0]
    ss.reliability_mitigations = [0, 0, 0]

# Original systems designs
if "df_systems" not in ss:
    # ss.df_systems = pd.read_csv("data/Systems.csv", sep=";", decimal=",")
    ss.df_systems = get_data("data/Systems.csv")
    calculate_ms()

if "group" not in ss:
    ss.group = None

if "consent" not in ss:
    ss.consent = None

if "role" not in ss:
    ss.role = []

if "sector" not in ss:
    ss.sector = []

if "experience" not in ss:
    ss.experience = None

if (
    "q1" not in ss
    and "q2" not in ss
    and "q3" not in ss
    and "q4" not in ss
    and "q5" not in ss
    and "q6" not in ss
    and "q7" not in ss
    and "q8" not in ss
):
    ss.q1 = None
    ss.q2 = None
    ss.q3 = None
    ss.q4 = None
    ss.q5 = None
    ss.q6 = None
    ss.q7 = None
    ss.q8 = None

if (
    "before_artic_s1" not in ss
    and "before_artic_s2" not in ss
    and "before_artic_s3" not in ss
    and "before_desert_s1" not in ss
    and "before_desert_s2" not in ss
    and "before_desert_s3" not in ss
    and "before_special_s1" not in ss
    and "before_special_s2" not in ss
    and "before_special_s3" not in ss
    and "after_artic_s1" not in ss
    and "after_artic_s2" not in ss
    and "after_artic_s3" not in ss
    and "after_desert_s1" not in ss
    and "after_desert_s2" not in ss
    and "after_desert_s3" not in ss
    and "after_special_s1" not in ss
    and "after_special_s2" not in ss
    and "after_special_s3" not in ss
):
    ss.before_artic_s1 = None
    ss.before_artic_s2 = None
    ss.before_artic_s3 = None
    ss.before_desert_s1 = None
    ss.before_desert_s2 = None
    ss.before_desert_s3 = None
    ss.before_special_s1 = None
    ss.before_special_s2 = None
    ss.before_special_s3 = None
    ss.after_artic_s1 = None
    ss.after_artic_s2 = None
    ss.after_artic_s3 = None
    ss.after_desert_s1 = None
    ss.after_desert_s2 = None
    ss.after_desert_s3 = None
    ss.after_special_s1 = None
    ss.after_special_s2 = None
    ss.after_special_s3 = None

if "cost_mitigations" not in ss and "reliability_mitigations" not in ss:
    ss.cost_mitigations = [0, 0, 0]
    ss.reliability_mitigations = [0, 0, 0]


###############################################################################
# Head
###############################################################################

# Logo and title
col_logo, col_title = st.columns([0.2, 1])
with col_logo:
    st.write("")
    st.write("")
    st.image("assets/logo_large.png", width=60)

with col_title:
    st.title("VISP Workshop")
    st.caption("**Workshop Facilitator** webapp")

with st.expander("Info", expanded=True):
    st.markdown(
        """
            Please fill in the following information to start the workshop.
            """
    )

    ss.group = st.radio(
        label="Workshop group",
        help="Select your assigned group here.",
        options=(
            "Select",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "Test",
        ),
        horizontal=True,
    )
    ss.consent = st.checkbox(
        label="I consent to the use of my data for research purposes.",
        help="Please check this box to consent to the use of data collected from your interaction with this website for research purposes. No identifying information will be collected.",
    )
    st.caption(
        f"To exercise your right to remove the collected data from the database at a later date, please contact the workshop facilitator with this Session ID: {get_session_id()}"
    )

    if not ((ss.group != "Select") and ss.consent):
        warning = st.warning(
            body="Please make sure to enter your group correctly.",
            icon="⚠️",
        )
    else:
        st.success(
            body="You are ready to go! Click on the top right arrow to minimize this section. The tabs bellow will guide you through the workshop.",
            icon="👍",
        )

# If the user has filled in the intro form correctly
is_ready = (ss.group != "Select") and ss.consent
if is_ready:
    # with st.expander("**Select system**", expanded=False):
    with st.sidebar:
        system_logo = st.empty()
        # Select system to display
        ss.system = st.selectbox(
            label="Select the system to analyze",
            options=["System 1", "System 2", "System 3"],
            index=0,
            help="Select the system to analyze",
            on_change=on_system_selection(),
        )
        system_logo.image(f"assets/system{ss.system[-1]}.png", width=245)

        systems_ids = {
            "System 1": 0,
            "System 2": 1,
            "System 3": 2,
        }
        systems_shorts = {
            "System 1": "s1",
            "System 2": "s2",
            "System 3": "s3",
        }

        st.markdown(
            f"""
            {ss["df_systems"]["description"][systems_ids[ss.system]]}

            **Min turning radius**: {ss["df_systems"]["min_R"][systems_ids[ss.system]]} m

            **Reliability**: {ss["df_systems"]["reliability"][systems_ids[ss.system]] + ss.reliability_mitigations[systems_ids[ss.system]]:0.3f}

            **Unit Price**: {ss["df_systems"]["price"][systems_ids[ss.system]]} k€

            **Unit Cost**: {ss["df_systems"]["cost"][systems_ids[ss.system]] + ss.cost_mitigations[systems_ids[ss.system]]} k€

            **Total Profit**: {ss["df_systems"]["total_profit"][systems_ids[ss.system]]/1000} M€
            """
        )

    # holder.empty()
    tab1, tab2, tab3 = st.tabs(
        [
            "(1) Analyze Value",
            "(2) Identify Risks",
            "(3) Mitigate Risks",
            # "Questionnaire",
            # "Help",
        ]
    )

    ###############################################################################
    # Tab 1
    ###############################################################################

    with tab1:
        st.subheader("Markets and systems")
        with st.expander("**Markets**", expanded=True):
            st.markdown(
                """Potential number of trucks sold each year per application)"""
            )
            tab_markets_1, tab_markets_2, tab_markets_3 = st.columns(3)

            with tab_markets_1:
                st.image("assets/artic.jpg")
                ss.market_sizes[0] = st.slider(
                    "Artic", 0, 200000, ss.market_sizes[0], disabled=True
                )
            with tab_markets_2:
                st.image("assets/desert.jpg")
                ss.market_sizes[1] = st.slider(
                    "Desert", 0, 200000, ss.market_sizes[1], disabled=True
                )
            with tab_markets_3:
                st.image("assets/special.jpg")
                ss.market_sizes[2] = st.slider(
                    "Special", 0, 200000, ss.market_sizes[2], disabled=True
                )

        with st.expander("**Systems under consideration**", expanded=True):
            st.markdown(
                """The following systems are under consideration for introduction into the market."""
            )

            tab_systems_1, tab_systems_2, tab_systems_3 = st.columns(3)

            with tab_systems_1:
                st.image("assets/system1.png")
                st.markdown(
                    f"""
                    **{ss["df_systems"]["name"][0]}**: {ss["df_systems"]["description"][0]}

                    **Minimum turning radius**: {ss["df_systems"]["min_R"][0]} m

                    **Reliability**: {ss["df_systems"]["reliability"][0] + ss.reliability_mitigations[0]:0.3f}

                    **Price**: {ss["df_systems"]["price"][0]} k€

                    **Cost**: {ss["df_systems"]["cost"][0] + ss.cost_mitigations[0]} k€
                    """
                )
            with tab_systems_2:
                st.image("assets/system2.png")
                st.markdown(
                    f"""
                    **{ss["df_systems"]["name"][1]}**: {ss["df_systems"]["description"][1]}

                    **Minimum turning radius**: {ss["df_systems"]["min_R"][1]}

                    **Reliability**: {ss["df_systems"]["reliability"][1] + ss.reliability_mitigations[1]:0.3f}

                    **Price**: {ss["df_systems"]["price"][1]} k€

                    **Cost**: {ss["df_systems"]["cost"][1] + ss.cost_mitigations[1]} k€
                    """
                )
            with tab_systems_3:
                st.image("assets/system3.png")
                st.markdown(
                    f"""
                    **{ss["df_systems"]["name"][2]}**: {ss["df_systems"]["description"][2]}

                    **Minimum turning radius**: {ss["df_systems"]["min_R"][2]}

                    **Reliability**: {ss["df_systems"]["reliability"][2] + ss.reliability_mitigations[2]:0.3f}

                    **Price**: {ss["df_systems"]["price"][2]} k€

                    **Cost**: {ss["df_systems"]["cost"][2] + ss.cost_mitigations[2]} k€
                    """
                )

            # calculate_ms(editable_df)
            calculate_ms(ss["df_systems"])
            editable_df = ss["df_systems"]

            market_shares_artic = []
            market_shares_desert = []
            market_shares_special = []

            for i in range(len(editable_df)):
                a = 1 / (1 + ((editable_df["min_R"][i] - 10) * 0.5) ** 2) - 0.5
                d = 1 - (0.5) ** (50 / editable_df["price"][i]) - 0.3
                e = 1 - (0.5) ** (1 / editable_df["reliability"][i])
                market_share = 0.2 * (a + d + e)
                market_shares_artic.append(market_share)
                # print(i, a, d, e, market_shares_artic)

            for i in range(len(editable_df)):
                a = 1 / (1 + ((editable_df["min_R"][i] - 10) * 0.5) ** 2) - 0.5
                d = 1 - (0.5) ** (50 / editable_df["price"][i]) - 0.3
                e = 1 - (0.5) ** (1 / editable_df["reliability"][i])
                market_share = 0.2 * (a + d + e)
                market_shares_desert.append(market_share)
                # print(i, a, d, e, market_shares_desert)

            for i in range(len(editable_df)):
                a = 1 - (0.5) ** (50 / editable_df["min_R"][i]) - 0.3
                d = 1 - (0.5) ** (500 / editable_df["price"][i]) - 0.3
                e = 1 - (0.5) ** (1 / editable_df["reliability"][i])
                market_share = 0.2 * (a + d + e)
                market_shares_special.append(market_share)
                # print(i, a, d, e, market_shares_special)

            units_artic = [
                editable_df["market_units_1"][0],
                editable_df["market_units_1"][1],
                editable_df["market_units_1"][2],
            ]
            units_desert = [
                editable_df["market_units_2"][0],
                editable_df["market_units_2"][1],
                editable_df["market_units_2"][2],
            ]
            units_special = [
                editable_df["market_units_3"][0],
                editable_df["market_units_3"][1],
                editable_df["market_units_3"][2],
            ]
            revenue_artic = [
                editable_df["market_revenue_1"][0],
                editable_df["market_revenue_1"][1],
                editable_df["market_revenue_1"][2],
            ]
            revenue_desert = [
                editable_df["market_revenue_2"][0],
                editable_df["market_revenue_2"][1],
                editable_df["market_revenue_2"][2],
            ]
            revenue_special = [
                editable_df["market_revenue_3"][0],
                editable_df["market_revenue_3"][1],
                editable_df["market_revenue_3"][2],
            ]
            profit_artic = [
                editable_df["market_profit_1"][0],
                editable_df["market_profit_1"][1],
                editable_df["market_profit_1"][2],
            ]
            profit_desert = [
                editable_df["market_profit_2"][0],
                editable_df["market_profit_2"][1],
                editable_df["market_profit_2"][2],
            ]
            profit_special = [
                editable_df["market_profit_3"][0],
                editable_df["market_profit_3"][1],
                editable_df["market_profit_3"][2],
            ]

            markets_df = pd.DataFrame(
                [
                    {
                        "market": "Artic",
                        "share_system_1": market_shares_artic[0],
                        "share_system_2": market_shares_artic[1],
                        "share_system_3": market_shares_artic[2],
                        "units_system_1": editable_df["market_units_1"][0],
                        "units_system_2": editable_df["market_units_1"][1],
                        "units_system_3": editable_df["market_units_1"][2],
                        "revenue_system_1": editable_df["market_revenue_1"][0],
                        "revenue_system_2": editable_df["market_revenue_1"][1],
                        "revenue_system_3": editable_df["market_revenue_1"][2],
                        "profit_system_1": editable_df["market_profit_1"][0],
                        "profit_system_2": editable_df["market_profit_1"][1],
                        "profit_system_3": editable_df["market_profit_1"][2],
                    },
                    {
                        "market": "Desert",
                        "share_system_1": market_shares_desert[0],
                        "share_system_2": market_shares_desert[1],
                        "share_system_3": market_shares_desert[2],
                        "units_system_1": editable_df["market_units_2"][0],
                        "units_system_2": editable_df["market_units_2"][1],
                        "units_system_3": editable_df["market_units_2"][2],
                        "revenue_system_1": editable_df["market_revenue_2"][0],
                        "revenue_system_2": editable_df["market_revenue_2"][1],
                        "revenue_system_3": editable_df["market_revenue_2"][2],
                        "profit_system_1": editable_df["market_profit_2"][0],
                        "profit_system_2": editable_df["market_profit_2"][1],
                        "profit_system_3": editable_df["market_profit_2"][2],
                    },
                    {
                        "market": "Special",
                        "share_system_1": market_shares_special[0],
                        "share_system_2": market_shares_special[1],
                        "share_system_3": market_shares_special[2],
                        "units_system_1": editable_df["market_units_3"][0],
                        "units_system_2": editable_df["market_units_3"][1],
                        "units_system_3": editable_df["market_units_3"][2],
                        "revenue_system_1": editable_df["market_revenue_3"][0],
                        "revenue_system_2": editable_df["market_revenue_3"][1],
                        "revenue_system_3": editable_df["market_revenue_3"][2],
                        "profit_system_1": editable_df["market_profit_3"][0],
                        "profit_system_2": editable_df["market_profit_3"][1],
                        "profit_system_3": editable_df["market_profit_3"][2],
                    },
                ]
            )

        st.subheader("1. Analyze Value")

        with st.expander("**Profits**", expanded=True):
            col_profits_1, col_profits_2, col_profits_3 = st.columns(3)

            with col_profits_1:
                st.metric(
                    label="System 1",
                    value=f"{editable_df['total_profit'][0]/1000000:.3f} M€",
                    delta="",
                )

            with col_profits_2:
                st.metric(
                    label="System 2",
                    value=f"{editable_df['total_profit'][1]/1000000:.3f} M€",
                    delta="",
                )

            with col_profits_3:
                st.metric(
                    label="System 3",
                    value=f"{editable_df['total_profit'][2]/1000000:.3f} M€",
                    delta="",
                )

        with st.expander("**Market share**", expanded=True):
            # Plotly group bars plot of market share per system
            st.plotly_chart(
                px.bar(
                    markets_df,
                    x="market",
                    y=["share_system_1", "share_system_2", "share_system_3"],
                    barmode="group",
                    color_discrete_sequence=systems_colors,
                    labels={
                        "value": "Market share",
                        "variable": "System",
                        "market": "Market",
                    },
                    height=400,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Questions Tab 1
        questions_tab1 = st.expander("**Questions**", expanded=True)

        form_tab1 = questions_tab1.form(key="form_tab1")

        with form_tab1:
            form_tab1.write(
                "I think that the most valuable systems for each of the markets are..."
            )
            form_tab1.caption(
                "Please rate from 1 to 10, where 1 means low potential and 10 high potential."
            )

            cont_systems = form_tab1.container()
            with cont_systems:
                (
                    col_cont_systems_0,
                    col_cont_systems_1,
                    col_cont_systems_2,
                    col_cont_systems_3,
                ) = st.columns(4)
                with col_cont_systems_1:
                    st.write("System 1")
                    st.image("assets/system1.png")
                with col_cont_systems_2:
                    st.write("System 2")
                    st.image("assets/system2.png")
                with col_cont_systems_3:
                    st.write("System 3")
                    st.image("assets/system3.png")
            cont_market_artic = form_tab1.container()
            with cont_market_artic:
                st.write("Artic Market")
                (
                    col_cont_artic_0,
                    col_cont_artic_1,
                    col_cont_artic_2,
                    col_cont_artic_3,
                ) = st.columns(4)
                with col_cont_artic_0:
                    st.image("assets/artic.jpg")
                with col_cont_artic_1:
                    ss.before_artic_s1 = st.slider("System 1 in Artic Market", 1, 10, 5)
                with col_cont_artic_2:
                    ss.before_artic_s2 = st.slider("System 2 in Artic Market", 1, 10, 5)
                with col_cont_artic_3:
                    ss.before_artic_s3 = st.slider("System 3 in Artic Market", 1, 10, 5)

            cont_market_desert = form_tab1.container()
            with cont_market_desert:
                st.write("Desert Market")
                (
                    col_cont_desert_0,
                    col_cont_desert_1,
                    col_cont_desert_2,
                    col_cont_desert_3,
                ) = st.columns(4)
                with col_cont_desert_0:
                    st.image("assets/desert.jpg")
                with col_cont_desert_1:
                    ss.before_desert_s1 = st.slider(
                        "System 1 in Desert Market", 1, 10, 5
                    )
                with col_cont_desert_2:
                    ss.before_desert_s2 = st.slider(
                        "System 2 in Desert Market", 1, 10, 5
                    )
                with col_cont_desert_3:
                    ss.before_desert_s3 = st.slider(
                        "System 3 in Desert Market", 1, 10, 5
                    )

            cont_market_special = form_tab1.container()
            with cont_market_special:
                st.write("Special Market")
                (
                    col_cont_special_0,
                    col_cont_special_1,
                    col_cont_special_2,
                    col_cont_special_3,
                ) = st.columns(4)
                with col_cont_special_0:
                    st.image("assets/special.jpg")
                with col_cont_special_1:
                    ss.before_special_s1 = st.slider(
                        "System 1 in Special Market", 1, 10, 5
                    )
                with col_cont_special_2:
                    ss.before_special_s2 = st.slider(
                        "System 2 in Special Market", 1, 10, 5
                    )
                with col_cont_special_3:
                    ss.before_special_s3 = st.slider(
                        "System 3 in Special Market", 1, 10, 5
                    )

            form_tab1_submitted = st.form_submit_button(
                label="Submit",
                help="Click here to submit your answers.",
                type="primary",
                use_container_width=True,
            )

    ###############################################################################
    # Tab 2
    ###############################################################################

    with tab2:
        st.subheader("2. Identify Risks")

        with st.expander(f"**Technical Risk Registry for {ss.system}**", expanded=True):
            st.markdown(
                """
                The following table lists the technical risks that have been identified for the system under consideration.

                Each failure has three potential originating mechanisms (mechanical, electromagnetic, and thermal) and a risk index.
                """
            )

            if ss.system == "System 1":
                df_risks_to_display = df_risks[df_risks["s1"] == True]
            elif ss.system == "System 2":
                df_risks_to_display = df_risks[df_risks["s2"] == True]
            elif ss.system == "System 3":
                df_risks_to_display = df_risks[df_risks["s3"] == True]
            else:
                df_risks_to_display = df_risks

            df_risks_table = st.dataframe(
                df_risks_to_display.style.background_gradient(cmap=cm_g2r).format(
                    {2: "{:.2f}"}, na_rep="MISS", precision=2
                ),
                height=400,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.TextColumn(
                        "Risk ID", help="Risk ID", width="small"
                    ),
                    "Name": st.column_config.TextColumn(
                        "Risk name", help="Risk description", width="large"
                    ),
                    "Mechanical": st.column_config.NumberColumn(
                        "Mechanical",
                        help="Mechanical risk",
                        format="%.2f",
                        width="small",
                    ),
                    "Electromagnetic": st.column_config.NumberColumn(
                        "Electromagnetic",
                        help="Electromagnetic risk",
                        format="%.2f",
                        width="small",
                    ),
                    "Thermal": st.column_config.NumberColumn(
                        "Thermal",
                        help="Thermal risk",
                        format="%.2f",
                        width="small",
                    ),
                    "Comments": None,
                    "s1": None,
                    "s2": None,
                    "s3": None,
                },
            )

        matrices = st.expander("**Matrices**", expanded=True)
        with matrices:
            ss.matrix = st.radio(
                label="Select the matrix to display",
                options=["Interfaces DSM", "Distance DSM", "Risk DSM"],
                index=0,
                captions=[
                    "Interfaces between components",
                    "Distances between components",
                    "Propagation of risks between components",
                ],
                horizontal=True,
            )

            if ss.matrix == "Interfaces DSM":
                if ss.system == "System 1":
                    st.image("assets/s1_interfaces.png")
                elif ss.system == "System 2":
                    st.image("assets/s2_interfaces.png")
                elif ss.system == "System 3":
                    st.image("assets/s3_interfaces.png")
            elif ss.matrix == "Distance DSM":
                if ss.system == "System 1":
                    st.image("assets/s1_distances.png")
                elif ss.system == "System 2":
                    st.image("assets/s2_distances.png")
                elif ss.system == "System 3":
                    st.image("assets/s3_distances.png")
            elif ss.matrix == "Risk DSM":
                if ss.system == "System 1":
                    st.image("assets/s1_risk.png")
                elif ss.system == "System 2":
                    st.image("assets/s2_risk.png")
                elif ss.system == "System 3":
                    st.image("assets/s3_risk.png")

        with st.expander(f"**Select {ss.system} risks for mitigation**", expanded=True):
            st.markdown(
                f"""
                Which of the risks present in **{ss.system}** would you select for mitigation?
                """
            )

            questions_tab2_col1, questions_tab2_col2 = st.columns(2)

            if ss.system == "System 1":
                questions_tab2_col1.multiselect(
                    label="Select the risks you would like to mitigate.",
                    options=df_risks[df_risks["s1"] == True].ID,
                    help="Select the risks you would like to mitigate.",
                    key="risks_selected_s1",
                    on_change=on_risks_selection(ss.risks_selected_s1),
                )
                questions_tab2_col2.dataframe(
                    df_risks[df_risks.ID.isin(ss.risks_selected_s1)],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Selected": None,
                        "ID": st.column_config.TextColumn(
                            "Risk ID", help="Risk ID", width="small"
                        ),
                        "Name": st.column_config.TextColumn(
                            "Risk name", help="Risk description", width="large"
                        ),
                        "Comments": None,
                        "Mechanical": None,
                        "Electromagnetic": None,
                        "Thermal": None,
                        "s1": None,
                        "s2": None,
                        "s3": None,
                    },
                )
            elif ss.system == "System 2":
                questions_tab2_col1.multiselect(
                    label="Select the risks you would like to mitigate.",
                    options=df_risks[df_risks["s2"] == True].ID,
                    help="Select the risks you would like to mitigate.",
                    key="risks_selected_s2",
                    on_change=on_risks_selection(ss.risks_selected_s2),
                )
                questions_tab2_col2.dataframe(
                    df_risks[df_risks.ID.isin(ss.risks_selected_s2)],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Selected": None,
                        "ID": st.column_config.TextColumn(
                            "Risk ID", help="Risk ID", width="small"
                        ),
                        "Name": st.column_config.TextColumn(
                            "Risk name", help="Risk description", width="large"
                        ),
                        "Comments": None,
                        "Mechanical": None,
                        "Electromagnetic": None,
                        "Thermal": None,
                        "s1": None,
                        "s2": None,
                        "s3": None,
                    },
                )
            elif ss.system == "System 3":
                questions_tab2_col1.multiselect(
                    label="Select the risks you would like to mitigate.",
                    options=df_risks[df_risks["s3"] == True].ID,
                    help="Select the risks you would like to mitigate.",
                    key="risks_selected_s3",
                    on_change=on_risks_selection(ss.risks_selected_s3),
                )
                questions_tab2_col2.dataframe(
                    df_risks[df_risks.ID.isin(ss.risks_selected_s3)],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Selected": None,
                        "ID": st.column_config.TextColumn(
                            "Risk ID", help="Risk ID", width="small"
                        ),
                        "Name": st.column_config.TextColumn(
                            "Risk name", help="Risk description", width="large"
                        ),
                        "Comments": None,
                        "Mechanical": None,
                        "Electromagnetic": None,
                        "Thermal": None,
                        "s1": None,
                        "s2": None,
                        "s3": None,
                    },
                )
            else:
                st.warning("Please select a system to explore.")

    ###############################################################################
    # Tab 3
    ###############################################################################

    with tab3:
        st.subheader("3. Mitigate Risks")

        with st.expander("**List of Mitigation Elements**", expanded=True):
            st.dataframe(
                df_mitigations.style.background_gradient(
                    cmap=cm_g2r, subset=["Cost (k€)"]
                )
                .background_gradient(cmap=cm_r2g, subset=["Reliability gain"])
                .format({3: "{:.2f}", 4: "{:.3f}"}, na_rep=False, precision=3),
                height=400,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.TextColumn(
                        "ID", help="Mitigation ID", width="small"
                    ),
                    "Risk Mitigation element": st.column_config.TextColumn(
                        "Risk Mitigation element",
                        help="Risk Mitigation element",
                        width="medium",
                    ),
                    "Affects the interactions between (A-B)": st.column_config.TextColumn(
                        "Affects the interactions between",
                        help="Affects the interactions between",
                        width="medium",
                    ),
                    "A": st.column_config.ListColumn(
                        "A",
                        help="A",
                        width="small",
                    ),
                    "B": st.column_config.ListColumn(
                        "B",
                        help="B",
                        width="small",
                    ),
                    "Cost (k€)": st.column_config.NumberColumn(
                        "Cost (k€)",
                        help="Cost (k€)",
                        width="small",
                    ),
                    "Reliability gain": st.column_config.NumberColumn(
                        "Reliability gain",
                        help="Reliability gain",
                        width="small",
                    ),
                    "Mechanical": None,
                    "Electromagnetic": None,
                    "Thermal": None,
                    "id2": None,
                    "x": None,
                    "y": None,
                    "z": None,
                    "force_e2": None,
                    "force_t": None,
                    "force_r": None,
                    "electro_e2": None,
                    "electro_t": None,
                    "electro_r": None,
                    "thermo_e2": None,
                    "thermo_t": None,
                    "thermo_r": None,
                    "s1": None,
                    "s2": None,
                    "s3": None,
                },
            )

        with st.expander(f"**Select mitigations for System 1**", expanded=True):
            st.markdown(
                f"""
                Which of the available mitigations would you add to **System 1**?
                """
            )

            questions_tab3_s1_col1, questions_tab3_s1_col2 = st.columns(2)

            questions_tab3_s1_col1.multiselect(
                label="Select the mitigations you would like to mitigate.",
                options=df_mitigations[df_mitigations["s1"] == True].ID,
                help="Select the mitigations you would like to mitigate.",
                key="mitigations_selected_s1",
                on_change=on_mitigations_selection(ss.mitigations_selected_s1),
            )
            questions_tab3_s1_col2.dataframe(
                df_mitigations[df_mitigations.ID.isin(ss.mitigations_selected_s1)],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Selected": None,
                    "ID": st.column_config.TextColumn(
                        "ID", help="ID", width="small"
                    ),
                    "Risk Mitigation element": st.column_config.TextColumn(
                        "Name", help="Risk Mitigation element", width="large"
                    ),
                    "Affects the interactions between": None,
                    "A": None,
                    "B": None,
                    "Cost (k€)": None,
                    "id2": None,
                    "x": None,
                    "y": None,
                    "z": None,
                    "force_e2": None,
                    "force_t": None,
                    "force_r": None,
                    "electro_e2": None,
                    "electro_t": None,
                    "electro_r": None,
                    "thermo_e2": None,
                    "thermo_t": None,
                    "thermo_r": None,
                    "Reliability gain": None,
                    "Mechanical": None,
                    "Electromagnetic": None,
                    "Thermal": None,
                    "s1": None,
                    "s2": None,
                    "s3": None,
                },
            )
            # Added cost and reliability by the selected mitigations
            ss.cost_mitigations[0] = df_mitigations[
                df_mitigations.ID.isin(ss.mitigations_selected_s1)
            ]["Cost (k€)"].sum()
            ss["df_systems"]["cost"][0] = ss["df_systems"]["cost"][0] + ss.cost_mitigations[0]
            ss.reliability_mitigations[0] = df_mitigations[
                df_mitigations.ID.isin(ss.mitigations_selected_s1)
            ]["Reliability gain"].sum()
            ss["df_systems"]["reliability"][0] = (
                ss["df_systems"]["reliability"][0] + ss.reliability_mitigations[0]
            )
            questions_tab3_s1_col2.markdown(
                f"""
                The total cost of the selected mitigations is **{ss.cost_mitigations[0]:.3f} k€** per unit.

                The total increase in reliability is **{ss.reliability_mitigations[0]:.3f}**.
                """
            )
        with st.expander(f"**Select mitigations for System 2**", expanded=True):
            st.markdown(
                f"""
                Which of the available mitigations would you add to **System 2**?
                """
            )

            questions_tab3_s2_col1, questions_tab3_s2_col2 = st.columns(2)

            questions_tab3_s2_col1.multiselect(
                label="Select the mitigations you would like to mitigate.",
                options=df_mitigations[df_mitigations["s2"] == True].ID,
                help="Select the mitigations you would like to mitigate.",
                key="mitigations_selected_s2",
                on_change=on_risks_selection(ss.mitigations_selected_s2),
            )
            questions_tab3_s2_col2.dataframe(
                df_mitigations[df_mitigations.ID.isin(ss.mitigations_selected_s2)],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Selected": None,
                    "ID": st.column_config.TextColumn(
                        "ID", help="ID", width="small"
                    ),
                    "Risk Mitigation element": st.column_config.TextColumn(
                        "Name", help="Risk Mitigation element", width="large"
                    ),
                    "Affects the interactions between": None,
                    "A": None,
                    "B": None,
                    "Cost (k€)": None,
                    "id2": None,
                    "x": None,
                    "y": None,
                    "z": None,
                    "force_e2": None,
                    "force_t": None,
                    "force_r": None,
                    "electro_e2": None,
                    "electro_t": None,
                    "electro_r": None,
                    "thermo_e2": None,
                    "thermo_t": None,
                    "thermo_r": None,
                    "Reliability gain": None,
                    "Mechanical": None,
                    "Electromagnetic": None,
                    "Thermal": None,
                    "s1": None,
                    "s2": None,
                    "s3": None,
                },
            )
            # Added cost and reliability by the selected mitigations
            ss.cost_mitigations[1] = df_mitigations[
                df_mitigations.ID.isin(ss.mitigations_selected_s2)
            ]["Cost (k€)"].sum()
            ss["df_systems"]["cost"][1] = ss["df_systems"]["cost"][1] + ss.cost_mitigations[1]
            ss.reliability_mitigations[1] = df_mitigations[
                df_mitigations.ID.isin(ss.mitigations_selected_s2)
            ]["Reliability gain"].sum()
            ss["df_systems"]["reliability"][1] = (
                ss["df_systems"]["reliability"][1] + ss.reliability_mitigations[1]
            )
            questions_tab3_s2_col2.markdown(
                f"""
                The total cost of the selected mitigations is **{ss.cost_mitigations[1]:.3f} k€** per unit.

                The total increase in reliability is **{ss.reliability_mitigations[1]:.3f}**.
                """
            )
        with st.expander(f"**Select mitigations for System 3**", expanded=True):
            st.markdown(
                f"""
                Which of the available mitigations would you add to **System 3**?
                """
            )

            questions_tab3_s3_col1, questions_tab3_s3_col2 = st.columns(2)

            questions_tab3_s3_col1.multiselect(
                label="Select the mitigations you would like to mitigate.",
                options=df_mitigations[df_mitigations["s3"] == True].ID,
                help="Select the mitigations you would like to mitigate.",
                key="mitigations_selected_s3",
                on_change=on_risks_selection(ss.mitigations_selected_s3),
            )
            questions_tab3_s3_col2.dataframe(
                df_mitigations[df_mitigations.ID.isin(ss.mitigations_selected_s3)],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Selected": None,
                    "ID": st.column_config.TextColumn(
                        "ID", help="ID", width="small"
                    ),
                    "Risk Mitigation element": st.column_config.TextColumn(
                        "Name", help="Risk Mitigation element", width="large"
                    ),
                    "Affects the interactions between": None,
                    "A": None,
                    "B": None,
                    "Cost (k€)": None,
                    "id2": None,
                    "x": None,
                    "y": None,
                    "z": None,
                    "force_e2": None,
                    "force_t": None,
                    "force_r": None,
                    "electro_e2": None,
                    "electro_t": None,
                    "electro_r": None,
                    "thermo_e2": None,
                    "thermo_t": None,
                    "thermo_r": None,
                    "Reliability gain": None,
                    "Mechanical": None,
                    "Electromagnetic": None,
                    "Thermal": None,
                    "s1": None,
                    "s2": None,
                    "s3": None,
                },
            )
            # Added cost and reliability by the selected mitigations
            ss.cost_mitigations[2] = df_mitigations[
                df_mitigations.ID.isin(ss.mitigations_selected_s3)
            ]["Cost (k€)"].sum()
            ss["df_systems"]["cost"][2] = ss["df_systems"]["cost"][2] + ss.cost_mitigations[2]
            ss.reliability_mitigations[2] = df_mitigations[
                df_mitigations.ID.isin(ss.mitigations_selected_s3)
            ]["Reliability gain"].sum()
            ss["df_systems"]["reliability"][2] = (
                ss["df_systems"]["reliability"][2] + ss.reliability_mitigations[2]
            )
            questions_tab3_s3_col2.markdown(
                f"""
                The total cost of the selected mitigations is **{ss.cost_mitigations[2]:.3f} k€** per unit.

                The total increase in reliability is **{ss.reliability_mitigations[2]:.3f}**.
                """
            )

        # Questions Tab 3
        questions_tab3 = st.expander("**Questions**", expanded=True)

        form_tab3 = questions_tab3.form(key="form_tab3")

        with form_tab3:
            form_tab3.write(
                "Please reasses the potential of the new designs with mitigations compared with the baseline designs:"
            )
            form_tab3.caption(
                "Please rate from 1 to 10, where 1 means low potential and 10 high potential."
            )

            cont_systems = form_tab3.container()
            with cont_systems:
                (
                    col_cont_systems_0,
                    col_cont_systems_1,
                    col_cont_systems_2,
                    col_cont_systems_3,
                ) = st.columns(4)
                with col_cont_systems_1:
                    st.write("System 1 + mitigations")
                    st.image("assets/system1.png")
                with col_cont_systems_2:
                    st.write("System 2 + mitigations")
                    st.image("assets/system2.png")
                with col_cont_systems_3:
                    st.write("System 3 + mitigations")
                    st.image("assets/system3.png")
            cont_market_artic = form_tab3.container()
            with cont_market_artic:
                st.write("Artic Market")
                (
                    col_cont_artic_0,
                    col_cont_artic_1,
                    col_cont_artic_2,
                    col_cont_artic_3,
                ) = st.columns(4)
                with col_cont_artic_0:
                    st.image("assets/artic.jpg")
                with col_cont_artic_1:
                    ss.after_artic_s1 = st.slider("System 1 in Artic Market", 1, 10, 5)
                with col_cont_artic_2:
                    ss.after_artic_s2 = st.slider("System 2 in Artic Market", 1, 10, 5)
                with col_cont_artic_3:
                    ss.after_artic_s3 = st.slider("System 3 in Artic Market", 1, 10, 5)

            cont_market_desert = form_tab3.container()
            with cont_market_desert:
                st.write("Desert Market")
                (
                    col_cont_desert_0,
                    col_cont_desert_1,
                    col_cont_desert_2,
                    col_cont_desert_3,
                ) = st.columns(4)
                with col_cont_desert_0:
                    st.image("assets/desert.jpg")
                with col_cont_desert_1:
                    ss.after_desert_s1 = st.slider(
                        "System 1 in Desert Market", 1, 10, 5
                    )
                with col_cont_desert_2:
                    ss.after_desert_s2 = st.slider(
                        "System 2 in Desert Market", 1, 10, 5
                    )
                with col_cont_desert_3:
                    ss.after_desert_s3 = st.slider(
                        "System 3 in Desert Market", 1, 10, 5
                    )

            cont_market_special = form_tab3.container()
            with cont_market_special:
                st.write("Special Market")
                (
                    col_cont_special_0,
                    col_cont_special_1,
                    col_cont_special_2,
                    col_cont_special_3,
                ) = st.columns(4)
                with col_cont_special_0:
                    st.image("assets/special.jpg")
                with col_cont_special_1:
                    ss.after_special_s1 = st.slider(
                        "System 1 in Special Market", 1, 10, 5
                    )
                with col_cont_special_2:
                    ss.after_special_s2 = st.slider(
                        "System 2 in Special Market", 1, 10, 5
                    )
                with col_cont_special_3:
                    ss.after_special_s3 = st.slider(
                        "System 3 in Special Market", 1, 10, 5
                    )

            form_tab3_submitted = st.form_submit_button(
                label="Submit",
                help="Click here to submit your answers.",
                type="primary",
                use_container_width=True,
            )


###############################################################################
# Session state
###############################################################################

# with st.expander("Session State", expanded=True):
#     st.write(ss)

# print("Here's the session state:")
# print([key for key in ss.keys()])
# print(ss)

print(f"Session ID: {get_session_id()}")

try:
    document_name = f"{get_session_id()}_{get_timestamp()}"
    session_state_ref = db.collection("session_states").document(document_name)
    # And then uploading the data to that reference
    session_state_ref.set(
        {
            "session_id": get_session_id(),
            "timestamp": get_timestamp(),
            "role": ss.role,
            "sector": ss.sector,
            "experience": ss.experience,
            "group": ss.group,
            "risks_selected_s1": ss.risks_selected_s1,
            "risks_selected_s2": ss.risks_selected_s2,
            "risks_selected_s3": ss.risks_selected_s3,
            "mitigations_selected_s1": ss.mitigations_selected_s1,
            "mitigations_selected_s2": ss.mitigations_selected_s2,
            "mitigations_selected_s3": ss.mitigations_selected_s3,
            "matrix": ss.matrix,
            "q1": ss.q1,
            "q2": ss.q2,
            "q3": ss.q3,
            "q4": ss.q4,
            "q5": ss.q5,
            "q6": ss.q6,
            "q7": ss.q7,
            "q8": ss.q8,
            "before": {
                "artic_s1": ss.before_artic_s1,
                "artic_s2": ss.before_artic_s2,
                "artic_s3": ss.before_artic_s3,
                "desert_s1": ss.before_desert_s1,
                "desert_s2": ss.before_desert_s2,
                "desert_s3": ss.before_desert_s3,
                "special_s1": ss.before_special_s1,
                "special_s2": ss.before_special_s2,
                "special_s3": ss.before_special_s3,
            },
            "after": {
                "artic_s1": ss.after_artic_s1,
                "artic_s2": ss.after_artic_s2,
                "artic_s3": ss.after_artic_s3,
                "desert_s1": ss.after_desert_s1,
                "desert_s2": ss.after_desert_s2,
                "desert_s3": ss.after_desert_s3,
                "special_s1": ss.after_special_s1,
                "special_s2": ss.after_special_s2,
                "special_s3": ss.after_special_s3,
            },
        }
    )
    st.toast("Session state data uploaded to database", icon="📡")
except Exception as e:
    st.toast(f"Session state data not uploaded to database: {e}", icon="📡")
    pass
