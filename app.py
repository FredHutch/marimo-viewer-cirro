import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Differential Expression Viewer""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys

    # If the script is running in WASM (instead of local development mode), load micropip
    if "pyodide" in sys.modules:
        import micropip
    else:
        micropip = None
    return micropip, sys


@app.cell
async def _(micropip, mo):
    with mo.status.spinner("Loading dependencies"):
        if micropip is not None:
            print("Installing via micropip")
            await micropip.install("ssl")
            await micropip.install(["urllib3==2.3.0"])
            print('added urllib3==2.3.0')
            await micropip.install([
                "boto3==1.36.3",
                "botocore==1.36.3"
            ], verbose=True)
            print('added boto 1.36.3')
            await micropip.install(["cirro>=1.2.13"], verbose=True)
            print('added cirro')
        import cirro
    return (cirro,)


@app.cell
def _(mo):
    with mo.status.spinner("Loading dependencies"):
        from cirro import DataPortal
        from cirro.config import AppConfig, list_tenants
        from cirro import CirroApi, DataPortal
        from cirro import DataPortalProject, DataPortalDataset
        from cirro.auth.device_code import DeviceCodeAuth
    return (
        AppConfig,
        CirroApi,
        DataPortal,
        DataPortalDataset,
        DataPortalProject,
        DeviceCodeAuth,
        list_tenants,
    )


@app.cell
def _(mo):
    with mo.status.spinner("Loading dependencies"):
        from io import StringIO
        from threading import Thread
        from queue import Queue
        from time import sleep
        from typing import Dict, Optional
        import plotly.express as px
        import pandas as pd
        import numpy as np
        from functools import lru_cache
        from io import BytesIO
        import base64
        from urllib.parse import quote_plus
    return (
        BytesIO,
        Dict,
        Optional,
        Queue,
        StringIO,
        Thread,
        base64,
        lru_cache,
        np,
        pd,
        px,
        quote_plus,
        sleep,
    )


@app.cell
def _(mo):
    # Get and set the query parameters
    query_params = mo.query_params()
    return (query_params,)


@app.cell
def _(list_tenants):
    # Get the tenants (organizations) available in Cirro
    tenants_by_name = {i["displayName"]: i for i in list_tenants()}
    tenants_by_domain = {i["domain"]: i for i in list_tenants()}


    def domain_to_name(domain):
        return tenants_by_domain.get(domain, {}).get("displayName")


    def name_to_domain(name):
        return tenants_by_name.get(name, {}).get("domain")
    return (
        domain_to_name,
        name_to_domain,
        tenants_by_domain,
        tenants_by_name,
    )


@app.cell
def _(mo):
    mo.md(r"""## Load Data""")
    return


@app.cell
def _(mo, query_params):
    # Give the user the option to load data from a URL or from Cirro
    # If a Cirro domain is provided in query params, then that will be the default
    data_source_ui = mo.ui.radio(
        {
            "URL": "url",
            "Cirro": "cirro"
        },
        label="Read Data From",
        value=("Cirro" if query_params.get("domain") is not None else "URL")
    )
    data_source_ui
    return (data_source_ui,)


@app.cell
def _(data_source_ui, mo, query_params):
    # Give the user the option to load data from a URL
    if data_source_ui.value == "url":
        url_ui = mo.ui.text(
            label="Load Data from URL (CSV)",
            placeholder="--",
            value=query_params.get("url", ""),
            on_change=lambda v: query_params.set("url", v)
        )
    else:
        url_ui = None
    url_ui
    return (url_ui,)


@app.cell
def _(data_source_ui, domain_to_name, mo, query_params, tenants_by_name):
    # If Cirro is selected, let the user select which tenant to log in to (using displayName)
    if data_source_ui.value == "cirro":
        domain_ui = mo.ui.dropdown(
            options=tenants_by_name,
            value=domain_to_name(query_params.get("domain")),
            on_change=lambda i: query_params.set("domain", i["domain"]),
            label="Load Data from Cirro",
        )
    else:
        domain_ui = None

    domain_ui
    return (domain_ui,)


@app.cell
def _(mo):
    # Use a state element to manage the Cirro client object
    get_client, set_client = mo.state(None)
    return get_client, set_client


@app.cell
def _(
    AppConfig,
    CirroApi,
    DataPortal,
    DeviceCodeAuth,
    Queue,
    StringIO,
    data_source_ui,
    domain_ui,
    get_client,
    mo,
    sleep,
):
    def _cirro_login(auth_io: StringIO, base_url: str, client_queue: Queue):
        """Process used within a thread to log in to Cirro."""

        app_config = AppConfig(base_url=base_url)

        auth_info = DeviceCodeAuth(
            region=app_config.region,
            client_id=app_config.client_id,
            auth_endpoint=app_config.auth_endpoint,
            enable_cache=False,
            auth_io=auth_io
        )

        cirro_client = CirroApi(
            auth_info=auth_info,
            base_url=base_url
        )
        cirro = DataPortal(
            client=cirro_client
        )
        client_queue.put(cirro)


    def cirro_login(domain):
        if domain is None:
            return None, None, None

        auth_io = StringIO()
        client_queue = Queue()
        cirro_login_thread = mo.Thread(
            target=_cirro_login,
            args=(auth_io, domain, client_queue)
        )
        cirro_login_thread.start()

        login_string = auth_io.getvalue()

        while len(login_string) == 0 and cirro_login_thread.is_alive():
            sleep(0.1)
            login_string = auth_io.getvalue()
        if login_string is None:
            return None, None, None

        text, url = login_string.rsplit(" ", 1)
        prompt_md = f"{text} [{url}]({url})"
        return prompt_md, cirro_login_thread, client_queue


    # If the user selected Cirro as the source of data, and a domain is selected
    if get_client() is None and data_source_ui.value == "cirro" and domain_ui.value is not None:
        with mo.status.spinner("Authenticating"):
            prompt_md, cirro_login_thread, client_queue = cirro_login(domain_ui.value["domain"])
        if prompt_md is not None:
            display_link = mo.center(mo.md(prompt_md))
        else:
            display_link = None
    else:
        display_link = None
        cirro_login_thread = None
        client_queue = None

    display_link
    return (
        cirro_login,
        cirro_login_thread,
        client_queue,
        display_link,
        prompt_md,
    )


@app.cell
def _(cirro_login_thread, client_queue, set_client):
    if cirro_login_thread is not None:
        cirro_login_thread.join()
        set_client(client_queue.get())
    return


@app.cell
def _(get_client):
    client = get_client()
    return (client,)


@app.cell
def _():
    # Helper functions for dealing with lists of objects that may be accessed by id or name
    def id_to_name(obj_list: list, id: str) -> str:
        if obj_list is not None:
            return {i.id: i.name for i in obj_list}.get(id)


    def name_to_id(obj_list: list) -> dict:
        if obj_list is not None:
            return {i.name: i.id for i in obj_list}
        else:
            return {}
    return id_to_name, name_to_id


@app.cell
def _(client):
    # Set the list of projects available to the user
    if client is not None:
        projects = client.list_projects()
        projects.sort(key=lambda i: i.name)
    else:
        projects = None
    return (projects,)


@app.cell
def _(id_to_name, mo, name_to_id, projects, query_params):
    # Let the user select which project to get data from
    if projects is not None:
        project_ui=mo.ui.dropdown(
            value=id_to_name(projects, query_params.get("project")),
            options=name_to_id(projects),
            on_change=lambda i: query_params.set("project", i)
        )
    else:
        project_ui = None
    project_ui
    return (project_ui,)


@app.cell
def _(client, project_ui):
    # Get the list of datasets available to the user
    if client is None or project_ui is None or project_ui.value is None:
        datasets = None
    else:
        # Filter the list of datasets by type (process_id)
        datasets = [
            dataset
            for dataset in client.get_project_by_id(project_ui.value).list_datasets()
            if dataset.process_id in [
                "process-hutch-differential-expression-1_0",
                "process-hutch-differential-expression-custom-1_0",
                "differential-expression-table",
                "process-nf-core-differentialabundance-1_5"
            ]
        ]
    return (datasets,)


@app.cell
def _(datasets, id_to_name, mo, name_to_id, query_params):
    # Let the user select which dataset to get data from
    if datasets is not None:
        dataset_ui=mo.ui.dropdown(
            value=id_to_name(datasets, query_params.get("dataset")),
            options=name_to_id(datasets),
            on_change=lambda i: query_params.set("dataset", i)
        )
    else:
        dataset_ui = None
    dataset_ui
    return (dataset_ui,)


@app.cell
def _(client, dataset_ui, project_ui):
    # Get the list of files within the selected dataset
    if client is None or dataset_ui is None or dataset_ui.value is None:
        file_list = None
    else:
        file_list = [
            file.name
            for file in (
                client
                .get_project_by_id(project_ui.value)
                .get_dataset_by_id(dataset_ui.value)
                .list_files()
            )
        ]
    return (file_list,)


@app.cell
def _(file_list, mo, query_params):
    # Let the user select which file to get data from
    if file_list is not None:
        file_ui=mo.ui.dropdown(
            value=(query_params.get("file") if query_params.get("file") in file_list else None),
            options=file_list,
            on_change=lambda i: query_params.set("file", i)
        )
    else:
        file_ui = None
    file_ui
    return (file_ui,)


@app.cell
def _(mo, query_params):
    # Let the user provide information about the file format
    sep_ui = mo.ui.dropdown(
        ["comma", "tab", "space"],
        value=query_params.get("sep", "comma"),
        label="Field Separator"
    )
    sep_ui
    return (sep_ui,)


@app.cell
def _(
    client,
    data_source_ui,
    dataset_ui,
    file_ui,
    mo,
    pd,
    project_ui,
    sep_ui,
    url_ui,
):
    # Here is where we read in the table as a DataFrame

    # If the URL was provided
    if data_source_ui.value == "url" and url_ui is not None and url_ui.value is not None:
        df = pd.read_csv(url_ui, sep=dict(comma=",", tab="\t", space=" ")[sep_ui.value])
    elif data_source_ui.value == "cirro" and file_ui is not None and file_ui.value is not None:
        df = (
            client
            .get_project_by_id(project_ui.value)
            .get_dataset_by_id(dataset_ui.value)
            .list_files()
            .get_by_id(file_ui.value)
            .read_csv(sep=dict(comma=",", tab="\t", space=" ")[sep_ui.value])
        )
    else:
        df = None

    mo.stop(df is None)
    return (df,)


@app.cell
def _():
    # Helper when trying to guess the right column in the input table to use
    def guess_column(l, options):
        for i in l:
            if isinstance(i, str) and i.lower() in options:
                return i
    return (guess_column,)


@app.cell
def _(df, guess_column, mo, query_params):
    # Let the user select parameters for the display
    params_ui = (
        mo.md("""
    ## Display Settings

    ### Columns

    - p-values: {pval_cname} (maximum threshold: {max_pval})
    - log-fold change: {lfc_cname} (minimum threshold (abs): {min_lfc})
    - mean abundance: {abund_cname} (display on log scale {abund_log})
    - gene name: {label_cname}

    ### Labels

    - {pval_label}
    - {lfc_label}
    - {abund_label}

    ### Display

    - Theme: {theme}
    - Width: {width}
    - Height: {height}
        """)
        .batch(
            pval_cname=mo.ui.dropdown(
                options=df.columns,
                value=query_params.get(
                    'pval_cname',
                    guess_column(df.columns.values, ['pval', 'pvalue', 'p-value'])
                )
            ),
            max_pval=mo.ui.text(
                value=query_params.get("max_pval", "0.05")
            ),
            abund_cname=mo.ui.dropdown(
                options=df.columns,
                value=query_params.get(
                    'abund_cname',
                    guess_column(df.columns.values, ['basemean', 'mean_abund', 'meanabund', 'base_mean'])
                )
            ),
            abund_log=mo.ui.checkbox(
                value=query_params.get("abund_log", True)
            ),
            lfc_cname=mo.ui.dropdown(
                options=df.columns,
                value=query_params.get(
                    'lfc_cname',
                    guess_column(df.columns.values, ['logfc', 'foldchange', 'fold_change', 'log_fold_change'])
                )
            ),
            min_lfc=mo.ui.text(
                value=query_params.get("min_lfc", "2")
            ),
            label_cname=mo.ui.dropdown(
                options=df.columns,
                value=query_params.get(
                    'label_cname',
                    guess_column(df.columns.values, ['gene_id', 'geneid', 'gene_name', 'genename'])
                )
            ),
            pval_label=mo.ui.text(
                value=query_params.get("pval_label", "p-value (-log10)")
            ),
            lfc_label=mo.ui.text(
                value=query_params.get("lfc_label", "Fold Change (log2)")
            ),
            abund_label=mo.ui.text(
                value=query_params.get("abund_label", "Mean Abundance (RPKM)")
            ),
            theme=mo.ui.dropdown(
                options=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"],
                value=query_params.get("theme", "none")
            ),
            width=mo.ui.number(
                value=query_params.get("width", 800),
                start=100,
                stop=2400,
                step=10
            ),
            height=mo.ui.number(
                value=query_params.get("height", 600),
                start=100,
                stop=2400,
                step=10
            )
        )
    )
    params_ui
    return (params_ui,)


@app.cell
def _(params_ui):
    # Map the params to a single dict for brevity
    params = params_ui.value
    return (params,)


@app.cell
def _(df, mo, np, params):
    max_pval = None
    min_lfc = None
    user_feedback = ""
    if df is not None:
        with mo.status.spinner("Preparing data"):
            prepared_df = df.copy()

            # Set any 0 pvalues to a minimum value (for plotting), and apply the -log10 transform
            min_pval = df.query(f'{params["pval_cname"]} > 0')[params["pval_cname"]].min()
            prepared_df = prepared_df.assign(**{
                "_neg_log10_pval": -prepared_df[params["pval_cname"]].clip(lower=min_pval).apply(np.log10)
            })

            # Set the thresholds for maximum pvalue and minimum fold change
            if params.get("max_pval") is not None:
                try:
                    max_pval = float(params["max_pval"])
                except:
                    user_feedback += f"Could not convert p-value threshold to a float ({params['max_pval']})\n\n"

            if params.get("min_lfc") is not None:
                try:
                    min_lfc = float(params["min_lfc"])
                except:
                    user_feedback += f"Could not convert fold change threshold to a float ({params['min_lfc']})\n\n"

            prepared_df = prepared_df.assign(
                passes_threshold=prepared_df.apply(
                    lambda r: (
                        ((max_pval is None) or (r[params["pval_cname"]] <= max_pval))
                        and
                        ((min_lfc is None) or (np.abs(r[params["lfc_cname"]]) >= min_lfc))
                    ),
                    axis=1
                )
            )

    else:
        prepared_df = None

    mo.md(user_feedback)
    return max_pval, min_lfc, min_pval, prepared_df, user_feedback


@app.cell
def _(mo, volcano_fig):
    if volcano_fig is not None:
        volcano_header = mo.md("## Volcano Plot")
    else:
        volcano_header = None

    volcano_header
    return (volcano_header,)


@app.cell
def _(df, max_pval, min_lfc, mo, np, params, prepared_df, px):
    if df is not None:
        with mo.status.spinner("Making plot"):
            volcano_fig = px.scatter(
                data_frame=prepared_df,
                x=params["lfc_cname"],
                y="_neg_log10_pval",
                labels={
                    "_neg_log10_pval": params["pval_label"],
                    params["lfc_cname"]: params["lfc_label"],
                    params["abund_cname"]: params["abund_label"],
                    "passes_threshold": "Passes Threshold",
                },
                color="passes_threshold",
                hover_name=params["label_cname"],
                hover_data=[params["pval_cname"], params["abund_cname"]],
                template=params["theme"],
                width=params["width"],
                height=params["height"]
            )
            volcano_fig.update_layout(
                plot_bgcolor="white",  # Set plot background color
                paper_bgcolor="white", # Set paper background color
                xaxis=dict(
                    showgrid=True,  # Show grid lines on the x-axis
                    gridcolor='lightgray'  # Set the color of the grid lines
                ),
                yaxis=dict(
                    showgrid=True,  # Show grid lines on the y-axis
                    gridcolor='lightgray'  # Set the color of the grid lines
                )
            )
            if max_pval is not None:
                volcano_fig.add_hline(-np.log10(max_pval), line_dash="dash", line_color="grey", opacity=0.5)
            if min_lfc is not None:
                volcano_fig.add_vline(min_lfc, line_dash="dash", line_color="grey", opacity=0.5)
                volcano_fig.add_vline(-min_lfc, line_dash="dash", line_color="grey", opacity=0.5)
    else:
        volcano_fig = None

    volcano_fig
    return (volcano_fig,)


@app.cell
def _(mo, volcano_fig):
    if volcano_fig is not None:
        volcano_download = mo.download(volcano_fig.to_image(), label="Download (png)")
    else:
        volcano_download = None

    volcano_download
    return (volcano_download,)


@app.cell
def _(ma_fig, mo):
    if ma_fig is not None:
        ma_header = mo.md("## M-A Plot")
    else:
        ma_header = None

    ma_header
    return (ma_header,)


@app.cell
def _(df, min_lfc, mo, params, prepared_df, px):
    if df is not None:
        with mo.status.spinner("Making plot"):
            ma_fig = px.scatter(
                data_frame=prepared_df,
                x=params["lfc_cname"],
                y=params["abund_cname"],
                color="passes_threshold",
                labels={
                    "_neg_log10_pval": params["pval_label"],
                    params["lfc_cname"]: params["lfc_label"],
                    params["abund_cname"]: params["abund_label"],
                    "passes_threshold": "Passes Threshold",
                },
                hover_name=params["label_cname"],
                hover_data=[params["pval_cname"], params["abund_cname"]],
                log_y=params["abund_log"],
                template=params["theme"],
                width=params["width"],
                height=params["height"]
            )
            ma_fig.update_layout(
                plot_bgcolor="white",  # Set plot background color
                paper_bgcolor="white", # Set paper background color
                xaxis=dict(
                    showgrid=True,  # Show grid lines on the x-axis
                    gridcolor='lightgray'  # Set the color of the grid lines
                ),
                yaxis=dict(
                    showgrid=True,  # Show grid lines on the y-axis
                    gridcolor='lightgray'  # Set the color of the grid lines
                )
            )
            if min_lfc is not None:
                ma_fig.add_vline(min_lfc, line_dash="dash", line_color="grey", opacity=0.5)
                ma_fig.add_vline(-min_lfc, line_dash="dash", line_color="grey", opacity=0.5)

    else:
        ma_fig = None

    ma_fig
    return (ma_fig,)


@app.cell
def _(ma_fig, mo):
    if ma_fig is not None:
        ma_download = mo.download(ma_fig.to_image(), label="Download (png)")
    else:
        ma_download = None

    ma_download
    return (ma_download,)


@app.cell
def _(mo, three_dim_fig):
    if three_dim_fig is not None:
        three_dim_header = mo.md("## 3D Plot")
    else:
        three_dim_header = None

    three_dim_header
    return (three_dim_header,)


@app.cell
def _(df, mo, params, prepared_df, px):
    if df is not None:
        with mo.status.spinner("Making plot"):
            three_dim_fig = px.scatter_3d(
                data_frame=prepared_df,
                x=params["lfc_cname"],
                y="_neg_log10_pval",
                z=params["abund_cname"],
                color="passes_threshold",
                labels={
                    "_neg_log10_pval": params["pval_label"],
                    params["lfc_cname"]: params["lfc_label"],
                    params["abund_cname"]: params["abund_label"],
                    "passes_threshold": "Passes Threshold",
                },
                hover_name=params["label_cname"],
                hover_data=[params["pval_cname"], params["abund_cname"]],
                log_z=params["abund_log"],
                template=params["theme"],
                width=params["width"],
                height=params["height"]
            )

    else:
        three_dim_fig = None

    three_dim_fig
    return (three_dim_fig,)


@app.cell
def _(mo, params, query_params, volcano_fig):
    if volcano_fig is not None:
        permalink = mo.md(
            (
                "[Permalink](https://fredhutch.github.io/differential-expression-viewer?{_query_params})"
                .format(
                    _query_params="&".join([
                        f"{kw}={str(val)}"
                        for kw, val in {
                            **{
                                kw: query_params.get(kw)
                                for kw in ["url", "domain", "project", "dataset", "file"]
                            },
                            **params
                        }.items()
                        if val is not None
                    ])
                )
            )
        )
    else:
        permalink = None
    permalink
    return (permalink,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
