import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


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
        px,
        quote_plus,
        sleep,
    )


@app.cell
def _(mo):
    # Get and set the query parameters
    query_params = mo.query_params()

    # Set up a state elements for user inputs selecting the data source,
    # which by default will use values from query_params
    get_domain, set_domain = mo.state(query_params.get("domain"))
    get_project, set_project = mo.state(query_params.get("project"))
    get_dataset, set_dataset = mo.state(query_params.get("dataset"))
    get_file, set_file = mo.state(query_params.get("file"))
    get_sep, set_sep = mo.state(query_params.get("sep", "Comma"))
    get_url, set_url = mo.state(query_params.get("url"))

    # Set up a state element for the cirro SDK client
    get_client, set_client = mo.state(None)
    return (
        get_client,
        get_dataset,
        get_domain,
        get_file,
        get_project,
        get_sep,
        get_url,
        query_params,
        set_client,
        set_dataset,
        set_domain,
        set_file,
        set_project,
        set_sep,
        set_url,
    )


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
def _(get_domain, mo, query_params, set_domain, tenants_by_name):
    # Let the user select which tenant to log in to (using displayName),
    # and once it is selected, remove the UI element
    def set_domain_and_query_params(selection):
        set_domain(selection["domain"])
        query_params.set("domain", selection["domain"])

    if get_domain() is None:
        domain_dropdown = mo.ui.dropdown(
            options=tenants_by_name,
            on_change=set_domain_and_query_params,
            label="Select Organization",
        )
        domain_dropdown_ui = mo.center(domain_dropdown)
    else:
        domain_dropdown = None
        domain_dropdown_ui = None

    domain_dropdown_ui
    return domain_dropdown, domain_dropdown_ui, set_domain_and_query_params


@app.cell
def _(
    AppConfig,
    CirroApi,
    DataPortal,
    DeviceCodeAuth,
    Queue,
    StringIO,
    get_client,
    get_domain,
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


    if get_client() is None:
        with mo.status.spinner("Authenticating"):
            prompt_md, cirro_login_thread, client_queue = cirro_login(get_domain())
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
        _client = client_queue.get()
        if _client is not None:
            set_client(_client)
    return


@app.cell
def _(get_client, mo):
    mo.stop(get_client() is None)
    return


@app.cell
def _(get_client, mo):
    # Set the list of projects available to the user
    if get_client() is not None:
        projects = get_client().list_projects()
        projects.sort(key=lambda i: i.name)
    else:
        mo.stop(get_client() is None)
    return (projects,)


@app.cell
def _(get_client, get_dataset, get_project, set_dataset):
    # Get the list of datasets available to the user
    if get_project() is None:
        datasets = []
    else:
        # Filter the list of datasets by type (process_id)
        datasets = [
            dataset
            for dataset in get_client().get_project_by_id(get_project()).list_datasets()
            if dataset.process_id in [
                "process-hutch-differential-expression-1_0",
                "process-hutch-differential-expression-custom-1_0",
                "differential-expression-table",
                "process-nf-core-differentialabundance-1_5"
            ]
        ]
        if get_dataset() not in [ds.id for ds in datasets]:
            set_dataset(None)
    return (datasets,)


@app.cell
def _(get_client, get_dataset, get_file, get_project, set_file):
    # Get the list of files within the selected dataset
    if get_dataset() is None:
        file_list = []
    else:
        file_list = [
            file.name
            for file in (
                get_client()
                .get_project_by_id(get_project())
                .get_dataset_by_id(get_dataset())
                .list_files()
            )
        ]
        if get_file() not in file_list:
            set_file(None)
    return (file_list,)


@app.cell
def _(
    datasets,
    file_list,
    get_dataset,
    get_file,
    get_project,
    get_sep,
    mo,
    projects,
    set_dataset,
    set_file,
    set_project,
    set_sep,
):
    # Let the user select the project, dataset, and file from Cirro containing the data to plot

    def id_to_name(obj_list: list, id: str) -> str:
        if obj_list is not None:
            return {i.id: i.name for i in obj_list}.get(id)


    def name_to_id(obj_list: list) -> dict:
        if obj_list is not None:
            return {i.name: i.id for i in obj_list}
        else:
            return {}


    source_info = (
        mo.md("""
    ## Source Data
    Project: {project}

    Dataset: {dataset} 

    Table: {file} ({sep} Separated Values)
        """)
        .batch(
            project=mo.ui.dropdown(
                value=id_to_name(projects, get_project()),
                options=name_to_id(projects),
                on_change=set_project
            ),
            dataset=mo.ui.dropdown(
                value=id_to_name(datasets, get_dataset()),
                options=name_to_id(datasets),
                on_change=set_dataset
            ),
            file=mo.ui.dropdown(
                value=get_file(),
                options=file_list,
                on_change=set_file
            ),
            sep=mo.ui.dropdown(
                options=["Comma", "Tab", "Space"],
                value=get_sep(),
                on_change=set_sep
            )
        )
    )
    source_info
    return id_to_name, name_to_id, source_info


@app.cell
def _(
    get_client,
    get_dataset,
    get_file,
    get_project,
    get_sep,
    source_info,
):
    # Read the file from Cirro
    if source_info.value["file"] is not None:
        df = (
            get_client()
            .get_project_by_id(get_project())
            .get_dataset_by_id(get_dataset())
            .list_files()
            .get_by_name(get_file())
            .read_csv(sep={
                "Comma": ",",
                "Tab": "\t",
                "Space": " ",
            }.get(get_sep, ","))
        )
    else:
        df = None
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
def _(
    get_dataset,
    get_domain,
    get_file,
    get_project,
    get_sep,
    get_url,
    mo,
    params,
    volcano_fig,
):
    if volcano_fig is not None:
        permalink = mo.md(
            (
                "[Permalink](https://fredhutch.github.io/differential-expression-viewer?{_query_params})"
                .format(
                    _query_params="&".join([
                        f"{kw}={str(val)}"
                        for kw, val in {
                            "domain": get_domain(),
                            "project": get_project(),
                            "dataset": get_dataset(),
                            "file": get_file(),
                            "sep": get_sep(),
                            "url": get_url(),
                            **params
                        }.items()
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
