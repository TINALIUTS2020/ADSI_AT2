from __future__ import annotations

# import typing

# if typing.TYPE_CHECKING:
import re

import plotly.express as px

def vis_histogram(data, col, split, relative=True, barmode="group"):
    if relative is True:
        histnorm ="percent"
    elif relative is False:
        histnorm = None

    if isinstance(data, AbstractDataset):
        data = data.data

    return px.histogram(data, x=col, color=split, barmode=barmode, histnorm=histnorm,  nbins=100)

def axis_title_get_from_parent(axis):
    axis_name = axis._plotly_name
    match = re.search(r'\d+$', axis_name)
    if match:
        number = match.group()
        number = int(number) - 1
    else:
        number = 0

    label_from_annotation = axis._parent.annotations[number].text
    label = label_from_annotation.split("=")[1].strip()

    return label

def viz_to_outcome(data, id_col, outcome_col):
    barmodes = ["group", "overlay"]

    if isinstance(id_col, str):
        id_col = [id_col]
    
    if isinstance(outcome_col, str):
        outcome_col = [outcome_col]

    splits = list(set(id_col + outcome_col))
    var_name = "columns"
    value_name = "value"

    # TODO: do the melt in the dataset class
    if isinstance(data, AbstractDataset):
        cols = [col for col in data.data.columns.to_list() if col not in splits]
        data = data.melt_internal(id_vars=splits, var_name=var_name, value_name=value_name)
    
    else:
        cols = [col for col in data.columns.to_list() if col not in splits]
        data = data.melt(id_vars=splits, var_name=var_name, value_name=value_name)

    fig = px.histogram(
        data,
        x=value_name,
        color=outcome_col[0],
        facet_col=var_name,
        facet_col_wrap=1,
        histnorm="percent",
        barmode="group",
        height=300*len(cols),
        width=1500,
        facet_col_spacing=0.03,
        facet_row_spacing=0.02,
    )
    fig.update_xaxes(matches=None, showticklabels=True, visible=True)
    fig.for_each_xaxis(lambda ax: ax.update(title=axis_title_get_from_parent(ax)))
    fig.update_yaxes(matches=None, title_text="percent_of_each_class")
    fig.update_annotations(visible=False)
    fig.show()

