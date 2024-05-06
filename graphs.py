import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np  # Add this line to import NumPy
import geopandas as gpd
import ipywidgets as widgets
import networkx as nx
from IPython.display import display, clear_output
import pandas as pd
import plotly.express as px


def update_plot(fields_growth_rate, field1, field2):
    import plotly.graph_objects as go
    fig = go.Figure()

    # Check if fields are selected and add traces accordingly
    if field1:
        fig.add_trace(go.Scatter(x=list(range(len(fields_growth_rate[field1]))),
                                 y=fields_growth_rate[field1],
                                 mode='lines+markers',
                                 name=field1))
    if field2:
        fig.add_trace(go.Scatter(x=list(range(len(fields_growth_rate[field2]))),
                                 y=fields_growth_rate[field2],
                                 mode='lines+markers',
                                 name=field2))

    # Update layout only if at least one field is selected
    if field1 or field2:
        fig.update_layout(title='Temporal Analysis of Field Growth',
                          xaxis_title='Year',
                          yaxis_title='Growth Rate (%)',
                          legend_title='Field',
                          template='plotly_white')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')

    return fig

def plot_publication_counts(data):
    professors_names = list(data.keys())
    publications_counts = [data[name].get('Paper_Count', 0) for name in professors_names]
    sorted_data = sorted(zip(professors_names, publications_counts), key=lambda x: x[1], reverse=True)
    fig = go.Figure(data=[go.Bar(x=[x[0] for x in sorted_data], y=[x[1] for x in sorted_data], marker_color='skyblue')])
    fig.update_layout(
        title='Publication Counts per Professor',
        xaxis_tickangle=-90,
        xaxis_title="Professor",
        yaxis_title="Number of Publications",
        template='plotly_white',
        autosize=True
    )
    return fig

def plot_h_index_distribution(data):
    professors_names = list(data.keys())
    h_indices = [data[name].get('Index_H', 0) for name in professors_names]
    fig = go.Figure(data=[go.Histogram(x=h_indices, nbinsx=10, marker_color='green', opacity=0.7)])
    fig.update_layout(
        title='Histogram of H-indices',
        xaxis_title="H-index",
        yaxis_title="Frequency",
        template='plotly_white',
        bargap=0.1,
        autosize=True
    )
    return fig

def plot_citation_impact_over_time(data):
    publication_years = []
    citation_counts = []
    for prof_data in data.values():
        for paper in prof_data.get('Papers', []):
            year = paper.get('Year_of_Publication')
            if year:
                publication_years.append(year)
                citation_counts.append(paper.get('Citation_Count', 0))
    avg_citation_counts = {}
    for year, citation_count in zip(publication_years, citation_counts):
        if year in avg_citation_counts:
            avg_citation_counts[year].append(citation_count)
        else:
            avg_citation_counts[year] = [citation_count]
    years = sorted(avg_citation_counts.keys())
    avg_citations = [np.mean(avg_citation_counts[year]) for year in years]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=avg_citations, mode='lines+markers', marker=dict(color='orange', size=8), line=dict(color='orange')))
    fig.update_layout(
        title='Citation Impact Over Time',
        xaxis_title='Year',
        yaxis_title='Average Citation Count',
        template='plotly_white',
        autosize=True
    )
    return fig

def plot_gender_distribution(data):
    gender_counts = {}
    for prof_data in data.values():
        gender = prof_data.get('Gender')
        if gender:
            if gender in gender_counts:
                gender_counts[gender] += 1
            else:
                gender_counts[gender] = 1
    fig = go.Figure(data=[go.Pie(labels=list(gender_counts.keys()), values=list(gender_counts.values()), hole=.3, hoverinfo='label+percent', textinfo='value')])
    fig.update_layout(
        title='Gender Distribution of Authors',
        template='plotly_white',
        autosize=True
    )
    return fig



def compute_field_publications(data):
    field_publications = {}
    for prof_data in data.values():
        for paper in prof_data.get('Papers', []):
            fields = paper.get('Fields_of_Study', [])
            year = paper.get('Year_of_Publication')
            for field in fields:
                if field not in field_publications:
                    field_publications.setdefault(field, {}).setdefault(year, 0)
                    field_publications[field][year] += 1
                else:
                    field_publications[field].setdefault(year, 0)
                    field_publications[field][year] += 1
    return field_publications

def plot_field_growth(field_publications):
    if not field_publications:  # Check if the dictionary is empty
        print("No publication data available.")
        return go.Figure()  # Return an empty figure

    fields_growth_rate = {}
    max_years = max(len(publications) for publications in field_publications.values()) if field_publications else 0

    for field, publications in field_publications.items():
        years = sorted(publications.keys())
        counts = [publications[year] for year in years]
        growth_rate = [(counts[i] - counts[i - 1]) / counts[i - 1] * 100 if i > 0 else 0 for i in range(len(counts))]
        growth_rate += [0] * (max_years - len(growth_rate))  # Normalize length of growth rates
        fields_growth_rate[field] = growth_rate

    fig = go.Figure()
    for field, growth_rate in fields_growth_rate.items():
        years = list(range(1, len(growth_rate) + 1))
        fig.add_trace(go.Scatter(
            x=years,
            y=growth_rate,
            mode='lines+markers',
            name=field
        ))

    fig.update_layout(title="Growth Rate of Publications by Field", xaxis_title="Year", yaxis_title="Growth Rate (%)")
    return fig
def calculate_publication_trends(data):
    field_publications = {}
    for prof_data in data.values():
        for paper in prof_data.get('Papers', []):
            fields = paper.get('Fields_of_Study', []) if paper.get('Fields_of_Study') is not None else []
            year = paper.get('Year_of_Publication')
            for field in fields:
                if field not in field_publications:
                    field_publications[field] = {year: 1}
                else:
                    if year in field_publications[field]:
                        field_publications[field][year] += 1
                    else:
                        field_publications[field][year] = 1
    return field_publications



def plot_citation_comparison(data):
    # Extract field publications from data
    field_publications = calculate_publication_trends(data)
    field_citation_counts = {field: sum(publications.values()) for field, publications in field_publications.items()}
    sorted_fields = sorted(field_citation_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_field_names = [field[0] for field in sorted_fields]
    sorted_citation_counts = [field[1] for field in sorted_fields]
    fig = go.Figure(data=go.Bar(x=sorted_field_names, y=sorted_citation_counts))
    fig.update_layout(
        title='Comparison of Citation Counts Across Fields',
        xaxis_tickangle=-45,
        xaxis_title='Field of Study',
        yaxis_title='Total Citation Counts',
        template='plotly_white',
        autosize=True
    )
    return fig


def plot_co_author_network(data, selected_prof):
    G = nx.Graph()
    professor_data = data.get(selected_prof, {})
    co_authors = professor_data.get('Co-authors', {})

    for co_author in co_authors.keys():
        G.add_edge(selected_prof, co_author)

    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure(data=go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                    line=dict(width=0.5, color='#888'), hoverinfo='none'),
                    layout=go.Layout(title='Network graph of co-authorships', hovermode='closest',
                                     showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                             marker=dict(showscale=True, color=[len(G.adj[node]) for node in G.nodes()],
                                         size=10, line=dict(width=2))))
    return fig


def plot_citation_distribution_treemap(data):
    rows = []
    others_citations = 0
    for professor_name, details in data.items():
        total_citations = details.get("Citation_Count", 0)
        paper_citations_sum = sum(paper.get("Citation_Count", 0) for paper in details.get("Papers", []))

        remaining_citations = total_citations - paper_citations_sum
        if remaining_citations < 100:
            others_citations += remaining_citations
        else:
            rows.append({"labels": professor_name, "parents": "", "values": remaining_citations})
        for paper in details.get("Papers", []):
            if paper.get("Citation_Count", 0) > 0:
                rows.append({"labels": paper["Title"], "parents": professor_name, "values": paper.get("Citation_Count", 0)})

    if others_citations > 0:
        rows.append({"labels": "Others", "parents": "", "values": others_citations})

    df = pd.DataFrame(rows)
    fig = px.treemap(df, path=['parents', 'labels'], values='values',
                     color='values', hover_data=['labels'],
                     color_continuous_scale='RdBu', title='Interactive Treemap of Citation Distribution')
    return fig


def prepare_data(data):
    professors_country = [{'Name': key, 'Country': value['Country']} for key, value in data.items()]
    professor_df = pd.DataFrame(professors_country)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    professor_df['Country'] = professor_df['Country'].replace({
        'USA': 'United States of America'
    })

    merged = world.merge(professor_df, how="right", left_on="name", right_on="Country")

    professor_counts = merged.groupby('iso_a3').size().reset_index(name='Professor_Count')
    df_geo = merged.merge(professor_counts, on='iso_a3', how='left')
    return df_geo

def adjust_projection_scale(country):
    scales = {
        'India': 6, 'Chile': 3, 'United States of America': 2,
        'Israel': 15, 'Albania': 20, 'Netherlands': 20, 'Germany': 15,
        'Italy': 15, 'Greece': 20,
        'Turkey': 15, 'Iran': 12, 'United Kingdom': 15, 'Bulgaria': 20,
    }
    return scales.get(country, 5)  # Default scale


def plot_global_distribution_of_professors(data):
    # Data preparation
    professors_country = [{'Name': key, 'Country': value['Country']} for key, value in data.items()]
    professor_df = pd.DataFrame(professors_country)

    # Load geographical data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    professor_df['Country'] = professor_df['Country'].replace({'USA': 'United States of America'})
    merged = world.merge(professor_df, how="right", left_on="name", right_on="Country")

    # Aggregate data by country
    professor_counts = merged.groupby('iso_a3').size().reset_index(name='Professor_Count')
    df_geo = merged.merge(professor_counts, on='iso_a3', how='left')

    # Load a built-in color scale
    original_scale = px.colors.sequential.Blues

    # Cut the first 30% of the colors
    cut_point = int(len(original_scale) * 0.4)
    new_scale = original_scale[cut_point:]

    # Create a custom continuous color scale
    custom_scale = [(i / (len(new_scale) - 1), color) for i, color in enumerate(new_scale)]

    # Create choropleth
    fig = go.Figure(data=[go.Choropleth(
        locations=df_geo['iso_a3'],  # Use the ISO A3 country codes
        z=df_geo['Professor_Count'],  # Data to be visualized
        text=df_geo['Country'],  # Hover text
        colorscale=custom_scale,  # Color scale
        autocolorscale=False,
        marker_line_color= "darkblue",
        marker_line_width=0.5,
        colorbar_title='Number of Professors',
        zmin=0,
        zmax=42,
        colorbar=dict(
            len=0.3,
            lenmode='fraction'
        )
    )])

    fig.update_layout(
        title_text='Global Distribution of Professors',
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    return fig


def prepare_professor_data(data):
    professors_country = [{'Name': key, 'Country': value['Country']} for key, value in data.items()]
    professor_df = pd.DataFrame(professors_country)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    professor_df['Country'] = professor_df['Country'].replace({'USA': 'United States of America'})
    merged = world.merge(professor_df, how="right", left_on="name", right_on="Country")
    professor_counts = merged.groupby('iso_a3').size().reset_index(name='Professor_Count')
    df_geo = merged.merge(professor_counts, on='iso_a3', how='left')

    country_coords = {
        "Albania": (41.153332, 20.168331),
        "Greece": (39.074208, 21.824312),
        "Italy": (41.871940, 12.567380),
        "India": (20.595164, 78.963060),
        "Israel": (31.046051, 34.851612),
        "United States of America": (37.090240, -95.712891),
        "South Korea": (35.907757, 127.766922),
        "Germany": (51.165691, 10.451526),
        "Turkey": (38.963745, 35.243322),
        "United Kingdom": (55.378051, -3.435973),
        "Bulgaria": (42.733883, 25.485830),
        "Netherlands": (52.132633, 5.291266),
        "Chile": (-35.675147, -71.542969),
        "Iran": (32.453814, 48.348936)
    }
    for country, coords in country_coords.items():
        df_geo.loc[df_geo['Country'] == country, 'latitude'] = coords[0]
        df_geo.loc[df_geo['Country'] == country, 'longitude'] = coords[1]

    return df_geo, professor_df


def create_choropleth(df_geo):
    fig = go.Figure(data=[go.Choropleth(
        locations=df_geo['iso_a3'],
        z=df_geo['Professor_Count'],
        text=df_geo['Country'],
        colorscale='Blues',
        marker_line_color="darkblue",
        marker_line_width=0.5,
        colorbar_title='Number of Professors',
        colorbar=dict(
            len=0.3,
            lenmode='fraction'
        )
    )])
    fig.update_layout(
        width=2000,
        height=800,
        title_text='Global Distribution of Professors',
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig