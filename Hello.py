import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import geopandas as gpd
from graphs import plot_publication_counts, plot_h_index_distribution, plot_citation_impact_over_time, plot_gender_distribution, calculate_publication_trends, plot_field_growth, update_plot, plot_citation_comparison, plot_co_author_network, plot_citation_distribution_treemap, plot_global_distribution_of_professors, prepare_data, adjust_projection_scale, create_choropleth, prepare_professor_data

st.set_page_config(page_title="Professor's Dashboard", layout="wide")

st.sidebar.title("Choose a Page")
page = st.sidebar.selectbox("", ["Main Page", "Dashboard 1: Professors", "Dashboard 2: Analysis of fields of study", "Interactive graphs"])

@st.cache_data()
def load_data():
    with open('professors_data.json', 'r') as file:
        data = json.load(file)
    return data

data = load_data()
professors_names = list(data.keys())

if page == "Main Page":
    st.markdown('<style>'
                '.container { display: flex; flex-direction: column; align-items: center; }'
                '.box { width: 80%; margin-bottom: 20px; padding: 20px; }'
                '.selector-box { border-bottom: none; }'
                '</style>', unsafe_allow_html=True)

    st.title("Main Page - Professor Selection and Metrics")

    selected_professor = st.selectbox('Select a Professor', professors_names)

    if selected_professor:
        prof_info = data[selected_professor]

        st.markdown('<div class="container">', unsafe_allow_html=True)

        st.markdown('<div class="box selector-box">', unsafe_allow_html=True)
        st.write(f"Professor: {selected_professor}")
        st.write(f"H-index: {prof_info.get('Index_H', 'N/A')}")
        st.write(f"Citations: {prof_info.get('Citation_Count', 'N/A')}")
        st.write(f"Publications: {prof_info.get('Paper_Count', 'N/A')}")
        st.write(f"Country: {prof_info.get('Country', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)

        if 'Papers' in prof_info:
            publications = prof_info['Papers']
            if publications:
                most_cited_publication = max(publications, key=lambda x: x.get('Citation_Count', 0))

                st.markdown('<div class="box">', unsafe_allow_html=True)
                st.markdown('<div class="info">', unsafe_allow_html=True)
                st.write("Publication with Most Citations:")
                st.write(f"Title: {most_cited_publication.get('Title', 'N/A')}")
                st.write(f"Citations: {most_cited_publication.get('Citation_Count', 'N/A')}")
                st.write(f"Year: {most_cited_publication.get('Year_of_Publication', 'N/A')}")
                st.write(f"Venue: {most_cited_publication.get('Venue_Name', 'N/A')}")
                st.write(f"URL: {most_cited_publication.get('Paper_URL', 'N/A')}")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        if 'Co-authors' in prof_info:
            coauthors = prof_info['Co-authors']
            top_coauthors = sorted(coauthors.items(), key=lambda item: item[1], reverse=True)[:5]

            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.markdown('<div class="info">', unsafe_allow_html=True)
            st.write("Top Co-Authors:")
            st.markdown('</div>', unsafe_allow_html=True)

            for coauthor, collaborations in top_coauthors:
                st.markdown('<div class="info">', unsafe_allow_html=True)
                st.write(f"Co-author: {coauthor}, Collaborations: {collaborations}")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)



elif page == "Dashboard 1: Professors":
    st.title("Dashboard 1: Professors")

    fig_pub = plot_publication_counts(data)
    st.plotly_chart(fig_pub, use_container_width=True)

    fig_h_index = plot_h_index_distribution(data)
    st.plotly_chart(fig_h_index, use_container_width=True)

    fig_citation = plot_citation_impact_over_time(data)
    st.plotly_chart(fig_citation, use_container_width=True)

    fig_gender = plot_gender_distribution(data)
    st.plotly_chart(fig_gender, use_container_width=True)

elif page == "Dashboard 2: Analysis of fields of study":
    st.title("Dashboard 2: Analysis of fields of study")

    field_publications = calculate_publication_trends(data)
    fields_growth_rate = plot_field_growth(data)
    fig = plot_field_growth(field_publications)
    st.plotly_chart(fig, use_container_width=True)

    dropdown1 = st.selectbox('Field 1', [None] + list(fields_growth_rate.keys()))
    dropdown2 = st.selectbox('Field 2', [None] + list(fields_growth_rate.keys()))

    fig_growth = update_plot(fields_growth_rate, dropdown1, dropdown2)
    st.plotly_chart(fig_growth, use_container_width=True)

    fig_field_citations = plot_citation_comparison(field_publications)
    st.plotly_chart(fig_field_citations, use_container_width=True)

elif page == "Interactive graphs":
    st.title("Interactive graphs")

    selected_prof = st.selectbox("Select a Professor", list(data.keys()))
    fig_network = plot_co_author_network(data, selected_prof)
    st.plotly_chart(fig_network, use_container_width=True)

    fig_treemap = plot_citation_distribution_treemap(data)
    st.plotly_chart(fig_treemap, use_container_width=True)

    df_geo, professor_df = prepare_professor_data(data)

    st.title("Global Distribution of Professors")

    with st.container():
        country = st.selectbox('Select a Country:', ['All'] + sorted(df_geo['Country'].unique().tolist()))
        col1, col3, col2 = st.columns([4, 0.3, 2])

    with col1:
        if country == 'All':
            fig = create_choropleth(df_geo)
        else:
            selected_data = df_geo[df_geo['Country'] == country]
            scale = adjust_projection_scale(country)
            lat = selected_data['latitude'].values[0]
            lon = selected_data['longitude'].values[0]
            fig = create_choropleth(selected_data)
            fig.update_geos(center={"lat": lat, "lon": lon}, projection_scale=scale)

        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.empty()

    with col2:
        st.subheader("Professors List")
        if country != 'All':
            displayed_professors = professor_df[professor_df['Country'] == country]
            st.dataframe(displayed_professors[['Name']], width=600,hide_index=True)
        else:
            st.dataframe(professor_df[['Name', 'Country']], width=600, height=600, hide_index=True)
