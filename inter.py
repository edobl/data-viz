import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np  # Add this line to import NumPy
import geopandas as gpd

# Set the page config as the very first Streamlit command
st.set_page_config(page_title="Professor's Dashboard", layout="wide")


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Page Navigation
st.sidebar.title("Choose a Page")
page = st.sidebar.selectbox("", ["Main Page", "Dashboard 1: Professors", "Dashboard 2: Analysis of fields of study", "Interactive graphs"])

# Load the JSON data with caching
@st.cache_data()  # Use caching to load the data only once
def load_data():
    with open('professors_data.json', 'r') as file:
        data = json.load(file)
    return data

data = load_data()
professors_names = list(data.keys())

# Page Navigation
if page == "Main Page":
    st.title("Main Page - Professor Selection and Metrics")

    # Professor selection and information display
    selected_professor = st.selectbox('Select a Professor', professors_names)
    if selected_professor:
        prof_info = data[selected_professor]
        st.write(f"Professor: {selected_professor}")
        st.write(f"H-index: {prof_info.get('Index_H', 'N/A')}")
        st.write(f"Citations: {prof_info.get('Citation_Count', 'N/A')}")  # Corrected key
        st.write(f"Publications: {prof_info.get('Paper_Count', 'N/A')}")  # Corrected key
        st.write(f"Country: {prof_info.get('Country', 'N/A')}")

        # Publication with the most citations
        if 'Papers' in prof_info:
            publications = prof_info['Papers']
            if publications:
                most_cited_publication = max(publications, key=lambda x: x.get('Citation_Count', 0))
                st.write("Publication with Most Citations:")
                st.write(f"Title: {most_cited_publication.get('Title', 'N/A')}")
                st.write(f"Citations: {most_cited_publication.get('Citation_Count', 'N/A')}")
                st.write(f"Year: {most_cited_publication.get('Year_of_Publication', 'N/A')}")

                # Add more info about the top publication if available
                # Add more info about the top publication if available
                authors_details = most_cited_publication.get('Authors_Details', [])
                if authors_details:
                    if isinstance(authors_details, list):
                        authors = ', '.join(author.get('Name', 'N/A') for author in authors_details)
                        if all(author.get('Name') == 'N/A' for author in authors_details):
                            authors = 'N/A'
                    else:
                        authors = ', '.join(authors_details)
                        st.write(f"Authors: {authors}")

                st.write(f"Venue: {most_cited_publication.get('Venue_Name', 'N/A')}")
                st.write(f"URL: {most_cited_publication.get('Paper_URL', 'N/A')}")

        # Top Co-Authors (showing only the top three)
        if 'Co-authors' in prof_info:
            coauthors = prof_info['Co-authors']
            # Sorting co-authors by collaborations, descending, and picking the top three
            top_coauthors = sorted(coauthors.items(), key=lambda item: item[1], reverse=True)[:5]
            st.write("Top Co-Authors:")
            for coauthor, collaborations in top_coauthors:
                st.write(f"Co-author: {coauthor}, Collaborations: {collaborations}")

        # Publication Types Distribution (assuming 'Publication_Types' structure remains the same)
        if 'Publication_Types' in prof_info:
            publication_types = prof_info['Publication_Types']
            st.write("Publication Types Distribution:")
            for pub_type, count in publication_types.items():
                st.write(f"{pub_type}: {count}")

elif page == "Dashboard 1: Professors":
    st.title("Dashboard 1: Professors")

    # Detailed graphs for all professors
    publications_counts = [data[name].get('Paper_Count', 0) for name in professors_names]
    h_indices = [data[name].get('Index_H', 0) for name in professors_names]

    # Publication counts per professor
    sorted_data = sorted(zip(professors_names, publications_counts), key=lambda x: x[1], reverse=True)
    fig_pub = go.Figure(data=[
        go.Bar(
            x=[x[0] for x in sorted_data],
            y=[x[1] for x in sorted_data],
            marker_color='skyblue'
        )
    ])
    fig_pub.update_layout(
        title='Publication Counts per Professor',
        xaxis_tickangle=-90,
        xaxis_title="Professor",
        yaxis_title="Number of Publications",
        template='plotly_white'
    )
    fig_pub.update_layout(autosize=True)
    st.plotly_chart(fig_pub, use_container_width=True)

    # H-index distribution
    fig_h_index = go.Figure(data=[
        go.Histogram(x=h_indices, nbinsx=10, marker_color='green', opacity=0.7)
    ])
    fig_h_index.update_layout(
        title='Histogram of H-indices',
        xaxis_title="H-index",
        yaxis_title="Frequency",
        template='plotly_white',
        bargap=0.1
    )
    fig_h_index.update_layout(autosize=True)
    st.plotly_chart(fig_h_index, use_container_width=True)

    # Extracting year and citation data
    publication_years = []
    citation_counts = []
    for prof_data in data.values():
        for paper in prof_data.get('Papers', []):
            year = paper.get('Year_of_Publication')
            if year:
                publication_years.append(year)
                citation_counts.append(paper.get('Citation_Count', 0))

    # Calculate average citation count per year
    avg_citation_counts = {}
    for year, citation_count in zip(publication_years, citation_counts):
        if year in avg_citation_counts:
            avg_citation_counts[year].append(citation_count)
        else:
            avg_citation_counts[year] = [citation_count]

    years = sorted(avg_citation_counts.keys())
    avg_citations = [np.mean(avg_citation_counts[year]) for year in years]

    # Create Plotly figure
    fig_citation = go.Figure()

    fig_citation.add_trace(
        go.Scatter(
            x=years,
            y=avg_citations,
            mode='lines+markers',
            marker=dict(color='orange', size=8),
            line=dict(color='orange')
        )
    )

    fig_citation.update_layout(
        title='Citation Impact Over Time',
        xaxis_title='Year',
        yaxis_title='Average Citation Count',
        template='plotly_white'
    )

    st.plotly_chart(fig_citation)
    fig_citation.update_layout(autosize=True)
    st.plotly_chart(fig_citation, use_container_width=True)

    # %% md
    ### Gender Distribution of Authors
    # %%
    # Extracting gender counts
    gender_counts = {}
    for prof_data in data.values():
        gender = prof_data.get('Gender')
        if gender:
            if gender in gender_counts:
                gender_counts[gender] += 1
            else:
                gender_counts[gender] = 1

    # Create Plotly pie chart
    fig_gender = go.Figure(data=[go.Pie(labels=list(gender_counts.keys()),
                                        values=list(gender_counts.values()),
                                        hole=.3,  # Creates a donut-shaped pie chart
                                        hoverinfo='label+percent',
                                        textinfo='value')])

    fig_gender.update_layout(
        title='Gender Distribution of Authors',
        template='plotly_white'
    )

    fig_gender.update_layout(autosize=True)
    st.plotly_chart(fig_gender, use_container_width=True)

elif page == "Dashboard 2: Analysis of fields of study":
    st.title("Dashboard 2: Analysis of fields of study")

    # Publication Trends by Field
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

    fields_growth_rate = {}
    max_years = max(len(publications) for publications in field_publications.values())
    for field, publications in field_publications.items():
        years = sorted(publications.keys())
        counts = [publications[year] for year in years]
        growth_rate = [(counts[i] - counts[i - 1]) / counts[i - 1] * 100 if i > 0 else 0 for i in range(len(counts))]
        growth_rate += [0] * (max_years - len(growth_rate))
        fields_growth_rate[field] = growth_rate[:max_years]

    # Comparison between temporal analysis of fields growth
    import plotly.graph_objects as go

    # Initialize the figure and output widget
    fig = go.Figure()

    # Function to update the plot based on dropdown selections
    def update_plot(field1, field2):
        # Initialize a new figure to start fresh
        fig = go.Figure()

        years = list(range(1, max_years + 1))
        if field1:  # Add the first field if selected
            fig.add_trace(go.Scatter(x=years, y=fields_growth_rate[field1],
                                     mode='lines+markers', name=field1))
        if field2:  # Add the second field if selected
            fig.add_trace(go.Scatter(x=years, y=fields_growth_rate[field2],
                                     mode='lines+markers', name=field2))

        # Update layout only if at least one field is selected
        if field1 or field2:
            fig.update_layout(
                title='Temporal Analysis of Field Growth',
                xaxis_title='Year',
                yaxis_title='Growth Rate (%)',
                legend_title='Field',
                template='plotly_white'
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')

        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)


    # Dropdown widgets for field selection
    dropdown1 = st.selectbox('Field 1', [None] + list(fields_growth_rate.keys()))
    dropdown2 = st.selectbox('Field 2', [None] + list(fields_growth_rate.keys()))

    # Arrange the dropdowns and the plot output in a vertical layout
    update_plot(dropdown1, dropdown2)

    # Citation count for each field of study
    # Comparison of Citation Counts Across Fields
    field_citation_counts = {}
    for field, publications in field_publications.items():
        counts = sum(publications.values())
        field_citation_counts[field] = counts

    # Sorting the fields by citation counts in descending order
    sorted_fields = sorted(field_citation_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_field_names = [field[0] for field in sorted_fields]
    sorted_citation_counts = [field[1] for field in sorted_fields]

    # Creating the bar chart
    fig = go.Figure(data=go.Bar(x=sorted_field_names, y=sorted_citation_counts))

    # Enhancing the chart appearance
    fig.update_layout(title='Comparison of Citation Counts Across Fields',
                      xaxis_tickangle=-45,
                      xaxis_title='Field of Study',
                      yaxis_title='Total Citation Counts',
                      template='plotly_white')

    # Show the plot
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)

    # Plotting widget output for Publication Trends by Field
    plot_output = st.empty()

    # Plotting widget output
    plot_output = st.empty()


    def update_plot(selected_field):
        plot_output.empty()  # Clear the current output
        fig = go.Figure()

        # Add data for selected field if any
        if selected_field:
            years = sorted(field_publications[selected_field].keys())
            counts = [field_publications[selected_field][year] for year in years]
            fig.add_trace(go.Scatter(x=years, y=counts, mode='lines+markers', name=selected_field))

        # Update plot layout
        fig.update_layout(
            title='Publication Trends by Field',
            xaxis_title='Year',
            yaxis_title='Number of Publications',
            legend_title='Field',
            template='plotly_white'
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')

        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)


    # Dropdown widget for field selection
    dropdown = st.selectbox('Select a Field:', [None] + list(field_publications.keys()))

    # Call update_plot function when dropdown selection changes
    update_plot(dropdown)

elif page == "Interactive graphs":
    st.title("Interactive graphs")

    import json
    import networkx as nx
    import plotly.graph_objects as go
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import pandas as pd
    import plotly.express as px

    # Dropdown widget
    selected_prof = st.selectbox("Select a Professor", list(data.keys()))

    G = nx.Graph()
    professor_data = data.get(selected_prof, {})
    co_authors = professor_data.get('Co-authors', {})

    # Populate your graph with edges
    for co_author in co_authors.keys():
        G.add_edge(selected_prof, co_author)

    # Position nodes using one of the layout options in NetworkX
    pos = nx.spring_layout(G)

    fig = go.FigureWidget()

    with fig.batch_update():
        fig.data = []  # Clear existing data

        # Extract node positions for plotting
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # line breaks
            edge_y.extend([y0, y1, None])  # line breaks

        # Create edge traces
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))

        # Create node traces
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                color=[len(G.adj[node]) for node in G.nodes()],
                size=10,
                line=dict(width=2)
            ),
            text=node
        ))

        # Update layout
        fig.update_layout(
            title='<br>Network graph of co-authorships',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="This graph represents the co-author network of the selected professor.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        # Set text for nodes
        for node in G.nodes():
            fig.data[1].text = list(G.nodes())

    # Display the network graph
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)

    # Display the Treemap graph
    rows = []
    others_citations = 0
    others_professors = []

    for professor_name, details in data.items():
        total_citations = details.get("Citation_Count", 0)

        # Track the total citations of papers to subtract from the professor's node
        paper_citations_sum = 0

        if "Papers" in details:
            for paper in details["Papers"]:
                paper_title = paper["Title"]
                paper_citations = paper.get("Citation_Count", 0)

                # Append a row for each paper
                rows.append({
                    "labels": paper_title,
                    "parents": professor_name,
                    "values": paper_citations,
                })

                paper_citations_sum += paper_citations

        # Subtract paper citations from the professor's total to avoid double-counting
        remaining_citations = total_citations - paper_citations_sum

        # Check if the remaining citations are less than 100
        if remaining_citations < 100:
            others_citations += remaining_citations
            others_professors.append(professor_name)
        else:
            # Append the professor's node only if there are remaining citations
            rows.append({
                "labels": professor_name,
                "parents": "",
                "values": remaining_citations,
            })

    # Add the "Others" category if there are citations
    if others_citations > 0:
        rows.append({
            "labels": "Others",
            "parents": "",
            "values": others_citations,
        })
        for prof in others_professors:
            rows.append({
                "labels": prof,
                "parents": "Others",
                "values": data[prof].get("Citation_Count", 0),
            })

    df = pd.DataFrame(rows)
    df['values'].fillna(0, inplace=True)
    df = df[df['values'] > 0]

    fig_treemap = px.treemap(df, path=['parents', 'labels'], values='values',
                     color='values', hover_data=['labels'],
                     color_continuous_scale='RdBu',
                     title='Interactive Treemap of Citation Distribution')

    fig_treemap.update_layout(autosize=True)
    st.plotly_chart(fig_treemap, use_container_width=True)

    # Prepare the data
    professors_country = [{'Name': key, 'Country': value['Country']} for key, value in data.items()]
    professor_df = pd.DataFrame(professors_country)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Correct the country names
    professor_df['Country'] = professor_df['Country'].replace({
        'Great Britain': 'United Kingdom',
        'USA': 'United States of America',
        'Korea': 'South Korea',
        'Bulgary': 'Bulgaria',
        'Holland': 'Netherlands'
    })

    # Merge dataframes
    merged = world.merge(professor_df, how="right", left_on="name", right_on="Country")

    # Count professors per country
    professor_counts = merged.groupby('iso_a3').size().reset_index(name='Professor_Count')

    # Create the initial map
    fig = go.Figure(data=[go.Choropleth(
        locations=professor_counts['iso_a3'],
        z=professor_counts['Professor_Count'],
        text=merged['name'],
        colorscale='Blues',
        autocolorscale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title='Number of Professors'
    )])

    fig.update_layout(
        title_text='Global Distribution of Professors',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        autosize=True
    )

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="RebeccaPurple"
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        width=1600,  # Adjust width to fit your screen setup
        height=800  # Adjust height based on your needs
    )

    st.plotly_chart(fig, use_container_width=True)