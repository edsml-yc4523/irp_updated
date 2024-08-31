"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import folium
import pandas as pd
from folium.plugins import TimestampedGeoJson
from datetime import datetime


def parse_date_range(date_str):
    """
    Parse a date range string into start and end datetime objects.

    Args:
        date_str (str): Date range string in the format 'YYYY.MM-YYYY.MM' or 'YYYY.MM-present'.

    Returns:
        tuple: A tuple containing start and end datetime objects.
    """
    if '-' in date_str:
        start, end = date_str.split('-')
        start = pd.to_datetime(start, format='%Y.%m')
        if end.lower() == "present":
            end = datetime.now()
        else:
            end = pd.to_datetime(end, format='%Y.%m')
    elif date_str.lower() == "unknown":
        start = end = pd.NaT  # Use NaT (Not a Time) for missing dates
    else:
        start = pd.to_datetime(date_str, format='%Y.%m')
        end = start
    return start, end


def process_dates(df):
    """
    Process the date range strings in a DataFrame and add start and end date columns.

    Args:
        df (pandas.DataFrame): DataFrame containing a 'date' column with date range strings.

    Returns:
        pandas.DataFrame: Updated DataFrame with 'date_start' and 'date_end' columns.
    """
    df['date_start'], df['date_end'] = zip(*df['date'].apply(parse_date_range))
    df['date_end'] = df['date_end'].fillna(datetime.now())
    df = df.sort_values(by='date_start')
    return df


def generate_map(df, map_file='map_with_timeline.html'):
    """
    Generate an interactive map with time series data using Folium and save it as an HTML file.

    Args:
        df (pandas.DataFrame): DataFrame containing geospatial data with latitude, longitude, status, and date information.
        map_file (str): File path to save the generated map as an HTML file.

    Returns:
        None
    """
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()],
                   zoom_start=14, tiles='CartoDB positron')

    status_colors = {
        'construction completed': 'green',
        'continuously expanding': 'blue',
        'demolished': 'black',
        'existing': 'gray',
        'expansion completed': 'purple',
        'newly constructed': 'orange',
        'non-existent': 'red',
        'under construction': '#008080',
    }

    location_shapes = {
        'harbor': {'radius': 7, 'fillOpacity': 1, 'weight': 2, 'fillColor': None},
        'jetty': {'radius': 3, 'fillOpacity': 0.5, 'weight': 2, 'fillColor': None},
        'resort': {'radius': 5, 'fillOpacity': 0, 'weight': 2, 'fillColor': None}
    }

    features = []
    for _, row in df.iterrows():
        status = row['status']
        color = status_colors.get(status, 'gray')
        shape_attr = location_shapes[row['true_label'].lower()]

        popup_content = (f"Label: {row['true_label']}<br>"
                         f"Status: {row['status']}<br>"
                         f"Date: {row['date']}")

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['longitude'], row['latitude']],
            },
            'properties': {
                'times': [row['date_start'].strftime('%Y-%m-%d'),
                          row['date_end'].strftime('%Y-%m-%d')],
                'popup': popup_content,
                'style': {
                    'radius': shape_attr['radius'],
                    'fillOpacity': shape_attr['fillOpacity'],
                    'fillColor': color if shape_attr['fillOpacity'] > 0 else None,
                    'stroke': True,
                    'color': color,
                    'weight': shape_attr['weight'],
                },
                'icon': 'circle'
            }
        }
        features.append(feature)

    timestamped_geojson = TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features,
        },
        period='P1M',
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD',
        time_slider_drag_update=True,
    )

    timestamped_geojson.add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; height: auto; 
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white;
                opacity: 0.8;
                padding: 5px 10px; margin: 0;">
    <h4 style="margin-top: 0; margin-bottom: 5px;">Status Legend</h4>
    <i style="background-color: gray; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Existing<br>
    <i style="background-color: red; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Non-existent<br>
    <i style="background-color: orange; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Newly Constructed<br>
    <i style="background-color: black; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Demolished<br>
    <i style="background-color: blue; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Continuously Expanding<br>
    <i style="background-color: purple; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Expansion Completed<br>
    <i style="background-color: #008080; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Under Construction<br>
    <i style="background-color: green; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></i> Construction Completed<br>
    <h4 style="margin-top: 10px; margin-bottom: 5px;">Location Types</h4>
    <svg width="12" height="12" style="margin-right: 5px;">
        <circle cx="6" cy="6" r="6" fill="black" stroke="black" stroke-width="2"></circle>
    </svg> Harbor (Fully Filled)<br>
    <svg width="12" height="12" style="margin-right: 5px;">
        <circle cx="6" cy="6" r="6" fill="black" fill-opacity="0.5" stroke="black" stroke-width="2"></circle>
    </svg> Jetty (Half Filled)<br>
    <svg width="12" height="12" style="margin-right: 5px;">
        <circle cx="6" cy="6" r="6" fill="none" stroke="black" stroke-width="2"></circle>
    </svg> Resort (No Fill)<br>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(map_file)
