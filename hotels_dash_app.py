import base64

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, dash_table, exceptions
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# font styles for title and labels
font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
numerical_columns = ['lead_time',
                     'stays_in_weekend_nights',
                     'stays_in_week_nights',
                     'adults',
                     'children',
                     'babies',
                     'previous_cancellations',
                     'booking_changes',
                     'days_in_waiting_list',
                     'adr',
                     'required_car_parking_spaces',
                     'total_of_special_requests']


# functions

def data_cleaning_hotels(hotels_df):
    # pre-processing
    dropped_columns = []
    droppable_columns = []
    if (hotels_df.isnull().sum().sum() != 0) or (hotels_df.isna().sum().sum() != 0):
        droppable_columns = hotels_df.columns[hotels_df.isna().any()].tolist()
        # Replace null values in the "children" column with 0
        hotels_df['children'].fillna(0, inplace=True)
        dropped_columns = hotels_df.columns[hotels_df.isna().any()].tolist()
        hotels_df.dropna(axis=1, inplace=True)

    return hotels_df


def outlier_detection_iqr(hotels):
    # Calculate Q1, Q3, and IQR for female height
    Q1 = hotels["lead_time"].quantile(0.25)
    Q3 = hotels["lead_time"].quantile(0.75)
    IQR = Q3 - Q1
    # Find the range for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    cleaned_lead_time = hotels["lead_time"][
        (hotels["lead_time"] >= lower_bound) & (hotels["lead_time"] <= upper_bound)]
    # Identify and remove outliers
    hotels_filtered = hotels[~hotels["lead_time"].isin(
        hotels[(hotels["lead_time"] < lower_bound) | (hotels["lead_time"] > upper_bound)]["lead_time"])]

    return hotels_filtered


# load dataset
hotels = pd.read_csv("hotel_bookings.csv")
hotels = data_cleaning_hotels(hotels)
# Outlier detection and removal
hotels = outlier_detection_iqr(hotels)

my_app = Dash('My app', external_stylesheets=external_stylesheets)
my_app.layout = html.Div(
    [html.Header(children=[html.H1("Hotel Booking Demand",
                                   style={"textAlign": "center", "color": "black"}),
                           html.H3("Information Visualization - Final Term Project",
                                   style={"textAlign": "center", "color": "#393e6f"}), ]),
     dcc.Tabs(id="iv-ftp-tabs",
              children=[
                  dcc.Tab(label="About Dataset", value="t1"),
                  dcc.Tab(label="Preprocessing", value="t2"),
                  dcc.Tab(label="Dynamic Plots", value="t3"),
                  dcc.Tab(label="Interactive Plots", value="t4"),
                  dcc.Tab(label="Tables", value="t5"),

              ], value='t1'),
     html.Div(id="layout")
     ],
    style={"background-color": "#9599bf"},
)


@my_app.callback(
    Output(component_id='layout', component_property='children'),
    Input(component_id='iv-ftp-tabs', component_property='value')
)
def update_layout(tab):
    if tab == 't1':
        return tab1_layout
    elif tab == 't2':
        return tab2_layout
    elif tab == 't3':
        return tab3_layout
    elif tab == 't4':
        return tab4_layout
    elif tab == 't5':
        return tab5_layout


# Tab 1
# Load image of the neural network diagram
img_path1 = 'hotels.png'
img_path2 = 'statistics.png'
encoded_img1 = base64.b64encode(open(img_path1, 'rb').read())
encoded_img2 = base64.b64encode(open(img_path2, 'rb').read())
CategoricalData = {
    'hotel': 'Type of hotel (Resort Hotel or City Hotel).',
    'is_canceled': 'Indicates if the booking was canceled (1) or not (0).',
    'meal': 'Type of meal booked (e.g., Bed & Breakfast, Half board).',
    'is_repeated_guest': 'Indicates if the booking name is from a repeated guest (1) or not (0).',
    'reserved_room_type': 'Code of room type reserved.',
    'assigned_room_type': 'Code for the type of room assigned to the booking.',
    'deposit_type': 'Indication if the customer made a deposit.',
    'reservation_status': 'Reservation last status (e.g., Canceled, Check-Out, No-Show).'
}

NumericalData = {
    'lead_time': 'Number of days between booking and arrival.',
    'stays_in_weekend_nights': 'Number of weekend nights the guest stayed.',
    'stays_in_week_nights': 'Number of week nights the guest stayed.',
    'adults': 'Number of adults in the booking.',
    'children': 'Number of children in the booking.',
    'babies': 'Number of babies in the booking.',
    'previous_cancellations': 'Number of previous bookings canceled by the customer.',
    'previous_bookings_not_canceled': 'Number of previous bookings not canceled by the customer.',
    'booking_changes': 'Number of changes/amendments made to the booking.',
    'days_in_waiting_list': 'Number of days the booking was in the waiting list before confirmation.',
    'adr': 'Average Daily Rate (average transaction amount per night).',
    'required_car_parking_spaces': 'Number of car parking spaces required by the customer.',
    'total_of_special_requests': 'Number of special requests made by the customer.'
}

TimeSeriesData = {
    'reservation_status_date': 'Date at which the last status was set.',
    'arrival_date_year': 'Year of arrival date.',
    'arrival_date_month': 'Month of arrival date.',
    'arrival_date_day_of_month': 'Day of arrival date.'
}
# Convert numericalData into table rows
num_table_rows = [html.Tr([html.Td(col), html.Td(desc)]) for col, desc in NumericalData.items()]
# Convert categoricalData into table rows
cat_table_rows = [html.Tr([html.Td(col), html.Td(desc)]) for col, desc in CategoricalData.items()]
# Convert timeSeriesData into table rows
time_table_rows = [html.Tr([html.Td(col), html.Td(desc)]) for col, desc in TimeSeriesData.items()]
# Create the numerical table
table_header = [html.Thead(html.Tr([html.Th("Feature"), html.Th("Description")]))]
num_table_body = [html.Tbody(num_table_rows)]
# Create the categorical table
cat_table_body = [html.Tbody(cat_table_rows)]
# Create the time series table
time_table_body = [html.Tbody(time_table_rows)]
tab1_layout = html.Div([
    html.H2('Hotel Booking Demand Dataset', style={"textAlign": "center", "color": "black"}),
    html.Br(),
    html.H5('Description:', style={"margin-left": "30px", "color": "black"}),
    html.Br(),
    # html.Img
    html.Img(src='data:image/png;base64,{}'.format(encoded_img1.decode()),
             style={'width': '30%', 'display': 'block', 'margin': 'auto'}),
    html.Br(),
    html.P(
        [
            "The dataset has 119390 ",
            html.Span(
                "observations ",
                id="tooltip-target1",
                style={"textDecoration": "underline", "cursor": "pointer"}, ),
            " and 32 ",
            html.Span("features.",
                      id="tooltip-target2",
                      style={"textDecoration": "underline", "cursor": "pointer"}, ),
        ], style={"textAlign": "center", "color": "black"}),
    html.Br(),
    html.P(
        'It is real-time booking information in city and resort hotels between the 1st of July of 2015 and the 31st of August 2017.',
        style={"textAlign": "center", "color": "black"}),
    html.Br(),
    html.P('Below are some of the features and you can find the raw dataset here: ',
           style={"textAlign": "center", "color": "black"}),
    html.Br(),
    html.P(
        html.A("https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/",
               href="https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/",
               target="_blank", style={"color": "black"}),
        style={"textAlign": "center", "color": "black"}),
    dbc.Tooltip(
        "rows",
        target="tooltip-target1",
        placement='bottom',
    ),
    dbc.Tooltip(
        "columns",
        target="tooltip-target2",
        placement='bottom',
    ),
    html.Br(),
    html.H5('Features of the dataset:', style={"margin-left": "30px", "color": "black"}),
    html.Br(),
    dcc.Dropdown(id="drop1",
                 options=[
                     {"label": "Numerical Data", "value": "numericalData"},
                     {"label": "Categorical Data", "value": "categoricalData"},
                     {"label": "Time-Series Data", "value": "timeSeriesData"},
                 ], value="numericalData"),
    html.Br(),
    html.Div(id='output1', style={"display": "flex", "justify-content": "center"}),
    html.Br(),
    html.H5('Dataset Statistics:', style={"margin-left": "30px", "color": "black"}),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(encoded_img2.decode()),
             style={'width': '70%', 'display': 'block', 'margin': 'auto'}),
    # dbc.Tab   le(table_header + num_table_body, bordered=True),
    # html.Br(),
    # dbc.Table(table_header + cat_table_body, bordered=True),
    # html.Br(),
    # dbc.Table(table_header + time_table_body, bordered=True),
], style={"margin": "0", "padding": "15px"})


@my_app.callback(
    Output(component_id='output1', component_property='children'),
    Input(component_id='drop1', component_property='value')
)
def update_drop(drop):
    if drop == "numericalData":
        return dbc.Table(table_header + num_table_body, bordered=True, color='light')
    elif drop == "categoricalData":
        return dbc.Table(table_header + cat_table_body, bordered=True, color='light')
    elif drop == "timeSeriesData":
        return dbc.Table(table_header + time_table_body, bordered=True, color='light')


# Tab 2

tab2_layout = html.Div([
    html.H3('Data Preprocessing'),
    html.Br(),
    dcc.RadioItems(
        id='preprocessing-radio',
        options=[
            {'label': 'Before Preprocessing', 'value': 'Before Preprocessing'},
            {'label': 'After Preprocessing', 'value': 'After Preprocessing'},
        ],
        value='Before Preprocessing',
    ),
    html.Br(),
    dcc.Graph(id='boxplot-graph'),
    html.Br(),
    html.Button("Download Data", id="btn_download",
                style={
                    'position': 'absolute',
                    'top': '10px',  # Adjust the distance from the top as needed
                    'right': '10px',  # Adjust the distance from the right as needed
                    'zIndex': '1000'  # Ensure the button is above other elements
                }),

    # Hidden component that will trigger the download
    dcc.Download(id="download-dataframe-csv"),
])


@my_app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_download", "n_clicks"),
    prevent_initial_call=True,
)
def download_data(n_clicks):
    hotels_filtered = outlier_detection_iqr(hotels)
    if n_clicks:
        return dcc.send_data_frame(hotels_filtered.to_csv, "hotel_dataset.csv", index=False)


# Callback to update box plots
@my_app.callback(
    Output('boxplot-graph', 'figure'),
    [Input('preprocessing-radio', 'value')]
)
def update_boxplots(selected_option):
    if selected_option == 'Before Preprocessing':
        fig = px.box(hotels, y='lead_time', title='Boxplot of lead_time (Before Preprocessing)')
    else:
        Q1 = hotels["lead_time"].quantile(0.25)
        Q3 = hotels["lead_time"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_lead_time = hotels["lead_time"][
            (hotels["lead_time"] >= lower_bound) & (hotels["lead_time"] <= upper_bound)]
        fig = px.box(y=cleaned_lead_time, title='Boxplot of Cleaned lead_time (After Preprocessing)')

    return fig


# Tab 3

tab3_layout = html.Div([
    html.H3('Dynamic plots based on selection'),
    html.Br(),
    # Slider for total_of_special_requests
    html.Label("Select Total number of special requests:"),
    html.Br(),
    dcc.Slider(
        id='total-slider',
        min=hotels['total_of_special_requests'].min(),
        max=hotels['total_of_special_requests'].max(),
        step=1,
        value=hotels['total_of_special_requests'].min(),
        marks={i: str(i) for i in
               range(hotels['total_of_special_requests'].min(), hotels['total_of_special_requests'].max() + 1)},
        tooltip={'placement': 'bottom'},
    ),
    html.Br(),

    # Loading for total_of_special_requests plot
    dcc.Loading(
        id="loading-total",
        type="circle",
        children=[
            html.Label("Lead Time Distribution"),
            dcc.Graph(id='lead-time-plot'),
        ],
    ),
    html.Br(),

    # Radio buttons for hotel type
    html.Label("Select Hotel Type"),
    html.Br(),
    dcc.RadioItems(
        id='hotel-radio',
        options=[
            {'label': 'City Hotel', 'value': 'City Hotel'},
            {'label': 'Resort Hotel', 'value': 'Resort Hotel'},
        ],
        value='City Hotel',
    ),
    html.Br(),
    # Loading for ADR plot
    dcc.Loading(
        id="loading-adr",
        type="circle",
        children=[
            # Label for ADR plot
            html.Label("Average Daily Rate (ADR)"),
            dcc.Graph(id='adr-plot'),
        ],
    ),
    html.Br(),
    # Checklist for customer types
    # Label for Checklist
    html.Label("Select Customer Types"),
    html.Br(),
    dcc.Checklist(
        id='customer-checklist',
        options=[
            {'label': customer_type, 'value': customer_type} for customer_type in hotels['customer_type'].unique()
        ],
        value=hotels['customer_type'].unique()[:2].tolist(),
    ),
    html.Br(),
    # Loading for customer types count plot
    dcc.Loading(
        id="loading-customer-count",
        type="circle",
        children=[
            html.Label("Customer Types Count"),
            dcc.Graph(id='customer-count-plot'),
        ],
    ),
    html.Br(),
    # RangeSlider for selecting months
    html.Label('Select Range of Months'),
    html.Br(),
    dcc.RangeSlider(
        id='month-slider',
        marks={i: month for i, month in enumerate(
            ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
             'November', 'December'])},
        min=0,
        max=11,
        step=1,
        value=[0, 11],
    ),
    html.Br(),
    # Loading for cancellations plot

    dcc.Loading(
        id="loading-cancellations",
        type="circle",
        children=[
            html.Label('Cancellations per Month'),
            dcc.Graph(id='cancellations-plot'),
        ],
    ),
])


# Define callback to update lead_time plot
@my_app.callback(
    Output('lead-time-plot', 'figure'),
    [Input('total-slider', 'value')]
)
def update_lead_time_plot(selected_special_requests):
    filtered_data = hotels[hotels['total_of_special_requests'] == selected_special_requests]
    fig = px.histogram(filtered_data, x='lead_time', nbins=20, title='Lead Time Distribution')
    return fig


# Define callback to update ADR plot
@my_app.callback(
    Output('adr-plot', 'figure'),
    Input('hotel-radio', 'value')
)
def update_adr_plot(selected_hotel):
    filtered_data = hotels[hotels['hotel'] == selected_hotel]
    fig = px.histogram(filtered_data, x='adr', nbins=20, title=f'ADR Distribution for {selected_hotel}')
    return fig


@my_app.callback(
    Output('customer-count-plot', 'figure'),
    Input('customer-checklist', 'value')
)
def update_customer_count_plot(selected_customer_types):
    if not selected_customer_types:
        raise exceptions.PreventUpdate

    # Filter the DataFrame based on selected customer types
    filtered_data = hotels[hotels['customer_type'].isin(selected_customer_types)]

    # Create a count plot
    fig = px.histogram(filtered_data, x='customer_type', text_auto=True)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(title_text='Customer Types Count', xaxis_title='Customer Type', yaxis_title='Count')

    return fig


# Define callback to update cancellations plot
@my_app.callback(
    Output('cancellations-plot', 'figure'),
    Input('month-slider', 'value')
)
def update_cancellations_plot(selected_months):
    # Map slider values to month names if necessary
    # Assuming you have a function or a dictionary to do this mapping
    month_mapping = {i: month for i, month in enumerate(
        ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December'], start=1)}
    selected_months_range = range(selected_months[0] + 1, selected_months[1] + 2)
    filtered_months = [month_mapping[month] for month in selected_months_range]

    filtered_data = hotels[hotels['arrival_date_month'].isin(filtered_months)]
    cancellations_count = filtered_data[filtered_data['is_canceled'] == 1].groupby('arrival_date_month')[
        'is_canceled'].count()

    fig = px.bar(cancellations_count, x=cancellations_count.index, y=cancellations_count,
                 title='Cancellations per Month')
    fig.update_layout(xaxis_title='Month', yaxis_title='Number of Cancellations')

    return fig


# Tab 4

tab4_layout = html.Div([
    html.H3('Interactive Plots'),
    html.Br(),
    html.Label('Select type of plot'),
    dcc.Dropdown(
        id='plot-type-dropdown',
        options=[
            {'label': 'Line Plot', 'value': 'line'},
            {'label': 'Bar Plot', 'value': 'bar'},
            {'label': 'Pie Chart', 'value': 'pie'},
            {'label': 'Strip Plot', 'value': 'strip'},
            {'label': 'Box Plot', 'value': 'box'},
            {'label': 'Violin Plot', 'value': 'violin'},
            {'label': 'Bubble Plot', 'value': 'bubble'}
        ],
        value='line',
    ),
    html.Br(),
    html.Label('Select numerical column'),
    dcc.Dropdown(
        id='numerical-data-dropdown',
        options=[{'label': key, 'value': key} for key in NumericalData.keys()],
        value='lead_time',
    ),
    html.Br(),
    html.Label('Select categorical column'),
    dcc.Dropdown(
        id='categorical-data-dropdown',
        options=[{'label': key, 'value': key} for key in CategoricalData.keys()],
        value='hotel',
    ),
    html.Br(),
    dcc.Graph(id='selected-plot'),
])


@my_app.callback(
    Output('selected-plot', 'figure'),
    [
        Input('plot-type-dropdown', 'value'),
        Input('numerical-data-dropdown', 'value'),
        Input('categorical-data-dropdown', 'value')
    ]
)
def update_selected_plot(plot_type, numerical_column, categorical_column):
    if plot_type and numerical_column and categorical_column:
        # Create the appropriate plot based on selected values
        if plot_type == 'line':
            fig = px.line(hotels, x=categorical_column, y=numerical_column,
                          title=f'{plot_type} Plot between {numerical_column} and {categorical_column}')
        elif plot_type == 'bar':
            fig = px.histogram(hotels, x=categorical_column, y=numerical_column,
                         title=f'{plot_type} Plot between {numerical_column} and {categorical_column}')
        elif plot_type == 'pie':
            fig = px.pie(hotels, names=categorical_column,
                         title=f'{plot_type} Chart of{categorical_column}')
        elif plot_type == 'strip':
            fig = px.strip(hotels, x=categorical_column, y=numerical_column,
                           title=f'{plot_type} Plot between {numerical_column} and {categorical_column}')
        elif plot_type == 'box':
            fig = px.box(hotels, x=categorical_column, y=numerical_column,
                         title=f'{plot_type} Plot between {numerical_column} and {categorical_column}')
        elif plot_type == 'violin':
            fig = px.violin(hotels, x=categorical_column, y=numerical_column,
                            title=f'{plot_type} Plot between {numerical_column} and {categorical_column}')
        elif plot_type == 'bubble':
            fig = px.scatter(hotels, x=categorical_column, y=numerical_column, size='pop', size_max=60,
                             title=f'{plot_type} Plot between {numerical_column} and {categorical_column}')
            fig.update_traces(mode='markers+lines')

        return fig

    return px.scatter(title='Select Plot Type, Numerical Column, and Categorical Column')


# Tab 5 layout
# Load image
img_path3 = 'acb_table.png'
img_path4 = 'stays_table.png'
img_path5 = 'booking_table.png'
encoded_img3 = base64.b64encode(open(img_path3, 'rb').read())
encoded_img4 = base64.b64encode(open(img_path4, 'rb').read())
encoded_img5 = base64.b64encode(open(img_path5, 'rb').read())
tab5_layout = html.Div([
    html.H4('Statistics of Numerical Columns:'),
    html.Br(),
    # html.Img
    html.Img(src='data:image/png;base64,{}'.format(encoded_img3.decode()),
             style={'width': '30%', 'display': 'block', 'margin': 'auto'}),
    html.Br(),
    # html.Img
    html.Img(src='data:image/png;base64,{}'.format(encoded_img4.decode()),
             style={'width': '30%', 'display': 'block', 'margin': 'auto'}),
    html.Br(),
    # html.Img
    html.Img(src='data:image/png;base64,{}'.format(encoded_img5.decode()),
             style={'width': '30%', 'display': 'block', 'margin': 'auto'}),
])

my_app.run_server(port=8026, host='0.0.0.0')
