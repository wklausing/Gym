# Import required libraries
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Initialize the Dash application
app = dash.Dash(__name__)

# Dash layout
app.layout = html.Div([
    dcc.Graph(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    ),
    html.Button('Update', id='hidden-button', style={'display': 'none'})
])

# Generate initial data
def generate_data():
    return pd.DataFrame({
        'type': np.random.choice(['CDN', 'Client'], 20, p=[0.3, 0.7]),
        'x': np.random.rand(20),
        'y': np.random.rand(20),
        'x_target': np.random.rand(20),
        'y_target': np.random.rand(20)
    })

# Function to find nearest server for each client
def drawLines(data):
    for _, row in data.iterrows():
        if row['type'] == 'CDN': continue

    servers = data[data['type'] == 'Server'][['x', 'y']]
    clients = data[data['type'] == 'Client'][['x', 'y']]
    if len(servers) == 0 or len(clients) == 0:
        return pd.DataFrame(columns=['x0', 'y0', 'x1', 'y1'])

    connections = pd.DataFrame({
        'x0': clients['x'].values,
        'y0': clients['y'].values,
        'x1': clients['x_target'].values,
        'y1': clients['y_target'].values
    })
    return connections

# Callback to update the graph
@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    print(n)
    # Update data
    data = generate_data()

    # Create lines
    connections = drawLines(data)

    # Create the Plotly figure
    fig = go.Figure()

    # Add server and client points
    for t in ['CDN', 'Client']:
        fig.add_trace(go.Scatter(x=data[data['type'] == t]['x'], 
                                 y=data[data['type'] == t]['y'], 
                                 mode='markers', 
                                 name=t))

    # Add lines for connections
    for i in range(len(connections)):
        fig.add_trace(go.Scatter(x=[connections.iloc[i]['x0'], connections.iloc[i]['x1']],
                                 y=[connections.iloc[i]['y0'], connections.iloc[i]['y1']],
                                 mode='lines',
                                 line=dict(color='blue', width=1),
                                 showlegend=False))

    fig.update_layout(title='Real-time update of Server and Client Locations with Connections',
                      xaxis_title='X Coordinate',
                      yaxis_title='Y Coordinate')

    return fig

def run_dash_app():
    app.run_server(debug=True)

# Run the app
if __name__ == '__main__':
    run_dash_app()

