 from flask import Flask, render_template, request, jsonify
import networkx as nx
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import random
import json
import logging
import numpy as np

app = Flask(__name__)


logging.basicConfig(level=logging.DEBUG)

G = None
node_positions = None
closed_edges = []
path = None
dynamic_traffic = False
show_detailed_labels = False
current_start = None
current_end = None

def create_city_graph():
    global G, node_positions
    G = nx.Graph()
    intersections = [str(i) for i in range(1, 51)]
    G.add_nodes_from(intersections)
    
    for i in range(1, 50):
        u, v = str(i), str(i + 1)
        weight = random.randint(5, 15) if 20 <= i <= 30 else random.randint(1, 10)
        traffic = 0 if random.random() < 0.2 else random.uniform(1.0, 3.0)
        G.add_edge(u, v, weight=weight, traffic=traffic, base_weight=weight)
    
    for i in range(1, 41, 10):
        u, v = str(i), str(i + 10)
        weight = random.randint(5, 15) if 20 <= i <= 30 else random.randint(1, 10)
        traffic = 0 if random.random() < 0.2 else random.uniform(1.0, 3.0)
        G.add_edge(u, v, weight=weight, traffic=traffic, base_weight=weight)
    
    G.add_edge('50', '30', weight=random.randint(1, 10), traffic=random.uniform(1.0, 3.0), base_weight=weight)
    G.add_edge('50', '6', weight=random.randint(1, 10), traffic=random.uniform(1.0, 3.0), base_weight=weight)
    
    current_edges = len(G.edges())
    additional_edges = max(0, 68 - current_edges)
    for _ in range(additional_edges):
        u, v = random.sample(intersections, 2)
        if not G.has_edge(u, v):
            weight = random.randint(1, 10)
            traffic = 0 if random.random() < 0.2 else random.uniform(1.0, 3.0)
            G.add_edge(u, v, weight=weight, traffic=traffic, base_weight=weight)
    
    for u, v in G.edges():
        edge_data = G[u][v]
        if 'weight' not in edge_data:
            edge_data['weight'] = edge_data.get('base_weight', 1)
        if 'traffic' not in edge_data:
            edge_data['traffic'] = 1.0
        if 'base_weight' not in edge_data:
            edge_data['base_weight'] = edge_data['weight']
    
    node_positions = nx.spring_layout(G, iterations=50)
    logging.debug(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    logging.debug(f"Initial node positions computed: {len(node_positions)} nodes")
    return G

create_city_graph()

def update_traffic_levels(G):
    for u, v in G.edges():
        if random.random() < 0.3:
            new_traffic = 0 if random.random() < 0.2 else random.uniform(1.0, 3.0)
            G[u][v]['traffic'] = round(new_traffic, 1)
            logging.debug(f"Updated traffic for edge {u}-{v}: {G[u][v]['traffic']}")
    logging.debug("Traffic levels updated")

def redistribute_traffic(G, path):
    pass

def find_shortest_path(G, start, end):
    try:
        path = nx.shortest_path(G, source=start, target=end, weight='weight')
        length = nx.shortest_path_length(G, source=start, target=end, weight='weight')
        edge_times = [(u, v, int(G[u][v]['weight'])) for u, v in zip(path[:-1], path[1:])]
        
        has_high_traffic = False
        for u, v in zip(path[:-1], path[1:]):
            traffic = G[u][v]['traffic']
            if traffic > 2.5:
                has_high_traffic = True
            logging.debug(f"Edge {u}-{v} in path: weight={G[u][v]['weight']}, traffic={traffic}")
        
        if has_high_traffic:
            G_temp = G.copy()
            for u, v in G_temp.edges():
                if G_temp[u][v]['traffic'] > 2.5:
                    G_temp[u][v]['weight'] = 1000
                    logging.debug(f"Penalized edge {u}-{v} due to high traffic: new weight=1000")
            try:
                alt_path = nx.shortest_path(G_temp, source=start, target=end, weight='weight')
                alt_length = nx.shortest_path_length(G_temp, source=start, target=end, weight='weight')
                alt_edge_times = [(u, v, int(G[u][v]['weight'])) for u, v in zip(alt_path[:-1], alt_path[1:])]
                logging.debug(f"Alternative path found: {alt_path}, total length={alt_length}")
                if alt_length < length:
                    delay = alt_length - length
                    return alt_path, int(alt_length), alt_edge_times, path, int(length), delay
                else:
                    logging.debug("Alternative path not better; sticking with original path")
            except nx.NetworkXNoPath:
                logging.debug("No alternative path found; using original path despite high traffic")
        
        return path, int(length), edge_times, None, None, None
    except nx.NetworkXNoPath:
        return None, float('inf'), [], None, None, None

def generate_plotly_graph(G, path=None, path_edge_weights=None):
    global node_positions
    pos = node_positions
    logging.debug(f"Using stored positions for {len(pos)} nodes")
    
    edge_traces = []
    edge_label_traces = []
    path_edges = set(zip(path, path[1:])) if path else set()
    for u, v in G.edges():
        traffic = G[u][v]['traffic']
        if traffic == 0:
            color = 'gray'
            dash = 'dash'
        else:
            color = 'green' if traffic < 1.5 else 'orange' if traffic < 2.5 else 'red'
            dash = 'solid'
        width = 2.0
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color, dash=dash),
            hoverinfo='text',
            hovertemplate='Base Weight: %{customdata[0]} min<br>Weight: %{text} min<br>Traffic: %{customdata[1]:.1f}',
            text=[int(G[u][v]["weight"])] * 3,
            customdata=[[int(G[u][v]["base_weight"]), G[u][v]["traffic"]] for _ in range(3)],
            mode='lines',
            showlegend=False
        )
        edge_traces.append(trace)
        
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2 + 0.05
        weight = int(G[u][v]['weight'])
        traffic = round(G[u][v]['traffic'], 1)
        label_text = f"W:{weight}, T:{traffic}" if show_detailed_labels else f"{weight}"
        logging.debug(f"Edge {u}-{v}: show_detailed_labels={show_detailed_labels}, label_text={label_text}")
        label_trace = go.Scatter(
            x=[mid_x],
            y=[mid_y],
            text=[label_text],
            mode='text',
            textposition='middle center',
            textfont=dict(size=12, color='yellow'),
            hoverinfo='none',
            showlegend=False
        )
        edge_label_traces.append(label_trace)
    
    logging.debug(f"Created {len(edge_traces)} edge traces and {len(edge_label_traces)} edge label traces")
    
    if path:
        path_edges = list(zip(path, path[1:]))
        path_x, path_y = [], []
        for u, v in path_edges:
            path_x.extend([pos[u][0], pos[v][0], None])
            path_y.extend([pos[u][1], pos[v][1], None])
        path_trace = go.Scatter(
            x=path_x,
            y=path_y,
            line=dict(width=4, color='#00008B'),
            mode='lines',
            name='Ambulance Route'
        )
        edge_traces.append(path_trace)
        logging.debug("Added path trace")
    
    node_x, node_y = [], []
    node_text = []
    for node in G.nodes():
        if node not in pos:
            pos[node] = (0, 0)
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(size=15, color='cyan'),
        hoverinfo='text',
        hovertemplate='Node: %{text}',
        textfont=dict(size=12, color='white'),
        customdata=list(G.nodes()),
        name='Intersections'
    )
    logging.debug(f"Created node trace with {len(node_x)} nodes")
    
    closure_x, closure_y = [], []
    for u, v in closed_edges:
        if G.has_edge(u, v):
            continue
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        closure_x.append(mid_x)
        closure_y.append(mid_y)
    closure_trace = go.Scatter(
        x=closure_x,
        y=closure_y,
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Road Closure'
    ) if closure_x else None
    logging.debug(f"Created closure trace with {len(closure_x)} closures")
    
    ambulance_trace = None
    if path:
        x, y = pos[path[0]]
        ambulance_trace = go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(size=12, color='red', symbol='triangle-up'),
            name='Ambulance',
            hoverinfo='text',
            hovertemplate=f'Ambulance at Node {path[0]}'
        )
        logging.debug("Added ambulance trace")
    
    traffic_legend_traces = [
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=2, color='gray', dash='dash'),
            name='Traffic: Empty'
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=2, color='green'),
            name='Traffic: Low (< 1.5)'
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=2, color='orange'),
            name='Traffic: Medium (1.5-2.5)'
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=2, color='red'),
            name='Traffic: High (> 2.5)'
        )
    ]
    logging.debug("Added traffic legend traces")
    
    traces = edge_traces + edge_label_traces + [trace for trace in [node_trace, closure_trace, ambulance_trace] if trace is not None] + traffic_legend_traces
    logging.debug(f"Total traces: {len(traces)}")
    
    layout = go.Layout(
        title=dict(
            text='Urban Traffic Optimization and Simulation',
            font=dict(size=20, color='#39FF14'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=50),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        clickmode='event+select',
        autosize=True,
        legend=dict(
            x=1,
            y=0,
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='white')
        )
    )
    
    fig = go.Figure(data=traces, layout=layout)
    logging.debug(f"Plotly figure created with show_detailed_labels={show_detailed_labels}")
    return fig

@app.route('/')
def index():
    global G, path
    fig = generate_plotly_graph(G, path)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    logging.debug(f"Generated graph_json: {graph_json[:500]}...")
    return render_template('index.html', graph_json=graph_json, intersections=list(G.nodes()))

@app.route('/get_nodes', methods=['GET'])
def get_nodes():
    return jsonify({'nodes': list(G.nodes())})

@app.route('/add_node', methods=['POST'])
def add_node():
    global G, path, node_positions
    new_node_id = request.form['new_node_id']
    if not new_node_id:
        return jsonify({'error': 'Please provide a node ID.'})
    if new_node_id in G.nodes():
        return jsonify({'error': f'Node {new_node_id} already exists.'})
    
    G.add_node(new_node_id)
    if node_positions:
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        new_x = random.uniform(min_x, max_x)
        new_y = random.uniform(min_y, max_y)
        node_positions[new_node_id] = (new_x, new_y)
    else:
        node_positions[new_node_id] = (0, 0)
    path = None
    fig = generate_plotly_graph(G, path)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({
        'message': f'Node {new_node_id} added at position ({node_positions[new_node_id][0]:.2f}, {node_positions[new_node_id][1]:.2f}).',
        'graph_json': graph_json,
        'path': None,
        'travel_time': None
    })

@app.route('/remove_node', methods=['POST'])
def remove_node():
    global G, path, node_positions, closed_edges, current_start, current_end
    node_id = request.form['node_id']
    if not node_id:
        return jsonify({'error': 'Please provide a node ID.'})
    if node_id not in G.nodes():
        return jsonify({'error': f'Node {node_id} does not exist.'})
    
    G.remove_node(node_id)
    if node_id in node_positions:
        del node_positions[node_id]
    if path and node_id in path:
        path = None
    closed_edges = [(u, v) for u, v in closed_edges if u != node_id and v != node_id]
    
    if current_start == node_id:
        current_start = None
    if current_end == node_id:
        current_end = None
    
    if (current_start and current_end and current_start != current_end and
        current_start in G.nodes() and current_end in G.nodes()):
        path, length, edge_times, _, _, _ = find_shortest_path(G, current_start, current_end)
        logging.debug(f"Recomputed path after removing node: {path}, length: {length}")
        if path:
            path_info = " -> ".join([f"{u} ({t})" for u, v, t in edge_times] + [path[-1]])
            fig = generate_plotly_graph(G, path, edge_times)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': f'Node {node_id} and its edges removed.',
                'graph_json': graph_json,
                'nodes': list(G.nodes()),
                'path': path_info,
                'travel_time': f"{length} min"
            })
    
    fig = generate_plotly_graph(G, path)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({
        'message': f'Node {node_id} and its edges removed.',
        'graph_json': graph_json,
        'nodes': list(G.nodes()),
        'path': None,
        'travel_time': None
    })

@app.route('/add_edge', methods=['POST'])
def add_edge():
    global G, path, current_start, current_end
    node1 = request.form['node1']
    node2 = request.form['node2']
    try:
        weight = int(request.form['weight'])
        if weight <= 0:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'Weight must be a positive integer.'})
    try:
        traffic = float(request.form['traffic'])
        if traffic < 0 or traffic > 3:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'Traffic must be between 0 and 3.'})
    
    if not node1 or not node2:
        return jsonify({'error': 'Please select both nodes.'})
    if node1 == node2:
        return jsonify({'error': 'Nodes must be different.'})
    if node1 not in G.nodes() or node2 not in G.nodes():
        return jsonify({'error': 'One or both nodes do not exist.'})
    if G.has_edge(node1, node2):
        return jsonify({'error': f'Edge between {node1} and {node2} already exists.'})
    
    G.add_edge(node1, node2, weight=weight, traffic=traffic, base_weight=weight)
    

    if (current_start and current_end and current_start != current_end and
        current_start in G.nodes() and current_end in G.nodes()):
        path, length, edge_times, _, _, _ = find_shortest_path(G, current_start, current_end)
        logging.debug(f"Recomputed path after adding edge: {path}, length: {length}")
        if path:
            path_info = " -> ".join([f"{u} ({t})" for u, v, t in edge_times] + [path[-1]])
            fig = generate_plotly_graph(G, path, edge_times)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': f'Edge between {node1} and {node2} added with weight {weight} and traffic {traffic}.',
                'graph_json': graph_json,
                'path': path_info,
                'travel_time': f"{length} min"
            })
        else:
            path = None
            fig = generate_plotly_graph(G, path)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': f'Edge between {node1} and {node2} added with weight {weight} and traffic {traffic}. No path exists between {current_start} and {current_end}.',
                'graph_json': graph_json,
                'path': None,
                'travel_time': None
            })
    
    path = None
    fig = generate_plotly_graph(G, path)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({
        'message': f'Edge between {node1} and {node2} added with weight {weight} and traffic {traffic}.',
        'graph_json': graph_json,
        'path': None,
        'travel_time': None
    })

@app.route('/check_edge_weight', methods=['POST'])
def check_edge_weight():
    node1 = request.form['node1']
    node2 = request.form['node2']
    if not node1 or not node2:
        return jsonify({'error': 'Please select both nodes.'})
    if node1 == node2:
        return jsonify({'error': 'Nodes must be different.'})
    if node1 not in G.nodes() or node2 not in G.nodes():
        return jsonify({'error': 'One or both nodes do not exist.'})
    if not G.has_edge(node1, node2):
        return jsonify({'error': f'No edge exists between {node1} and {node2}.'})
    
    edge_data = G[node1][node2]
    return jsonify({
        'weight': int(edge_data['weight']),
        'traffic': round(edge_data['traffic'], 1)
    })

@app.route('/calculate_path', methods=['POST'])
def calculate_path():
    global G, path, current_start, current_end
    start = request.form['start']
    end = request.form['end']
    if not start or not end:
        return jsonify({
            'error': 'Please select both source and destination.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    if start == end:
        return jsonify({
            'error': 'Source and destination must be different.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    
    if start not in G.nodes() or end not in G.nodes():
        return jsonify({
            'error': 'One or both nodes do not exist.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    
    current_start = start
    current_end = end
    
    path, length, edge_times, alt_path, alt_length, delay = find_shortest_path(G, start, end)
    logging.debug(f"Calculated path: {path}, length: {length}")
    
    if path:
        path_info = " -> ".join([f"{u} ({t})" for u, v, t in edge_times] + [path[-1]])
        fig = generate_plotly_graph(G, path, edge_times)
        graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        response = {
            'message': 'Path calculated successfully.',
            'path': path_info,
            'travel_time': f"{length} min",
            'graph_json': graph_json
        }
        
        if alt_path and alt_length:
            logging.debug(f"Alternative path: {alt_path}")
            for u, v in zip(alt_path[:-1], alt_path[1:]):
                if not G.has_edge(u, v):
                    logging.error(f"Edge {u}-{v} in alt_path does not exist in G")
                else:
                    logging.debug(f"Edge {u}-{v} weight: {G[u][v]['weight']}")
            alt_path_info = " -> ".join([f"{u} ({int(G[u][v]['weight'])})" for u, v in zip(alt_path[:-1], alt_path[1:])] + [alt_path[-1]])
            response['message'] = f"If the ambulance takes the shortest path with high traffic, it could face a delay of {delay} min. Recommended path: {path_info} ({length} min)."
        
        return jsonify(response)
    
    fig = generate_plotly_graph(G, None)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({
        'error': 'No path exists between the selected nodes.',
        'graph_json': graph_json,
        'path': None,
        'travel_time': None
    })

@app.route('/close_road', methods=['POST'])
def close_road():
    global G, closed_edges, path, current_start, current_end
    node1 = request.form['node1']
    node2 = request.form['node2']
    if not node1 or not node2:
        return jsonify({
            'error': 'Please select both nodes for road closure.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    if node1 == node2:
        return jsonify({
            'error': 'Nodes for road closure must be different.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    if not G.has_edge(node1, node2):
        return jsonify({
            'error': f'Road between {node1} and {node2} does not exist or is already closed.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    
    G.remove_edge(node1, node2)
    closed_edges.append((node1, node2))
    
    if (current_start and current_end and current_start != current_end and
        current_start in G.nodes() and current_end in G.nodes()):
        path, length, edge_times, _, _, _ = find_shortest_path(G, current_start, current_end)
        logging.debug(f"Recomputed path after closing road: {path}, length: {length}")
        if path:
            path_info = " -> ".join([f"{u} ({t})" for u, v, t in edge_times] + [path[-1]])
            fig = generate_plotly_graph(G, path, edge_times)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': f'Road between {node1} and {node2} closed.',
                'graph_json': graph_json,
                'path': path_info,
                'travel_time': f"{length} min"
            })
        else:
            path = None
            fig = generate_plotly_graph(G, path)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': f'Road between {node1} and {node2} closed. No path exists between {current_start} and {current_end}.',
                'graph_json': graph_json,
                'path': None,
                'travel_time': None
            })
    
    fig = generate_plotly_graph(G, path)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({
        'message': f'Road between {node1} and {node2} closed.',
        'graph_json': graph_json,
        'path': None,
        'travel_time': None
    })

@app.route('/change_weight', methods=['POST'])
def change_weight():
    global G, path, current_start, current_end
    node1 = request.form['node1']
    node2 = request.form['node2']
    weight = request.form['weight']
    try:
        new_weight = int(weight)
        if new_weight <= 0:
            raise ValueError
    except ValueError:
        return jsonify({
            'error': 'Weight must be a positive integer.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    if not node1 or not node2:
        return jsonify({
            'error': 'Please select both nodes for weight change.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    if node1 == node2:
        return jsonify({
            'error': 'Nodes for weight change must be different.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    if not G.has_edge(node1, node2):
        return jsonify({
            'error': f'No edge exists between {node1} and {node2}.',
            'graph_json': None,
            'path': None,
            'travel_time': None
        })
    
    G[node1][node2]['base_weight'] = new_weight
    G[node1][node2]['weight'] = new_weight
    G[node1][node2]['traffic'] = 1.0
    logging.debug(f"Updated edge {node1}-{node2}: base_weight={new_weight}, weight={G[node1][node2]['weight']}, traffic={G[node1][node2]['traffic']}")
    

    if (current_start and current_end and current_start != current_end and
        current_start in G.nodes() and current_end in G.nodes()):
        path, length, edge_times, _, _, _ = find_shortest_path(G, current_start, current_end)
        logging.debug(f"Recomputed path after changing weight: {path}, length: {length}")
        if path:
            path_info = " -> ".join([f"{u} ({t})" for u, v, t in edge_times] + [path[-1]])
            fig = generate_plotly_graph(G, path, edge_times)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': f'Weight of edge {node1}-{node2} updated to {new_weight}.',
                'graph_json': graph_json,
                'path': path_info,
                'travel_time': f"{length} min"
            })
        else:
            path = None
            fig = generate_plotly_graph(G, path)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': f'Weight of edge {node1}-{node2} updated to {new_weight}. No path exists between {current_start} and {current_end}.',
                'graph_json': graph_json,
                'path': None,
                'travel_time': None
            })
    
    fig = generate_plotly_graph(G, path)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({
        'message': f'Weight of edge {node1}-{node2} updated to {new_weight}.',
        'graph_json': graph_json,
        'path': None,
        'travel_time': None
    })

@app.route('/toggle_labels', methods=['POST'])
def toggle_labels():
    global show_detailed_labels, G, path
    show_detailed_labels = not show_detailed_labels
    logging.debug(f"Toggled detailed labels to {show_detailed_labels}")
    fig = generate_plotly_graph(G, path)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({
        'message': f'Detailed labels {"enabled" if show_detailed_labels else "disabled"}.',
        'graph_json': graph_json,
        'path': None,
        'travel_time': None
    })

@app.route('/update_traffic', methods=['POST'])
def update_traffic():
    global G, path, current_start, current_end
    update_traffic_levels(G)
    
    if (current_start and current_end and current_start != current_end and
        current_start in G.nodes() and current_end in G.nodes()):
        path, length, edge_times, _, _, _ = find_shortest_path(G, current_start, current_end)
        logging.debug(f"Recomputed path after updating traffic: {path}, length: {length}")
        if path:
            path_info = " -> ".join([f"{u} ({t})" for u, v, t in edge_times] + [path[-1]])
            fig = generate_plotly_graph(G, path, edge_times)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': 'Traffic updated.',
                'graph_json': graph_json,
                'path': path_info,
                'travel_time': f"{length} min"
            })
        else:
            path = None
            fig = generate_plotly_graph(G, path)
            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return jsonify({
                'message': 'Traffic updated. No path exists between {current_start} and {current_end}.',
                'graph_json': graph_json,
                'path': None,
                'travel_time': None
            })
    
    fig = generate_plotly_graph(G, path)
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({
        'message': 'Traffic updated.',
        'graph_json': graph_json,
        'path': None,
        'travel_time': None
    })

if __name__ == '__main__':
    app.run(debug=True)
