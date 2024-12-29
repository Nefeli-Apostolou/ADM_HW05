import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import heapq
import functions 
import matplotlib.pyplot as plt
import folium
from folium import FeatureGroup, LayerControl


# Function to build a graph from a DataFrame
def build_graph(df):
    """
    Builds a directed graph from the data in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns 'Origin_airport', 'Destination_airport', and 'Distance'.

    Returns:
        networkx.DiGraph: A directed graph with edge weights representing distances.
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Origin_airport'], row['Destination_airport'], weight=row['Distance'])
    return G

# Function to prepare the DataFrame for conversion into a graph object
def build_df_for_network(df):
    """
    Prepares a DataFrame for constructing a directed graph based on flight data.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns 'Origin_airport', 'Destination_airport', and 'Distance'.

    Returns:
        pd.DataFrame: A new DataFrame with unique airport pairs and the calculated modal distance.
    """
    # Calculate the mode considering the pairs as equal regardless of the order
    df['Airport_pair'] = df.apply(
        lambda row: tuple(sorted([row['Origin_airport'], row['Destination_airport']])), axis=1
    )

    # Calculate the mode of the distance for each ordered pair
    pair_mode_distances = (
        df.groupby('Airport_pair')['Distance']
        .transform(lambda x: x.mode()[0])  # Gets the most frequent value (mode)
    )

    # Replace the distance with the calculated mode
    df['Distance_mode'] = pair_mode_distances

    # Create a new DataFrame with origin airport, destination airport, and distance
    df_util = df[['Origin_airport', 'Destination_airport', 'Distance_mode']].drop_duplicates().rename(columns={'Distance_mode': 'Distance'})

    # Remove the temporary Airport_pair column
    df_util = df_util.drop(columns=['Airport_pair'], errors='ignore')

    return df_util

# Function for calculating the degrees of each node in a graph
def calculate_degree_centrality(graph):
    """
    Calculates the in-degrees, out-degrees, total degree, and normalized total degree for each node in a graph.

    Parameters:
        graph (networkx.DiGraph): The graph representing the flight network.

    Returns:
        pd.DataFrame: DataFrame with in-degrees, out-degrees, and normalized degrees for each node.
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # Calculate out-degree
    out_degree = {node: len(list(graph.successors(node))) for node in nodes}

    # Calculate in-degree
    in_degree = {node: len(list(graph.predecessors(node))) for node in nodes}

    # Create the DataFrame
    degree_data = []
    for node in nodes:
        out_d = out_degree.get(node, 0)
        in_d = in_degree.get(node, 0)
        total_d = out_d + in_d
        normalized_d = total_d / (2 * (n - 1)) if n > 1 else 0
        degree_data.append({
            'Airport': node,
            'Out_degree': out_d,
            'In_degree': in_d,
            'Total_degree': total_d,
            'Normalized_degree': normalized_d
        })

    return pd.DataFrame(degree_data)


# Function to calculate weighted Betweenness Centrality
def calculate_betweenness_centrality(graph):
    """
    Calculates the weighted Betweenness Centrality for a directed graph.

    Parameters:
        graph (networkx.DiGraph): The graph representing the flight network.

    Returns:
        dict: Dictionary containing the Betweenness Centrality for each node.
    """
    centrality = defaultdict(float)
    nodes = list(graph.nodes)

    for s in nodes:  # For each source node
        # Initialize structures for shortest paths
        sigma = defaultdict(int)  # Number of shortest paths
        sigma[s] = 1
        dist = defaultdict(lambda: float('inf'))  # Distance initialized to infinity
        dist[s] = 0
        pred = defaultdict(list)  # Predecessors
        queue = []  # Priority queue for Dijkstra
        heapq.heappush(queue, (0, s))  # (distance, node)
        stack = []

        # Compute shortest paths (Dijkstra for weighted graphs)
        while queue:
            d, v = heapq.heappop(queue)
            if dist[v] < d:
                continue
            stack.append(v)
            for w in graph.successors(v):
                weight = graph[v][w]['weight']
                if dist[w] > dist[v] + weight:  # New shortest path found
                    dist[w] = dist[v] + weight
                    heapq.heappush(queue, (dist[w], w))
                    sigma[w] = sigma[v]
                    pred[w] = [v]
                elif dist[w] == dist[v] + weight:  # Another equally short path found
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Accumulate dependencies
        dependency = defaultdict(float)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                dependency[v] += (sigma[v] / sigma[w]) * (1 + dependency[w])
            if w != s:
                centrality[w] += dependency[w]

    # Normalization (for directed graphs)
    normalization_factor = (len(nodes) - 1) * (len(nodes) - 2)
    if normalization_factor > 0:
        for node in centrality:
            centrality[node] /= normalization_factor

    return dict(centrality)


# Function to calculate weighted Closeness Centrality
def calculate_closeness_centrality(graph):
    """
    Calculates the weighted Closeness Centrality for a directed graph.

    Parameters:
        graph (networkx.DiGraph): The graph representing the flight network.

    Returns:
        dict: Dictionary containing the Closeness Centrality for each node.
    """
    closeness = {}
    n = len(graph.nodes)  # Total number of nodes in the graph

    for node in graph.nodes:
        # Initialize distances
        dist = defaultdict(lambda: float('inf'))
        dist[node] = 0
        queue = []
        heapq.heappush(queue, (0, node))  # (distance, node)

        # Dijkstra to calculate minimum distances
        while queue:
            d, current = heapq.heappop(queue)
            if d > dist[current]:
                continue
            for neighbor in graph.successors(current):
                weight = graph[current][neighbor]['weight']
                if dist[neighbor] > dist[current] + weight:
                    dist[neighbor] = dist[current] + weight
                    heapq.heappush(queue, (dist[neighbor], neighbor))

        # Calculate the sum of distances to all other nodes
        reachable_nodes = [d for d in dist.values() if d < float('inf')]
        sum_distances = sum(reachable_nodes)

        # Calculate Closeness Centrality (avoid division by 0)
        if sum_distances > 0 and len(reachable_nodes) > 1:
            closeness[node] = (len(reachable_nodes) - 1) / sum_distances
        else:
            closeness[node] = 0.0  # No reachable nodes or isolated node

    return closeness

# Function to calculate PageRank for each node in a graph
def calculate_pagerank(graph, alpha=0.85, max_iter=100, tol=1.0e-6):
    """
    Calculates the PageRank for a directed graph.

    Parameters:
        graph (networkx.DiGraph): The graph representing the flight network.
        alpha (float): Damping factor (default: 0.85).
        max_iter (int): Maximum number of iterations (default: 100).
        tol (float): Tolerance for convergence (default: 1.0e-6).

    Returns:
        dict: Dictionary containing the PageRank of each node.
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    pagerank = {node: 1 / n for node in nodes}  # Initialize PR(v) = 1 / N
    damping_value = (1 - alpha) / n

    # Calculate out-degree using calculate_degree_centrality
    degree_df = calculate_degree_centrality(graph)
    out_degree = degree_df.set_index('Airport')['Out_degree'].to_dict()

    for iteration in range(max_iter):
        new_pagerank = {}
        for node in nodes:
            # Sum of contributions from predecessor nodes
            rank_sum = sum(
                pagerank[neighbor] / out_degree.get(neighbor, 1)  # Use the calculated out-degree
                for neighbor in graph.predecessors(node)
            )
            # Calculate the new PageRank value
            new_pagerank[node] = damping_value + alpha * rank_sum

        # Check for convergence
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in nodes)
        if diff < tol:
            break

        pagerank = new_pagerank

    return pagerank

# Function that returns the four centrality measures (Betweenness Centrality, Closeness Centrality, Degree Centrality, PageRank) for a specific airport (node in the graph)
def analyze_centrality(flight_network, airport):
    """
    Compute only the Betweenness Centrality for a given airport.

    Parameters:
        flight_network (networkx.DiGraph): The flight network as a directed graph.
        airport (str): The airport for which centrality measures are computed.

    Returns:
        dict: A dictionary with Betweenness Centrality.
    """
    # Calculate weighted Betweenness Centrality
    betweenness_centrality = calculate_betweenness_centrality(flight_network)
    # Calculate weighted Closeness Centrality
    closeness_centrality = calculate_closeness_centrality(flight_network)
    # Calculate Degree Centrality
    degree_df = calculate_degree_centrality(flight_network)
    degree_centrality = degree_df.set_index('Airport').to_dict(orient='index')
    # Calculate PageRank
    pagerank = calculate_pagerank(flight_network)

    return {
        'Betweenness Centrality': betweenness_centrality.get(airport, "Airport Not Found"),
        'Closeness Centrality': closeness_centrality.get(airport, "Airport Not Found"), 
        'Degree Centrality': degree_centrality.get(airport, {}).get('Total_degree', "Airport Not Found"),
        'PageRank': pagerank.get(airport, "Airport Not Found")
    }



# Function that visualizes the trends of the four centrality measures using histograms and returns the 5 airports (graph nodes) with the highest respective values.
def compare_centralities(flight_network):
    """
    Calculates and compares centrality values for all nodes in the graph.
    Generates histograms for the distributions of centralities.
    Returns the top 5 airports for each centrality measure with their centrality values.

    Parameters:
        flight_network (networkx.DiGraph): The directed graph representing the flight network.

    Returns:
        dict: A dictionary containing the top 5 airports and their centrality values for each centrality measure.
    """
    # Calculate centralities
    betweenness_centrality = calculate_betweenness_centrality(flight_network)
    closeness_centrality = calculate_closeness_centrality(flight_network)
    degree_df = calculate_degree_centrality(flight_network)
    degree_centrality = degree_df.set_index('Airport')['Total_degree'].to_dict()
    pagerank = calculate_pagerank(flight_network)
    
    # Create a DataFrame for centralities
    centralities_df = pd.DataFrame({
        'Betweenness Centrality': betweenness_centrality,
        'Closeness Centrality': closeness_centrality,
        'Degree Centrality': degree_centrality,
        'PageRank': pagerank 
    }).fillna(0)
    
    # Plot histograms for centrality distributions
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    colors = "#0000FF"

    for ax, centrality in zip(axes, centralities_df.columns):
        centralities_df[centrality].hist(bins=20, alpha=0.7, ax=ax, color=colors)
        ax.set_title(f'Distribution of {centrality}')
        ax.set_xlabel('Centrality Value')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
    # Get top 5 airports for each centrality measure with their values
    top_5_airports = {
        centrality: centralities_df[centrality].nlargest(5).to_dict()
        for centrality in centralities_df.columns
    }
    
    return top_5_airports




def build_transport_network(df):
    
    """
    Construct a directed graph from a DataFrame containing transportation and route information.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the route and node data.

    Returns:
    nx.DiGraph: Directed graph representing the transportation network.
    """
    
    G = nx.DiGraph() 

    for _, row in df.iterrows():
        
        origin = row['Origin_airport']
        destination = row['Destination_airport']

        # Adding nodes with attributes for source airport
        G.add_node(origin, 
                    city=row['Origin_city'], 
                    population=int(row['Origin_population']), 
                    lat=float(row['Org_airport_lat']), 
                    long=float(row['Org_airport_long']))

        # Adding nodes with attributes for destination airport
        G.add_node(destination, 
                    city=row['Destination_city'], 
                    population=int(row['Destination_population']), 
                    lat=float(row['Dest_airport_lat']), 
                    long=float(row['Dest_airport_long']))
        
        # Adding edges with attributes
        G.add_edge(origin, destination, 
                    passengers=int(row['Passengers']), 
                    flights=int(row['Flights']), 
                    seats=int(row['Seats']), 
                    weight=int(row['Distance']), 
                    fly_date=row['Fly_date'])

    return G



def compute_optimal_path(network_graph, start_city, end_city, travel_date):
    """
    Determines the optimal path (shortest route) between the start and end cities based on the provided travel date.

    Parameters:
    network_graph (nx.Graph): The graph representing the flight network.
    start_city (str): The starting city or airport.
    end_city (str): The target city or airport.
    travel_date (str): The date for which the optimal path is computed.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the start city, target city, and the optimal route.
    """

    # Filter the graph to include only edges relevant to the given travel date.
    date_filtered_graph = filter_network_by_date(network_graph, "fly_date", travel_date)

    # Identify all airports in the start and target cities within the filtered graph.
    starting_airports = [node for node, attributes in date_filtered_graph.nodes(data=True) if start_city in attributes.get("city", "")]
    destination_airports = [node for node, attributes in date_filtered_graph.nodes(data=True) if end_city in attributes.get("city", "")]

    minimal_distance = np.inf  # Set initial minimal distance to infinity.
    best_path = ''  # Initialize best path as an empty string.

    # Iterate through each airport in the start city to find the shortest route to the target airports.
    for source_airport in starting_airports:
        distance_map, predecessor_nodes = compute_shortest_paths(date_filtered_graph, source_airport)  # Compute shortest paths.

        # Evaluate paths to each target airport.
        for target_airport in destination_airports:
            current_distance = distance_map[target_airport]

            # Update minimal distance and best path if a better route is found.
            if current_distance < minimal_distance:
                minimal_distance = current_distance
                best_path = "→".join(trace_path(predecessor_nodes, target_airport, source_airport))

    # If no valid route is found, set the result to indicate no route.
    if best_path == '':
        best_path = 'No route available.'

    # Prepare a pandas DataFrame with the start, target, and route information.
    result_data = {
        'Starting_City_Airport': [start_city],
        'Destination_City_Airport': [end_city],
        'Best_Route': [best_path]
    }

    # Return the result as a DataFrame.
    return pd.DataFrame(result_data)

def filter_network_by_date(network, attribute, specific_date):
    """
    Filters the provided network graph based on a specified date. This function retains only edges that match the
    given date according to the specified attribute and returns a new graph with relevant edges and nodes.

    Parameters:
    network (nx.Graph): The original graph to filter.
    attribute (str): The edge attribute to filter on (e.g., "fly_date").
    specific_date (str): The date used to filter edges.

    Returns:
    nx.Graph: A new graph containing only edges that match the specified date.
    """
    filtered_graph = nx.Graph()  # Create a new graph for filtered edges and nodes.

    # Iterate through each edge in the original graph.
    for start_node, end_node, edge_data in network.edges(data=True):
        # If the edge matches the specified date, add it to the filtered graph.
        if edge_data.get(attribute, None) == specific_date:
            filtered_graph.add_edge(start_node, end_node, **edge_data)  # Add the edge to the filtered graph.
            filtered_graph.add_node(start_node, **network.nodes[start_node])  # Add the source node.
            filtered_graph.add_node(end_node, **network.nodes[end_node])  # Add the target node.

    return filtered_graph  # Return the filtered graph.

def compute_shortest_paths(flight_network, starting_point):
    """
    Computes the shortest paths from the starting node to all other nodes in the graph using Dijkstra's algorithm.

    Parameters:
    flight_network (nx.Graph): The graph representing the flight network with weighted edges (flight routes).
    starting_point (str): The node (airport) from which the shortest paths will be calculated.

    Returns:
    tuple: A tuple containing:
        - distances (dict): A dictionary with the shortest distance from the starting point to each node.
        - predecessors (dict): A dictionary mapping each node to its predecessor in the shortest path.
    """

    # Initialize distances and predecessors.
    distances = {starting_point: 0}  # Distance to the starting point is 0.
    predecessors = {starting_point: None}  # No predecessor for the starting point.
    priority_queue = [(0, starting_point)]  # Priority queue for selecting the node with the smallest distance.

    # Set initial distances to infinity for all other nodes.
    for node in flight_network.nodes:
        if node != starting_point:
            distances[node] = np.inf
        predecessors[node] = None

    # Process the graph using Dijkstra's algorithm.
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)  # Get the node with the smallest distance.

        # Check each neighbor of the current node.
        for neighbor, attributes in flight_network[current_node].items():
            path_distance = current_distance + attributes["weight"]

            # Update if a shorter path is found.
            if path_distance < distances[neighbor]:
                distances[neighbor] = path_distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (path_distance, neighbor))

    return distances, predecessors

def trace_path(predecessors, destination, origin):
    """
    Reconstructs the shortest path from the origin to the destination node based on the predecessors dictionary.

    Parameters:
    predecessors (dict): A dictionary mapping each node to its predecessor in the shortest path.
    destination (str): The destination node where the path ends.
    origin (str): The origin node where the path starts.

    Returns:
    list: A list of nodes representing the shortest path from origin to destination.
    """
    path = []
    current_node = destination

    # Backtrack from destination to origin using predecessors.
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors.get(current_node)

    # Reverse the path if it starts with the origin.
    if path and path[-1] == origin:
        path.reverse()
        return path
    else:
        return []
    


def analyze_graph_features(flight_network):
    # Count the number of airports (nodes) and flights (edges)
    num_nodes = flight_network.number_of_nodes()
    num_edges = flight_network.number_of_edges()

    print(f"Number of airports (nodes): {num_nodes}")
    print(f"Number of flights (edges): {num_edges}")

    # Compute the density of the graph
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    print(f"Graph density: {density:.4f}")

    # Compute degree for each airport
    degrees = dict(flight_network.degree())  # Dictionary {node: degree}

    # Plot histogram for degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(degrees.values(), bins=20, color='r', edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

    # Identify hubs (airports with degree higher than the 90th percentile)
    degree_values = list(degrees.values())
    degree_90th_percentile = np.percentile(degree_values, 90)
    hubs = [node for node, degree in degrees.items() if degree > degree_90th_percentile]

    print(f"Hubs (airports with degree > 90th percentile): {hubs}")

    # Determine if the graph is sparse or dense
    if density < 0.5:
        print("The graph is sparse.")
    else:
        print("The graph is dense.")

def summarize_graph_features(flight_network):
    # Summary dictionary to collect data
    summary = {}

    # 1. Number of nodes and edges
    num_nodes = flight_network.number_of_nodes()
    num_edges = flight_network.number_of_edges()
    summary['Number of nodes'] = num_nodes
    summary['Number of edges'] = num_edges

    # 2. Graph density
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    summary['Graph density'] = density

    # 3. Degree distribution
    degrees = dict(flight_network.degree())  # Degree distribution

    # Plot histogram for degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(degrees.values(), bins=20, color='skyblue', edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

    # 4. Identify hubs (airports with degree higher than the 90th percentile)
    degree_values = list(degrees.values())
    degree_90th_percentile = np.percentile(degree_values, 90)
    hubs = [(node, degree) for node, degree in degrees.items() if degree > degree_90th_percentile]
    hubs_df = pd.DataFrame(hubs, columns=['Airport', 'Degree']).sort_values(by='Degree', ascending=False)

    summary['Hubs'] = hubs_df

    # Print summary report
    print("\n--- Graph Summary Report ---")
    for key, value in summary.items():
        if key == 'Hubs':
            print(f"\n{key}:")
            print(value.to_string(index=False))
        else:
            print(f"{key}: {value}")

    return summary

def create_flight_network_map(df):
    """
    Create an interactive map visualizing flight network geographic spread.

    Parameters:
    -----------
    df : pandas.DataFrame
        Processed flight route data

    Returns:
    --------
    folium.Map
        Interactive map of flight network
    """
    # Calculate map center
    center_lat = df[['Org_airport_lat', 'Dest_airport_lat']].mean().mean()
    center_lon = df[['Org_airport_long', 'Dest_airport_long']].mean().mean()

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='CartoDB positron'  # Clean, minimalist map style
    )

    # Create feature groups for better layer control
    routes_layer = FeatureGroup(name='Flight Routes')
    airports_layer = FeatureGroup(name='Airports')

    # Normalize route thickness and color based on passenger volume
    max_passengers = df['Passengers'].max()
    min_passengers = df['Passengers'].min()

    # Track unique airports to avoid duplicate markers
    unique_airports = set()

    for _, route in df.iterrows():
        # Route line
        route_line = folium.PolyLine(
            locations=[
                [route['Org_airport_lat'], route['Org_airport_long']],
                [route['Dest_airport_lat'], route['Dest_airport_long']]
            ],
            # Normalize line thickness and color based on passenger volume
            weight=1 + 5 * (route['Passengers'] - min_passengers) / (max_passengers - min_passengers),
            color='blue',
            opacity=0.5,
            tooltip=(
                f"Route: {route['Origin_airport']} → {route['Destination_airport']}<br>"
                f"Cities: {route['Origin_city']} → {route['Destination_city']}<br>"
                f"Passengers: {route['Passengers']:,}"
            )
        )
        routes_layer.add_child(route_line)

        # Add origin airport marker if not already added
        if route['Origin_airport'] not in unique_airports:
            folium.CircleMarker(
                location=[route['Org_airport_lat'], route['Org_airport_long']],
                radius=3,
                popup=f"{route['Origin_airport']} - {route['Origin_city']}",
                color='red',
                fill=True,
                fillColor='red'
            ).add_to(airports_layer)
            unique_airports.add(route['Origin_airport'])

        # Add destination airport marker if not already added
        if route['Destination_airport'] not in unique_airports:
            folium.CircleMarker(
                location=[route['Dest_airport_lat'], route['Dest_airport_long']],
                radius=3,
                popup=f"{route['Destination_airport']} - {route['Destination_city']}",
                color='green',
                fill=True,
                fillColor='green'
            ).add_to(airports_layer)
            unique_airports.add(route['Destination_airport'])

    # Add layers to map
    routes_layer.add_to(m)
    airports_layer.add_to(m)

    # Add layer control
    LayerControl().add_to(m)

    return m