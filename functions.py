import pandas as pd
import networkx as nx
from collections import defaultdict
import heapq
import functions 
import matplotlib.pyplot as plt

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




