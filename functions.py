import pandas as pd
import networkx as nx
from collections import defaultdict
import heapq
import functions 

# Funzione per costruire un grafo da un DataFrame
def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Origin_airport'], row['Destination_airport'], weight=row['Distance'])
    return G

def build_df_for_network(df):
    # Calcolo della moda considerando le coppie come uguali indipendentemente dall'ordine
    df['Airport_pair'] = df.apply(
        lambda row: tuple(sorted([row['Origin_airport'], row['Destination_airport']])), axis=1
    )

    # Calcola la moda della distanza per ogni coppia ordinata
    pair_mode_distances = (
        df.groupby('Airport_pair')['Distance']
        .transform(lambda x: x.mode()[0])  # Prendi il valore più frequente (moda)
    )

    # Sostituisci la distanza con la moda calcolata
    df['Distance_mode'] = pair_mode_distances

    # Crea un nuovo DataFrame con le coppie distinte e le distanze aggiornate
    df_util = df[['Origin_airport', 'Destination_airport', 'Distance_mode']].drop_duplicates().rename(columns={'Distance_mode': 'Distance'})

    # Calcolo del conteggio per ciascuna coppia distinta di origine e destinazione
    #pair_counts = df.groupby(['Origin_airport', 'Destination_airport']).size().reset_index(name='Count')
    # Aggiungo il conteggio al DataFrame distinct_airports
    #df_util = df_util.merge(pair_counts, on=['Origin_airport', 'Destination_airport'], how='left')

    # Rimuovo la colonna temporanea Airport_pair
    df_util = df_util.drop(columns=['Airport_pair'], errors='ignore')

    return df_util


def calculate_degree_centrality(graph):
    """
    Calcola i gradi entranti, uscenti e normalizzati per ciascun nodo in un grafo.

    Parameters:
        graph (networkx.DiGraph): Il grafo rappresentante la rete di voli.

    Returns:
        pd.DataFrame: DataFrame con gradi entranti, uscenti e normalizzati per ogni nodo.
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # Calcolo del grado uscente (out-degree)
    out_degree = {node: len(list(graph.successors(node))) for node in nodes}

    # Calcolo del grado entrante (in-degree)
    in_degree = {node: len(list(graph.predecessors(node))) for node in nodes}

    # Creazione del DataFrame
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


# Funzione per calcolare la Betweenness Centrality ponderata
def calculate_betweenness_centrality(graph):
    """
    Calcola la Betweenness Centrality ponderata per un grafo orientato.

    Parameters:
        graph (networkx.DiGraph): Il grafo rappresentante la rete di voli.

    Returns:
        dict: Dizionario contenente la Betweenness Centrality per ogni nodo.
    """
    centrality = defaultdict(float)
    nodes = list(graph.nodes)

    for s in nodes:  # Per ogni nodo sorgente
        # Inizializza strutture per percorsi più brevi
        sigma = defaultdict(int)  # Numero di percorsi più brevi
        sigma[s] = 1
        dist = defaultdict(lambda: float('inf'))  # Distanza inizializzata a infinito
        dist[s] = 0
        pred = defaultdict(list)  # Predecessori
        queue = []  # Priority queue per Dijkstra
        heapq.heappush(queue, (0, s))  # (distanza, nodo)
        stack = []

        # Calcolo dei percorsi più brevi (Dijkstra per grafi ponderati)
        while queue:
            d, v = heapq.heappop(queue)
            if dist[v] < d:
                continue
            stack.append(v)
            for w in graph.successors(v):
                weight = graph[v][w]['weight']
                if dist[w] > dist[v] + weight:  # Nuovo percorso più breve trovato
                    dist[w] = dist[v] + weight
                    heapq.heappush(queue, (dist[w], w))
                    sigma[w] = sigma[v]
                    pred[w] = [v]
                elif dist[w] == dist[v] + weight:  # Percorso altrettanto breve trovato
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Accumula dipendenze
        dependency = defaultdict(float)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                dependency[v] += (sigma[v] / sigma[w]) * (1 + dependency[w])
            if w != s:
                centrality[w] += dependency[w]

    # Normalizzazione (per grafi orientati)
    normalization_factor = (len(nodes) - 1) * (len(nodes) - 2)
    if normalization_factor > 0:
        for node in centrality:
            centrality[node] /= normalization_factor

    return dict(centrality)


# Funzione per calcolare la Closeness Centrality ponderata
def calculate_closeness_centrality(graph):
    """
    Calcola la Closeness Centrality ponderata per un grafo orientato.

    Parameters:
        graph (networkx.DiGraph): Il grafo rappresentante la rete di voli.

    Returns:
        dict: Dizionario contenente la Closeness Centrality per ogni nodo.
    """
    closeness = {}
    n = len(graph.nodes)  # Numero totale di nodi nel grafo

    for node in graph.nodes:
        # Inizializza le distanze
        dist = defaultdict(lambda: float('inf'))
        dist[node] = 0
        queue = []
        heapq.heappush(queue, (0, node))  # (distanza, nodo)

        # Dijkstra per calcolare le distanze minime
        while queue:
            d, current = heapq.heappop(queue)
            if d > dist[current]:
                continue
            for neighbor in graph.successors(current):
                weight = graph[current][neighbor]['weight']
                if dist[neighbor] > dist[current] + weight:
                    dist[neighbor] = dist[current] + weight
                    heapq.heappush(queue, (dist[neighbor], neighbor))

        # Calcola la somma delle distanze verso tutti gli altri nodi
        reachable_nodes = [d for d in dist.values() if d < float('inf')]
        sum_distances = sum(reachable_nodes)

        # Calcola la Closeness Centrality (evita la divisione per 0)
        if sum_distances > 0 and len(reachable_nodes) > 1:
            closeness[node] = (len(reachable_nodes) - 1) / sum_distances
        else:
            closeness[node] = 0.0  # Nessun nodo raggiungibile o nodo isolato

    return closeness


def calculate_pagerank(graph, alpha=0.85, max_iter=100, tol=1.0e-6):
    """
    Calcola il PageRank utilizzando il grado uscente calcolato con calculate_degree_centrality.

    Parameters:
        graph (networkx.DiGraph): Il grafo rappresentante la rete di voli.
        alpha (float): Fattore di damping (default: 0.85).
        max_iter (int): Numero massimo di iterazioni (default: 100).
        tol (float): Tolleranza per la convergenza (default: 1.0e-6).

    Returns:
        dict: Dizionario contenente il PageRank di ogni nodo.
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    pagerank = {node: 1 / n for node in nodes}  # Inizializza PR(v) = 1 / N
    damping_value = (1 - alpha) / n

    # Calcolo del grado uscente usando calculate_degree_centrality
    degree_df = calculate_degree_centrality(graph)
    out_degree = degree_df.set_index('Airport')['Out_degree'].to_dict()

    for iteration in range(max_iter):
        new_pagerank = {}
        for node in nodes:
            # Somma dei contributi dai nodi predecessori
            rank_sum = sum(
                pagerank[neighbor] / out_degree.get(neighbor, 1)  # Usa il grado uscente calcolato
                for neighbor in graph.predecessors(node)
            )
            # Calcolo del nuovo valore di PageRank
            new_pagerank[node] = damping_value + alpha * rank_sum

        # Verifica della convergenza
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in nodes)
        if diff < tol:
            break

        pagerank = new_pagerank

    return pagerank


def analyze_centrality(flight_network, airport):
    """
    Compute only the Betweenness Centrality for a given airport.

    Parameters:
        flight_network (networkx.DiGraph): The flight network as a directed graph.
        airport (str): The airport for which centrality measures are computed.

    Returns:
        dict: A dictionary with Betweenness Centrality.
    """
    # Calcolo della Betweenness Centrality ponderata
    betweenness_centrality_weighted = calculate_betweenness_centrality(flight_network)
    # Calcolo della Closeness Centrality ponderata
    closeness_centrality = calculate_closeness_centrality(flight_network)
    # Calcolo del Degree Centrality
    degree_df = calculate_degree_centrality(flight_network)
    degree_centrality = degree_df.set_index('Airport').to_dict(orient='index')
    # Calcolo del PageRank
    pagerank = calculate_pagerank(flight_network)

    return {
        'Betweenness Centrality': betweenness_centrality_weighted.get(airport, "Airport Not Found"),
        'Closeness Centrality': closeness_centrality.get(airport, "Airport Not Found"), 
        'Degree Centrality': degree_centrality.get(airport, {}).get('Total_degree', "Airport Not Found"),
        'PageRank': pagerank.get(airport, "Airport Not Found")
    }