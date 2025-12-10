import pandas as pd
import numpy as np
import networkx as nx
from causalnex.structure.notears import from_pandas

def create_dummy_data(n_samples=1000):
    """
    Creates a dummy dataset with known causal relationships.
    Structure: A -> B -> C
    """
    np.random.seed(42)
    
    # A is a random variable
    df = pd.DataFrame()
    df['A'] = np.random.normal(0, 1, n_samples)
    
    # B depends on A
    df['B'] = 2 * df['A'] + np.random.normal(0, 0.5, n_samples)
    
    # C depends on B
    df['C'] = -1.5 * df['B'] + np.random.normal(0, 0.5, n_samples)
    
    return df

def main():
    print("Generating synthetic data (A -> B -> C)...")
    df = create_dummy_data()
    
    print("\nLearning structure using NOTEARS...")
    # Learn the structure
    # threshold removes edges with weight below value (helps remove noise)
    sm = from_pandas(df, w_threshold=0.5)
    
    print("\nAdjacency Matrix (Causality Matrix):")
    # The adjacency matrix is the causality matrix we are looking for
    # Rows are source nodes, Columns are target nodes.
    # Value represents edge weight (strength of causal relationship).
    adj_matrix = pd.DataFrame(
        nx.adjacency_matrix(sm).todense(),
        index=sm.nodes,
        columns=sm.nodes
    )
    
    print(adj_matrix)
    
    print("\nInterpretation:")
    print("Row A, Column B: {:.2f} (Expected ~2.0)".format(adj_matrix.loc['A', 'B']))
    print("Row B, Column C: {:.2f} (Expected ~-1.5)".format(adj_matrix.loc['B', 'C']))

if __name__ == "__main__":
    main()
