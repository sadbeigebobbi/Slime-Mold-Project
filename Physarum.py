#%%
import numpy as np

class PhysarumNetwork:
    def __init__(self, num_nodes, edges, source, sink, dt=0.1, initial_D=1.0):

        # num_nodes: int
        # edges: list of (i, j, length)
        # source, sink: node indices
        # dt: time step
        # initial_D: initial conductivity on existing edges

        self.num_nodes = num_nodes
        self.source = source
        self.sink = sink
        self.dt = dt
        
        # Length matrix
        self.L_len = np.zeros((num_nodes, num_nodes), dtype=float)
        for i, j, length in edges:
            self.L_len[i, j] = length
            self.L_len[j, i] = length       # undirected
            
        # Conductivity matrix
        self.D = np.zeros_like(self.L_len)
        # Initialize D only where edges exist
        self.D[self.L_len > 0] = initial_D
        
    def build_conductance(self):
        # Don't divide by zero!
        C = np.zeros_like(self.D)
        mask = self.L_len > 0
        C[mask] = self.D[mask] / self.L_len[mask]
        return C
    
    def build_laplacian(self, C):
        N = self.num_nodes
        L = np.zeros((N, N), dtype=float)
    
        # Laplacian: L_ii = sum_j C_ij, L_ij = -C_ij for i != j
        L[np.arange(N), np.arange(N)] = np.sum(C, axis=1)
        L -= C
        return L
    
    def build_rhs(self):
        # b vector: +1 at source, -1 at sink
        b = np.zeros(self.num_nodes, dtype=float)
        b[self.source] = 1.0
        b[self.sink] = -1.0
        return b
    
    def solve_pressures(self, L, b, ref_node=0):
        # Solve L p = b with p[ref_node] to fix gauge
        
        N = self.num_nodes
        
        # Indices of nodes (except reference)
        idx = [i for i in range(N) if i != ref_node]
        
        L_red = L[np.ix_(idx, idx)]
        b_red = b[idx]
        
        # Solve the system
        p_red = np.linalg.solve(L_red, b_red)
        
        # Put back
        p = np.zeros(N, dtype=float)
        p[ref_node] = 0.0
        p[idx] = p_red
        return p
    
    def step(self):
        # Perform one update step
        # Retrun max change in D to monitor for convergence
        
        C = self.build_conductance()
        L = self.build_laplacian(C)
        b = self.build_rhs()
        p = self.solve_pressures(L, b, ref_node=self.sink)
        
        # Compute flows Q_ij = C_ij * (p_i - p_j)
        dp = p[:, None] - p[None, :]
        Q = C * dp
        
        # Update D
        D_old = self.D.copy()
        growth = np.abs(Q)
        self.D += self.dt * (growth - self.D)
        
        # Fix issues with small negative numbers during computation
        self.D[self.D < 0] = 0.0
        
        # Enforce symmetry
        self.D = 0.5 * (self.D + self.D.T)
        
        max_change = np.max(np.abs(self.D - D_old))
        return max_change, Q, p
    
    def run(self, max_iters=1000, tol=1e-4, verbose=False):
        # Run until convergence or max iterations completed
        # Returns D and history of max_change
        
        history = []
        for iter in range(max_iters):
            max_change, Q, p = self.step()
            history.append(max_change)
            if verbose and iter % 50 == 0:
                print(f"Iteration {iter}, max_change = {max_change:0.6f}")
            if max_change < tol:
                if verbose:
                    print(f"Converged at iteration {iter}.")
                break
        return self.D, history

#%%
if __name__ == "__main__":
    # Simple graph:
    # 0 --- 1
    # | \   |
    # |  \  |
    # 2 --- 3
    #
    # Source = 0, sink = 3

    edges = [
        (0, 1, 1.0),
        (1, 3, 1.0),
        (0, 2, 1.0),
        (2, 3, 1.0),
        (0, 3, 1.4),  # diagonal, slightly longer than straight through 1 or 2
    ]

    phys = PhysarumNetwork(
        num_nodes=4,
        edges=edges,
        source=0,
        sink=3,
        dt=0.1,
        initial_D=1.0
    )

    D_final, history = phys.run(max_iters=1000, tol=1e-5, verbose=True)

    print("Final D matrix:")
    print(D_final)

#%%
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# Add nodes
G.add_nodes_from(range(4))

# Add weighted edges
for (i, j, length) in edges:
    weight = D_final[i, j]
    if weight > 1e-5:  # ignore edges that have fully decayed
        G.add_edge(i, j, weight=weight)

# Get positions (manual or spring layout)
pos = {0:(0,1), 1:(1,1), 2:(0,0), 3:(1,0)}  # square layout

# Draw
nx.draw(G, pos,
        with_labels=True,
        width=[G[u][v]['weight']*10 for u,v in G.edges()],
        node_color='lightblue',
        node_size=700,
        font_size=12)

plt.title("Physarum Network After Convergence")
plt.show()

