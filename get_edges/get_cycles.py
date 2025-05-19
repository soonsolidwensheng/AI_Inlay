# ---------------------------------------------------------------------------
# Created by: Chuanbo Wang
# Created on: 05/11/2022
# ---------------------------------------------------------------------------
# This script contains utils that return circles in a undirected graph
# ---------------------------------------------------------------------------

cyclenumber = 0


def get_cycles(edges, N):
    """
    edges: [[v1, v2], [v2, v3], ...]
    N: largest vert id in the crown mesh
    Return: cycles
    """
    # variables to be used in both functions
    graph = [[] for i in range(N)]
    cycles = [[] for i in range(N)]

    def addEdge(u, v):
        graph[u].append(v)
        graph[v].append(u)

    # add edges
    for e in edges:
        addEdge(e[0], e[1])
 
    # arrays required to color the
    # graph, store the parent of node
    color = [0] * N
    par = [0] * N

    def dfs_cycle(u, p, color: list, par: list):
        """
        Function to mark the vertex with different colors for different cycles

        Args:
            u (_type_): _description_
            p (_type_): _description_
            color (list): _description_
            par (list): _description_
        """
        global cyclenumber
    
        # already (completely) visited vertex.
        if color[u] == 2:
            return
    
        # seen vertex, but was not completely visited -> cycle detected.
        # backtrack based on parents to find the complete cycle.
        if color[u] == 1:
            v = []
            cur = p
            v.append(cur)
    
            # backtrack the vertex which are
            # in the current cycle thats found
            while cur != u:
                cur = par[cur]
                v.append(cur)
            cycles[cyclenumber] = v
            cyclenumber += 1
    
            return
    
        par[u] = p
    
        # partially visited.
        color[u] = 1
    
        # simple dfs on graph
        for v in graph[u]:
    
            # if it has not been visited previously
            if v == par[u]:
                continue
            dfs_cycle(v, u, color, par)
    
        # completely visited.
        color[u] = 2

    # find the ndoe ids in the sparse graph as starters
    node_ids = [i for i, node in enumerate(graph) if node]

    # call DFS to mark the cycles
    dfs_cycle(node_ids[1], node_ids[0], color, par)
 
    # function to print the cycles
    # printCycles()
    return [c for c in cycles if c]