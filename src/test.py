
from algo import get_layout, recourse, show_graph
from helper import load_dataframe, transfrom_dataframe

index = 0
size = 4

df = load_dataframe()
scalar, df_new, source = transfrom_dataframe(df, 0, size, seed=42)

graph, ts, dists = recourse(df_new, 3, source, limit=10, verbose=False)

graph.vs.find("t").delete()
coord = get_layout(df_new)
show_graph(graph, coord, source, ts)

# backtracking(graph, dists, source, size)
