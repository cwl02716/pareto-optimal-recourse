from algo import get_layout, recourse, show_graph
from main import preprocess

df, df_small, source = preprocess(0, 16)
graph, ts, dists = recourse(df_small, 3, source, limit=100000)
graph.vs.find('t').delete()
coord = get_layout(df_small)
show_graph(graph, coord, source, ts)
