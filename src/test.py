from contextlib import redirect_stdout

from algo import get_layout, recourse, show_graph
from main import preprocess

df, df_small, source = preprocess(0, 5)
with redirect_stdout(open("log.txt", "w")):
    graph, ts, dists = recourse(df_small, 3, source, limit=100000, verbose=True)
graph.vs.find("t").delete()
coord = get_layout(df_small)
show_graph(graph, coord, source, ts)
