from contextlib import redirect_stdout

from algo import backtracking, recourse
from main import preprocess

size = 4
df, df_small, source = preprocess(0, size)

with redirect_stdout(open("log.txt", "w")):
    graph, ts, dists = recourse(df_small, 3, source, limit=100000, verbose=True)

# graph.vs.find("t").delete()
# coord = get_layout(df_small)
# show_graph(graph, coord, source, ts)

backtracking(graph, dists, source, size)
