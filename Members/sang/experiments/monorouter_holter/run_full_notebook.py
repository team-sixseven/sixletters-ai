import nbformat

n = nbformat.read("Members/sang/notebooks/monorouter_holter_experiment.ipynb", 4)
g = {}
for i in [2, 3, 5, 7, 9, 11, 13]:
    print("\n실행 셀:", i, flush=True)
    exec(compile(n.cells[i].source, f"notebook cell {i}", "exec"), g)
