import os
import numpy as np
# https://github.com/cyang-kth/fmm
from fmm import Network, NetworkGraph, STMATCH, STMATCHConfig, FastMapMatch, FastMapMatchConfig, UBODT, \
    UBODTGenAlgorithm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--time", type=int, default=60)
parser.add_argument("--model", type=str, default="STM")
args = parser.parse_args()

network = Network("beijing_5thring_01.pbf")
graph = NetworkGraph(network)
print(graph.get_num_vertices())

if args.model == "STM":
    model = STMATCH(network, graph)
elif args.model == "FMM":
    ubodt_gen = UBODTGenAlgorithm(network, graph)
    if os.path.exists("ubodt.txt"):
        pass
    else:
        status = ubodt_gen.generate_ubodt("ubodt.txt", 4, binary=False, use_omp=True)
    ubodt = UBODT.read_ubodt_csv("ubodt.txt")

    model = FastMapMatch(network, graph, ubodt)

# wkt = "LINESTRING(0.200812146892656 2.14088983050848,1.44262005649717 2.14879943502825,3.06408898305084 2.16066384180791,3.06408898305084 2.7103813559322,3.70872175141242 2.97930790960452,4.11606638418078 2.62337570621469)"
if args.model == "STM":
    config = STMATCHConfig()
elif args.model == "FMM":
    config = FastMapMatchConfig()

config.k = 8
config.gps_error = 50
config.radius = 0.004
config.vmax = 50
config.factor = 1.5

wkt_list = []
wkt_list_simple = []
with open("real_test.trace") as fid:
    for line in fid:
        traj = line.split(":")[1]
        seq_loc = []
        seq_loc2 = []
        for i, pt in enumerate(traj.split(",")):
            lat, lon, tid = pt.split(" ")
            seq_loc.append(lat + " " + lon)
            if i == 0:
                seq_loc2.append(lat + " " + lon)
            else:
                if (float(tid) - float(tid_last)) > args.time:
                    seq_loc2.append(lat + " " + lon)
            tid_last = tid
        wkt_list.append("LINESTRING (" + ",".join(seq_loc) + ")")
        wkt_list_simple.append(("LINESTRING (" + ",".join(seq_loc2) + ")"))

acc_list = []
for i, (wkt, wkt_simple) in enumerate(zip(wkt_list, wkt_list_simple)):
    result_complete = model.match_wkt(wkt, config)
    rc = set(list(result_complete.opath))

    result_simple = model.match_wkt(wkt_simple, config)
    rs = set(list(result_simple.opath))

    if max(len(rc), len(rs)) > 0:
        acc = len(rc & rs) / max(len(rc), len(rs))
        acc_list.append(acc)

    if i % 100 == 0:
        print(i, np.mean(acc_list))
