import argparse
from ModelInterface import ModelInterface
import neuralmodels as nm
import pickle
import random
import torch
from torch_geometric.nn.pool import EdgePooling
from cluster_pool import ClusterPooling

import sys

# Define arg parse
def parser_function() -> argparse.ArgumentParser:
    """Define the command line arguments."""
    parser = argparse.ArgumentParser(
        description=("Runs selected Graph Neural Network on specified task for specified data set"),
        )
    parser.add_argument("--model", default="GCN", type=str, help="Model type. Default: GCN. Available: GCN-Diehl, GUNET, GUNET-Diehl")
    parser.add_argument("--dataset", default="PROTEIN", type=str, help="Data set name. Default: PROTEIN. Available: ")
    parser.add_argument("--task", default="node", type=str, help="Classification task for the model. Default: node. Available: graph")
    return parser

def build_model(parser: argparse) -> ModelInterface:
    data = None
    test_set_ids = None
    labels = None
    type = nm.GraphConvPoolNN
    task_type_node = True
    if parser.dataset == "PROTEIN":
        #print(os.listdir("Datasets"))
        with open("Datasets/PROTEINS/PROTEINS_full/PROTEINS.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], prodict["node_label_tensor_tuple"])]
                labels = prodict["node_label_range"]
            
        test_set_ids = [16, 21, 22, 24, 27, 34, 80, 106, 124, 125, 127, 137, 146, 147, 153, 157, 187, 194, 195, 202, 204, 223, 246, 267, 270, 285, 290, 310, 315, 317, 334, 337, 347, 349, 389, 391, 407, 410, 421, 436, 439, 448, 467, 471, 503, 515, 531, 533, 541, 543, 546, 554, 578, 598, 605, 608, 614, 615, 619, 635, 652, 666, 676, 681, 695, 696, 714, 719, 722, 726, 745, 753, 769, 777, 780, 805, 808, 810, 812, 813, 814, 816, 835, 848, 852, 856, 882, 888, 891, 898, 902, 919, 934, 936, 946, 948, 965, 984, 992, 1016, 1017, 1019, 1024, 1033, 1061, 1067, 1079, 1089, 1094, 1099, 1112]
    elif parser.dataset == "COLLAB":
        with open("Datasets/COLLAB/COLLAB.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
        test_set_ids = [5, 20, 24, 36, 38, 43, 58, 70, 98, 108, 115, 120, 127, 138, 142, 151, 164, 165, 167, 174, 175, 178, 181, 203, 212, 215, 236, 239, 241, 243, 267, 275, 278, 293, 312, 325, 336, 351, 352, 360, 365, 369, 396, 397, 400, 401, 405, 408, 430, 432, 441, 451, 460, 462, 480, 492, 495, 499, 510, 525, 532, 534, 537, 540, 546, 561, 567, 575, 579, 589, 596, 629, 660, 662, 670, 674, 690, 698, 713, 718, 722, 733, 736, 751, 760, 764, 765, 784, 799, 804, 811, 830, 831, 841, 844, 856, 859, 865, 873, 874, 877, 889, 907, 923, 944, 947, 957, 965, 998, 1005, 1013, 1014, 1022, 1029, 1038, 1050, 1058, 1069, 1110, 1115, 1135, 1160, 1164, 1166, 1172, 1183, 1200, 1210, 1211, 1217, 1218, 1222, 1259, 1265, 1266, 1268, 1277, 1288, 1294, 1300, 1308, 1312, 1315, 1333, 1337, 1339, 1341, 1366, 1369, 1394, 1412, 1414, 1418, 1430, 1455, 1470, 1472, 1479, 1500, 1516, 1529, 1532, 1537, 1613, 1625, 1637, 1642, 1651, 1653, 1656, 1687, 1688, 1690, 1696, 1700, 1701, 1707, 1709, 1712, 1713, 1723, 1727, 1732, 1738, 1741, 1742, 1754, 1760, 1761, 1764, 1777, 1794, 1803, 1805, 1815, 1817, 1830, 1845, 1855, 1879, 1885, 1892, 1895, 1900, 1904, 1908, 1917, 1921, 1927, 1928, 1929, 1944, 1945, 1947, 1957, 2005, 2020, 2024, 2025, 2041, 2052, 2066, 2073, 2086, 2122, 2134, 2145, 2150, 2186, 2190, 2191, 2192, 2193, 2202, 2212, 2229, 2232, 2233, 2236, 2244, 2247, 2261, 2268, 2272, 2278, 2287, 2294, 2296, 2300, 2306, 2307, 2316, 2318, 2321, 2323, 2330, 2331, 2335, 2336, 2344, 2353, 2362, 2406, 2409, 2414, 2426, 2432, 2438, 2443, 2462, 2471, 2474, 2491, 2507, 2515, 2516, 2526, 2548, 2557, 2562, 2574, 2577, 2579, 2593, 2603, 2611, 2624, 2627, 2628, 2644, 2648, 2662, 2663, 2683, 2684, 2687, 2696, 2705, 2711, 2713, 2720, 2722, 2757, 2781, 2782, 2800, 2804, 2808, 2831, 2864, 2866, 2870, 2890, 2901, 2903, 2906, 2909, 2919, 2939, 2979, 2984, 2994, 3001, 3030, 3036, 3058, 3071, 3093, 3112, 3122, 3131, 3138, 3141, 3168, 3178, 3182, 3201, 3214, 3243, 3259, 3260, 3271, 3274, 3278, 3283, 3293, 3295, 3299, 3305, 3309, 3312, 3322, 3327, 3329, 3341, 3356, 3358, 3361, 3362, 3363, 3380, 3397, 3398, 3400, 3410, 3413, 3419, 3430, 3478, 3499, 3516, 3532, 3554, 3573, 3574, 3582, 3584, 3592, 3602, 3604, 3612, 3616, 3629, 3647, 3656, 3681, 3689, 3729, 3742, 3752, 3755, 3775, 3795, 3805, 3811, 3814, 3851, 3867, 3872, 3885, 3931, 3950, 3960, 3961, 3976, 3985, 4015, 4017, 4019, 4023, 4027, 4041, 4049, 4052, 4056, 4059, 4066, 4070, 4086, 4103, 4106, 4117, 4127, 4131, 4135, 4137, 4151, 4158, 4162, 4164, 4165, 4174, 4201, 4205, 4215, 4216, 4228, 4240, 4259, 4284, 4287, 4299, 4333, 4355, 4356, 4358, 4389, 4390, 4403, 4407, 4430, 4435, 4438, 4450, 4451, 4463, 4491, 4495, 4499, 4507, 4510, 4524, 4525, 4538, 4547, 4553, 4558, 4560, 4562, 4581, 4598, 4606, 4609, 4657, 4673, 4676, 4677, 4680, 4681, 4689, 4690, 4710, 4733, 4734, 4740, 4752, 4779, 4791, 4805, 4810, 4853, 4867, 4875, 4886, 4893, 4909, 4942, 4948, 4985, 4986]
    elif parser.dataset == "REDDIT-BINARY":
        with open("Datasets/COLLAB/REDDIT-BINARY.pkl", 'rb') as pkl:
            prodict = pickle.load(pkl)
            if args.task == "graph":
                lbls = prodict["graph_label_tensor"]
                data = [list(a) for a in zip(prodict["node_tensor_tuple"], prodict["edge_tensor_tuple"], lbls.tensor_split([i for i in range(1, lbls.size(0))]) )]
                labels = prodict["graph_label_range"]
            else:
                print(f"ERROR NO NODE TASK FOR {parser.dataset}")
                sys.exit(-1)
        test_set_ids = [12, 15, 17, 35, 43, 46, 49, 60, 75, 97, 105, 113, 120, 124, 130, 134, 139, 175, 178, 180, 181, 187, 192, 201, 204, 205, 209, 210, 224, 230, 231, 237, 242, 262, 273, 277, 281, 284, 285, 287, 295, 296, 299, 309, 310, 316, 337, 342, 357, 372, 417, 423, 432, 451, 457, 459, 469, 483, 493, 501, 515, 521, 532, 536, 537, 542, 557, 566, 600, 603, 622, 629, 631, 649, 654, 662, 667, 670, 672, 693, 695, 726, 728, 730, 734, 748, 769, 773, 810, 889, 894, 898, 908, 925, 932, 937, 966, 976, 981, 991, 1020, 1028, 1029, 1060, 1068, 1075, 1077, 1080, 1086, 1107, 1113, 1117, 1127, 1133, 1140, 1146, 1157, 1174, 1175, 1180, 1200, 1220, 1224, 1235, 1240, 1241, 1245, 1269, 1298, 1299, 1318, 1323, 1333, 1336, 1344, 1346, 1351, 1375, 1384, 1393, 1395, 1443, 1449, 1461, 1465, 1469, 1489, 1498, 1499, 1500, 1509, 1526, 1532, 1544, 1558, 1569, 1586, 1589, 1595, 1611, 1616, 1624, 1658, 1660, 1669, 1702, 1728, 1744, 1756, 1757, 1763, 1765, 1777, 1781, 1789, 1792, 1798, 1820, 1858, 1862, 1863, 1875, 1894, 1896, 1901, 1905, 1913, 1919, 1921, 1925, 1936, 1939, 1941, 1944, 1947, 1949, 1956, 1966, 1981, 1994]
        
        
    if parser.model.startswith("GCN"):
        type = nm.GraphConvPoolNN
    elif parser.model.startswith("GUNET"):
        type = nm.GUNET
    else:
        print("Error. model argument not recognized:", parser.model)
        sys.exit(-1)
    if parser.model.endswith("Diehl"):
        pooltype = EdgePooling
    else:
        pooltype = ClusterPooling
    if parser.task == "graph":
        task_type_node = False
    return nm.GCNModel(data=data, labels=labels, task_type_node=task_type_node, test_set_idx=test_set_ids, type=type, pooltype=pooltype)

if __name__ == "__main__":
    # Define command line arguments
    parser = parser_function()

    # Process command line arguments
    args = parser.parse_args()
    

    model = build_model(args)
    print("Model has been built")
    model.run_folds(folds=1, kCross=False)