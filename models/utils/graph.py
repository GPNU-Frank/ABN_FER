import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge()
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_node = 55
        # neighbor_link = [(i, j) for i in range(self.num_node) for j in range(self.num_node) if i != j]
        neighbor_link = [(0, 1), (0, 19), (0, 20), (1, 0), (1, 2), (1, 19), (2, 1), (2, 3), (2, 20), (3, 21), (3, 2), (3, 4), (4, 10), (4, 22), (4, 3), (5, 10), (5, 6), (5, 25), (6, 26), (6, 5), (6, 7), (7, 6), (7, 8), (7, 27), (8, 9), (8, 7), (8, 28), (9, 8), (9, 28), (9, 27), (10, 4), (10, 5), (10, 11), (11, 12), (11, 10), (11, 25), (12, 11), (12, 13), (12, 10), (13, 12), (13, 16), (13, 
17), (14, 15), (14, 33), (14, 32), (15, 14), (15, 16), (15, 33), (16, 17), (16, 15), (16, 34), (17, 18), (17, 16), (17, 35), (18, 17), (18, 16), (18, 35), (19, 24), (19, 20), (19, 23), (20, 24), (20, 21), (20, 19), (21, 23), (21, 20), (21, 22), (22, 23), (22, 21), (22, 4), (23, 21), (23, 22), (23, 24), (24, 20), (24, 19), (24, 23), (25, 26), (25, 30), (25, 5), (26, 
30), (26, 27), (26, 25), (27, 29), (27, 26), (27, 28), (28, 29), (28, 27), (28, 30), (29, 27), (29, 28), (29, 30), (30, 26), (30, 29), (30, 27), (31, 49), (31, 42), (31, 50), (32, 42), (32, 50), (32, 33), (33, 41), (33, 15), (33, 34), (34, 40), (34, 16), (34, 33), (35, 39), (35, 17), (35, 53), (36, 38), (36, 54), (36, 35), (37, 43), (37, 38), (37, 54), (38, 54), (38, 37), (38, 43), (39, 53), (39, 35), (39, 40), (40, 52), (40, 53), (40, 34), (41, 51), (41, 33), (41, 40), (42, 50), (42, 31), (42, 49), (43, 37), (43, 38), (43, 54), (44, 38), (44, 54), (44, 45), (45, 46), (45, 44), (45, 53), (46, 45), (46, 47), (46, 52), (47, 46), (47, 48), (47, 51), (48, 42), (48, 50), (48, 47), (49, 31), (49, 42), (49, 50), (50, 42), (50, 31), (50, 49), (51, 41), (51, 40), (51, 52), (52, 40), (52, 53), (52, 39), (53, 39), (53, 40), (53, 52), (54, 38), (54, 37), (54, 43)]
        self.edge = neighbor_link
        self.center = 1
    # def get_edge(self, layout):
    #     if layout == 'openpose':
    #         self.num_node = 18
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
    #                                                                     11),
    #                          (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
    #                          (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
    #         self.edge = self_link + neighbor_link
    #         self.center = 1
    #     elif layout == 'ntu-rgb+d':
    #         self.num_node = 25
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
    #                           (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
    #                           (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    #                           (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
    #                           (22, 23), (23, 8), (24, 25), (25, 12)]
    #         neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    #         self.edge = self_link + neighbor_link
    #         self.center = 21 - 1
    #     elif layout == 'ntu_edge':
    #         self.num_node = 24
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
    #                           (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
    #                           (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
    #                           (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
    #                           (23, 24), (24, 12)]
    #         neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    #         self.edge = self_link + neighbor_link
    #         self.center = 2
    #     # elif layout=='customer settings'
    #     #     pass
    #     else:
    #         raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD