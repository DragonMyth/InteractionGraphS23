from matplotlib.colors import same_color
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from fairmotion.ops import conversions

class Interaction(object):
    def __init__(self,vert,self_cnt=0) -> None:
        self.vert = vert
        self.self_cnt = self_cnt
    def build_interaction_graph(self, type="interaction_mesh"):
        edges = []
        scales = []
        if type == "interaction_mesh":
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T
        elif type == "fully_connected":
            full_matrix = np.ones((self.vert.shape[0],self.vert.shape[0]))
            matrix = coo_matrix(full_matrix)
            edges = np.array([matrix.row,matrix.col])
        elif type == "full_bipartite":
            full_matrix = np.ones((self.vert.shape[0],self.vert.shape[0]))
            full_matrix[:self.self_cnt,:self.self_cnt] = 0
            full_matrix[self.self_cnt:,self.self_cnt:] = 0
            matrix = coo_matrix(full_matrix)
            edges = np.array([matrix.row,matrix.col])     
        elif type == "interaction_mesh_bipartite":
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T   
            cross_edges_idx = np.where((((edges[0]>self.self_cnt) & (edges[1]<self.self_cnt)) | ((edges[0]<self.self_cnt) & (edges[1]>self.self_cnt))))
            
            edges = edges[:,cross_edges_idx[0]]
        elif type == 'interaction_mesh_filtered':
            filted_pairs = [

                (0,8), # hip - right upper leg
                (0,11), # hip - left upper leg
                (8,0),  # hip - right upper leg
                (11,0), # hip - left upper leg
                (8,11), # left upper leg - right upper leg
                (11,8), # left upper leg - right upper leg
                
                (15,23), # hip - right upper leg (oppo)
                (15,26), # hip - left upper leg (oppo)
                (23,15), # hip - right upper leg (oppo)
                (26,15), # hip - left upper leg (oppo)
                (23,26), # left upper leg - right upper leg (oppo)
                (26,23), # left upper leg - right upper leg (oppo)

            ]
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if (i,neighbor_vert) in filted_pairs:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T   
        elif type == 'interaction_mesh_filtered_2':
            filted_pairs = [

                (0,8), # hip - right upper leg
                (0,11), # hip - left upper leg
                (8,0),  # hip - right upper leg
                (11,0), # hip - left upper leg
                (8,11), # left upper leg - right upper leg
                (11,8), # left upper leg - right upper leg
                (14,2), # head - right shoulder
                (14,5), # head - left shoulder
                (2,14), # right shoulder - head
                (5,14), # left shoulder - head

                (15,23), # hip - right upper leg (oppo)
                (15,26), # hip - left upper leg (oppo)
                (23,15), # hip - right upper leg (oppo)
                (26,15), # hip - left upper leg (oppo)
                (23,26), # left upper leg - right upper leg (oppo)
                (26,23), # left upper leg - right upper leg (oppo)

                (29,17), # head - right shoulder (oppo)
                (29,20), # head - left shoulder (oppo)
                (17,29), # right shoulder - head (oppo)
                (20,29), # left shoulder - head (oppo)

            ]
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if (i,neighbor_vert) in filted_pairs:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T    
        elif type == "interaction_mesh_filtered_3":
            filted_pairs = [

                (0,8), # hip - right upper leg
                (0,11), # hip - left upper leg
                (8,0),  # hip - right upper leg
                (11,0), # hip - left upper leg
                (8,11), # left upper leg - right upper leg
                (11,8), # left upper leg - right upper leg
                (14,2), # head - right shoulder
                (14,5), # head - left shoulder
                (2,14), # right shoulder - head
                (5,14), # left shoulder - head

                (15,23), # hip - right upper leg (oppo)
                (15,26), # hip - left upper leg (oppo)
                (23,15), # hip - right upper leg (oppo)
                (26,15), # hip - left upper leg (oppo)
                (23,26), # left upper leg - right upper leg (oppo)
                (26,23), # left upper leg - right upper leg (oppo)

                (29,17), # head - right shoulder (oppo)
                (29,20), # head - left shoulder (oppo)
                (17,29), # right shoulder - head (oppo)
                (20,29), # left shoulder - head (oppo)
            ]   

            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if (i,neighbor_vert) in filted_pairs:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
                    
            edges = np.array(edges).T   
            cross_edges_idx = np.where(((edges[0]<self.self_cnt) & (edges[1]<self.self_cnt)))
            
            edges = edges[:,cross_edges_idx[0]]
        elif type == "interaction_mesh_filtered_4":
            filted_pairs = [
            
                (0,8), # hip - right upper leg
                (0,11), # hip - left upper leg
                (8,0),  # hip - right upper leg
                (11,0), # hip - left upper leg
                (8,11), # left upper leg - right upper leg
                (11,8), # left upper leg - right upper leg
                (14,2), # head - right shoulder
                (14,5), # head - left shoulder
                (2,14), # right shoulder - head
                (5,14), # left shoulder - head
                ## Bone Edges
                # (2,3),
                # (3,4),
                # (5,6),
                # (6,7),
                # (8,9),
                # (9,10),
                # (11,12),
                # (12,13),

                # (3,2),
                # (4,3),
                # (6,5),
                # (7,6),
                # (9,8),
                # (10,9),
                # (12,11),
                # (13,12),
            ]   

            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if (i,neighbor_vert) in filted_pairs:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
                    
            edges = np.array(edges).T   
            cross_edges_idx = np.where(((edges[0]<self.self_cnt) & (edges[1]<self.self_cnt)))
            
            # edges = edges[:,cross_edges_idx[0]]
            edges = [
                [0],
                [13]
                ]
        elif type == "interaction_mesh_remove_self":
            filtered_verts = [15,16,17,18]
            tet = Delaunay(self.vert)
            neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
            for i in range(len(tet.points)):
                neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
                for neighbor_vert in neighbors:
                    if i in filtered_verts and neighbor_vert in filtered_verts:
                        continue
                    edge = [i,neighbor_vert]
                    edges.append(edge)
            edges = np.array(edges).T
        return edges
class Mesh(object):
    def __init__(self):
        pass
    def construct_from_poses(self, poses, sample_infos, compute_neightbor_info=True, weighted_avg=True):
        def sample_vertices(pose, sample_info):
            vertices = []
            for j1, j2, alpha in sample_info:
                p1 = conversions.T2p(pose.get_transform(j1, local=False))
                p2 = p1 if j2 is None else conversions.T2p(pose.get_transform(j2, local=False))
                vertices.append((1.0 - alpha) * p1 + alpha * p2)
            return vertices
        vertices_all = []
        for pose, sample_info in zip(poses, sample_infos):
            vertices_all += sample_vertices(pose, sample_info)
        self.construct_from_vertices(vertices_all, compute_neightbor_info, weighted_avg)
    def construct_from_vertices(self, vertices, compute_neightbor_info=True, weighted_avg=True):
        self.vertices = vertices
        self.tet = Delaunay(vertices)
        self.num_simplex = len(self.tet.simplices)
        self.neighbor_indicies = []
        self.neighbor_weights = []
        self.laplacian_coordinates = []
        if compute_neightbor_info:
            for i in range(len(self.vertices)):
                neighbor = set([])
                for j in range(self.num_simplex):
                    indices = self.tet.simplices[j]
                    if i in indices:
                        neighbor.update(indices)
                neighbor.remove(i)
                neighbor = list(neighbor)
                if weighted_avg:
                    dist = np.array([np.linalg.norm(self.vertices[j]-self.vertices[i]) for j in neighbor])
                    exp_neg_dist = np.exp(-5*dist)
                    neighbor_weight = exp_neg_dist/np.sum(exp_neg_dist)
                else:
                    neighbor_weight = np.ones_like(neighbor)/len(neighbor)
                # print(dist)
                # print(neighbor_weight)
                # print(np.sum(neighbor_weight))
                # print('---------------')
                self.neighbor_indicies.append(neighbor)
                self.neighbor_weights.append(neighbor_weight)
                self.laplacian_coordinates.append(self._get_laplacian_coordinates(
                    self.vertices, i, neighbor, neighbor_weight))
    def get_simplex(self, idx):
        assert idx < self.num_simplex
        idx1, idx2, idx3, idx4 = self.tet.simplices[idx]
        return self.vertices[idx1], self.vertices[idx2], self.vertices[idx3], self.vertices[idx4]
    def get_simplex_indices(self, idx):
        assert idx < self.num_simplex
        return self.tet.simplices[idx]
    def _get_laplacian_coordinates(self, vertices, idx, indices_neighbor, weights_neighbor=None):
        assert len(indices_neighbor) > 0
        vertices_neighbor = [vertices[i] for i in indices_neighbor]
        return vertices[idx] - np.average(vertices_neighbor, axis=0, weights=weights_neighbor)
    def get_laplacian_coordinates(self, vertices):
        ''' Return Laplacian coordinates given a new set of vertices '''
        assert len(self.vertices) == len(self.neighbor_indicies)
        assert len(self.vertices) == len(self.neighbor_weights)
        assert len(self.vertices) == len(self.laplacian_coordinates)
        assert len(self.vertices) == len(vertices)
        coordinates = []
        for i in range(len(vertices)):
            coordinates.append(self._get_laplacian_coordinates(
                vertices, i, self.neighbor_indicies[i], self.neighbor_weights[i]))
        return coordinates
    def get_laplacian_deformation_energy(self, vertices):
        ''' Measure Laplacian deformation energy given a new set of vertices '''
        coordinates_new = self.get_laplacian_coordinates(vertices)
        coordinates_old = self.laplacian_coordinates
        energy = 0.0
        for i in range(len(vertices)):
            diff = coordinates_old[i] - coordinates_new[i]
            energy += np.dot(diff, diff)
        return energy
