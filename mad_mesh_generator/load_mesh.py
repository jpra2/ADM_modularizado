# -*- coding: utf-8 -*-

from pymoab import core, types, rng, topo_util
from utils import pymoab_utils as utpy
import numpy as np
import scipy.sparse as sp
import pdb
import os
import numpy as np

class Mesh:

    def __init__(self, mesh_file):
        self.mb = core.Core()
        self.mtu = topo_util.MeshTopoUtil(self.mb)
        self.mb.load_file(mesh_file)
        self.all_nodes, self.all_edges, self.all_faces, self.all_volumes = utpy.get_all_entities(self.mb)
        self.tags = LoadMesh.load_tags(self.mb)
        self.matrices = LoadMesh.load_matrices()
        self.ADJs = np.array([self.mb.get_adjacencies(face, 3) for face in self.all_faces])
        self.all_centroids = np.array([self.mtu.get_average_position([v]) for v in self.all_volumes])
        self.wirebasket_elems, self.wirebasket_numbers, self.mv = LoadMesh.load_wirebasket_elems(self.mb, self.tags)
        self.meshset_levels, self.L2_meshset = LoadMesh.load_meshsets_level(self.mb, self.tags)
        self.all_boundary_faces = LoadMesh.load_all_bound_faces(self.mb, self.tags)

class LoadMesh:

    @staticmethod
    def load_tags(mb):
        list_names_tags = np.load('list_names_tags.npy')

        tags = dict()

        for name in list_names_tags:
            try:
                tags[name] = mb.tag_get_handle(str(name))
            except:
                print(f'A tag {name} nao existe no arquivo')
                pdb.set_trace()

        return tags

    @staticmethod
    def load_matrices():
        list_names_variables_npy = np.load('list_names_variables_npy.npy')
        list_names_variables_npz = np.load('list_names_variables_npz.npy')

        matrices = dict()

        ext_npy = '.npy'
        ext_npz = '.npz'

        for name in list_names_variables_npy:
            ext = name + ext_npy
            try:
                matrices[name] = np.load(ext)
            except:
                print(f'O vetor {name} nao existe')
                pdb.set_trace()


        for name in list_names_variables_npz:
            ext = name + ext_npz
            try:
                matrices[name] = sp.load_npz(ext)
            except:
                print(f'A matriz {name} nao existe')
                pdb.set_trace()

        return matrices

    @staticmethod
    def load_wirebasket_elems(mb, tags):

        D1_tag = tags['d1']
        internos=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
        faces=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
        arestas=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
        vertices=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
        wire_elems1 = [internos, faces, arestas, vertices]
        wire_numbers1 = [len(internos), len(faces), len(arestas), len(vertices)]

        mv=mb.create_meshset()
        mb.add_entities(mv,vertices)

        D2_tag = tags['d2']
        inte=mb.get_entities_by_type_and_tag(mv, types.MBHEX, np.array([D2_tag]), np.array([0]))
        fac=mb.get_entities_by_type_and_tag(mv, types.MBHEX, np.array([D2_tag]), np.array([1]))
        are=mb.get_entities_by_type_and_tag(mv, types.MBHEX, np.array([D2_tag]), np.array([2]))
        ver=mb.get_entities_by_type_and_tag(mv, types.MBHEX, np.array([D2_tag]), np.array([3]))
        wire_elems2 = [inte, fac, are, ver]
        wire_numbers2 = [len(inte), len(fac), len(are), len(ver)]

        wirebasket_elems = [wire_elems1, wire_elems2]
        wirebasket_numbers = [wire_numbers1, wire_numbers2]
        meshset_vertices1 = mv

        return wirebasket_elems, wirebasket_numbers, meshset_vertices1

    @staticmethod
    def load_meshsets_level(mb, tags):

        meshsets_nv1 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([tags['PRIMAL_ID_1']]), np.array([None]))
        meshsets_nv2 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([tags['PRIMAL_ID_2']]), np.array([None]))
        L2_meshset = mb.tag_get_data(tags['L2_MESHSET'], 0, flat=True)[0]

        return [meshsets_nv1, meshsets_nv2], L2_meshset

    @staticmethod
    def load_all_bound_faces(mb, tags):
        boundary_faces_tag = tags['FACES_BOUNDARY']
        bound_faces_meshset = mb.tag_get_data(boundary_faces_tag, 0, flat=True)[0]
        all_boundary_faces = mb.get_entities_by_handle(bound_faces_meshset)
        return all_boundary_faces
