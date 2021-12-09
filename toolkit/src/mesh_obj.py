"""
Copyright 2020, Hao Zhu, Haotian Yang, NJU.
OBJ file loader and writer.
"""

import numpy as np, os

class mesh_obj:
    def __init__(self, filename=None):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.vert_colors = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.adjacent_list = []
        material = None
        
        if filename != None:
            for line in open(filename, "r"):
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    if len(values) == 4:
                        self.vertices.append(list(map(float, values[1:4])))
                    elif len(values) == 7:
                        self.vertices.append(list(map(float, values[1:4])))      
                        self.vert_colors.append(list(map(float, values[4:7])))
                elif values[0] == 'vn':
                    self.normals.append(list(map(float, values[1:4])))
                elif values[0] == 'vt':
                    self.texcoords.append(list(map(float, values[1:3])))
                elif values[0] in ('usemtl', 'usemat'):
                    material = values[1]
                elif values[0] == 'mtllib':
                    fn = os.path.dirname(filename) + '/' + os.path.basename(values[1])
                    if os.path.isfile(fn) is True:
                        self.mtl = self.read_mtl()
                    else:
                        print("mtl file not found: %s" % fn)
                elif values[0] == 'f':
                    face = []
                    texcoords = []
                    norms = []
                    for v in values[1:]:
                        w = v.split('/')
                        face.append(int(w[0]))
                        if len(w) >= 2 and len(w[1]) > 0:
                            texcoords.append(int(w[1]))
                        else:
                            texcoords.append(0)
                        if len(w) >= 3 and len(w[2]) > 0:
                            norms.append(int(w[2]))
                        else:
                            norms.append(0)
                    self.faces.append((face, norms, texcoords, material))

    def create(self, vertices = [], vert_colors = [], normals = [], texcoords = [], 
               faces_v = [], faces_vn = [], faces_vt = []):
        self.vertices = vertices
        self.vert_colors = vert_colors
        self.normals = normals
        self.texcoords = texcoords
        self.faces_v = faces_v
        self.faces_vn = faces_vn
        self.faces_vt = faces_vt
        self.faces = []
        material = None
        
        face_num = max(len(faces_v), len(faces_vn), len(faces_vt))
        if face_num > 0:
            if len(faces_v) != face_num:
                faces_v = [[-1, -1, -1]] * face_num
            if len(faces_vn) != face_num:
                faces_vn = [[-1, -1, -1]] * face_num
            if len(faces_vt) != face_num:
                faces_vt = [[-1, -1, -1]] * face_num                
            for i in range(face_num):
                self.faces.append((faces_v[i], faces_vn[i], faces_vt[i], material))
        
    def get_adjacent(self, index):
        if not self.adjacent_list:
            adjacent_list = [[] for i in range(len(self.vertices))]
            for face in self.faces:
                face_vertices, face_normals, face_texture_coords, material = face
                adjacent_list[face_vertices[0] - 1].append(face_vertices[1] - 1)
                adjacent_list[face_vertices[0] - 1].append(face_vertices[2] - 1)
                adjacent_list[face_vertices[1] - 1].append(face_vertices[0] - 1)
                adjacent_list[face_vertices[1] - 1].append(face_vertices[2] - 1)
                adjacent_list[face_vertices[2] - 1].append(face_vertices[0] - 1)
                adjacent_list[face_vertices[2] - 1].append(face_vertices[1] - 1)

            adjacent_list = list(map(set, adjacent_list))
            self.adjacent_list = list(map(list, adjacent_list))
        return self.adjacent_list[index]
    
    def export(self, output_dir, file_name, texture_name=None, enable_vc=False, enable_vt=True):

        tgt_dir = os.path.dirname(output_dir)
        output_file = os.path.join(output_dir, file_name) + '.obj'
        mtl_file = os.path.join(output_dir, file_name) + '.mtl'
        
        if len(tgt_dir) != 0:
            os.makedirs(tgt_dir, exist_ok=True)

        with open(output_file, "w") as f:
            if texture_name is not None:
                f.write('mtllib ./%s.mtl\n' % file_name)
            if enable_vc is True:
                for idx, vert in enumerate(self.vertices):
                    f.write("v %f %f %f %f %f %f\n" % (vert[0], vert[1], vert[2],
                                                       self.vert_colors[idx][0], 
                                                       self.vert_colors[idx][1],
                                                       self.vert_colors[idx][2]))
            else:
                for vert in self.vertices:
                    f.write("v %f %f %f\n" % (vert[0], vert[1], vert[2]))
            if enable_vt is True:
                for tc in self.texcoords:
                    f.write("vt %.6f %.6f\n" % (tc[0], tc[1]))
                if texture_name is not None:
                    f.write('usemtl material_0\n')
                for face in self.faces:
                    face_vertices, face_normals, face_texture_coords, material = face
                    f.write("f %d/%d %d/%d %d/%d\n" % (face_vertices[0], face_texture_coords[0], 
                                                       face_vertices[1], face_texture_coords[1], 
                                                       face_vertices[2], face_texture_coords[2]))
            else:
                for face in self.faces:
                    face_vertices, face_normals, face_texture_coords, material = face
                    f.write("f %d %d %d\n" % (face_vertices[0], face_vertices[1], face_vertices[2]))

        if texture_name is not None:
            with open(mtl_file, 'w') as f:
                f.write('newmtl material_0\nKa 0.200000 0.200000 0.200000\nKd 0.000000 0.000000 0.000000\n')
                f.write(
                    'Ks 1.000000 1.000000 1.000000\nTr 0.000000\nillum 2\nNs 0.000000\nmap_Kd %s' % texture_name)

    def read_mtl(self, filename):
        contents = {}
        mtl = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'newmtl':
                mtl = contents[values[1]] = {}
            elif mtl is None:
                raise ValueError('mtl file doesn\'t start with newmtl stmt')
            elif values[0] == 'map_Kd':
                # load the texture referred to by this declaration
                mtl[values[0]] = values[1]
            else:
                mtl[values[0]] = map(float, values[1:])
        return contents
    
