import numpy as np, os

class facescape_bm():
    def __init__(self, filename):
        bm_model = np.load(filename, allow_pickle = True) 
        self.shape_bm_core = bm_model['shape_bm_core'] # shape core
        self.color_bm_core = bm_model['color_bm_core'] # color core
        self.color_bm_mean = bm_model['color_bm_mean'] # color mean
        self.fv_indices = bm_model['fv_indices'] # face - vertex indices
        self.ft_indices = bm_model['ft_indices'] # face - texture_coordinate indices
        self.fv_indices_front = bm_model['fv_indices_front'] # frontal face-vertex indices
        self.ft_indices_front = bm_model['ft_indices_front'] # frontal face-texture_coordinate indices
        self.vc_dict_front = bm_model['vc_dict_front'] # frontal vertex color dictionary
        self.v_indices_front = bm_model['v_indices_front'] # frontal vertex indices
        self.vert_num = bm_model['vert_num'] # vertex number
        self.face_num = bm_model['face_num'] # face number
        self.frontal_vert_num = bm_model['frontal_vert_num'] # frontal vertex number
        self.frontal_face_num = bm_model['frontal_face_num'] # frontal face number
        self.texcoords = bm_model['texcoords'] # texture coordinates (constant)
        self.facial_mask = bm_model['facial_mask'] # UV facial mask
        self.sym_dict = bm_model['sym_dict'] # symmetry dictionary
        self.lm_list_v16 = bm_model['lm_list_v16'] # landmark indices
        self.vert_10to16_dict = bm_model['vert_10to16_dict'] # vertex indices dictionary (v1.0 to v1.6)
        self.vert_16to10_dict = bm_model['vert_16to10_dict'] # vertex indices dictionary (v1.6 to v1.0)
        
    def gen_full(self, id_vec, exp_vec):
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        mesh = mesh_obj()
        mesh.create(vertices = verts, 
                    texcoords = self.texcoords, 
                    faces_v = self.fv_indices, 
                    faces_vt = self.ft_indices)
        return mesh
    
    def gen_face(self, id_vec, exp_vec):
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        mesh = mesh_obj()
        mesh.create(vertices = verts[self.v_indices_front], 
                    texcoords = self.texcoords[self.v_indices_front], 
                    faces_v = self.fv_indices_front, 
                    faces_vt = self.ft_indices_front)
        return mesh
    
    def gen_face_color(self, id_vec, exp_vec, vc_vec):
        
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        vert_colors = self.color_bm_mean + self.color_bm_core.dot(vc_vec)
        vert_colors = vert_colors.reshape((-1, 3)) / 255
        mesh = mesh_obj()
        
        new_vert_colors = vert_colors[self.vc_dict_front][:,[2,1,0]]
        new_vert_colors[(self.vc_dict_front == -1)] = np.array([0, 0, 0], dtype = np.float32)
        
        mesh.create(vertices = verts[self.v_indices_front], 
                    vert_colors = new_vert_colors,
                    texcoords = self.texcoords[self.v_indices_front], 
                    faces_v = self.fv_indices_front, 
                    faces_vt = self.ft_indices_front)
        return mesh

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
    
    def export(self, tgt_dirname, enable_vc = False, enable_vt = False):
        
        if tgt_dirname[-4:] != '.obj' and tgt_dirname[-4:] != '.OBJ':
            print("ERROR: suffix to save should be '.obj' or '.OBJ': %s" % tgt_dirname)
            return False
        tgt_dir = os.path.dirname(tgt_dirname)
        filename = os.path.basename(tgt_dirname)
        
        if len(tgt_dir) != 0:
            os.makedirs(tgt_dir, exist_ok = True)

        with open(tgt_dirname, "w") as f:
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
                for face in self.faces:
                    face_vertices, face_normals, face_texture_coords, material = face
                    f.write("f %d/%d %d/%d %d/%d\n" % (face_vertices[0], face_texture_coords[0], 
                                                       face_vertices[1], face_texture_coords[1], 
                                                       face_vertices[2], face_texture_coords[2]))
            else:
                for face in self.faces:
                    face_vertices, face_normals, face_texture_coords, material = face
                    f.write("f %d %d %d\n" % (face_vertices[0], face_vertices[1], face_vertices[2]))
    
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
    