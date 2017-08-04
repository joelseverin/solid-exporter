import bpy
import mathutils
import gpu

import json
import collections
import base64
import functools
import os
import shutil
import struct
import zlib

default_settings = {
    'solidjson_output_dir': '',
    'buffers_embed_data': True,
    'buffers_combine_data': False,
    'nodes_export_hidden': False,
    'nodes_global_matrix': mathutils.Matrix.Identity(4),
    'nodes_selected_only': False,
    'shaders_data_storage': 'NONE',
    'meshes_apply_modifiers': True,
    'meshes_interleave_vertex_data' : True,
    'images_data_storage': 'COPY',
    'asset_profile': 'WEB',
    'ext_export_physics': False,
    'ext_export_actions': False,
}

if 'imported' in locals():
    import imp
    import bpy
else:
    imported = True

g_buffer = bytearray()

def append_buffer(buf):
    global g_buffer
    start = len(g_buffer)
    length = len(buf)
    g_buffer.extend(buf)
    return [start, length]
    
class Vertex:
    __slots__ = (
        "co",
        "normal",
        "uvs",
        "colors",
        "loop_indices",
        "index",
        "weights",
        "joint_indexes",
        )
    def __init__(self, mesh, loop):
        vi = loop.vertex_index
        i = loop.index
        self.co = mesh.vertices[vi].co.freeze()
        self.normal = loop.normal.freeze()
        self.uvs = tuple(layer.data[i].uv.freeze() for layer in mesh.uv_layers)
        self.colors = tuple(layer.data[i].color.freeze() for layer in mesh.vertex_colors)
        self.loop_indices = [i]

        # Take the four most influential groups
        groups = sorted(mesh.vertices[vi].groups, key=lambda group: group.weight, reverse=True)
        if len(groups) > 4:
            groups = groups[:4]

        self.weights = [group.weight for group in groups]
        self.joint_indexes = [group.group for group in groups]

        if len(self.weights) < 4:
            for i in range(len(self.weights), 4):
                self.weights.append(0.0)
                self.joint_indexes.append(0)

        self.index = 0

    def __hash__(self):
        return hash((self.co, self.normal, self.uvs, self.colors))

    def __eq__(self, other):
        eq = (
            (self.co == other.co) and
            (self.normal == other.normal) and
            (self.uvs == other.uvs) and
            (self.colors == other.colors)
            )

        if eq:
            indices = self.loop_indices + other.loop_indices
            self.loop_indices = indices
            other.loop_indices = indices
        return eq

def togl(matrix):
    return [i for col in matrix.col for i in col]

def export_mesh(mesh, skinned_meshes):
    data = {
        'name': mesh.name
    }
    
    is_skinned = mesh.name in skinned_meshes

    mesh.calc_normals_split()
    mesh.calc_tessface()
    
    if len(mesh.vertex_colors) > 0:
        print("Warning: mesh", mesh.name, "contains vertex colors, which are ignored.")
    
    has_uv_set = True
    if len(mesh.uv_layers) < 1:
        has_uv_set = False
        print("Warning: mesh", mesh.name, "is not UV unwrapped, using zeroes")
    
    vert_list = { Vertex(mesh, loop) : 0 for loop in mesh.loops}.keys()
    num_verts = len(vert_list)
    
    vertex_size = (3 + 3 + 2) * 4 # position, normal, chosen UV set; 4 bytes per float
    skin_vertex_size = (4 + 4) * 4
    
    buf = bytearray(num_verts * vertex_size)
    if is_skinned:
        skin_buf = bytearray(num_verts * skin_vertex_size)

    chosen_uv_set = mesh.uv_layers.find('Baked')
    if chosen_uv_set < 0:
        chosen_uv_set = 0 # Pick first. Remember to read has_uv_set too!
    
    # Copy vertex data
    for i, vtx in enumerate(vert_list):
        base = i * vertex_size
        base_position = base
        base_normal = base + 3*4
        base_texcoord = base + (3 + 2)*4
        
        skin_base = i * skin_vertex_size
        skin_base_joints = skin_base
        skin_base_weights = skin_base + 4*4
        
        vtx.index = i
        co = vtx.co
        normal = vtx.normal

        for j in range(3):
            struct.pack_into('<f', buf, base_position + j*4, co[j])
            struct.pack_into('<f', buf, base_normal + j*4, normal[j])

        if has_uv_set:
            uv = vtx.uvs[chosen_uv_set]
            struct.pack_into('<f', buf, base_texcoord + 0*4, uv.x)
            struct.pack_into('<f', buf, base_texcoord + 1*4, uv.y)
        else:
            struct.pack_into('<f', buf, base_texcoord + 0*4, 0)
            struct.pack_into('<f', buf, base_texcoord + 1*4, 0)

    if is_skinned:
        for i, vtx in enumerate(vert_list):
            joints = vtx.joint_indexes
            weights = vtx.weights

            for j in range(4):
                struct.pack_into('<f', skin_buf, skin_base_joints + j*4, joints[j])
                struct.pack_into('<f', skin_buf, skin_base_weights + j*4, weights[j])

    # Note that mesh.materials may contain multiple materials in blender. We use the first for everything.
    
    # Map loop indices to vertices for index data extraction
    vert_dict = {i : v for v in vert_list for i in v.loop_indices}
    
    # Collect primitive vertices by triangulation
    prim = []
    max_vert_index = 0
    for poly in mesh.polygons:
        # Find the (vertex) index associated with each loop in the polygon.
        indices = [vert_dict[i].index for i in poly.loop_indices]

        max_vert_index = max(max_vert_index, max(indices))
        
        if len(indices) == 3:
            # No triangulation necessary
            prim += indices
        elif len(indices) > 3:
            # Triangulation necessary
            for i in range(len(indices) - 2):
                prim += (indices[-1], indices[i], indices[i + 1])
        else:
            # Bad polygon
            raise RuntimeError(
                "Invalid polygon with {} vertices.".format(len(indices))
            )

    index_stride = 4
    index_buf = bytearray(index_stride * len(prim))
    
    for i, v in enumerate(prim):
        # <I is 32-bit unsigned int (we may downgrade this in later stages)
        struct.pack_into('<I', index_buf, i*index_stride, v)

    data['buffers'] = {
        'vertices': append_buffer(buf),
        'indices': append_buffer(index_buf)
    }

    if is_skinned:
        data['buffers']['skinning'] = append_buffer(skin_buf)

    # 65535-1 is optimal, as most WebGL implementations that make use of Direct3D
    # have problems with handling the 65535, that would otherwise be OpenGL's max.
    if max_vert_index > 65534:
        print("Warning: Exported mesh", mesh.name, "has index higher than 65534:", max_vert_index)
    
    return data

def export_image(image, settings):
    def check_image(image):
        errors = []
        if image.size[0] == 0:
            errors.append('x dimension is 0')
        if image.size[1] == 0:
            errors.append('y dimension is 0')
        if image.type != 'IMAGE':
            errors.append('not an image')

        if errors:
            err_list = '\n\t'.join(errors)
            print('Unable to export image {} due to the following errors:\n\t{}'.format(image.name, err_list))
            return False

        return True

    #extMap = {'BMP': 'bmp', 'JPEG': 'jpg', 'PNG': 'png', 'TARGA': 'tga'}
    extMap = {'JPEG': 'jpg', 'PNG': 'png'}
    uri = ''

    if check_image(image):
        if image.packed_file != None:
            if image.file_format in extMap:
                # save the file to the output directory
                uri = '.'.join([image.name, extMap[image.file_format]])
                temp = image.filepath
                image.filepath = os.path.join(settings['solidjson_output_dir'], uri)
                image.save()
                image.filepath = temp
            else:
                print("Warning: image", image.name, " is not jpg or png, will not be exported")
        else:
            try:
                shutil.copy(bpy.path.abspath(image.filepath), settings['solidjson_output_dir'])
            except shutil.SameFileError:
                # If the file already exists, no need to copy
                pass
            
            path = image.filepath
            
            # No idea why, but it starts with // at least sometimes, and then the basename
            # becomes an empty string. Let's just remove the // part and hope for the best.
            if path.startswith("//"):
                path = path[2:]
            
            uri = os.path.basename(path)
    return uri # image.name is also available

def make_image_data_uri(value):
    # R G B A
    write = [0, 0, 0, 1]
    if type(value) is float:
        for k in range(3):
            write[k] = value
    else:
        for j, v in enumerate(value):
            write[j] = v
    
    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
            chunk_head +
            struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    width = 1
    height = 1
    raw_data = b'\x00' + bytearray([int(p * 255) for p in write])
    png_bytes = b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

    return 'data:image/png;base64,' + base64.b64encode(png_bytes).decode()

def find_solidrendering_group(material):
    if material.node_tree is None:
        print("Warning: material", material.name, "has no Cycles node tree, skipping it")
        return None
    
    if 'Group' not in material.node_tree.nodes:
        print("Warning: material", material.name, "has no (SolidRendering) node named Group, skipping it")
        return None
    
    # Todo: Group is just the name of the node. We should probably iterate and match type instead.
    group = material.node_tree.nodes['Group']
    
    if type(group.interface) is not bpy.types.NodeTreeInterface_SolidRendering:
        print("Warning: material", material.name, "has Group but is not SolidRendering, skipping it")
        return None
    
    return group

def image_from_solidrendering_group(group, name, settings, default = None):
    prop = group.inputs[name]
    links = prop.links
    if len(links) > 0:
        node = links[0].from_node
        if type(node) is bpy.types.ShaderNodeTexImage:
            # Todo is to make sure the same image is only exported once
            return export_image(node.image, settings)
        else:
            print('Warning: material', material.name, 'has attachment that is not an Image Texture on slot', name)
            return ''
    else:
        # default is so strong that it overrides values set on the node group directly ("default_value" they are called)
        if default is None:
            return make_image_data_uri(prop.default_value)
        else:
            return make_image_data_uri(default)

def export_material(material, settings):
    group = find_solidrendering_group(material)
    if group is None:
        return None
    
    data = {
        'base_color': image_from_solidrendering_group(group, 'BaseColor', settings),
        'metalness': image_from_solidrendering_group(group, 'Metalness', settings),
        'roughness': image_from_solidrendering_group(group, 'Roughness', settings),
        'reflectivity': image_from_solidrendering_group(group, 'Reflectivity', settings),
        'emissive': image_from_solidrendering_group(group, 'Emissive', settings),
        'sky_visibility': image_from_solidrendering_group(group, 'SkyVisibility', settings),
        'normal': image_from_solidrendering_group(group, 'Normal', settings, [0, 0, 1])
    }
    
    return data

def export_solidjson(settings={}):
    global g_buffer
    g_buffer = bytearray()

    # todo not use explicit context
    context = bpy.context
    
    actions = list(bpy.data.actions)
    cameras = list(bpy.data.cameras)
    lamps = list(bpy.data.lamps)
    images = list(bpy.data.images)
    materials = list(bpy.data.materials)
    meshes = list(bpy.data.meshes)
    objects = list(bpy.data.objects)
    scenes = list(bpy.data.scenes)
    textures = list(bpy.data.textures)
    
    # Fill in any missing settings with defaults
    for key, value in default_settings.items():
        settings.setdefault(key, value)

    skinned_meshes = {}
    
    # Collect meshes and apply modifiers
    mesh_list = []
    mod_meshes = {}
    scene = context.scene
    mod_obs = [ob for ob in objects if ob.is_modified(scene, 'PREVIEW')]
    for mesh in meshes:
        mod_users = [ob for ob in mod_obs if ob.data == mesh]

        # Only convert meshes with modifiers, otherwise each non-modifier
        # user ends up with a copy of the mesh and we lose instancing
        mod_meshes.update({ob.name: ob.to_mesh(scene, True, 'PREVIEW') for ob in mod_users})

        # Add unmodified meshes directly to the mesh list
        if len(mod_users) < mesh.users:
            mesh_list.append(mesh)
    mesh_list.extend(mod_meshes.values())

    def found_image(image):
        """todo"""
    used_materials = {}
    def found_material(material):
        if not material.name in used_materials:
            exported = export_material(material, settings)
            if exported is not None:
                used_materials[material.name] = exported
    
    used_meshes = {}
    def found_mesh(mesh):
        if not mesh.name in used_meshes:
            used_meshes[mesh.name] = export_mesh(mesh, skinned_meshes)
    
    # Filter nodes based on our criteria
    # Below, we also report what we find, lazily instancing the other used_* vars
    should_include_node = lambda node: any(node.is_visible(scene) and (node.type == 'MESH' or node.type == 'EMPTY') for scene in scenes)
    def generate_used_node(node):
        used_node = {
            #'name': node.name,
            'transform': togl(node.matrix_world)
        }
        
        children = list(node.children)
        if node.type == 'EMPTY' and node.dupli_group is not None:
            children += node.dupli_group.objects
        used_node['children'] = {child.name: generate_used_node(child) for child in children if should_include_node(child)}
        
        if node.type == 'MESH':
            mesh = mod_meshes.get(node.name, node.data)
            
            armature = node.find_armature()
            if armature:
                skinned_meshes[mesh.name] = node
                # found_armature(armature.data.name)
            
            found_mesh(mesh)
            
            used_node['mesh'] = mesh.name
        
        num_materials = len(node.material_slots)
        if num_materials == 0:
            print("Warning: no material for node", node.name)
        else:
            material = node.material_slots[0].material
            if num_materials > 1:
                print("Warning: too many materials for node", node.name, ", using first:", material.name)
            found_material(material)
            used_node['material'] = material.name
            
        return used_node
    used_nodes = {node.name: generate_used_node(node) for node in objects if node.parent is None and should_include_node(node)}
    
    # Find lights
    
    # Find skybox and found_image() it
    
    solid_root = {
        'materials': used_materials,
        #'lights': used_lights,
        'meshes': used_meshes,
        'nodes': used_nodes,
        'physics': 'todo',
        'skins': 'todo'
    }
    
    # Retroactively add skins attribute to nodes
    #for mesh_name, obj in skinned_meshes.items():
    #    gltf['nodes']['node_' + obj.name]['skin'] = 'skin_{}'.format(mesh_name)

    # Write out the large binary buffer containing all meshes
    with open(os.path.join(settings['solidjson_output_dir'], 'meshdata.bin'), 'wb') as fout:
        fout.write(g_buffer)
    g_buffer = bytearray()

    # gltf = {key: value for key, value in gltf.items() if value}

    # Remove any temporary meshes from applying modifiers
    for mesh in mod_meshes.values():
        bpy.data.meshes.remove(mesh)

    print("Solid JSON export completed. Review any warnings above.")

    return solid_root