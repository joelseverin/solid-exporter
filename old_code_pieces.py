# Texture formats
GL_ALPHA = 6406
GL_RGB = 6407
GL_RGBA = 6408
GL_LUMINANCE = 6409
GL_LUMINANCE_ALPHA = 6410

# sRGB texture formats (not actually part of WebGL 1.0 or glTF 1.0)
GL_SRGB = 0x8C40
GL_SRGB_ALPHA = 0x8C42

OES_ELEMENT_INDEX_UINT = 'OES_element_index_uint'

profile_map = {
    'WEB': {'api': 'WebGL', 'version': '1.0.3'},
    'DESKTOP': {'api': 'OpenGL', 'version': '3.0'}
}

g_glExtensionsUsed = []

if 'imported' in locals():
    import imp
    import bpy
    #imp.reload(gpu_luts)
    #imp.reload(shader_converter)
else:
    imported = True
    #from . import gpu_luts
    #from . import shader_converter

class Buffer:
    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963

    BYTE = 5120
    UNSIGNED_BYTE = 5121
    SHORT = 5122
    UNSIGNED_SHORT = 5123
    INT = 5124
    UNSIGNED_INT = 5125

    FLOAT = 5126

    MAT4 = 'MAT4'
    VEC4 = 'VEC4'
    VEC3 = 'VEC3'
    VEC2 = 'VEC2'
    SCALAR = 'SCALAR'

    class Accessor:
        __slots__ = (
            "name",
            "buffer",
            "buffer_view",
            "byte_offset",
            "byte_stride",
            "component_type",
            "count",
            "min",
            "max",
            "type",
            "type_size",
            "_ctype",
            "_ctype_size",
            "_buffer_data",
            )
        def __init__(self,
                     name,
                     buffer,
                     buffer_view,
                     byte_offset,
                     byte_stride,
                     component_type,
                     count,
                     type):
            self.name = name
            self.buffer = buffer
            self.buffer_view = buffer_view
            self.byte_offset = byte_offset
            self.byte_stride = byte_stride
            self.component_type = component_type
            self.count = count
            self.min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.type = type

            if self.type == Buffer.MAT4:
                self.type_size = 16
            elif self.type == Buffer.VEC4:
                self.type_size = 4
            elif self.type == Buffer.VEC3:
                self.type_size = 3
            elif self.type == Buffer.VEC2:
                self.type_size = 2
            else:
                self.type_size = 1

            if component_type == Buffer.BYTE:
                self._ctype = '<b'
            elif component_type == Buffer.UNSIGNED_BYTE:
                self._ctype = '<B'
            elif component_type == Buffer.SHORT:
                self._ctype = '<h'
            elif component_type == Buffer.UNSIGNED_SHORT:
                self._ctype = '<H'
            elif component_type == Buffer.INT:
                self._ctype = '<i'
            elif component_type == Buffer.UNSIGNED_INT:
                self._ctype = '<I'
            elif component_type == Buffer.FLOAT:
                self._ctype = '<f'
            else:
                raise ValueError("Bad component type")

            self._ctype_size = struct.calcsize(self._ctype)
            self._buffer_data = self.buffer._get_buffer_data(self.buffer_view)

        # Inlined for performance, leaving this here as reference
        # def _get_ptr(self, idx):
            # addr = ((idx % self.type_size) * self._ctype_size + idx // self.type_size * self.byte_stride) + self.byte_offset
            # return addr

        def __len__(self):
            return self.count

        def __getitem__(self, idx):
            if not isinstance(idx, int):
                raise TypeError("Expected an integer index")

            ptr = ((idx % self.type_size) * self._ctype_size + idx // self.type_size * self.byte_stride) + self.byte_offset

            return struct.unpack_from(self._ctype, self._buffer_data, ptr)[0]

        def __setitem__(self, idx, value):
            if not isinstance(idx, int):
                raise TypeError("Expected an integer index")

            i = idx % self.type_size
            self.min[i] = value if value < self.min[i] else self.min[i]
            self.max[i] = value if value > self.max[i] else self.max[i]

            ptr = (i * self._ctype_size + idx // self.type_size * self.byte_stride) + self.byte_offset

            struct.pack_into(self._ctype, self._buffer_data, ptr, value)

    __slots__ = (
        "name",
        "type",
        "bytelength",
        "buffer_views",
        "accessors",
        )
    def __init__(self, name, uri=None):
        self.name = 'buffer_{}'.format(name)
        self.type = 'arraybuffer'
        self.bytelength = 0
        self.buffer_views = collections.OrderedDict()
        self.accessors = {}

    def export_buffer(self, settings):
        data = bytearray()
        for bn, bv in self.buffer_views.items():
            data.extend(bv['data'])

        if settings['buffers_embed_data']:
            uri = 'data:text/plain;base64,' + base64.b64encode(data).decode('ascii')
        else:
            uri = bpy.path.clean_name(self.name) + '.bin'
            with open(os.path.join(settings['solidjson_output_dir'], uri), 'wb') as fout:
                fout.write(data)

        return {
            'byteLength': self.bytelength,
            'type': self.type,
            'uri': uri,
        }

    def add_view(self, bytelength, target):
        buffer_name = 'bufferView_{}_{}'.format(self.name, len(self.buffer_views))
        self.buffer_views[buffer_name] = {
                'data': bytearray(bytelength),
                'target': target,
                'bytelength': bytelength,
                'byteoffset': self.bytelength,
            }
        self.bytelength += bytelength
        return buffer_name

    def export_views(self):
        gltf = {}

        for k, v in self.buffer_views.items():
            gltf[k] = {
                'buffer': self.name,
                'byteLength': v['bytelength'],
                'byteOffset': v['byteoffset'],
            }

            if v['target'] is not None:
                gltf[k]['target'] = v['target']

        return gltf

    def _get_buffer_data(self, buffer_view):
        return self.buffer_views[buffer_view]['data']

    def add_accessor(self,
                     buffer_view,
                     byte_offset,
                     byte_stride,
                     component_type,
                     count,
                     type):
        accessor_name = 'accessor_{}_{}'.format(self.name, len(self.accessors))
        self.accessors[accessor_name] = self.Accessor(accessor_name, self, buffer_view, byte_offset, byte_stride, component_type, count, type)
        return self.accessors[accessor_name]

    def export_accessors(self):
        gltf = {}

        for k, v in self.accessors.items():
            # Do not export an empty accessor
            if v.count == 0:
                continue

            gltf[k] = {
                'bufferView': v.buffer_view,
                'byteOffset': v.byte_offset,
                'byteStride': v.byte_stride,
                'componentType': v.component_type,
                'count': v.count,
                'min': v.min[:v.type_size],
                'max': v.max[:v.type_size],
                'type': v.type,
            }

        return gltf

    def __add__(self, other):
        # Handle the simple stuff
        combined = Buffer('combined')
        combined.bytelength = self.bytelength + other.bytelength
        combined.accessors = {**self.accessors, **other.accessors}

        # Need to update byte offsets in buffer views
        combined.buffer_views = self.buffer_views.copy()
        other_views = other.buffer_views.copy()
        for key in other_views.keys():
            other_views[key]['byteoffset'] += self.bytelength
        combined.buffer_views.update(other_views)

        return combined


g_buffers = []

def export_materials(settings, materials, shaders, programs, techniques):
    def export_material(material):
        all_textures = [ts for ts in material.texture_slots if ts and ts.texture.type == 'IMAGE']
        diffuse_textures = ['texture_' + t.texture.name for t in all_textures if t.use_map_color_diffuse]
        emission_textures = ['texture_' + t.texture.name for t in all_textures if t.use_map_emit]
        specular_textures = ['texture_' + t.texture.name for t in all_textures if t.use_map_color_spec]
        diffuse_color = list((material.diffuse_color * material.diffuse_intensity)[:]) + [material.alpha]
        emission_color = list((material.diffuse_color * material.emit)[:]) + [material.alpha]
        specular_color = list((material.specular_color * material.specular_intensity)[:]) + [material.specular_alpha]
        technique = 'PHONG'
        if material.use_shadeless:
            technique = 'CONSTANT'
        elif material.specular_intensity == 0.0:
            technique = 'LAMBERT'
        elif material.specular_shader == 'BLINN':
            technique = 'BLINN'
        return {
                'extensions': {
                    'KHR_materials_common': {
                        'technique': technique,
                        'values': {
                            'ambient': ([material.ambient]*3) + [1.0],
                            'diffuse': diffuse_textures[-1] if diffuse_textures else diffuse_color,
                            'doubleSided': not material.game_settings.use_backface_culling,
                            'emission': emission_textures[-1] if emission_textures else emission_color,
                            'specular': specular_textures[-1] if specular_textures else specular_color,
                            'shininess': material.specular_hardness,
                            'transparency': material.alpha,
                            'transparent': material.use_transparency,
                        }
                    }
                },
                'name': material.name,
            }
    exp_materials = {}
    for material in materials:
        if settings['shaders_data_storage'] == 'NONE':
            exp_materials['material_' + material.name] = export_material(material)
        else:
            # Handle shaders
            shader_data = gpu.export_shader(bpy.context.scene, material)
            if settings['asset_profile'] == 'DESKTOP':
                shader_converter.to_130(shader_data)
            else:
                shader_converter.to_web(shader_data)

            fs_name = 'shader_{}_FS'.format(material.name)
            vs_name = 'shader_{}_VS'.format(material.name)
            storage_setting = settings['shaders_data_storage']
            if storage_setting == 'EMBED':
                fs_bytes = shader_data['fragment'].encode()
                fs_uri = 'data:text/plain;base64,' + base64.b64encode(fs_bytes).decode('ascii')
                vs_bytes = shader_data['vertex'].encode()
                vs_uri = 'data:text/plain;base64,' + base64.b64encode(vs_bytes).decode('ascii')
            elif storage_setting == 'EXTERNAL':
                names = [bpy.path.clean_name(name) + '.glsl' for name in (material.name+'VS', material.name+'FS')]
                data = (shader_data['vertex'], shader_data['fragment'])
                for name, data in zip(names, data):
                    filename = os.path.join(settings['solidjson_output_dir'], name)
                    with open(filename, 'w') as fout:
                        fout.write(data)
                vs_uri, fs_uri = names
            else:
                print('Encountered unknown option ({}) for shaders_data_storage setting'.format(storage_setting));

            shaders[fs_name] = {'type': 35632, 'uri': fs_uri}
            shaders[vs_name] = {'type': 35633, 'uri': vs_uri}

            # Handle programs
            programs['program_' + material.name] = {
                'attributes' : [a['varname'] for a in shader_data['attributes']],
                'fragmentShader' : 'shader_{}_FS'.format(material.name),
                'vertexShader' : 'shader_{}_VS'.format(material.name),
            }

            # Handle parameters/values
            values = {}
            parameters = {}
            for attribute in shader_data['attributes']:
                name = attribute['varname']
                semantic = gpu_luts.TYPE_TO_SEMANTIC[attribute['type']]
                _type = gpu_luts.DATATYPE_TO_GLTF_TYPE[attribute['datatype']]
                parameters[name] = {'semantic': semantic, 'type': _type}

            for uniform in shader_data['uniforms']:
                valname = gpu_luts.TYPE_TO_NAME.get(uniform['type'], uniform['varname'])
                rnaname = valname
                semantic = None
                node = None
                value = None

                if uniform['varname'] == 'bl_ModelViewMatrix':
                    semantic = 'MODELVIEW'
                elif uniform['varname'] == 'bl_ProjectionMatrix':
                    semantic = 'PROJECTION'
                elif uniform['varname'] == 'bl_NormalMatrix':
                    semantic = 'MODELVIEWINVERSETRANSPOSE'
                else:
                    if uniform['type'] in gpu_luts.LAMP_TYPES:
                        node = uniform['lamp'].name
                        valname = node + '_' + valname
                        semantic = gpu_luts.TYPE_TO_SEMANTIC.get(uniform['type'], None)
                        if not semantic:
                            lamp_obj = bpy.data.objects[node]
                            value = getattr(lamp_obj.data, rnaname)
                    elif uniform['type'] in gpu_luts.MIST_TYPES:
                        valname = 'mist_' + valname
                        mist_settings = bpy.context.scene.world.mist_settings
                        if valname == 'mist_color':
                            value = bpy.context.scene.world.horizon_color
                        else:
                            value = getattr(mist_settings, rnaname)

                        if valname == 'mist_falloff':
                            value = 0.0 if value == 'QUADRATIC' else 1.0 if 'LINEAR' else 2.0
                    elif uniform['type'] in gpu_luts.WORLD_TYPES:
                        world = bpy.context.scene.world
                        value = getattr(world, rnaname)
                    elif uniform['type'] in gpu_luts.MATERIAL_TYPES:
                        value = gpu_luts.DATATYPE_TO_CONVERTER[uniform['datatype']](getattr(material, rnaname))
                        values[valname] = value
                    elif uniform['type'] == gpu.GPU_DYNAMIC_SAMPLER_2DIMAGE:
                        for ts in [ts for ts in material.texture_slots if ts and ts.texture.type == 'IMAGE']:
                            if ts.texture.image.name == uniform['image'].name:
                                value = 'texture_' + ts.texture.name
                                values[uniform['varname']] = value
                    else:
                        print('Unconverted uniform:', uniform)

                parameter = {}
                if semantic:
                    parameter['semantic'] = semantic
                    if node:
                        parameter['node'] = 'node_' + node
                else:
                    parameter['value'] = gpu_luts.DATATYPE_TO_CONVERTER[uniform['datatype']](value)
                if uniform['type'] == gpu.GPU_DYNAMIC_SAMPLER_2DIMAGE:
                    parameter['type'] = 35678 #SAMPLER_2D
                else:
                    parameter['type'] = gpu_luts.DATATYPE_TO_GLTF_TYPE[uniform['datatype']]
                parameters[valname] = parameter
                uniform['valname'] = valname

            # Handle techniques
            tech_name = 'technique_' + material.name
            techniques[tech_name] = {
                'parameters' : parameters,
                'program' : 'program_' + material.name,
                'attributes' : {a['varname'] : a['varname'] for a in shader_data['attributes']},
                'uniforms' : {u['varname'] : u['valname'] for u in shader_data['uniforms']},
            }

            exp_materials['material_' + material.name] = {'technique': tech_name, 'values': values}
            # exp_materials[material.name] = {}

    return exp_materials

def export_skins(skinned_meshes):
    def export_skin(obj):
        gltf_skin = {
            'bindShapeMatrix': togl(mathutils.Matrix.Identity(4)),
            'name': obj.name,
        }
        arm = obj.find_armature()
        gltf_skin['jointNames'] = ['node_{}_{}'.format(arm.name, group.name) for group in obj.vertex_groups]

        element_size = 16 * 4
        num_elements = len(obj.vertex_groups)
        buf = Buffer('IBM_{}_skin'.format(obj.name))
        buf_view = buf.add_view(element_size * num_elements, None)
        idata = buf.add_accessor(buf_view, 0, element_size, Buffer.FLOAT, num_elements, Buffer.MAT4)

        for i in range(num_elements):
            mat = togl(mathutils.Matrix.Identity(4))
            for j in range(16):
                idata[(i * 16) + j] = mat[j]

        gltf_skin['inverseBindMatrices'] = idata.name
        g_buffers.append(buf)

        return gltf_skin

    return {'skin_' + mesh_name: export_skin(obj) for mesh_name, obj in skinned_meshes.items()}


def export_lights(lamps):
    def export_light(light):
        def calc_att():
            kl = 0
            kq = 0

            if light.falloff_type == 'INVERSE_LINEAR':
                kl = 1 / light.distance
            elif light.falloff_type == 'INVERSE_SQUARE':
                kq = 1 / light.distance
            elif light.falloff_type == 'LINEAR_QUADRATIC_WEIGHTED':
                kl = light.linear_attenuation * (1 / light.distance)
                kq = light.quadratic_attenuation * (1 / (light.distance * light.distance))

            return kl, kq

        if light.type == 'SUN':
            return {
                'directional': {
                    'color': (light.color * light.energy)[:],
                },
                'type': 'directional',
            }
        elif light.type == 'POINT':
            kl, kq = calc_att()
            return {
                'point': {
                    'color': (light.color * light.energy)[:],

                    # TODO: grab values from Blender lamps
                    'constantAttenuation': 1,
                    'linearAttenuation': kl,
                    'quadraticAttenuation': kq,
                },
                'type': 'point',
            }
        elif light.type == 'SPOT':
            kl, kq = calc_att()
            return {
                'spot': {
                    'color': (light.color * light.energy)[:],

                    # TODO: grab values from Blender lamps
                    'constantAttenuation': 1.0,
                    'fallOffAngle': 3.14159265,
                    'fallOffExponent': 0.0,
                    'linearAttenuation': kl,
                    'quadraticAttenuation': kq,
                },
                'type': 'spot',
            }
        else:
            print("Unsupported lamp type on {}: {}".format(light.name, light.type))
            return {'type': 'unsupported'}

    gltf = {'light_' + lamp.name: export_light(lamp) for lamp in lamps}

    return gltf


def export_buffers(settings):
    gltf = {
        'buffers': {},
        'bufferViews': {},
        'accessors': {},
    }

    if settings['buffers_combine_data']:
        buffers = [functools.reduce(lambda x, y: x+y, g_buffers)]
    else:
        buffers = g_buffers

    for buf in buffers:
        gltf['buffers'][buf.name] = buf.export_buffer(settings)
        gltf['bufferViews'].update(buf.export_views())
        gltf['accessors'].update(buf.export_accessors())

    return gltf


def insert_root_nodes(gltf_data, root_matrix):
    for name, scene in gltf_data['scenes'].items():
        node_name = 'node_{}_root'.format(name)
        # Generate a new root node for each scene
        gltf_data['nodes'][node_name] = {
            'children': scene['nodes'],
            'matrix': root_matrix,
            'name': node_name,
        }

        # Replace scene node lists to just point to the new root nodes
        scene['nodes'] = [node_name]

def old_export(scene_delta, settings={}):
    global g_buffers
    global g_glExtensionsUsed

    # Fill in any missing settings with defaults
    for key, value in default_settings.items():
        settings.setdefault(key, value)

    shaders = {}
    programs = {}
    techniques = {}
    mesh_list = []
    mod_meshes = {}
    skinned_meshes = {}

    # Clear globals
    g_buffers = []
    g_glExtensionsUsed = []

    object_list = list(scene_delta.get('objects', []))

    # Apply modifiers
    if settings['meshes_apply_modifiers']:
        scene = bpy.context.scene
        mod_obs = [ob for ob in object_list if ob.is_modified(scene, 'PREVIEW')]
        for mesh in scene_delta.get('meshes', []):
            mod_users = [ob for ob in mod_obs if ob.data == mesh]

            # Only convert meshes with modifiers, otherwise each non-modifier
            # user ends up with a copy of the mesh and we lose instancing
            mod_meshes.update({ob.name: ob.to_mesh(scene, True, 'PREVIEW') for ob in mod_users})

            # Add unmodified meshes directly to the mesh list
            if len(mod_users) < mesh.users:
                mesh_list.append(mesh)
        mesh_list.extend(mod_meshes.values())
    else:
        mesh_list = scene_delta.get('meshes', [])

    scenes = scene_delta.get('scenes', [])
    gltf = {
        'asset': {
            'version': '1.0',
            'profile': profile_map[settings['asset_profile']]
        },
        'cameras': export_cameras(scene_delta.get('cameras', [])),
        'extensions': {},
        'extensionsUsed': [],
        'extras': {},
        'images': export_images(settings, scene_delta.get('images', [])),
        'materials': export_materials(settings, scene_delta.get('materials', []),
            shaders, programs, techniques),
        'nodes': export_nodes(settings, scenes, object_list, skinned_meshes, mod_meshes),
        # Make sure meshes come after nodes to detect which meshes are skinned
        'meshes': export_meshes(settings, mesh_list, skinned_meshes),
        'skins': export_skins(skinned_meshes),
        'programs': programs,
        'samplers': {'sampler_default':{}},
        'scene': 'scene_' + bpy.context.scene.name,
        'scenes': export_scenes(settings, scenes),
        'shaders': shaders,
        'techniques': techniques,
        'textures': export_textures(scene_delta.get('textures', [])),

        # TODO
        'animations': {},
    }

    if settings['shaders_data_storage'] == 'NONE':
        gltf['extensionsUsed'].append('KHR_materials_common')
        gltf['extensions']['KHR_materials_common'] = {
            'lights' : export_lights(scene_delta.get('lamps', []))
        }

    if settings['ext_export_actions']:
        gltf['extensionsUsed'].append('BLENDER_actions')
        gltf['extensions']['BLENDER_actions'] = {
            'actions': export_actions(scene_delta.get('actions', [])),
        }

    if settings['ext_export_physics']:
        gltf['extensionsUsed'].append('BLENDER_physics')

    # Retroactively add skins attribute to nodes
    for mesh_name, obj in skinned_meshes.items():
        gltf['nodes']['node_' + obj.name]['skin'] = 'skin_{}'.format(mesh_name)

    # Insert root nodes if axis conversion is needed
    if settings['nodes_global_matrix'] != mathutils.Matrix.Identity(4):
        insert_root_nodes(gltf, togl(settings['nodes_global_matrix']))

    gltf.update(export_buffers(settings))
    gltf.update({'glExtensionsUsed': g_glExtensionsUsed})
    g_buffers = []
    g_glExtensionsUsed = []

    gltf = {key: value for key, value in gltf.items() if value}

    # Remove any temporary meshes from applying modifiers
    for mesh in mod_meshes.values():
        bpy.data.meshes.remove(mesh)

    return gltf