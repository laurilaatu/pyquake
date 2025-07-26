# Copyright (c) 2020 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


__all__ = (
    'array_ims_from_indices',
    'im_from_array',
    'setup_diffuse_material',
    'setup_flat_material',
    'setup_fullbright_material',
    'setup_lightmap_material',
    'setup_light_style_node_groups',
    'setup_sky_material',
    'setup_transparent_fullbright_material',
    'setup_explosion_particle_material',
    'setup_teleport_particle_material'
)


from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict

import bpy
import numpy as np


_MAX_LIGHT_STYLES = 64


def im_from_array(name, array_im):
    im = bpy.data.images.new(name, alpha=True, width=array_im.shape[1], height=array_im.shape[0])
    im.pixels = np.ravel(array_im)
    im.pack()
    return im


def array_ims_from_indices(pal, im_indices, gamma=1.0, light_tint=(1, 1, 1, 1), force_fullbright=False):
    if force_fullbright:
        fullbright_array = np.full_like(im_indices, True)
    else:
        fullbright_array = (im_indices >= 224)

    array_im = pal[im_indices]
    array_im = array_im ** gamma

    if np.any(fullbright_array):
        fullbright_array_im = array_im * fullbright_array[..., None]
        fullbright_array_im *= light_tint
        fullbright_array_im = np.clip(fullbright_array_im, 0., 1.)
    else:
        fullbright_array_im = None

    return array_im, fullbright_array_im, fullbright_array


def setup_light_style_node_groups():
    groups = {}
    for style_idx in range(_MAX_LIGHT_STYLES):
        group = bpy.data.node_groups.new(f'style_{style_idx}', 'ShaderNodeTree')
        group.outputs.new('NodeSocketFloat', 'Value')
        input_node = group.nodes.new('NodeGroupInput')
        output_node = group.nodes.new('NodeGroupOutput')
        value_node = group.nodes.new('ShaderNodeValue')
        value_node.outputs['Value'].default_value = 1.0
        group.links.new(output_node.inputs['Value'], value_node.outputs['Value'])

        groups[style_idx] = group

    return groups


def _new_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = 'OPAQUE' # Default blend method

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()
    links.clear()

    return mat, nodes, links


@dataclass(eq=False)
class BlendMatImagePair:
    im: bpy.types.Image
    fullbright_im: Optional[bpy.types.Image]


@dataclass(eq=False)
class BlendMatImages:
    frames: List[BlendMatImagePair]
    alt_frames: List[BlendMatImagePair]

    @property
    def width(self):
        return self.frames[0].im.size[0]

    @property
    def height(self):
        return self.frames[0].im.size[1]

    @classmethod
    def from_single_diffuse(cls, im: bpy.types.Image):
        return cls(
            frames=[BlendMatImagePair(im, None)],
            alt_frames=[]
        )

    @classmethod
    def from_single_pair(cls, im: bpy.types.Image, fullbright_im: bpy.types.Image):
        return cls(
            frames=[BlendMatImagePair(im, fullbright_im)],
            alt_frames=[]
        )

    @property
    def any_fullbright(self):
        return any(p.fullbright_im is not None for l in [self.frames, self.alt_frames] for p in l)

    @property
    def is_animated(self):
        return len(self.frames) > 1 or len(self.alt_frames) > 1

    @property
    def is_posable(self):
        return len(self.alt_frames) > 0


@dataclass(eq=False)
class BlendMat:
    mat: bpy.types.Material

    def add_time_keyframe(self, time: float, blender_frame: int):
        time_input = self.mat.node_tree.nodes['time'].outputs['Value']
        time_input.default_value = time
        time_input.keyframe_insert('default_value', frame=blender_frame)

        if self.mat.node_tree.animation_data and self.mat.node_tree.animation_data.action:
            fcurve = self.mat.node_tree.animation_data.action.fcurves.find(
                'nodes["time"].outputs[0].default_value'
            )
            if fcurve:
                fcurve.keyframe_points[-1].interpolation = 'LINEAR'

    def add_frame_keyframe(self, frame: int, blender_frame: int):
        frame_input = self.mat.node_tree.nodes['frame'].outputs['Value']
        frame_input.default_value = frame
        frame_input.keyframe_insert('default_value', frame=blender_frame)

    def add_sample_as_light_keyframe(self, vis, blender_frame):
        if not self.mat.use_nodes:
            return

        emission_node = None
        for node in self.mat.node_tree.nodes:
            if node.type == 'EMISSION':
                emission_node = node
                break
        
        if emission_node:
            strength = 10.0 if vis else 0.0
            strength_input = emission_node.inputs.get('Strength')
            if strength_input:
                strength_input.default_value = strength
                strength_input.keyframe_insert(data_path='default_value', frame=blender_frame)
        else:
            print(f"Warning: Material '{self.mat.name}' has no 'Emission' node to keyframe.")

    @property
    def is_animated(self):
        return 'time' in self.mat.node_tree.nodes

    @property
    def is_posable(self):
        return 'frame' in self.mat.node_tree.nodes


def _setup_warp_uv_single(nodes, links, dim1_output, dim2_output, size1, size2):
    mul_node = nodes.new('ShaderNodeMath')
    mul_node.operation = 'MULTIPLY'
    mul_node.inputs[1].default_value = size2 * 2 * np.pi / 128
    links.new(mul_node.inputs[0], dim2_output)

    add_node = nodes.new('ShaderNodeMath')
    add_node.operation = 'ADD'
    links.new(add_node.inputs[0], mul_node.outputs['Value'])

    sine_node = nodes.new('ShaderNodeMath')
    sine_node.operation = 'SINE'
    links.new(sine_node.inputs[0], add_node.outputs['Value'])

    mul2_node = nodes.new('ShaderNodeMath')
    mul2_node.operation = 'MULTIPLY'
    mul2_node.inputs[1].default_value = 8 / size1
    links.new(mul2_node.inputs[0], sine_node.outputs['Value'])

    add2_node = nodes.new('ShaderNodeMath')
    add2_node.operation = 'ADD'
    links.new(add2_node.inputs[0], mul2_node.outputs['Value'])
    links.new(add2_node.inputs[1], dim1_output)

    return [add_node.inputs[1]], add2_node.outputs['Value']


def _setup_warp_uv(nodes, links, width, height):
    uv_node = nodes.new('ShaderNodeUVMap')

    sep_node = nodes.new('ShaderNodeSeparateXYZ')
    links.new(sep_node.inputs['Vector'], uv_node.outputs['UV'])

    u_time_inputs, u_output = _setup_warp_uv_single(
        nodes, links,
        sep_node.outputs['X'], sep_node.outputs['Y'],
        width, height
    )

    v_time_inputs, v_output = _setup_warp_uv_single(
        nodes, links,
        sep_node.outputs['Y'], sep_node.outputs['X'],
        height, width
    )

    combine_node = nodes.new('ShaderNodeCombineXYZ')
    links.new(combine_node.inputs['X'], u_output)
    links.new(combine_node.inputs['Y'], v_output)

    return u_time_inputs + v_time_inputs, combine_node.outputs['Vector']


def _setup_image_nodes(ims: Iterable[Optional[bpy.types.Image]], nodes, links, output_name) -> \
        Tuple[Optional[bpy.types.NodeSocket], List[bpy.types.NodeSocket], List[bpy.types.NodeSocket]]:
    texture_nodes = []
    for im in ims:
        if im is not None:
            texture_node = nodes.new('ShaderNodeTexImage')
            texture_node.image = im
            texture_node.interpolation = 'Closest'
            texture_nodes.append(texture_node)
        else:
            texture_nodes.append(None)

    if len(texture_nodes) == 1:
        if texture_nodes[0] is None:
            return [], [], None
        return [], [texture_nodes[0].inputs['Vector']], texture_nodes[0].outputs[output_name]
    elif len(texture_nodes) > 1:
        prev_output = texture_nodes[0].outputs[output_name] if texture_nodes[0] else None

        mul_node = nodes.new('ShaderNodeMath')
        mul_node.operation = 'MULTIPLY'
        mul_node.inputs[1].default_value = 1.0 / 0.1
        time_input = mul_node.inputs[0]

        mod_node = nodes.new('ShaderNodeMath')
        mod_node.operation = 'MODULO'
        links.new(mod_node.inputs[0], mul_node.outputs['Value'])
        mod_node.inputs[1].default_value = len(texture_nodes)

        floor_node = nodes.new('ShaderNodeMath')
        floor_node.operation = 'FLOOR'
        links.new(floor_node.inputs[0], mod_node.outputs['Value'])
        frame_output = floor_node.outputs['Value']

        for frame_num, texture_node in enumerate(texture_nodes[1:], 1):
            compare_node = nodes.new('ShaderNodeMath')
            compare_node.operation = 'COMPARE'
            compare_node.inputs[0].default_value = frame_num
            links.new(compare_node.inputs[1], frame_output)

            mix_node = nodes.new('ShaderNodeMix')
            mix_node.data_type = 'RGBA'
            
            if prev_output:
                links.new(mix_node.inputs['A'], prev_output)
            else:
                mix_node.inputs['A'].default_value = (0, 0, 0, 1)
                
            if texture_node:
                links.new(mix_node.inputs['B'], texture_node.outputs[output_name])
            else:
                mix_node.inputs['B'].default_value = (0, 0, 0, 1)

            links.new(mix_node.inputs['Factor'], compare_node.outputs['Value'])
            prev_output = mix_node.outputs['Result']

        uv_inputs = [tn.inputs['Vector'] for tn in texture_nodes if tn is not None]
        return [time_input], uv_inputs, prev_output
    else:
        raise ValueError('No images passed')


def _reduce(node_type: str, operation: str, it: Iterable[bpy.types.NodeSocket], nodes, links):
    iter_ = iter(it)
    try:
        accum = next(iter_)
    except StopIteration:
        return None

    for ns in iter_:
        op_node = nodes.new(node_type)
        if hasattr(op_node, 'operation'):
            op_node.operation = operation
        if node_type == 'ShaderNodeMix':
             op_node.data_type = 'RGBA'
             links.new(op_node.inputs['A'], accum)
             links.new(op_node.inputs['B'], ns)
             accum = op_node.outputs['Result']
        else:
            links.new(op_node.inputs[0], accum)
            links.new(op_node.inputs[1], ns)
            output_socket_name = 'Vector' if 'Vector' in op_node.outputs else 'Value'
            accum = op_node.outputs[output_socket_name]
            
    return accum


def _setup_alt_image_nodes(ims: BlendMatImages, nodes, links, warp: bool, fullbright: bool,
                           output_name: str = 'Color') -> \
        Tuple[Optional[bpy.types.NodeSocket], List[bpy.types.NodeSocket], List[bpy.types.NodeSocket]]:
    main_time_inputs, main_uv_inputs, main_output = _setup_image_nodes(
        (p.fullbright_im if fullbright else p.im for p in ims.frames),
        nodes, links, output_name
    )

    time_inputs = main_time_inputs
    uv_inputs = main_uv_inputs
    frame_inputs = []

    if not ims.alt_frames:
        output = main_output
    else:
        alt_time_inputs, alt_uv_inputs, alt_output = _setup_image_nodes(
            (p.fullbright_im if fullbright else p.im for p in ims.alt_frames),
            nodes, links, output_name
        )
        
        mix_node = nodes.new('ShaderNodeMix')
        mix_node.data_type = 'RGBA'
        
        if main_output:
            links.new(mix_node.inputs['A'], main_output)
        else:
            mix_node.inputs['A'].default_value = (0, 0, 0, 1)

        if alt_output:
            links.new(mix_node.inputs['B'], alt_output)
        else:
            mix_node.inputs['B'].default_value = (0, 0, 0, 1)

        output = mix_node.outputs['Result']
        time_inputs.extend(alt_time_inputs)
        uv_inputs.extend(alt_uv_inputs)
        frame_inputs.append(mix_node.inputs['Factor'])

    if warp:
        warp_time_inputs, uv_output = _setup_warp_uv(nodes, links, ims.width, ims.height)
        for uv_input in uv_inputs:
            links.new(uv_input, uv_output)
        time_inputs.extend(warp_time_inputs)

    return output, time_inputs, frame_inputs


def _create_value_node(inputs, nodes, links, name):
    value_node = nodes.new('ShaderNodeValue')
    value_node.name = name
    for inp in inputs:
        links.new(inp, value_node.outputs['Value'])


def _create_inputs(frame_inputs, time_inputs, nodes, links):
    if frame_inputs:
        _create_value_node(frame_inputs, nodes, links, 'frame')
    if time_inputs:
        _create_value_node(time_inputs, nodes, links, 'time')


def setup_sky_material(ims: BlendMatImages, mat_name: str):
    mat, nodes, links = _new_mat(mat_name)
    image = ims.frames[0].im
    
    output_node = nodes.new('ShaderNodeOutputMaterial')
    mix_shader = nodes.new('ShaderNodeMixShader')
    light_path = nodes.new('ShaderNodeLightPath')
    transparent_bsdf = nodes.new('ShaderNodeBsdfTransparent')
    emission = nodes.new('ShaderNodeEmission')
    
    links.new(output_node.inputs['Surface'], mix_shader.outputs['Shader'])
    links.new(mix_shader.inputs['Fac'], light_path.outputs['Is Camera Ray'])
    links.new(mix_shader.inputs[1], transparent_bsdf.outputs['BSDF'])
    links.new(mix_shader.inputs[2], emission.outputs['Emission'])
    
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_front = nodes.new('ShaderNodeTexImage')
    tex_front.image = image
    tex_front.extension = 'REPEAT'
    tex_back = nodes.new('ShaderNodeTexImage')
    tex_back.image = image
    tex_back.extension = 'REPEAT'
    
    map_front = nodes.new('ShaderNodeMapping')
    map_front.inputs['Scale'].default_value = (0.5, 1, 1)
    map_back = nodes.new('ShaderNodeMapping')
    map_back.inputs['Location'].default_value = (0.5, 0, 0)
    map_back.inputs['Scale'].default_value = (0.5, 1, 1)
    
    links.new(map_front.inputs['Vector'], tex_coord.outputs['Generated'])
    links.new(map_back.inputs['Vector'], tex_coord.outputs['Generated'])
    links.new(tex_front.inputs['Vector'], map_front.outputs['Vector'])
    links.new(tex_back.inputs['Vector'], map_back.outputs['Vector'])
    
    time_node = nodes.new('ShaderNodeValue')
    time_node.name = 'time'
    
    add_front_uv = nodes.new('ShaderNodeVectorMath')
    add_front_uv.operation = 'ADD'
    add_front_uv.inputs[1].default_value[0] = 0.25
    links.new(add_front_uv.inputs[0], time_node.outputs['Value'])
    
    add_back_uv = nodes.new('ShaderNodeVectorMath')
    add_back_uv.operation = 'ADD'
    add_back_uv.inputs[1].default_value[0] = 0.125
    links.new(add_back_uv.inputs[0], time_node.outputs['Value'])

    add_front_loc = nodes.new('ShaderNodeVectorMath')
    add_front_loc.operation = 'ADD'
    links.new(add_front_loc.inputs[0], map_front.outputs['Vector'])
    links.new(add_front_loc.inputs[1], add_front_uv.outputs['Vector'])
    links.new(tex_front.inputs['Vector'], add_front_loc.outputs['Vector'])
    
    add_back_loc = nodes.new('ShaderNodeVectorMath')
    add_back_loc.operation = 'ADD'
    links.new(add_back_loc.inputs[0], map_back.outputs['Vector'])
    links.new(add_back_loc.inputs[1], add_back_uv.outputs['Vector'])
    links.new(tex_back.inputs['Vector'], add_back_loc.outputs['Vector'])
    
    mix_color = nodes.new('ShaderNodeMix')
    mix_color.data_type = 'RGBA'
    mix_color.blend_type = 'MIX'
    
    links.new(mix_color.inputs['A'], tex_back.outputs['Color'])
    links.new(mix_color.inputs['B'], tex_front.outputs['Color'])
    mix_color.inputs['Factor'].default_value = 0.5 
    
    links.new(emission.inputs['Color'], mix_color.outputs['Result'])

    return BlendMat(mat)


def setup_diffuse_material(ims: BlendMatImages, mat_name: str, warp: bool):
    mat, nodes, links = _new_mat(mat_name)
    im_output, time_inputs, frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=False)
    
    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    principled_node.inputs['Roughness'].default_value = 1.0
    principled_node.inputs['Specular IOR Level'].default_value = 0.0

    output_node = nodes.new('ShaderNodeOutputMaterial')
    
    if im_output:
        links.new(principled_node.inputs['Base Color'], im_output)
    links.new(output_node.inputs['Surface'], principled_node.outputs['BSDF'])
    
    _create_inputs(frame_inputs, time_inputs, nodes, links)
    return BlendMat(mat)


def setup_fullbright_material(ims: BlendMatImages, mat_name: str, strength: float, cam_strength: float, warp: bool):
    mat, nodes, links = _new_mat(mat_name)
    
    diffuse_im_output, diffuse_time_inputs, diffuse_frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=False)
    fullbright_im_output, fullbright_time_inputs, fullbright_frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=True)
    time_inputs = diffuse_time_inputs + fullbright_time_inputs
    frame_inputs = diffuse_frame_inputs + fullbright_frame_inputs

    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    principled_node.inputs['Roughness'].default_value = 1.0
    principled_node.inputs['Specular IOR Level'].default_value = 0.0

    output_node = nodes.new('ShaderNodeOutputMaterial')
    add_node = nodes.new('ShaderNodeAddShader')
    emission_node = nodes.new('ShaderNodeEmission')
    
    if strength == cam_strength:
        emission_node.inputs['Strength'].default_value = strength
    else:
        map_range_node = nodes.new('ShaderNodeMapRange')
        map_range_node.inputs[1].default_value = 0
        map_range_node.inputs[2].default_value = 1
        map_range_node.inputs[3].default_value = strength
        map_range_node.inputs[4].default_value = cam_strength
        links.new(emission_node.inputs['Strength'], map_range_node.outputs['Result'])
        light_path_node = nodes.new('ShaderNodeLightPath')
        links.new(map_range_node.inputs['Value'], light_path_node.outputs['Is Camera Ray'])

    if diffuse_im_output:
        links.new(principled_node.inputs['Base Color'], diffuse_im_output)
    if fullbright_im_output:
        links.new(emission_node.inputs['Color'], fullbright_im_output)
    
    links.new(add_node.inputs[0], principled_node.outputs['BSDF'])
    links.new(add_node.inputs[1], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], add_node.outputs['Shader'])
    
    _create_inputs(frame_inputs, time_inputs, nodes, links)
    return BlendMat(mat)


def setup_transparent_fullbright_material(ims: BlendMatImages, mat_name: str, strength: float, warp: bool):
    mat, nodes, links = _new_mat(mat_name)
    mat.blend_method = 'BLEND'
    #mat.shadow_method = 'NONE'

    diffuse_im_output, diffuse_time_inputs, diffuse_frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=False)
    alpha_output, alpha_time_inputs, alpha_frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=True, output_name='Alpha')
    time_inputs = diffuse_time_inputs + alpha_time_inputs
    frame_inputs = diffuse_frame_inputs + alpha_frame_inputs

    emission_node = nodes.new('ShaderNodeEmission')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    mix_shader_node = nodes.new('ShaderNodeMixShader')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    
    emission_node.inputs['Strength'].default_value = strength
    
    if diffuse_im_output:
        links.new(emission_node.inputs['Color'], diffuse_im_output)
    if alpha_output:
        links.new(mix_shader_node.inputs['Fac'], alpha_output)
        
    links.new(mix_shader_node.inputs[1], transparent_node.outputs['BSDF'])
    links.new(mix_shader_node.inputs[2], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], mix_shader_node.outputs['Shader'])
    
    _create_inputs(frame_inputs, time_inputs, nodes, links)
    return BlendMat(mat)


def setup_explosion_particle_material(mat_name):
    mat, nodes, links = _new_mat(mat_name)
    mat.blend_method = 'BLEND'

    output_node = nodes.new('ShaderNodeOutputMaterial')
    mix_node = nodes.new('ShaderNodeMixShader')
    emission_node = nodes.new('ShaderNodeEmission')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    blackbody_node = nodes.new('ShaderNodeBlackbody')
    map_range_node = nodes.new('ShaderNodeMapRange')
    div_node = nodes.new('ShaderNodeMath')
    particle_info_node = nodes.new('ShaderNodeParticleInfo')

    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])
    links.new(mix_node.inputs[1], emission_node.outputs['Emission'])
    links.new(mix_node.inputs[2], transparent_node.outputs['BSDF'])
    links.new(emission_node.inputs['Color'], blackbody_node.outputs['Color'])
    links.new(blackbody_node.inputs['Temperature'], map_range_node.outputs['Result'])
    links.new(map_range_node.inputs['Value'], div_node.outputs['Value'])
    
    links.new(mix_node.inputs['Fac'], div_node.outputs['Value'])
    
    links.new(div_node.inputs[0], particle_info_node.outputs['Age'])
    links.new(div_node.inputs[1], particle_info_node.outputs['Lifetime'])

    emission_node.inputs['Strength'].default_value = 10
    map_range_node.inputs[1].default_value = 0
    map_range_node.inputs[2].default_value = 1
    map_range_node.inputs[3].default_value = 4000
    map_range_node.inputs[4].default_value = 500
    div_node.operation = 'DIVIDE'
    
    return BlendMat(mat)


def setup_teleport_particle_material(mat_name):
    mat, nodes, links = _new_mat(mat_name)
    mat.blend_method = 'BLEND'

    output_node = nodes.new('ShaderNodeOutputMaterial')
    mix_node = nodes.new('ShaderNodeMixShader')
    emission_node = nodes.new('ShaderNodeEmission')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    color_ramp_node = nodes.new('ShaderNodeValToRGB')
    div_node = nodes.new('ShaderNodeMath')
    particle_info_node = nodes.new('ShaderNodeParticleInfo')
    
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])
    links.new(mix_node.inputs[1], emission_node.outputs['Emission'])
    links.new(mix_node.inputs[2], transparent_node.outputs['BSDF'])
    links.new(emission_node.inputs['Color'], color_ramp_node.outputs['Color'])

    links.new(mix_node.inputs['Fac'], div_node.outputs['Value'])
    
    links.new(div_node.inputs[0], particle_info_node.outputs['Age'])
    links.new(div_node.inputs[1], particle_info_node.outputs['Lifetime'])
    links.new(color_ramp_node.inputs['Fac'], particle_info_node.outputs['Random'])
    
    emission_node.inputs['Strength'].default_value = 0.5
    div_node.operation = 'DIVIDE'
    
    return BlendMat(mat)


def setup_lightmap_material(mat_name: str, ims: BlendMatImages,
                            lightmap_ims: List[bpy.types.Image], lightmap_uv_layer_name: str,
                            warp: bool,
                            lightmap_styles: Tuple[int],
                            style_node_groups: Dict[int, bpy.types.ShaderNodeTree]):
    mat, nodes, links = _new_mat(mat_name)

    im_output, time_inputs, frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=False)
    if ims.any_fullbright:
        fullbright_im_output, fullbright_time_inputs, fullbright_frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=True, output_name='Alpha')
        time_inputs.extend(fullbright_time_inputs)
        frame_inputs.extend(fullbright_frame_inputs)

    output_node = nodes.new('ShaderNodeOutputMaterial')
    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    principled_node.inputs['Roughness'].default_value = 1.0
    principled_node.inputs['Specular IOR Level'].default_value = 0.0
    links.new(output_node.inputs['Surface'], principled_node.outputs['BSDF'])

    color_mul_node = nodes.new('ShaderNodeMix')
    color_mul_node.data_type = 'RGBA'
    color_mul_node.blend_type = 'MULTIPLY'
    color_mul_node.inputs['Factor'].default_value = 1.0
    links.new(principled_node.inputs['Base Color'], color_mul_node.outputs['Result'])
    if im_output:
        links.new(color_mul_node.inputs['A'], im_output)

    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = lightmap_uv_layer_name

    lightmap_outputs = []
    for lightmap_idx in (idx for idx in range(min(4, len(lightmap_ims))) if lightmap_styles[idx] != 255):
        lightmap_mul_node = nodes.new('ShaderNodeMix')
        lightmap_mul_node.data_type = 'RGBA'
        lightmap_mul_node.blend_type = 'MULTIPLY'
        lightmap_mul_node.inputs['Factor'].default_value = 1.0
        lightmap_outputs.append(lightmap_mul_node.outputs['Result'])

        lightmap_texture_node = nodes.new('ShaderNodeTexImage')
        lightmap_texture_node.image = lightmap_ims[lightmap_idx]
        lightmap_texture_node.interpolation = 'Linear'
        group_node = nodes.new('ShaderNodeGroup')
        group_node.node_tree = style_node_groups[lightmap_styles[lightmap_idx]]
        
        links.new(lightmap_mul_node.inputs['A'], lightmap_texture_node.outputs['Color'])
        links.new(lightmap_mul_node.inputs['B'], group_node.outputs['Value'])
        links.new(lightmap_texture_node.inputs['Vector'], uv_node.outputs['UV'])

    lightmap_sum = _reduce('ShaderNodeMix', 'ADD', lightmap_outputs, nodes, links)

    if not ims.any_fullbright:
        if lightmap_sum:
            links.new(color_mul_node.inputs['B'], lightmap_sum)
        else:
            color_mul_node.inputs['B'].default_value = (1, 1, 1, 1)
    else:
        mix_rgb_node = nodes.new('ShaderNodeMix')
        mix_rgb_node.data_type = 'RGBA'
        mix_rgb_node.blend_type = 'MIX'
        mix_rgb_node.inputs['B'].default_value = (1, 1, 1, 1)
        links.new(color_mul_node.inputs['B'], mix_rgb_node.outputs['Result'])
        if lightmap_sum:
            links.new(mix_rgb_node.inputs['A'], lightmap_sum)
        else:
            mix_rgb_node.inputs['A'].default_value = (0, 0, 0, 1)
        if fullbright_im_output:
            links.new(mix_rgb_node.inputs['Factor'], fullbright_im_output)
            
    _create_inputs(frame_inputs, time_inputs, nodes, links)
    return BlendMat(mat)


def setup_flat_material(mat_name: str, ims: BlendMatImages, warp: bool):
    mat, nodes, links = _new_mat(mat_name)

    im_output, time_inputs, frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=False)
    output_node = nodes.new('ShaderNodeOutputMaterial')
    emission_node = nodes.new('ShaderNodeEmission')
    links.new(output_node.inputs['Surface'], emission_node.outputs['Emission'])

    if not warp:
        if im_output:
            links.new(emission_node.inputs['Color'], im_output)
    else:
        color_mul_node = nodes.new('ShaderNodeMix')
        color_mul_node.data_type = 'RGBA'
        color_mul_node.blend_type = 'MULTIPLY'
        color_mul_node.inputs['Factor'].default_value = 1.0
        links.new(emission_node.inputs['Color'], color_mul_node.outputs['Result'])

        if im_output:
            links.new(color_mul_node.inputs['A'], im_output)
        color_mul_node.inputs['B'].default_value = (0.25, 0.25, 0.25, 1.0)
        
    _create_inputs(frame_inputs, time_inputs, nodes, links)
    return BlendMat(mat)


def setup_sky_material(ims: BlendMatImages, mat_name: str):
    mat, nodes, links = _new_mat(mat_name)
    image = ims.frames[0].im
    
    output_node = nodes.new('ShaderNodeOutputMaterial')
    emission_node = nodes.new('ShaderNodeEmission')
    links.new(output_node.inputs['Surface'], emission_node.outputs['Emission'])
    
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = image
    links.new(emission_node.inputs['Color'], tex_node.outputs['Color'])
    
    tex_coord = nodes.new('ShaderNodeTexCoord')
    links.new(tex_node.inputs['Vector'], tex_coord.outputs['Window'])
    
    # Add time node to make it animated
    time_node = nodes.new('ShaderNodeValue')
    time_node.name = 'time'
    
    return BlendMat(mat)