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


import pickle

import bpy
import numpy as np


def _dir_to_matrix(d):
    """Return an orthogonal whose y-axis points in the given direction."""

    y_vec = d
    x_vec = np.cross(d, np.array([0, 0, 1]))
    x_vec /= np.linalg.norm(x_vec, axis=1)[:, None]
    z_vec = np.cross(x_vec, y_vec)

    return np.stack([x_vec, y_vec, z_vec], axis=2)


def _get_path_face_and_verts(points, n_sides, radius):
    dirs = np.concatenate([
        (points[1] - points[0])[None, :],
        points[2:] - points[:-2],
        (points[-1] - points[-2])[None, :],
    ])
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    assert dirs.shape == (len(points), 3)

    mxs = _dir_to_matrix(dirs)
    assert mxs.shape == (len(points), 3, 3)

    theta = np.linspace(0, 2. * np.pi, n_sides, endpoint=False)
    poly = radius * np.stack([np.sin(theta),
                              np.zeros_like(theta),
                              np.cos(theta)], axis=1)
    assert poly.shape == (n_sides, 3)

    verts = (mxs @ poly.T[None, :, :]).transpose(0, 2, 1) + points[:, None, :]
    assert verts.shape == (len(points), n_sides, 3)
    verts = verts.reshape(len(points) * n_sides, 3)

    face_inds1 = np.stack([np.arange(0, n_sides),
                           np.concatenate([np.arange(1, n_sides), [0]]),
                           np.concatenate([np.arange(1, n_sides), [0]]),
                           np.arange(0, n_sides)],
                          axis=1)
    assert face_inds1.shape == (n_sides, 4)

    face_inds2 = np.stack([np.arange(0, len(points) - 1),
                           np.arange(0, len(points) - 1),
                           np.arange(1, len(points)),
                           np.arange(1, len(points))], axis=1)
    assert face_inds2.shape == (len(points) - 1, 4)

    face_inds = np.stack(np.broadcast_arrays(face_inds1[:, None, :],
                                             face_inds2[None, :, :]),
                         axis=3)
    assert face_inds.shape == (n_sides, len(points) - 1, 4, 2)

    face_inds = np.sum(face_inds * [1, n_sides], axis=3)
    face_inds = face_inds.reshape(n_sides * (len(points) - 1), 4)

    return [list(v) for v in verts], [list(f) for f in face_inds]


def _make_path_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()
    links.clear()

    # UPDATED: Use the standard Principled BSDF node instead of the legacy Diffuse BSDF.
    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    principled_node.inputs['Base Color'].default_value = (1, 0, 1, 1)
    principled_node.inputs['Roughness'].default_value = 1.0
    principled_node.inputs['Specular IOR Level'].default_value = 0.0 # Renamed in Blender 4.0

    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    transparent_node.inputs['Color'].default_value = (1, 1, 1, 1)

    mix_node = nodes.new('ShaderNodeMixShader')
    mix_node.inputs['Fac'].default_value = 0.2
    
    output_node = nodes.new('ShaderNodeOutputMaterial')

    links.new(mix_node.inputs[1], transparent_node.outputs['BSDF'])
    links.new(mix_node.inputs[2], principled_node.outputs['BSDF'])
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])

    return mat, mix_node


def _make_emission_path_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()
    links.clear()

    emission_node = nodes.new('ShaderNodeEmission')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    mix_node = nodes.new('ShaderNodeMixShader')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    transparent_node.inputs['Color'].default_value = (1, 1, 1, 1)
    emission_node.inputs['Strength'].default_value = 1
    emission_node.inputs['Color'].default_value = (1, 0, 1, 1)
    mix_node.inputs['Fac'].default_value = 0.0001

    links.new(mix_node.inputs[1], transparent_node.outputs['BSDF'])
    links.new(mix_node.inputs[2], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])

    return mat


def make_path_object(points, n_sides, radius, obj_name, vis_start, vis_mid, vis_end):
    # Create mesh
    verts, face_inds = _get_path_face_and_verts(points, n_sides, radius)

    mesh = bpy.data.meshes.new(obj_name)
    mesh.from_pydata(verts, [], face_inds)
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # Create material
    mat, mix_node = _make_path_material('path_mat')
    mesh.materials.append(mat)

    # Animate the mix node
    inp = mix_node.inputs['Fac']

    inp.default_value = 0.
    inp.keyframe_insert('default_value', frame=vis_start)
    inp.default_value = 0.2
    inp.keyframe_insert('default_value', frame=vis_mid)
    inp.default_value = 0.
    inp.keyframe_insert('default_value', frame=vis_end)

    # Animate render visibility
    obj.hide_render = True
    obj.keyframe_insert('hide_render', frame=0)
    obj.hide_render = False
    obj.keyframe_insert('hide_render', frame=vis_start)
    obj.hide_render = True
    obj.keyframe_insert('hide_render', frame=vis_end)


def get_paths_as_array(pickle_paths):
    for pickle_path in pickle_paths:
        with pickle_path.open('rb') as f:
            paths = pickle.load(f)
        for path in paths: 
            yield np.stack([x['pos'] for x in path])


def path_objects_from_pickles(pickle_paths):
    for i, points in enumerate(get_paths_as_array(pickle_paths)): 
        obj_name = f'path_{i:05d}'
        make_path_object(points, 5, 10, obj_name, i, i + 10, i + 20)
        bpy.data.objects[obj_name].scale = 0.01, 0.01, 0.01