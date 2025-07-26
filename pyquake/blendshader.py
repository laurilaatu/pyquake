__all__ = (
    'im_from_file',
    'setup_diffuse_material',
)


import io

import bpy
import numpy as np
import PIL.Image
import simplejpeg


def im_from_file(f, name):
    b = f.read()

    if simplejpeg.is_jpeg(b):
        # For some reason PIL doesn't like reading JPEGs from within Blender.  Use `simplejpeg` in this case.
        a = simplejpeg.decode_jpeg(b) / 255.
    else:
        bf = io.BytesIO(b)
        pil_im = PIL.Image.open(bf)
        # Convert PIL image to a NumPy array
        a = (np.frombuffer(pil_im.tobytes(), dtype=np.uint8)
             .reshape((pil_im.size[1], pil_im.size[0], -1))) / 255.

    # Ensure image has an alpha channel
    if a.shape[-1] == 3:
        a = np.concatenate([
            a,
            np.ones((a.shape[0], a.shape[1]))[:, :, None]
        ], axis=2)
    elif a.shape[-1] != 4:
        raise Exception('Only RGB and RGBA images are supported')

    blend_im = bpy.data.images.new(name, alpha=True, width=a.shape[1], height=a.shape[0])
    blend_im.pixels = a.ravel()
    blend_im.pack()

    return blend_im


def _new_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # UPDATED: Use nodes.clear() for a cleaner way to remove default nodes
    nodes.clear()
    links.clear()

    return mat, nodes, links


def setup_diffuse_material(im: bpy.types.Image, mat_name: str):
    mat, nodes, links = _new_mat(mat_name)

    output_node = nodes.new('ShaderNodeOutputMaterial')

    # UPDATED: Use Principled BSDF for modern PBR workflows.
    # This provides a result almost identical to the old Diffuse BSDF
    # while being the standard for current Blender versions.
    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    principled_node.inputs['Roughness'].default_value = 1.0
    principled_node.inputs['Specular IOR Level'].default_value = 0.0 # 'Specular' was renamed in Blender 4.0
    links.new(output_node.inputs['Surface'], principled_node.outputs['BSDF'])

    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.image = im
    texture_node.interpolation = 'Linear'
    links.new(principled_node.inputs['Base Color'], texture_node.outputs['Color'])

    return mat