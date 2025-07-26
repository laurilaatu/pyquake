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


import collections
import dataclasses
import functools
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import bpy
import bpy_types
import numpy as np

from . import proto, bsp, mdl, blendmdl, blendbsp, simplex, blendpart, blendmat


logger = logging.getLogger(__name__)
Vec3 = Tuple[float, float, float]


_NEAR_CLIP_PLANE = 8
_FAR_CLIP_PLANE = 2048
_EYE_HEIGHT = 22

# --- 3rd Person Camera Settings ---
THIRD_PERSON_DISTANCE = 120.0
THIRD_PERSON_HEIGHT = 32.0
# ------------------------------------


def _patch_vec(old_vec: Vec3, update):
    return tuple(v if u is None else u for v, u in zip(old_vec, update))


def _quake_to_blender_angles(quake_angles: Vec3) -> Vec3:
    return (math.pi * (90. - quake_angles[0]) / 180,
            math.pi * quake_angles[2] / 180,
            math.pi * (quake_angles[1] - 90.) / 180)


def _fix_angles(old_angles, new_angles, degrees=False):
    if degrees:
        t = 360
    else:
        t = 2 * np.pi

    old_yaw = old_angles[1] / t
    new_yaw = new_angles[1] / t
    i = old_yaw // 1
    j = np.argmin(np.abs(np.array([i - 1, i, i + 1]) + new_yaw - old_yaw))
    return (new_angles[0], (new_yaw + i + j - 1) * t, new_angles[2])


def _quake_angles_to_mat(angles):
    pitch = angles[0] * (np.pi / 180);
    yaw = angles[1] * (np.pi / 180);
    roll = angles[2] * (np.pi / 180);

    sy, cy = np.sin(yaw), np.cos(yaw)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sr, cr = np.sin(roll), np.cos(roll)

    right = np.array([-1*sr*sp*cy + -1*cr*-sy, -1*sr*sp*sy + -1*cr*cy, -1*sr*cp])
    forward = np.array([cp*cy, cp*sy, -sp])
    up = np.array([cr*sp*cy + -sr*-sy, cr*sp*sy + -sr*cy, cr*cp])

    return np.stack([right, forward, up], axis=1)


def _get_model_config(mdl_name, config):
    mdls_cfg = config['models']
    cfg = dict(mdls_cfg['__default__'])
    cfg.update(mdls_cfg.get(mdl_name, {}))
    return cfg


@dataclass
class _EntityInfo:
    model_num: int
    frame: int
    skin: int
    origin: Vec3
    angles: Vec3

    def update(self, msg: proto.ServerMessageUpdate, baseline: "_EntityInfo"):
        def none_or(a, b):
            return a if a is not None else b

        angles = _fix_angles(self.angles, _patch_vec(baseline.angles, msg.angle))

        return dataclasses.replace(
            baseline,
            model_num=none_or(msg.model_num, baseline.model_num),
            frame=none_or(msg.frame, baseline.frame),
            skin=none_or(msg.skin, baseline.skin),
            origin=_patch_vec(baseline.origin, msg.origin),
            angles=angles
        )


_DEFAULT_BASELINE = _EntityInfo(0, 0, 0, (0, 0, 0), (0, 0, 0))


class SampleAsLightObject:
    @property
    def bbox(self): raise NotImplementedError
    @property
    def leaf(self): raise NotImplementedError
    def add_keyframe(self, vis: bool, blender_frame: int): raise NotImplementedError


class AliasModelSampleAsLightObject:
    _bm: blendmdl.BlendMdl
    _bb: blendbsp.BlendBsp

    def __init__(self, bm, bb, mdl_cfg):
        self._bm = bm
        self._bb = bb
        self._bbox = np.array(mdl_cfg['bbox'])

    @property
    def bbox(self):
        return np.array(self._bm.obj.location) + self._bbox

    @property
    def leaf(self):
        return self._bb.bsp.models[0].get_leaf_from_point(self._bm.obj.location)

    def add_keyframe(self, vis: bool, blender_frame: int):
        for bmat in self._bm.sample_as_light_mats:
            bmat.add_sample_as_light_keyframe(vis, blender_frame)


@dataclass(eq=False)
class LeafSampleAsLightObject:
    _leaf: bsp.Leaf
    _mat: blendmat.BlendMat
    _tex_cfg: Dict
    _model_idx: int
    _bb: blendbsp.BlendBsp

    @property
    def bbox(self):
        if "bbox" not in self._tex_cfg:
            raise Exception("Sample as light textures must have bounding boxes")
        tex_bbox = np.array(self._tex_cfg['bbox'])
        return np.stack([self._leaf.bbox.mins + tex_bbox[0],
                         self._leaf.bbox.maxs + tex_bbox[1]]) + self._model_origin

    @property
    def _model(self):
        return self._bb.bsp.models[self._model_idx]

    @property
    def _model_origin(self):
        model_obj = self._bb.model_objs.get(self._model)
        return np.array(model_obj.location) if model_obj else np.array((0., 0., 0.))

    @property
    def leaf(self):
        if self._model_idx == 0:
            return self._leaf
        else:
            leaf_origin = 0.5 * (np.array(self._leaf.bbox.mins) +
                                 np.array(self._leaf.bbox.maxs))
            origin = leaf_origin + self._model_origin
            return self._bb.bsp.models[0].get_leaf_from_point(origin)

    def add_keyframe(self, vis: bool, blender_frame: int):
        self._mat.add_sample_as_light_keyframe(vis, blender_frame)

    @classmethod
    def create_from_bsp(cls, bb: blendbsp.BlendBsp):
        if bb.sample_as_light_info:
            return (
                cls(leaf, mat, tex_cfg, model.id_, bb)
                for model, model_info in bb.sample_as_light_info.items()
                for leaf, leaf_info in model_info.items()
                for mat, tex_cfg in leaf_info.items()
            )
        return []


@dataclass
class ManagedObject:
    fps: float
    def _get_blender_frame(self, time): return int(round(self.fps * time))
    @property
    def ignore_duplicate_updates(self) -> bool: raise NotImplementedError
    def add_pose_keyframe(self, pose_num: int, time: float): raise NotImplementedError
    def add_visible_keyframe(self, visible: bool, time: float): raise NotImplementedError
    def add_origin_keyframe(self, origin: Vec3, time: float): raise NotImplementedError
    def add_angles_keyframe(self, angles: Vec3, time: float): raise NotImplementedError
    def set_invisible_to_camera(self): raise NotImplementedError
    def done(self, final_time: float): raise NotImplementedError


@dataclass
class AliasModelManagedObject(ManagedObject):
    bm: blendmdl.BlendMdl
    @property
    def ignore_duplicate_updates(self) -> bool: return True
    def add_pose_keyframe(self, pose_num: int, time: float): self.bm.add_pose_keyframe(pose_num, time, self.fps)
    def add_visible_keyframe(self, visible: bool, time: float):
        blender_frame = self._get_blender_frame(time)
        for sub_obj in self.bm.sub_objs:
            sub_obj.hide_render = not visible
            sub_obj.keyframe_insert('hide_render', frame=blender_frame)
            sub_obj.hide_viewport = not visible
            sub_obj.keyframe_insert('hide_viewport', frame=blender_frame)
    def add_origin_keyframe(self, origin: Vec3, time: float):
        self.bm.obj.location = origin
        self.bm.obj.keyframe_insert('location', frame=self._get_blender_frame(time))
    def add_angles_keyframe(self, angles: Vec3, time: float):
        self.bm.obj.rotation_euler = (0., 0., angles[1])
        if self.bm.am.header['flags'] & mdl.ModelFlags.ROTATE:
            self.bm.obj.rotation_euler.z = time * 100. * np.pi / 180
        self.bm.obj.keyframe_insert('rotation_euler', frame=self._get_blender_frame(time))
    def set_invisible_to_camera(self): self.bm.set_invisible_to_camera()
    def done(self, final_time: float): self.bm.done(final_time, self.fps)


@dataclass
class BspModelManagedObject(ManagedObject):
    _bb: blendbsp.BlendBsp
    _model_num: int
    @property
    def ignore_duplicate_updates(self) -> bool: return False
    def add_pose_keyframe(self, pose_num: int, time: float):
        model = self._bb.bsp.models[self._model_num]
        self._bb.add_material_frame_keyframe(model, pose_num, self._get_blender_frame(time))
    @property
    def _model(self): return self._bb.bsp.models[self._model_num]
    def add_visible_keyframe(self, visible: bool, time: float):
        blender_frame = self._get_blender_frame(time)
        self._bb.add_visible_keyframe(self._model, visible, blender_frame)
    def add_origin_keyframe(self, origin: Vec3, time: float):
        obj = self._bb.model_objs[self._model]
        obj.location = origin
        obj.keyframe_insert('location', frame=self._get_blender_frame(time))
    def add_angles_keyframe(self, angles: Vec3, time: float): pass
    def done(self, final_time: float):
        self._bb.add_animated_material_keyframes(self._get_blender_frame(final_time), final_time)


@dataclass
class NullManagedObject(ManagedObject):
    @property
    def ignore_duplicate_updates(self) -> bool: return True
    def add_pose_keyframe(self, pose_num: int, time: float): pass
    def add_visible_keyframe(self, visible: bool, time: float): pass
    def add_origin_keyframe(self, origin: Vec3, time: float): pass
    def add_angles_keyframe(self, angles: Vec3, time: float): pass
    def set_invisible_to_camera(self): pass
    def done(self, final_time: float): pass


class ObjectManager:
    def __init__(self, fs, config, fps, fov, width, height, camera_mode, world_obj_name='demo', load_level=True):
        self._fs = fs
        self._fps = fps
        self._config = config
        self._load_level = load_level
        self._camera_mode = camera_mode

        self._pal = np.frombuffer(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255

        self._bb: Optional[blendbsp.BlendBsp] = None
        self._objs: Dict[Tuple[int, int], ManagedObject] = {}
        self._model_paths: Optional[List[str]] = None
        self._view_entity_num: Optional[int] = None
        self._static_objects: List[ManagedObject] = []
        self._sample_as_light_objects: List[SampleAsLightObject] = []
        self._sal_time: float = 0.
        self._first_update_time: Optional[float] = None
        self._intermission = False
        self._num_explosions: int = 0
        self._num_teleports: int = 0

        self.world_obj = bpy.data.objects.new(world_obj_name, None)
        bpy.context.scene.collection.objects.link(self.world_obj)

        self._particles = blendpart.Particles()
        self._particles.root.parent = self.world_obj

        self._width, self._height = width, height
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height

        self._fov = fov
        demo_cam = bpy.data.cameras.new(name="demo_cam")
        demo_cam.lens_unit = 'FOV'
        demo_cam.angle_x = fov * np.pi / 180
        demo_cam.clip_start = 0.04
        self._demo_cam_obj = bpy.data.objects.new(name="demo_cam", object_data=demo_cam)
        bpy.context.scene.collection.objects.link(self._demo_cam_obj)
        self._demo_cam_obj.parent = self.world_obj

    def set_intermission(self, i: bool):
        self._intermission = i

    def _path_to_bsp_name(self, bsp_path):
        m = re.match(r"maps/([a-zA-Z0-9_]+).bsp", bsp_path)
        if m is None:
            raise Exception(f"Unexpected BSP path {bsp_path}")
        return m.group(1)

    def set_model_paths(self, model_paths: List[str]):
        if self._model_paths is not None:
            raise Exception("Model paths already set")
        self._model_paths = model_paths
        if self._load_level:
            map_path = self._model_paths[0]
            logger.info('Parsing bsp %s', map_path)
            b = bsp.Bsp(self._fs.open(map_path))
            map_name = self._path_to_bsp_name(map_path)
            logger.info('Adding bsp %s', map_path)
            self._bb = blendbsp.add_bsp(b, self._pal, map_name, self._config)
            self._bb.map_obj.parent = self.world_obj
            self._bb.hide_all_but_main()
            self._sample_as_light_objects.extend(LeafSampleAsLightObject.create_from_bsp(self._bb))

    def _path_to_model_name(self, mdl_path):
        m = re.match(r"progs/([A-Za-z0-9-_]*)\.mdl", mdl_path)
        if m is None:
            raise Exception(f"Unexpected model path {mdl_path}")
        return m.group(1)

    @functools.lru_cache(None)
    def _load_alias_model(self, model_path):
        return mdl.AliasModel(self._fs.open(model_path))

    def set_view_entity(self, entity_num):
        self._view_entity_num = entity_num

    def create_static_object(self, model_num, frame, origin, angles, skin):
        model_path = self._model_paths[model_num - 1]
        if model_path.endswith('.spr'): return
        am = mdl.AliasModel(self._fs.open(model_path))
        mdl_name = self._path_to_model_name(model_path)
        mdl_cfg = _get_model_config(mdl_name, self._config)
        bm = blendmdl.add_model(am, self._pal, mdl_name, f"static{len(self._static_objects)}", skin, mdl_cfg, frame, self._config['do_materials'])
        bm.obj.parent = self.world_obj
        bm.obj.location = origin
        bm.obj.rotation_euler = (0., 0., angles[1])
        self._static_objects.append(bm)
        if bm.sample_as_light_mats:
            self._sample_as_light_objects.append(AliasModelSampleAsLightObject(bm, self._bb, mdl_cfg))

    def create_teleport(self, pos, time):
        obj_name = f'teleport{self._num_teleports}'
        self._particles.create_teleport(time, obj_name, pos, self._fps)
        self._num_teleports += 1

    def create_explosion(self, pos, time):
        obj_name = f'explosion{self._num_explosions}'
        self._particles.create_explosion(time, obj_name, pos, self._fps)
        self._num_explosions += 1

    def _create_managed_object(self, entity_num, model_num, skin_num, initial_pose_num):
        model_path = self._model_paths[model_num - 1] if model_num != 0 else None
        if model_num == 0:
            managed_obj = NullManagedObject(self._fps)
        elif model_path.startswith('*'):
            if self._load_level:
                map_model_idx = int(model_path[1:])
                managed_obj = BspModelManagedObject(self._fps, self._bb, map_model_idx)
            else:
                managed_obj = NullManagedObject(self._fps)
        elif model_path.endswith('.mdl'):
            am = self._load_alias_model(model_path)
            mdl_name = self._path_to_model_name(model_path)
            logger.info('Loading alias model %s', mdl_name)
            mdl_cfg = _get_model_config(mdl_name, self._config)
            bm = blendmdl.add_model(am, self._pal, mdl_name, f'ent{entity_num}_{mdl_name}', skin_num, mdl_cfg, initial_pose_num, self._config['do_materials'])
            bm.obj.parent = self.world_obj
            managed_obj = AliasModelManagedObject(self._fps, bm)
            if bm.sample_as_light_mats:
                self._sample_as_light_objects.append(AliasModelSampleAsLightObject(bm, self._bb, mdl_cfg))
        elif model_path.endswith('.bsp'):
            bsp_name = self._path_to_bsp_name(model_path)
            logger.info('Loading bsp model %s', bsp_name)
            b = bsp.Bsp(self._fs.open(model_path))
            if len(b.models) != 1:
                raise Exception(f"Expected one model in bsp model {bsp_name}, not {len(b.models)}")
            bb = blendbsp.add_bsp(b, self._pal, bsp_name, self._config, f'ent{entity_num}_')
            bb.map_obj.parent = self.world_obj
            managed_obj = BspModelManagedObject(self._fps, bb, 0)
        else:
            logging.warning('Cannot handle model %r', model_path)
            managed_obj = NullManagedObject(self._fps)
        return managed_obj

    def _view_simplex(self, view_origin, view_angles):
        view_origin = np.array(view_origin)
        aspect_ratio = self._width / self._height
        tan_fov = np.tan(0.5 * self._fov * np.pi / 180)
        h_tan = tan_fov if aspect_ratio > 1 else v_tan * aspect_ratio
        v_tan = h_tan / aspect_ratio if aspect_ratio > 1 else tan_fov
        constraints = np.array([[-1, h_tan, 0, 0], [1, h_tan, 0, 0], [0, v_tan, 1, 0], [0, v_tan, -1, 0], [0, 1, 0, -_NEAR_CLIP_PLANE], [0, -1, 0, _FAR_CLIP_PLANE]])
        constraints[:, :3] /= np.linalg.norm(constraints[:, :3], axis=1)[:, None]
        rotation_matrix = _quake_angles_to_mat(view_angles)
        constraints[:, :3] = constraints[:, :3] @ rotation_matrix.T
        constraints[:, 3] -= np.einsum('ij,j->i', constraints[:, :3], view_origin)
        return simplex.Simplex(3, constraints, np.array([False, True, False, True, False, True]))

    def _simplex_bbox_test(self, bbox: np.ndarray, sx: simplex.Simplex):
        bbox_simplex = simplex.Simplex.from_bbox(*bbox)
        try:
            bbox_simplex.intersect(sx)
            return True
        except simplex.Infeasible:
            return False

    def _update_sample_as_light(self, view_origin, view_angles, blender_frame, crude_test=True):
        if not self._load_level or self._bb is None: return
        
        view_sx = self._view_simplex(view_origin, view_angles)

        for sal_obj in self._sample_as_light_objects:
            # --- UPDATED TO DISABLE PVS CULLING ---
            # The original PVS check is commented out. We now force `vis = True`
            # to ensure all lights are considered, and let the frustum culling below handle visibility.
            # view_pvs = set(self._bb.bsp.models[0].get_leaf_from_point(view_origin).visible_leaves)
            # pvs = view_pvs & set(sal_obj.leaf.visible_leaves)
            # vis = bool(pvs)
            vis = True
            # --- END UPDATE ---

            if vis:
                leaf_bboxes = np.stack([[leaf.bbox.mins, leaf.bbox.maxs] for leaf in sal_obj.leaf.visible_leaves if leaf in self._bb.bsp.leaves])
                if leaf_bboxes.size == 0:
                    vis = False
                else:
                    bboxes = np.stack([np.maximum(leaf_bboxes[:, 0, :], sal_obj.bbox[0][None, :]), np.minimum(leaf_bboxes[:, 1, :], sal_obj.bbox[1][None, :])], axis=1)
                    bboxes = bboxes[np.all(bboxes[:, 0] < bboxes[:, 1], axis=1)]
                    vis = bboxes.shape[0] != 0

            if vis:
                crude_bbox = np.stack([np.min(bboxes[:, 0, :], axis=0), np.max(bboxes[:, 1, :], axis=0)])
                vis = self._simplex_bbox_test(crude_bbox, view_sx)

            if not crude_test and vis:
                vis = any(self._simplex_bbox_test(bbox, view_sx) for bbox in bboxes)

            sal_obj.add_keyframe(vis, blender_frame)

    def update(self, time, prev_entities, entities, prev_updated, updated, view_angles):
        blender_frame = int(round(self._fps * time))
        if self._intermission and self._view_entity_num in entities:
            view_angles = tuple(x * 180 / np.pi for x in entities[self._view_entity_num].angles)
        for entity_num in prev_updated:
            if (entity_num, prev_entities[entity_num].model_num) not in self._objs: continue
            obj = self._objs[entity_num, prev_entities[entity_num].model_num]
            if (prev_entities[entity_num].model_num != entities[entity_num].model_num or entity_num not in updated):
                obj.add_visible_keyframe(False, time)
                obj.add_origin_keyframe(prev_entities[entity_num].origin, time)
                obj.add_angles_keyframe(prev_entities[entity_num].angles, time)
        for entity_num in updated:
            ent = entities[entity_num]
            key = (entity_num, ent.model_num)
            if key not in self._objs:
                obj = self._create_managed_object(entity_num, ent.model_num, ent.skin, ent.frame)
                obj.add_visible_keyframe(False, 0)
                self._objs[key] = obj
            else:
                obj = self._objs[key]
            prev_ent = prev_entities.get(entity_num)
            if not obj.ignore_duplicate_updates or prev_ent is None or prev_ent.origin != ent.origin or prev_ent.angles != ent.angles:
                obj.add_origin_keyframe(ent.origin, time)
                obj.add_angles_keyframe(ent.angles, time)
            obj.add_pose_keyframe(ent.frame, time)
        for entity_num in updated:
            if entity_num not in prev_updated or (entity_num in prev_entities and prev_entities[entity_num].model_num != entities[entity_num].model_num):
                self._objs[entity_num, entities[entity_num].model_num].add_visible_keyframe(True, time)

        if self._view_entity_num in entities:
            view_origin = np.array(entities[self._view_entity_num].origin)
            
            if self._camera_mode == 'THIRD_PERSON':
                rotation_matrix = _quake_angles_to_mat(view_angles)
                forward_vector = rotation_matrix[:, 1]
                offset = forward_vector * THIRD_PERSON_DISTANCE
                camera_pos = view_origin - offset
                camera_pos[2] += THIRD_PERSON_HEIGHT
                self._demo_cam_obj.location = camera_pos
            else:
                if not self._intermission:
                    view_origin = (view_origin[0], view_origin[1], view_origin[2] + _EYE_HEIGHT)
                self._demo_cam_obj.location = view_origin
                
            self._demo_cam_obj.keyframe_insert('location', frame=blender_frame)
            self._demo_cam_obj.rotation_euler = _quake_to_blender_angles(view_angles)
            self._demo_cam_obj.keyframe_insert('rotation_euler', frame=blender_frame)
            self._update_sample_as_light(view_origin, view_angles, blender_frame)

        if self._first_update_time is None: self._first_update_time = time

    def done(self, final_time: float):
        for bm in self._static_objects: bm.done(final_time, self._fps)
        for obj in self._objs.values(): obj.done(final_time)
        if self._config.get('hide_view_entity', False):
            for (entity_num, model_num), obj in self._objs.items():
                if entity_num == self._view_entity_num and model_num != 0:
                    if self._camera_mode == 'FIRST_PERSON':
                        obj.set_invisible_to_camera()

        scene = bpy.context.scene
        if self._first_update_time is not None:
            scene.frame_start = int(round(self._first_update_time * self._fps))
        scene.frame_end = int(round(final_time * self._fps))


def add_demo(demo_file, fs, config, fps=30, world_obj_name='demo',
             load_level=True, view_entity_only=False, relative_time=False,
             fov=120, width=1920, height=1080, camera_mode='FIRST_PERSON'):
    assert not relative_time, "Not yet supported"
    if camera_mode not in ['FIRST_PERSON', 'THIRD_PERSON']:
        raise ValueError("camera_mode must be 'FIRST_PERSON' or 'THIRD_PERSON'")

    baseline_entities: Dict[int, _EntityInfo] = collections.defaultdict(lambda: _DEFAULT_BASELINE)
    entities: Dict[int, _EntityInfo] = {}
    fixed_view_angles: Vec3 = (0, 0, 0)
    prev_updated = set()
    demo_done = False
    obj_mgr = ObjectManager(fs, config, fps, fov, width, height, camera_mode, world_obj_name, load_level)
    last_time = 0.
    view_entity_num = None

    msg_iter = proto.read_demo_file(demo_file)
    while not demo_done:
        time = None
        update_done = False
        updated = set()
        prev_entities = dict(entities)
        while not update_done and not demo_done:
            try:
                msg_end, view_angles, parsed = next(msg_iter)
            except StopIteration:
                demo_done = True
                break
            if msg_end and time is not None and entities: update_done = True
            fixed_view_angles = _fix_angles(fixed_view_angles, view_angles, degrees=True)
            if parsed.msg_type == proto.ServerMessageType.TIME: time = parsed.time
            if parsed.msg_type == proto.ServerMessageType.SERVERINFO: obj_mgr.set_model_paths(parsed.models)
            if parsed.msg_type == proto.ServerMessageType.SETVIEW:
                obj_mgr.set_view_entity(parsed.viewentity)
                view_entity_num = parsed.viewentity
            if parsed.msg_type == proto.ServerMessageType.SETANGLE: fixed_view_angles = parsed.view_angles
            if parsed.msg_type == proto.ServerMessageType.SPAWNSTATIC and not view_entity_only:
                obj_mgr.create_static_object(parsed.model_num, parsed.frame, parsed.origin, parsed.angles, parsed.skin)
            if parsed.msg_type == proto.ServerMessageType.SPAWNBASELINE:
                baseline_entities[parsed.entity_num] = _EntityInfo(model_num=parsed.model_num, frame=parsed.frame, skin=parsed.skin, origin=parsed.origin, angles=parsed.angles)
            if parsed.msg_type == proto.ServerMessageType.UPDATE:
                if not view_entity_only or parsed.entity_num == view_entity_num:
                    baseline = baseline_entities[parsed.entity_num]
                    prev_info = entities.get(parsed.entity_num, baseline)
                    entities[parsed.entity_num] = prev_info.update(parsed, baseline)
                    updated.add(parsed.entity_num)
            if parsed.msg_type == proto.ServerMessageType.TEMP_ENTITY and not view_entity_only:
                if time is not None:
                    if parsed.temp_entity_type == proto.TempEntityTypes.EXPLOSION: obj_mgr.create_explosion(parsed.origin, time)
                    elif parsed.temp_entity_type == proto.TempEntityTypes.TELEPORT: obj_mgr.create_teleport(parsed.origin, time)
            if parsed.msg_type in (proto.ServerMessageType.INTERMISSION, proto.ServerMessageType.FINALE, proto.ServerMessageType.CUTSCENE):
                obj_mgr.set_intermission(True)
        if update_done:
            logger.debug('Handling update. time=%s', time)
            obj_mgr.update(time, prev_entities, entities, prev_updated, updated, fixed_view_angles)
            last_time = time
            prev_updated = updated
    obj_mgr.done(last_time)
    return obj_mgr.world_obj, obj_mgr