# Setup path.  Change the paths here to point to your virtualenv's
# site-packages, and your local pyquake install.
from os.path import expanduser
import sys
import bpy

for x in [
     expanduser('~/miniconda3/envs/blend/lib/python3.11/site-packages'),
     expanduser('~/pyquake'),
      ]:
    if x not in sys.path:
        sys.path.append(x)


# Import everything, and reload modules that we're likely to change>
import importlib
import logging
import json
import pyquake.blenddemo
import pyquake.blendmdl
import pyquake.blendbsp
from pyquake import pak

importlib.reload(pyquake.blendbsp)
importlib.reload(pyquake.blendmdl)
blenddemo = importlib.reload(pyquake.blenddemo)

# Log messages should appear on the terminal running Blender
logging.getLogger().setLevel(logging.INFO)


# =================================================================
# 1. SCENE AND RENDER SETUP
# =================================================================

# --- More Robust Scene Clearing ---
# Ensure we are in Object Mode
if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

# Delete all objects in the scene
for obj in bpy.data.objects:
    obj.select_set(True)
bpy.ops.object.delete()
print("Cleared all objects from the scene.")
# --------------------------------

# Set render engine to Cycles
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.fps = 30
scene.cycles.samples = 256
scene.cycles.device = 'GPU'

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.get_devices()
cuda_devices = [d for d in prefs.devices if d.type == 'CUDA']

if cuda_devices:
    print("CUDA devices found, enabling.")
    prefs.compute_device_type = 'CUDA'
    for device in prefs.devices:
        device.use = device.type == 'CUDA'
else:
    print("No CUDA devices found. Rendering will use CPU.")


# =================================================================
# 2. DEMO LOADING
# =================================================================

CAMERA_VIEW_MODE = 'THIRD_PERSON'

with open(expanduser('~/pyquake/config.json')) as f:
  config = json.load(f)

fs = pak.Filesystem(expanduser('/home/lauri/JoeQuake-1/build/trunk/joequake/'))
demo_fname = expanduser('/home/lauri/JoeQuake-1/build/trunk/joequake/joequake/t1.dem')

with open(demo_fname, 'rb') as demo_file:
    world_obj, obj_mgr = blenddemo.add_demo(demo_file,
                                            fs,
                                            config,
                                            fps=30,
                                            width=1280,
                                            height=720,
                                            camera_mode=CAMERA_VIEW_MODE)

world_obj.scale = (0.01,) * 3

print("Demo loading finished.")


# =================================================================
# 3. RENDER FRAME RANGE
# =================================================================

# Set the active camera for the scene
scene = bpy.context.scene
scene.camera = obj_mgr._demo_cam_obj

# Set the output file path and format
scene.render.filepath = expanduser('~/blender_video_100-125.mp4')
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
print(f"Render output path: {scene.render.filepath}")

# --- SET THE FRAME RANGE TO RENDER ---
scene.frame_start = 100
scene.frame_end = 125
print(f"Rendering frames from {scene.frame_start} to {scene.frame_end}")
# -------------------------------------

# Render the specified frame range
print("Starting animation render...")
bpy.ops.render.render(animation=True)
print("Render finished.")