"""
Converts lite6_full.urdf -> lite6_full.xml (MuJoCo MJCF)

Usage:
    python transform.py

MuJoCo 3.x strips subdirectories from mesh filenames in URDF, so this script:
  1. Creates a temporary work dir with meshes renamed flat (visual_X.stl, collision_X.stl)
  2. Generates a patched URDF that references those flat names
  3. Runs the MuJoCo conversion
  4. Adds meshdir="meshes_mujoco" to the compiler tag in the output XML
  5. Copies the collision meshes to meshes_mujoco/ (flat names)
  6. Cleans up the temp work dir
"""

import mujoco
import os
import re
import shutil
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
URDF_SRC   = os.path.join(BASE, "lite6_full.urdf")
XML_OUT    = os.path.join(BASE, "lite6_full.xml")
MESH_OUT   = os.path.join(BASE, "meshes_mujoco")


def patch_and_convert():
    with open(URDF_SRC) as f:
        content = f.read()

    work_dir = tempfile.mkdtemp(prefix="mj_convert_")
    try:
        def copy_mesh(match):
            orig_rel = match.group(1)               # e.g. meshes/visual/link1.stl
            orig_abs = os.path.join(BASE, orig_rel)
            # meshes/visual/link1.stl -> visual_link1.stl
            new_name = orig_rel.replace("meshes/", "").replace("/", "_")
            shutil.copy2(orig_abs, os.path.join(work_dir, new_name))
            return f'filename="{new_name}"'

        patched = re.sub(r'filename="([^"]+)"', copy_mesh, content)

        tmp_urdf = os.path.join(work_dir, "lite6_full.urdf")
        with open(tmp_urdf, "w") as f:
            f.write(patched)

        prev_dir = os.getcwd()
        os.chdir(work_dir)
        model = mujoco.MjModel.from_xml_path("lite6_full.urdf")
        mujoco.mj_saveLastXML(XML_OUT, model)
        os.chdir(prev_dir)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Add meshdir to compiler tag
    with open(XML_OUT) as f:
        xml = f.read()
    xml = xml.replace(
        '<compiler angle="radian"/>',
        '<compiler angle="radian" meshdir="meshes_mujoco"/>'
    )
    with open(XML_OUT, "w") as f:
        f.write(xml)

    # Copy collision meshes to meshes_mujoco/ with flat names
    os.makedirs(MESH_OUT, exist_ok=True)
    collision_dir = os.path.join(BASE, "meshes", "collision")
    for stl in os.listdir(collision_dir):
        if stl.endswith(".stl"):
            src = os.path.join(collision_dir, stl)
            dst = os.path.join(MESH_OUT, f"collision_{stl}")
            shutil.copy2(src, dst)

    print(f"Done: {XML_OUT}")
    print(f"Meshes: {MESH_OUT}/")


if __name__ == "__main__":
    patch_and_convert()
