"""Convert K1's physical articulation; retain the immutable source MJCF for FK.

MJCF's massless leaf reference frames are not rigid bodies. Exclude them from
PhysX import rather than letting the importer invent masses/inertias for them.
The importer-created worldBody is not part of the robot articulation.
"""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument('--xml',required=True)
AppLauncher.add_app_launcher_args(p); a=p.parse_args(); app=AppLauncher(a).app
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import mujoco,numpy as np
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.asset.importer.mjcf')
from isaaclab.sim.converters import MjcfConverter,MjcfConverterCfg
from pxr import Usd,UsdPhysics
source=Path(a.xml).resolve(); tree=ET.parse(source); removed=[]
for parent in list(tree.iter('body')):
    for child in list(parent.findall('body')):
        if len(child)==0:
            removed.append(child.attrib['name']);parent.remove(child)
for world in tree.findall('worldbody'):
    for child in list(world):
        if child.tag!='body': world.remove(child)
for parent in tree.iter():
    for child in list(parent.findall('site')): parent.remove(child)
for sensor in tree.findall('sensor'): tree.getroot().remove(sensor)
converted=source.with_suffix('.physics.xml');tree.write(converted)
original=mujoco.MjModel.from_xml_path(str(source)); physical=mujoco.MjModel.from_xml_path(str(converted))
assert original.nq==physical.nq and original.nv==physical.nv
for name in [original.body(i).name for i in range(original.nbody) if original.body(i).name not in removed]:
    i,j=original.body(name).id,physical.body(name).id
    for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat'):
        np.testing.assert_array_equal(getattr(original,field)[i],getattr(physical,field)[j])
# MjcfConverter expects a basename with one dot, so use a separate staging directory.
staging=source.parent/'import';staging.mkdir(exist_ok=True)
import_xml=staging/'K1_22dof.xml'
if not (staging/'meshes').exists(): (staging/'meshes').symlink_to('../meshes', target_is_directory=True)
root=tree.getroot();root.find('compiler').set('meshdir','meshes/');tree.write(import_xml)
c=MjcfConverter(MjcfConverterCfg(asset_path=str(import_xml),usd_dir=str(source.parent),usd_file_name='K1_22dof.usd',fix_base=False,make_instanceable=True,force_usd_conversion=True,self_collision=True))
stage=Usd.Stage.Open(c.usd_path)
world=[prim for prim in stage.Traverse() if prim.GetName()=='worldBody']
for prim in world: prim.SetActive(False)
stage.GetRootLayer().Save()
assert sum(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in stage.Traverse())==1
report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),removed_massless_leaf_frames=removed,
            worldbody_deactivated=[str(x.GetPath()) for x in world],physical_mass_kg=float(physical.body_mass.sum()),
            validation='exact equality of retained body transforms, masses, inertias and generalized-coordinate dimensions')
(source.parent/'conversion.json').write_text(json.dumps(report,indent=2))
print('K1_ASSET_CONVERSION_PASS',report,flush=True)
app.close()
