# How to use this file
You need to consider at least one glb function which has to be used to drop the waste on it.

Also it is important to download and use the [LFG Main tool](https://github.com/AthanasiosPetsanis/LFG) and remember to install all the libraries needed to run the script.

The main steps are the following:

- Choose .glb baseline mesh.
- Define what is the terrain in the mesh and make it passive.
- Add tag to all of them.
- Import waste objects through LFG main tool.
- Create pile and drop it in the environment created.
- Sample the barycenter point of each face of each object in the mesh. We take coordinate and color information.
- Export points as a point cloud in .las format with also tag as extra_field.

