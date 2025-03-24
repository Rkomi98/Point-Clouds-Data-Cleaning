import os
import bpy
import random
from itertools import combinations
from mathutils import Vector
import numpy as np
import laspy

'''
For the server

Paths needed:
    - output path
    - input environment path
    - LFG tool path
    - output path
    
ToDo:
    - upload LFG tool
    - upload script
    - upload main environment
    - create output folder
'''

# TODO To be changed
output_filepath = os.path.expanduser("/scratch/project/dd-24-47/PERIVALLON_UAV/pcwaste/Outputs")

# TODO to be checked!
tag_mapping = {
    "Vegetation": 1,
    "Terrain": 2,
    "Metals": 3,
    "Asbestos": 4,
    "Tyres": 5,
    "Plastics": 6,
    "default": 0  # Default if no tag is provided
}

#%% Clear all
# Thanks to stack Overflow XD
def clean_scene():
    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    
    # Delete all selected objects
    bpy.ops.object.delete(use_global=False)
    
    print("All objects have been deleted from the scene.")    

#%% Import environment

def obj_Passive(obj_name):
    if obj_name in bpy.data.objects:
        obj = bpy.data.objects[obj_name]
        # Apply any transformations
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        
        # Add rigid body physics if not already added
        if obj.rigid_body is None:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_add()
        
        # Set the rigid body type to PASSIVE
        obj.rigid_body.type = 'PASSIVE'
        obj.rigid_body.collision_shape = 'MESH'
        # Set the Rigid Body as Deformable (for soft body/mesh deforming behavior)
        obj["tag"] = "Terrain"
        print(obj["tag"])
        
        print(f"The object '{obj_name}' is now set to passive.")
    else:
        print(f"The object '{obj_name}' does not exist in the scene.")

def reset_object_state(obj_name):
    if obj_name in bpy.data.objects:
        obj = bpy.data.objects[obj_name]
        obj.rotation_euler = (0, 180, 0)  # Reset rotation
        #obj.location = (2, -2.5, -1)        # Reset location
        print(f"Reset the state of '{obj_name}'.")


def assign_tag_to_all_objects(tag_name, tag_value):
    for obj in bpy.data.objects:
        obj[tag_name] = tag_value
        print(f"Assigned tag '{tag_name}': {tag_value} to '{obj.name}'.")

# Sample a point inside a triangle (Barycentric sampling)
def sample_point_in_triangle(v0, v1, v2):
    r1 = random.random()
    r2 = random.random()
    sqrt_r1 = np.sqrt(r1)
    return (1 - sqrt_r1) * v0 + (sqrt_r1 * (1 - r2)) * v1 + (sqrt_r1 * r2) * v2

# Get the UV interpolated color from the texture
def get_texture_color(uv, image, image_pixels):
    u = int(uv[0] * image.size[0]) % image.size[0]  # Access uv[0] for x coordinate
    v = int(uv[1] * image.size[1]) % image.size[1]  # Access uv[1] for y coordinate
    return image_pixels[v, u][:3]  # Extract RGB and ignore alpha

def import_environment():
    filepath1 = False
    filepath2 = False
    filepath3 = False
    filepath4 = False
    #filepath1 = r"C:\Users\emmal\OneDrive - Politecnico di Milano\LM\TESI\Blender\Environments\grass_trees.glb"
    # TODO change the path depending on baseline chosen
    #filepath1 = r"C:\Users\Legion-pc-polimi\OneDrive - Politecnico di Milano\TESI\Blender\TyrePrato.glb"
    #filepath1 = r"C:\Users\Legion-pc-polimi\OneDrive - Politecnico di Milano\TESI\Blender\Trees_Bush.glb"
    #filepath1 = "/scratch/project/dd-24-47/PERIVALLON_UAV/pcwaste/Environments/grass_trees_bushes.glb"
    #filepath2 = "/scratch/project/dd-24-47/PERIVALLON_UAV/pcwaste/Environments/pineHD_T.glb"
    #filepath3 = r"C:\Users\Legion-pc-polimi\OneDrive - Politecnico di Milano\TESI\Blender\Hill\hill_treesJ.glb"
    #filepath4 = "/scratch/project/dd-24-47/PERIVALLON_UAV/pcwaste/Environments/forest.glb"
    filepath3 = "/scratch/project/dd-24-47/PERIVALLON_UAV/pcwaste/Environments/hill.glb"

    if filepath1:
        obj1 = "cesticka_01_cesticka_01_0"
    elif filepath2:
        obj1 = "obj1"
    elif filepath3:
        obj1 = "Plane_Gen"
    elif filepath4:
        obj1 = "Object_2"
    else:
        print("No File path found")
        return
    # Import the GLB file
    #bpy.ops.import_scene.gltf(filepath=filepath1)
    bpy.ops.import_scene.gltf(filepath=filepath3)
    assign_tag_to_all_objects("tag", "Vegetation")
    # Ensure the object exists in the scene
      
    obj_Passive(obj1)
    reset_object_state(obj1)
    #obj2 = "cesticka_01"
    # Assign the "vegetation" tag to all objects in the scene
    
    
#%% Other useful functions: Load and pile functions

# Function to load a collection from a Blender file
def load_collection_from_blend(filepath, collection_name):
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        if collection_name in data_from.collections:
            data_to.collections = [collection_name]

    return bpy.data.collections.get(collection_name)


# Function to create the pile from collections
def create_pile_from_collections(collections,x,y):
    ## TODO Remember to decide which waste you want in the environments
    print(collections)

    all_objects = []

    z_position = 5  # Initial z position to start stacking objects

    for collection_name in collections:
        # Set the path of the tool
        #tool_path = r"C:\Users\emmal\OneDrive - Politecnico di Milano\LM\TESI\Blender\LFG-main"
        tool_path = "/scratch/project/dd-24-47/PERIVALLON_UAV/pcwaste/LFG-main"
        blend_file_path = os.path.join(tool_path, "waste.blend")
        # Load the collection from the blend file
        # collection_name = collection_name[0]
        collection = load_collection_from_blend(blend_file_path, collection_name)
        if collection is None:
            print(f"Collection '{collection_name}' not found in the blend file.")
            continue

        if collection_name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(collection)

        # Apply scale factor if collection is "Asbestos". I chose 4 by attempt :-)
        if collection_name == "Asbestos":
            object_scale = 4 #0.5
            num_copies = 5 #3
        else:
            object_scale = 1.93 #0.3 
            num_copies = 10 #6
        
        # Randomize the position of objects within the collection
        # TODO: You can also increase the number of objects in the scene, depending on the environment  
        for obj in collection.objects:
            for _ in range(num_copies):
                # Create a new copy of the object
                obj_copy = obj.copy()
                obj_copy.data = obj.data.copy()
                bpy.context.collection.objects.link(obj_copy)
                
                # Randomize position within a larger area
                obj_copy.location.x = random.uniform(x - 3, x + 3)
                obj_copy.location.y = random.uniform(y - 3, y + 3)
                obj_copy.location.z = z_position
                
                obj_copy.scale *= object_scale  # Apply scale factor
                
                # Randomize rotation
                obj_copy.rotation_euler = (
                    random.uniform(0, 2 * 3.14159),
                    random.uniform(0, 2 * 3.14159),
                    random.uniform(0, 2 * 3.14159)
                )
                
                # Apply Rigid Body physics to each object
                if collection_name != "Asbestos":
                    obj_copy.rigid_body.type = 'ACTIVE'
                bpy.context.view_layer.objects.active = obj_copy
                bpy.ops.rigidbody.object_add()
                
                # Add a custom property (tag) to the object with the collection_name
                obj_copy["tag"] = collection_name
                
                all_objects.append(obj_copy)
                
                z_position += 0.2  # Smaller increment for more stacking
                
        bpy.data.collections.remove(collection)
        
    return all_objects
    
def vector_combination(collections_tot):
    ## This function is used to make a combination of the waste vector
    lista = list()
    for i in range(len(collections_tot)):
        i = i+1
        comb = combinations(collections_tot,i)
        # Print the obtained combinations
        for j in list(comb):
            print(j)
            lista.append(j)
    return lista
    
def set_scene():
    # Run the simulation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 250

    # Bake the physics simulation
    bpy.ops.ptcache.free_bake_all()  # Free any previous bakes
    bpy.ops.ptcache.bake_all(bake=True)

    # Set the scene to the last frame to check final object positions
    bpy.context.scene.frame_set(250)

def under_terrain_checker():
    # Iterate through all objects in the scene
    for obj in bpy.data.objects:
        print(obj)
        # Check if the z-location of the object is less than -1
        z_obj = obj.location[2]
        print(z_obj)
        if z_obj < 0:
            print('Rippppp')
            # Remove the object from the scene
            bpy.data.objects.remove(obj, do_unlink=True)
            #bpy.ops.object.delete(use_global=False, confirm=False)


def save(collections_list, iteration):
    # Define output path (consider using os.path.join for better path handling)
    #base_path = 'C:/Users/Legion-pc-polimi/OneDrive - Politecnico di Milano/TESI/Blender/Prova/'
    base_path = "/scratch/project/dd-24-47/PERIVALLON_UAV/pcwaste/Outputs"
    #for collection in collections_list:
    # Initialize arrays to store points, colors, and tags
    all_sampled_points = []
    all_sampled_colors = []
    all_sampled_tags = []
    environment_name = 'Hill'

    filename = collections_list[0]  # Use the collection name for the filename

    # Process each mesh object in the scene
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue  # Skip non-mesh objects

        # Ensure the mesh has UVs
        uv_layer = obj.data.uv_layers.active
        if uv_layer is None:
            print(f"Object {obj.name} has no active UV layer, skipping...")
            continue

        # Get the active texture image from the object's material
        material = obj.active_material
        if not material or not material.use_nodes:
            print(f"Object {obj.name} has no valid material with nodes, skipping...")
            continue

        # Get the first texture node and its image
        texture_node = None
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                texture_node = node
                break

        if texture_node is None:
            print(f"No image texture found in the material for object {obj.name}, skipping...")
            continue

        image = texture_node.image
        image_pixels = np.array(image.pixels).reshape(image.size[1], image.size[0], 4)  # RGBA

        # Get the transformation matrix for converting local coordinates to world space
        matrix_world = obj.matrix_world

        # Get the custom tag property (assuming it's stored as a string or number)
        tag_string = obj.get("tag", "default")  # Default to 'default' if no tag is found
        tag_int = tag_mapping.get(tag_string, tag_mapping["default"])  # Map string to int

        # Loop over each triangle in the mesh
        mesh = obj.data
        mesh.calc_loop_triangles()
        triangles = mesh.loop_triangles

        # Sample points from each triangle in the mesh
        for tri in triangles:
            print(tri)
            # Get triangle vertices in world space
            v0 = matrix_world @ mesh.vertices[tri.vertices[0]].co
            v1 = matrix_world @ mesh.vertices[tri.vertices[1]].co
            v2 = matrix_world @ mesh.vertices[tri.vertices[2]].co

            # Sample a random point within the triangle
            sampled_point = sample_point_in_triangle(v0, v1, v2)
            all_sampled_points.append(sampled_point)

            # Get the UV coordinates for the triangle's vertices
            uv0 = uv_layer.data[tri.loops[0]].uv
            uv1 = uv_layer.data[tri.loops[1]].uv
            uv2 = uv_layer.data[tri.loops[2]].uv

            # Convert the UV vectors into tuples for easier NumPy access
            uv0 = np.array(uv0)
            uv1 = np.array(uv1)
            uv2 = np.array(uv2)

            # Sample UV coordinates at the sampled point using the same Barycentric sampling
            sampled_uv = sample_point_in_triangle(uv0, uv1, uv2)

            # Get the color from the texture at the sampled UV coordinate
            sampled_color = get_texture_color(sampled_uv, image, image_pixels)
            all_sampled_colors.append(sampled_color)

            # Append the tag for each point
            all_sampled_tags.append(tag_int)

    # Convert to numpy arrays
    points_array = np.array(all_sampled_points)
    colors_array = np.array(all_sampled_colors)
    tags_array = np.array(all_sampled_tags, dtype=np.uint16)

    # Create a las file using laspy
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)

    # Assign XYZ coordinates from the sampled points
    las.x = points_array[:, 0]
    las.y = points_array[:, 1]
    las.z = points_array[:, 2]

    # Normalize color values to [0, 65535] and assign to las
    las.red = (colors_array[:, 0] * 65535).astype(np.uint16)
    las.green = (colors_array[:, 1] * 65535).astype(np.uint16)
    las.blue = (colors_array[:, 2] * 65535).astype(np.uint16)

    # Add custom dimension for tags (use uint16 for tag storage)
    extra_tag_dimension = laspy.ExtraBytesParams(name="tag", type=np.uint64, description="Object tag")
    las.add_extra_dim(extra_tag_dimension)
    las['tag'] = tags_array  # Assign the tags to the new dimension

    # Define output path
    output_filepath = os.path.join(base_path, "%s%d%s.las" % (environment_name, iteration, filename))

    # Write the las file
    las.write(output_filepath)

    print(f"Exported scene with sampled points and texture colors to {output_filepath}")


#%% Randomize environment
def get_random_point_on_mesh(mesh_obj):
    """
    Get a random point on the surface of the specified mesh object.
    
    :param mesh_obj: The Blender object to sample points from.
    :return: A random point (Vector) on the surface of the mesh.
    """
    if mesh_obj is None or mesh_obj.type != 'MESH':
        raise ValueError("The provided object is not a valid mesh object.")

    mesh = mesh_obj.data
    face = random.choice(mesh.polygons)
    verts_in_face = face.vertices[:]
    v1, v2, v3 = [mesh.vertices[i].co for i in verts_in_face]
    
    # Barycentric coordinates for random point in the triangle
    r1 = random.random()
    r2 = random.random()
    sqrt_r1 = r1 ** 0.5
    u = 1 - sqrt_r1
    v = r2 * sqrt_r1
    
    point = u * v1 + v * v2 + (1 - u - v) * v3
    return mesh_obj.matrix_world @ point

def is_point_inside_bounds(point, bounds_min, bounds_max):
    """
    Check if a point is within given bounds.
    
    :param point: The point to check (Vector).
    :param bounds_min: The minimum bounds (Vector).
    :param bounds_max: The maximum bounds (Vector).
    :return: True if the point is inside the bounds, False otherwise.
    """
    return (bounds_min.x <= point.x <= bounds_max.x and
            bounds_min.y <= point.y <= bounds_max.y and
            bounds_min.z <= point.z <= bounds_max.z)

def get_random_point_in_prato(prato_collection):
    """
    Get a random point on the surface of all mesh objects in the Prato collection.
    
    :param prato_collection: The Blender collection containing Prato objects.
    :return: A random point (Vector) on the surface of any mesh in the collection.
    """
    prato_meshes = [obj for obj in prato_collection.objects if obj.type == 'MESH']
    if not prato_meshes:
        raise ValueError("No mesh objects found in the Prato collection.")
    
    mesh_obj = random.choice(prato_meshes)
    point = get_random_point_on_mesh(mesh_obj)

    # Get the bounding box of the Prato mesh object
    bounds_min = Vector(mesh_obj.bound_box[0])
    bounds_max = Vector(mesh_obj.bound_box[6])
    
    if is_point_inside_bounds(point, bounds_min, bounds_max):
        return point
    else:
        return get_random_point_in_prato(prato_collection)

def randomize_objects_in_collection(cambia_collection, prato_collection):
    """
    Randomizes the location of objects in Cambia collection to the surface of objects
    in Prato collection and randomly rotates them around the z-axis.
    
    :param cambia_collection: The Blender collection to be transformed.
    :param prato_collection: The Blender collection to use for surface points.
    """
    for obj in cambia_collection.objects:
        point = get_random_point_in_prato(prato_collection)
        rotation_z = random.uniform(0, 3.14)
        obj.location = point
        obj.rotation_euler.z = rotation_z

def randomize_environment():
    # Get the Prato and Cambia collections
    grass_collection_name = "Prato"
    tree_collection_name = "Cambia"
    # Get the Prato and Cambia collections
    grass_collection = bpy.data.collections.get(grass_collection_name)
    tree_collection = bpy.data.collections.get(tree_collection_name)
    if grass_collection and tree_collection:
        randomize_objects_in_collection(tree_collection, grass_collection)
        print("Randomization of Tree collection completed.")
    else:
        if not grass_collection:
            print(f"Prato collection '{grass_collection_name}' not found.")
        if not tree_collection:
            print(f"Tree collection '{tree_collection_name}' not found.")

#%% Randomize waste
def randomize_waste():  
    x = random.randint(-6,6)
    y = random.randint(-6,6)
    # TODO To be changed if waste type are different from the one decided above
    collections_tot = ["Metals" "Asbestos","Tyres","Plastics"]
    
    collections_selected = vector_combination(collections_tot) 
    # combination of the waste selected among the one available
    combination_waste = [list(elem) for elem in collections_selected]
    iteration = 0
    
    for scene in combination_waste:    
        # Clear all the collection in the scene
        clean_scene()
        import_environment()
        # create_PC_by_collections_list(collections_tot, 0)
        # Create the pile of objects
        all_objects = create_pile_from_collections(scene,x,y)
        set_scene()
        #under_terrain_checker()
        save(scene, iteration)
        #export_csv(scene,iteration)
        iteration +=1
    # TODO It was the old function to check if there was something under the terrain. It still doesn't work.
        # under_terrain_checker()
    print(x,y)

randomize_waste()