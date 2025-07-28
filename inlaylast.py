import bpy
import bmesh
import mathutils
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree
try:
    import debug_view
    debug_view_exist = True
except:
    debug_view_exist = False
    
import os
import trimesh

import numpy as np
g_boundary_kd = None

def blender_mesh_to_trimesh(blender_mesh):

    blender_mesh.calc_loop_triangles()
    vertices = np.array([[v.co.x, v.co.y, v.co.z ] for v in blender_mesh.vertices], dtype=np.float64)

    faces = np.array([tri.vertices[:] for tri in blender_mesh.loop_triangles], dtype=np.int64)

    return trimesh.Trimesh(vertices=vertices, faces=faces)

def selectObj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

def area_inside(points, bm):

    rpoints = []
    bvh = BVHTree.FromBMesh(bm, epsilon=0.0001)

    # return points on polygons
    for (i,point) in enumerate(points):
        fco, normal, _, _ = bvh.find_nearest(point)
        p2 = fco - Vector(point)
        v = p2.dot(normal)
        
        if  v < 0.0:
            rpoints.append((i, point))  # addp(v >= 0.0) ?

    return rpoints

def getProjectDist(point, plane_p, plane_normal):
    v = point - plane_p
    dist = v.dot( plane_normal)
    #projectPoint = point - dist * normal
    return dist
def bmesh_face_points_random(f, num_points=1, margin=0.05):
    import random
    from random import uniform

    # for pradictable results
    random.seed(f.index)

    uniform_args = 0.0 + margin, 1.0 - margin
    vecs = [v.co for v in f.verts]

    for _ in range(num_points):
        u1 = uniform(*uniform_args)
        u2 = uniform(*uniform_args)
        u_tot = u1 + u2

        if u_tot > 1.0:
            u1 = 1.0 - u1
            u2 = 1.0 - u2

        side1 = vecs[1] - vecs[0]
        side2 = vecs[2] - vecs[0]

        yield vecs[0] + u1 * side1 + u2 * side2    
def findNearFace(points, bm, inlayObj, bvh):
    
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    count = 0
    for i in bm.faces:
        if i.select:
            count+=1
    kd = mathutils.kdtree.KDTree( count * 6)
    result_verts = []

    for i,f in enumerate(bm.faces):
        if f.select:
            pointList = bmesh_face_points_random(f, num_points=6)
            for p in pointList:
                kd.insert(p, i )
    kd.balance()    
    co_normal_pair=[]
    for i,p in points:
      
        co,index,dist = kd.find(p)
        f = bm.faces[index]
        
        
        moveDist = getProjectDist( p, co, f.normal)
        co_normal_pair.append( (p, -f.normal) )
        #debug_view.drawLine(p, -f.normal*moveDist + p) 
def findMoveVerts( inlay_bvh, bm ):
    #debug_view.open()
    vert_dist_list = []
    for i in bm.verts:
        result = inlay_bvh.ray_cast( i.co,  -i.normal)
        if result[0] != None:
            if i.normal.dot( result[1] ) > 0:
                vert_dist_list.append( (i, result[3]) )
    return vert_dist_list    
    #debug_view.drawPoints(show_verts)  
def getVertToBoundaryDist(inlay_bm, v ):
    kd = getBoundaryKd(inlay_bm)
    co, index, dist = kd.find( v.co )
    return dist


def getBoundaryKd( inlay_bm ):

    global g_boundary_kd
    if g_boundary_kd == None:
        count = 0
        for i in inlay_bm.verts:
            if i.is_boundary:
                count+=1
                
    
        g_boundary_kd = mathutils.kdtree.KDTree(count)
        for i,v in enumerate(inlay_bm.verts):
            if v.is_boundary:
                g_boundary_kd.insert(v.co, i )
        g_boundary_kd.balance()        
        
    return g_boundary_kd

def check(inlay_bm,inlay_bvh, bm):
    vert_thinkness_list = findMoveVerts(inlay_bvh, bm)
    #debug_view.open()       
    theLines = []
    
    final_move_list = []
    
    for (v, thinkness ) in vert_thinkness_list:
        if thinkness < 0.8:
            dist = getVertToBoundaryDist(inlay_bm, v)
            #debug_view.drawLine(v.co, v.co + -v.normal * thinkness)
            #debug_view.drawText( str(thinkness)[0:5] +  ',' + str(dist)[0:5], v.co )  


def scaleEdge(inlay_bm,inlay_bvh, bm):
    vert_thinkness_list = findMoveVerts(inlay_bvh, bm)
    #debug_view.close()   
    theLines = []
    
    final_move_list = []
    
    for (v, thinkness ) in vert_thinkness_list:
        
        dist = getVertToBoundaryDist(inlay_bm, v)
        if  dist < 0.1:
            final_move_list.append( v )
    
            
    for v in final_move_list:
        result = inlay_bvh.ray_cast( v.co, -v.normal )
        #print( result )
        if result[0] != None:
            
            offset =  result[3] 
            #print('offset', offset )
            if offset > 0:
                bpy.ops.mesh.select_all(action='DESELECT')
                v.select_set(True)

                bpy.ops.transform.translate(value=(0.0, 0.0, -offset), orient_type='NORMAL', orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=2.0, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        

    
def calcVertMoveDist(inlay_bm,inlay_bvh, bm):
    vert_thinkness_list = findMoveVerts(inlay_bvh, bm)
    #debug_view.close()   
    theLines = []
    
    final_move_list = []
    coo = []
    #~ for (v, thinkness ) in vert_thinkness_list:
        #~ if thinkness < 0.8:
            #~ dist = getVertToBoundaryDist(inlay_bm, v)
            #~ if  dist > 0.8:
                #~ final_move_list.append( v )
    for (v, thinkness ) in vert_thinkness_list:
        dist = getVertToBoundaryDist(inlay_bm, v)
        #if thinkness < 0.8 and thinkness < dist:
        if dist > 0.8:
                
                final_move_list.append( v )
            #v.co = v.co + v.normal * (dist-thinkness)
            #~ bpy.ops.mesh.select_all(action='DESELECT')
            #~ v.select_set(True)
    smooth_verts = []

    for v in final_move_list:
        result = inlay_bvh.ray_cast( v.co, -v.normal )
        if result[0] != None:
            
            #offset =  0.9 - result[3] 
            offset = 0.01
            #print('offset', offset )
            if offset > 0:
                bpy.ops.mesh.select_all(action='DESELECT')
                v.select_set(True)
                smooth_verts.append( v.index )
                bpy.ops.transform.translate(value=(0.0, 0.0, offset), orient_type='NORMAL', orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=1.0, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

    bExec = False
    if (bExec ):
        for v in final_move_list:
            result = inlay_bvh.ray_cast( v.co, -v.normal )
            if result[0] != None:
                
                offset =  0.9 - result[3] 
                
                #print('offset', offset )
                if offset > 0:
                    coo.append( (v.co[0], v.co[1], v.co[2]) )
                    bpy.ops.mesh.select_all(action='DESELECT')
                    v.select_set(True)
                    smooth_verts.append( v.index )
                    bpy.ops.transform.translate(value=(0.0, 0.0, offset),  orient_type='NORMAL', orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=1.5, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)



        bpy.ops.mesh.select_all(action='DESELECT')
        for i in smooth_verts:
            bm.verts[i].select_set(True)
        #~ bpy.ops.mesh.select_more()
        bpy.ops.mesh.select_more() 
        
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=2)

    #debug_view.drawPoints( coo )  
    #scaleEdge(inlay_bm,inlay_bvh, bm)
    #check(inlay_bm,inlay_bvh, bm)



def findNearFaceByPoints(points, bm, inlayObj, p_size=1.0 ):
    size = len(bm.verts)
    kd = mathutils.kdtree.KDTree(size)
    result_verts = []
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for i,v in enumerate(bm.verts):
        kd.insert(v.co, i )
    kd.balance()

        
    inlay_bm = bmesh.new()   # create an empty BMesh
    inlay_bm.from_mesh(inlayObj.data)     
    
    for (i, p ) in points:
        co, index, dist = kd.find(p)
        result_verts.append( index )
        
    for i in result_verts:
        bm.verts[i].select_set(True)
    bpy.ops.mesh.select_mode(use_expand=True, type='FACE')
      
        
    inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)
    bvh = mathutils.bvhtree.BVHTree.FromBMesh(bm)
    
    overlapList = bvh.overlap(inlay_bvh)
    
    for (i,j) in overlapList:
        bm.faces[i].select_set(True)
    aSelFaces = []
    
    bExec = True
    if bExec:
        for f in bm.faces:
            if f.select:
                aSelFaces.append( f )
        for f in aSelFaces:
            pointList = bmesh_face_points_random(f, num_points=20)
            count = 0
            for p in pointList:
                result = inlay_bvh.ray_cast(p, f.normal )
                if result[0] != None:
                    bpy.ops.mesh.select_all(action='DESELECT')
                    count += 1
                    f.select_set(True)
                    dist = result[3]

                    bpy.ops.transform.translate(value=(0.0, 0.0, dist), orient_type='NORMAL', orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=p_size, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
                    f.select_set(False)
            if count > 0:
                # print( 'count', count )
                pass

   
                
                
    #~ for i in result_verts:
        #~ #bm.verts[i].select_set(True)
        #~ result = bvh.ray_cast(bm.verts[i].co, bm.verts[i].normal )
        
        #~ if result[0] != None:
            #~ bpy.ops.mesh.select_all(action='DESELECT')

            #~ bm.verts[i].select_set(True)
            #~ dist = result[3]
            #~ print( dist )
            #~ targetPos = bm.verts[i].normal * ( dist + 0.1 )
            #~ bpy.ops.transform.translate(value=(0.0, 0.0, dist + 0.1), orient_type='NORMAL', orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    #~ bpy.ops.mesh.select_all(action='DESELECT')
    
#debug_view.close()   


def convertMesh(name, verts, faces):
    mesh = bpy.data.meshes.new(name=name)
    bm = bmesh.new()
    bResult = True
    num_vertices = verts.shape[0]

    for v_co in verts:
        bm.verts.new( v_co )

        
    bm.verts.ensure_lookup_table()
    
    try:
        for f_idx in faces:
            bm.faces.new( bm.verts[i] for i in f_idx )
    except:
        bResult = False
    bm.to_mesh( mesh )
    
    mesh.update()
    
    from bpy_extras import object_utils
    object_utils.object_data_add( bpy.context, mesh )
    bpy.context.active_object.name = name
    return bResult


class InlayAdjust:
    def __init__(self, inlayObj, crownObj):
        self.crownObj = crownObj
        self.inlayObj = inlayObj
        self.grooveFaces = []
        self.oc_faces = []
    def isNearBoundary(self, co, max_dist=1.0):
        v_co,idx,dist = self.boundary_kd.find( co )
        if dist < max_dist:
            return True
            
        return False
        
        
    def findNextSphere(self, bm, face_center_list, face_center ):
        pass
        #find_group = kd.find_range( start_face_center, 2.0 )
        
    def grooveSphereSort(self, bm ):
        kd = mathutils.kdtree.KDTree(len( self.grooveFaces ))
        for f in self.grooveFaces:
            #print(f)
            face_center = bm.faces[f].calc_center_median()
            kd.insert(face_center, f )
        kd.balance()
        
        last_find_count = 0
        start_f = self.grooveFaces[0]
        start_face_center = mathutils.Vector((0,0,0))
        last_find_list = []
        for f in self.grooveFaces:
            face_center = bm.faces[f].calc_center_median()
          
            face_center_list = kd.find_range( face_center, 1.0 )
            find_count = len(face_center_list)
            if find_count > last_find_count:
                start_f = f
                last_find_count = find_count
                start_face_center = face_center
                last_find_list = face_center_list
        # debug_view.open()
        # debug_view.drawPoint( [start_face_center] )
        
        
        #find next
        
           
        
             
    def isGrooveFace(self, f):
        for i in self.groove_faces:
            if i == f:
                return True
                
        return False
        
        
    def onlyCheck(self, thickness = 0.8):
        inlayObj = self.inlayObj
        crownObj = self.crownObj
        if bpy.context.mode != 'EDIT_MESH':
            selectObj(inlayObj)
            bpy.ops.object.editmode_toggle()    
        bm = bmesh.from_edit_mesh(inlayObj.data)
        bm.verts.ensure_lookup_table()
        bpy.ops.mesh.select_mode(use_expand=True, type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')
        
        
        boundaryCount = 0
        for i,v in enumerate(bm.verts):
            if v.is_boundary:
                boundaryCount += 1
 
        self.boundary_kd = mathutils.kdtree.KDTree(boundaryCount)
        for i,v in enumerate(bm.verts):
            if v.is_boundary:
                self.boundary_kd.insert(v.co, i )
        self.boundary_kd.balance()        
        
        ray_co = []
        ray_dir = []
        for i in bm.verts:
            bNear = self.isNearBoundary( i.co )
            if bNear == False:
                i.select_set(True)
                ray_co.append( (i.co[0], i.co[1], i.co[2]) )
                ray_dir.append( i.normal )
                
    
        if bpy.context.mode == 'EDIT_MESH':
            bpy.ops.object.editmode_toggle()    

        selectObj(crownObj)
        bpy.ops.object.editmode_toggle()  
        
        bpy.ops.mesh.select_mode(use_expand=True, type='FACE')
        crown_bm = bmesh.from_edit_mesh(crownObj.data)
        crown_bm.faces.ensure_lookup_table()
        bpy.ops.mesh.select_all(action='DESELECT')
        
        bpy.ops.mesh.edges_select_sharp(sharpness=0.139626)
        
        self.groove_faces = []
        for i in crown_bm.faces:
            if i.select:
                self.groove_faces.append(i.index)
        
        crown_bvh = mathutils.bvhtree.BVHTree.FromBMesh(crown_bm)
       
        #debug_view.open()
        #debug_view.drawPoints( ray_co )              
        # for i in ray_co:
            # face_list = crown_bvh.find_nearest_range( i, thickness )
            # for loc,nor,f_id,dist in face_list:
                # f = crown_bm.faces[f_id]
                # if f.select == False:
                    # debug_view.drawLine( i, loc )    
                    # pp = mathutils.Vector(( i[0] + loc[0], i[1] + loc[1], i[2] + loc[2]))
                    # debug_view.drawText( str(dist)[0:4], pp * 0.5)
                # f.select_set(True)
        move_dist = {}
        for i in ray_co:
            loc, nor, idx, dist = crown_bvh.find_nearest( i, thickness )
  
            if loc != None:
                crown_bm.faces[idx].select_set(True)
                #debug_view.drawLine( i, loc )    
                pp = mathutils.Vector(( i[0] + loc[0], i[1] + loc[1], i[2] + loc[2]))
                #debug_view.drawText( str(dist)[0:4], loc)
                
                if move_dist.get(idx) == None:
                    move_dist[idx] = dist
                else:
                    if dist > move_dist[idx]:
                        move_dist[idx] = dist
                        
        bpy.ops.mesh.select_all(action='DESELECT')                
        for f in ( move_dist ):
            if self.isGrooveFace( f ):
                crown_bm.faces[f].select_set(True)
                
    def set_oc_center_faces(self, crown_bm, center ):
    
        kd =  mathutils.kdtree.KDTree( len(self.groove_faces) )
        for i in self.groove_faces:
            
            f_center = crown_bm.faces[i].calc_center_median()

            kd.insert( f_center, i )
        kd.balance()        
            
        # rrr = kd.find_range(center, 1.0) 
        # print( rrr )
        for (co,index,dist) in kd.find_range(center, 1.0):
            crown_bm.faces[index].select_set(True)
    
    def set_oc_verts(self, verts_idx ):
        crownObj.select_set(True)
        bpy.context.view_layer.objects.active = crownObj
        
        bpy.ops.object.editmode_toggle()   
        bm = bmesh.from_edit_mesh(crownObj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bpy.ops.mesh.select_all(action='DESELECT')

        bpy.ops.mesh.select_mode(use_expand=True, type='VERT')    
        
        for i in verts_idx:
            bm.verts[i].select = True
            
        bpy.ops.mesh.select_mode(use_expand=True, type='FACE') 

        for i, f in enumerate(bm.faces):
            if f.select:
                self.oc_faces.append( i )  

        bpy.ops.object.editmode_toggle()  
         
    def set_oc_faces(self, oc_obj):
        
        crownObj.select_set(True)
        bpy.context.view_layer.objects.active = crownObj
        oc_obj.select_set(True)
        bpy.ops.object.editmode_toggle()  
        
        bm = bmesh.from_edit_mesh(crownObj.data)
        bm.faces.ensure_lookup_table()
        bpy.ops.mesh.select_mode(use_expand=True, type='FACE')
        bpy.ops.mesh.select_all(action='DESELECT')
        kd =  mathutils.kdtree.KDTree( len(bm.faces) )
        for i, f in enumerate(bm.faces):
            center = f.calc_center_median()

            kd.insert( center, i )
        kd.balance()        
        
        oc_bm = bmesh.from_edit_mesh(oc_obj.data)
        oc_bm.faces.ensure_lookup_table()
        bpy.ops.mesh.select_mode(use_expand=True, type='FACE')
        bpy.ops.mesh.select_all(action='DESELECT')
        
        
      
        for f in oc_bm.faces:
            center = f.calc_center_median()
            co, index, dist = kd.find( center ) 
            self.oc_faces.append( index )
            

        bpy.ops.object.editmode_toggle()  
   
        
        
                
    def refCheck(self, thickness = 0.8):
        pass
        
        
    def build_group_face(self, bm, group_faces):
        layer = bm.faces.layers.int.get("region_id")
        if not layer:
            layer = bm.faces.layers.int.new("region_id")

        for face in bm.faces:
            face[layer] = 0

        selected_faces = [f for f in bm.faces if f.select]

        region_id = 1
        visited = set()

        def flood_fill(start_face):
            stack = [start_face]
            connected_faces = set()

            while stack:
                f = stack.pop()
                if f in visited:
                    continue
                visited.add(f)
                connected_faces.add(f)

                for edge in f.edges:
                    for linked_face in edge.link_faces:
                        if linked_face in selected_faces and linked_face not in visited:
                            stack.append(linked_face)

            return connected_faces

        for face in selected_faces:
            if face not in visited:
                region_faces = flood_fill(face)
                for f in region_faces:
                    f[layer] = region_id
                region_id += 1
        
        for i in range(0,region_id-1):
            group_faces.append( [] )
        for face in selected_faces:
            region_id = face[layer]
            group_faces[region_id-1].append( face.index )
                

        # print( group_faces )
        
    
        
    def check(self, thickness = 0.8):
        inlayObj = self.inlayObj
        crownObj = self.crownObj
        if bpy.context.mode != 'EDIT_MESH':
            selectObj(inlayObj)
            bpy.ops.object.editmode_toggle()    
        bm = bmesh.from_edit_mesh(inlayObj.data)
        bm.verts.ensure_lookup_table()
        bpy.ops.mesh.select_mode(use_expand=True, type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')
        
        
        boundaryCount = 0
        for i,v in enumerate(bm.verts):
            if v.is_boundary:
                boundaryCount += 1
 
        self.boundary_kd = mathutils.kdtree.KDTree(boundaryCount)
        for i,v in enumerate(bm.verts):
            if v.is_boundary:
                self.boundary_kd.insert(v.co, i )
        self.boundary_kd.balance()        
        
        ray_co = []
        ray_dir = []
        for i in bm.verts:
            bNear = self.isNearBoundary( i.co )
            if bNear == False:
                i.select_set(True)
                ray_co.append( (i.co[0], i.co[1], i.co[2]) )
                ray_dir.append( i.normal )
                
    
        if bpy.context.mode == 'EDIT_MESH':
            bpy.ops.object.editmode_toggle()    

        selectObj(crownObj)
        bpy.ops.object.editmode_toggle()  
        
        bpy.ops.mesh.select_mode(use_expand=True, type='FACE')
        crown_bm = bmesh.from_edit_mesh(crownObj.data)
        crown_bm.faces.ensure_lookup_table()
        bpy.ops.mesh.select_all(action='DESELECT')
        
        
        
        self.groove_faces = []
        use_groove_center = False
        groove_center = mathutils.Vector((0,0,0))
        if len(self.oc_faces) > 0:
            # print( '****************' )
            for i in self.oc_faces:
                self.groove_faces.append(i)
                groove_center += crown_bm.faces[i].calc_center_median()
                
            groove_center = groove_center / len(self.oc_faces)
            use_groove_center = True    
        else:
            bpy.ops.mesh.edges_select_sharp(sharpness=0.139626)
            for i in crown_bm.faces:
                if i.select:
                    self.groove_faces.append(i.index)
        
        crown_bvh = mathutils.bvhtree.BVHTree.FromBMesh(crown_bm)
        
        #debug_view.open()
        #debug_view.drawPoints( ray_co )              
        # for i in ray_co:
            # face_list = crown_bvh.find_nearest_range( i, thickness )
            # for loc,nor,f_id,dist in face_list:
                # f = crown_bm.faces[f_id]
                # if f.select == False:
                    # debug_view.drawLine( i, loc )    
                    # pp = mathutils.Vector(( i[0] + loc[0], i[1] + loc[1], i[2] + loc[2]))
                    # debug_view.drawText( str(dist)[0:4], pp * 0.5)
                # f.select_set(True)
        move_dist = {}
        for i in ray_co:
            loc, nor, idx, dist = crown_bvh.find_nearest( i, thickness )
  
            if loc != None:
                crown_bm.faces[idx].select_set(True)
                #debug_view.drawLine( i, loc )    
                pp = mathutils.Vector(( i[0] + loc[0], i[1] + loc[1], i[2] + loc[2]))
                #debug_view.drawText( str(dist)[0:4], loc)
                
                if move_dist.get(idx) == None:
                    move_dist[idx] = dist
                else:
                    if dist > move_dist[idx]:
                        move_dist[idx] = dist
                        
        bpy.ops.mesh.select_all(action='DESELECT')                
        for f in ( move_dist ):
            if self.isGrooveFace( f ):
                crown_bm.faces[f].select_set(True)
                self.grooveFaces.append( f )
        #self.grooveSphereSort(crown_bm)
        # return
        bSeparte = False
        group_faces = []
        obj_name = bpy.context.active_object.name
        if bSeparte:
            
            bpy.ops.mesh.duplicate()
            bpy.ops.mesh.separate()
            bpy.ops.mesh.select_all(action='DESELECT')                
            bpy.context.active_object.select_set(False)
            
            bpy.ops.object.editmode_toggle()  
            bpy.ops.mesh.separate(type='LOOSE')
            
            ref_objs = []
            for i in bpy.context.selected_objects:
                ref_objs.append( i )
            
            
            crownObj.select_set(True)
            bpy.context.view_layer.objects.active = crownObj         
          
             
            bpy.ops.object.editmode_toggle() 
            bpy.ops.mesh.select_all(action='DESELECT')  
            
            ref_bm_list = []
            for i in ref_objs:
                bm = bmesh.from_edit_mesh( i.data )
                bm.faces.ensure_lookup_table()
                ref_bm_list.append( bm )
                     
            bm = bmesh.from_edit_mesh( crownObj.data )
            bm.faces.ensure_lookup_table()
            
            
            
            for ref_bm in ref_bm_list:
                faces = []
                for f in ref_bm.faces:
                    face_center = f.calc_center_median()
                    co, nor, idx, dist = crown_bvh.find_nearest( face_center )
                    
                    if co != None:
                        faces.append( idx )
                        #bm.faces[idx].select_set(True)
                if len(faces) > 0:
                    group_faces.append( faces )
            # print( group_faces )       
            for i in group_faces:
                for j in i:
                    bm.faces[j].select_set(True)
        else:
            self.build_group_face(crown_bm, group_faces)
            bm = crown_bm 
        # return
        group_faces_center = []   
        group_faces_dir = []
        #~ debug_view.open()
        max_dist = 0.0
        
        ref_rad = -100
        ref_center = mathutils.Vector((0,0,0))
        
        
        ref_numFaces = 0
        
        r_i = 0
        
        for i, faces in enumerate( group_faces ):
            bpy.ops.mesh.select_all(action='DESELECT')  
            v = 0.0
            
            center_loc = mathutils.Vector((0,0,0))
            
            local_dir = mathutils.Vector((0,0,0))
            for f in faces:
                bm.faces[f].select_set(True)
                center_loc += bm.faces[f].calc_center_median()
                local_dir += bm.faces[f].normal
                if move_dist.get(f) != None:
                    if move_dist[f] > v:
                        v = move_dist[f]
            if v > thickness:
                v = 0.0
            center_loc = center_loc / len(faces)  
            local_dir = local_dir.normalized()
            
            # print( 'num_faces', len(faces) )
            # rad = local_dir.dot( mathutils.Vector((0,1,0)) ) 
            if len(faces) > ref_numFaces:
                ref_numFaces = len(faces)
                ref_center = center_loc
                r_i = i
                
                
            group_faces_center.append( center_loc)
            group_faces_dir.append( local_dir )
            #bpy.ops.view3d.snap_cursor_to_selected()
            co, idx, dist = self.boundary_kd.find(center_loc)
            if dist > max_dist:
                max_dist = dist
         
        test = []
        test.append( groove_center )
        #~ debug_view.drawPoints( test )  


            
        # for i, faces in enumerate( group_faces ):
            # bpy.ops.mesh.select_all(action='DESELECT')  
            # v = 0.0
            # for f in faces:
                # bm.faces[f].select_set(True)
                # if move_dist.get(f) != None:
                    # if move_dist[f] > v:
                        # v = move_dist[f]
            # if v > thickness:
                # v = 0.0
            # center_loc = group_faces_center[i]
            
            # co, idx, dist = self.boundary_kd.find( center_loc )
            
            # frac = dist / max_dist
            
  
        
            # vv = ( thickness - v + 0.3 * frac ) * frac
            
            # movePos = mathutils.Vector((0,1,0)) * vv + ref_center
            # bpy.ops.transform.translate(value=(0,-vv,0), orient_type='GLOBAL', orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=(3.0 * frac), use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

        bpy.ops.mesh.select_all(action='DESELECT')  
        
        if use_groove_center:
            self.set_oc_center_faces( bm, groove_center )

        else:        
            for i, faces in enumerate( group_faces ):
                
                v = 0.0
                
                if i == r_i:
                    for f in faces:
                        bm.faces[f].select_set(True)
                    
                    
        move_max_thickness = 0.0
        max_proportional_size = 2.0
        
        
        
    
            
        
        
        for i, faces in enumerate( group_faces ):
            v = 0.0
            for f in faces:
                if move_dist.get(f) != None:
                    if move_dist[f] > v:
                        v = move_dist[f]
            if v > thickness:
                v = 0.0
            center_loc = group_faces_center[i]
            
            co, idx, dist = self.boundary_kd.find( center_loc )
            
            frac = dist / max_dist
            
            proportional_size = ( center_loc - ref_center ).length
        
            vv = thickness - v + 0.4 
            
            if vv > move_max_thickness :
                move_max_thickness = vv
                
            if proportional_size > max_proportional_size:
                max_proportional_size = proportional_size
            
        bpy.ops.transform.translate(value=(0,0,move_max_thickness), orient_type='NORMAL', orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=max_proportional_size, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

                
        bpy.ops.object.editmode_toggle() 
        bpy.ops.object.select_all(action='DESELECT')
        
       
        # for i in ref_objs:
            # i.select_set(True)
        
        # bpy.context.view_layer.objects.active = ref_objs[0]

        # bpy.ops.object.delete()
        crownObj.select_set(True)
        bpy.context.view_layer.objects.active = crownObj    
        bpy.ops.outliner.orphans_purge()
        
# def inlayAdjust(crown_verts, crown_faces, inlay_verts, inlay_faces ):
    # convertMesh( 'inlay', inlay_verts, inlay_faces )
    # inlayObj = bpy.context.active_object
    # convertMesh( 'crown', crown_verts, crown_faces )
    # crownObj = bpy.context.active_object


def inlayAdjust_fn(inlayObj, crownObj, p_size=1.0):
  
    #print(type(inlayObj))
    inlay_points = []


    for i in inlayObj.data.vertices:
        inlay_points.append( i.co )
        


    if bpy.context.mode != 'EDIT_MESH':
        selectObj(crownObj)
        bpy.ops.object.editmode_toggle()
    bm = bmesh.from_edit_mesh(crownObj.data)
    bpy.ops.mesh.select_all(action='DESELECT')  
    out_i_p = area_inside(inlay_points, bm)
    #bpy.ops.object.editmode_toggle()



    findNearFaceByPoints(out_i_p, bm, inlayObj, p_size=p_size )
    
    bExec = False
    if bExec:
        inlay_bm = bmesh.new()   # create an empty BMesh
        inlay_bm.from_mesh(inlayObj.data)   
        inlay_bm.verts.ensure_lookup_table()

        boundaryCount = 0
        for i in inlay_bm.verts:
            if i.is_boundary:
                boundaryCount += 1

        boundary_kd = mathutils.kdtree.KDTree(boundaryCount)
        for i,v in enumerate(inlay_bm.verts):
            if v.is_boundary:
                boundary_kd.insert(v.co, i )
        boundary_kd.balance()

        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='VERT')


        size = len(bm.verts)
        kd = mathutils.kdtree.KDTree(size)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for i,v in enumerate(bm.verts):
            kd.insert(v.co, i )
        kd.balance()

        #~ for i in inlay_bm.verts:
            #~ if i.is_boundary:
                #~ theList = kd.find_range(i.co, 1.8)
                #~ for (co,index,dist) in theList:
                   
                    #~ bm.verts[index].select_set(True)
                    
        coo = []
        inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)
        moveVerts = []            


        calcVertMoveDist(inlay_bm,inlay_bvh, bm)
    # crownObj = bpy.data.objects['crown']
    
    # bpy.ops.wm.save_as_mainfile(filepath='./inlay_test.blend')
    
    out_verts = np.array([ v.co for v in crownObj.data.vertices ])
    out_faces = np.array([tuple(polygon.vertices) for polygon in crownObj.data.polygons[:]], dtype="object")
    bpy.ops.object.editmode_toggle()
    # bpy.ops.wm.read_homefile(app_template="")
    # bpy.ops.outliner.orphans_purge()
    return out_verts, out_faces
         
def loadStl(filepath):
    bpy.ops.import_mesh.stl( filepath=filepath )
    obj = bpy.context.active_object
    verts = np.array([ v.co for v in bpy.context.object.data.vertices ])
    faces = np.array([tuple(polygon.vertices) for polygon in obj.data.polygons[:]], dtype="object")
    return verts, faces


def to_objects( crown_verts, crown_faces, inlay_verts, inlay_faces ):
    convertMesh( 'inlay', inlay_verts, inlay_faces )
    inlayObj = bpy.context.active_object
    convertMesh( 'crown', crown_verts, crown_faces )
    crownObj = bpy.context.active_object    
    return inlayObj, crownObj
    
def inlayAdjust_file(crown_filepath, inlay_filepath):
    crown_verts, crown_faces = loadStl( crown_filepath )
    
    
    inlay_verts, inlay_faces = loadStl( inlay_filepath )
    #return inlayAdjust(crown_verts, crown_faces, inlay_verts, inlay_faces )
    
def firstFindCrownObj():
    for i in bpy.data.objects:
        if i.name.find('boundary_TPSed') >= 0:
            return i

def firstFindInlayObj():
    for i in bpy.data.objects:
        if i.name.find('thickness_shell') >= 0:
            return i    
            

    
def inlayAdjust(crown_verts, crown_faces, inlay_verts, inlay_faces, oc_verts = None ):   
    inlayObj, crownObj = to_objects( crown_verts, crown_faces, inlay_verts, inlay_faces )
 
    testFirst(crownObj, inlayObj)


    #~ inlayAdjust_obj = InlayAdjust(inlayObj, crownObj)

    
    #~ if oc_verts != None:
        #~ inlayAdjust_obj.set_oc_verts(oc_verts)
    #~ inlayAdjust_obj.check()
    #~ inlayAdjust_obj.check()
    #~ inlayAdjust_fn( inlayObj, crownObj, p_size=1.0  )    
    #~ inlayAdjust_fn( inlayObj, crownObj, p_size=0.2  )    

    
    out_verts = np.array([ v.co for v in crownObj.data.vertices ])
    out_faces = np.array([tuple(polygon.vertices) for polygon in crownObj.data.polygons[:]], dtype="object")

    bpy.ops.outliner.orphans_purge()
    
    return out_verts, out_faces
    
    
def first(crownObj, inlayObj):
    inlay_bm
    inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)
    
def createObjectBVH(obj):
    vertices = [ v.co for v in obj.data.vertices]
    polygons = [ p.vertices for p in obj.data.polygons]
    
    bvh = mathutils.bvhtree.BVHTree.FromPolygons(vertices, polygons) 

    return bvh
    
    
def isNearBoundary( kd, origin):
    co, index, dist = kd.find(origin)
    if dist < 0.8:
        return True
    
    return False
    
    
def testFirst(crownObj, inlayObj):    
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.editmode_toggle()    
    selectObj(crownObj)
    
    inlay_bm = bmesh.new()
    inlay_bm.from_mesh(inlayObj.data)
    
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.editmode_toggle()
    
    crown_bm = bmesh.from_edit_mesh( crownObj.data)
    
    crown_bm.faces.ensure_lookup_table()
    
    #~ crown_bm = bmesh.new()
    #~ crown_bm.from_mesh( crownObj.data )
    
    boundary_count = 0
    #~ debug_view.close()
    #~ debug_view.open()
    
    for i in inlay_bm.verts:
        if i.is_boundary:
            boundary_count += 1
    
    inlay_boundary_kd = mathutils.kdtree.KDTree(boundary_count)
    for i in inlay_bm.verts:
        if i.is_boundary:
            inlay_boundary_kd.insert(i.co, i.index)
    inlay_boundary_kd.balance()
    
    
    bDebug = False
    
    for c in range(0, 80):
        # print( c )
        bpy.ops.mesh.select_all(action='DESELECT')
        #crown_bvh = createObjectBVH(crownObj)
        crown_bvh = mathutils.bvhtree.BVHTree.FromBMesh(crown_bm)
        move_faces = []
        move_faces_dist = []
        move_faces_dir = []
        inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)
        for f in crown_bm.faces:
            origin =  f.calc_center_median()
            direction = f.normal
            loc, normal, idx, dist = inlay_bvh.ray_cast( origin, direction )
            if loc != None and direction.dot(normal) > 0.6:

                
                bBoundary = isNearBoundary(inlay_boundary_kd, origin)
                if not bBoundary and dist < 2:
                    #~ f.select_set(True)
                    ray_dir = mathutils.Vector((0,1,0))
                    s_loc, s_nor, s_idx, s_dist = crown_bvh.ray_cast( origin + ray_dir*0.1, ray_dir )
                    if s_loc == None:
                  
                        move_faces.append( f.index )
                        move_faces_dist.append( dist )
                        move_faces_dir.append( direction )
                    

        if bDebug:            
                
            for i in move_faces:
                crown_bm.faces[i].select_set(True)
                
            return
        if len(move_faces) == 0 :
            break
        #~ return
                    #debug_view.drawPoint( n_loc )
                    
        mf_dist, mf, mf_dir = sort_arrays_based_on_first_desc( move_faces_dist, move_faces, move_faces_dir )
        bpy.ops.mesh.select_all(action='DESELECT')
        for i, f in enumerate(mf):
            #~ crown_bm.faces[f].select_set(True)
            upDir = mathutils.Vector((0,1,0))
            if crown_bm.faces[f].normal.dot( upDir ) > 0.5:
                #~ bpy.ops.mesh.select_all(action='DESELECT')
                crown_bm.faces[f].select_set(True)
                
            #center = crown_bm.faces[f].calc_center_median()
                # print( '***', mf_dist[0] )
                bpy.ops.transform.translate(value=(0,0.2,0), orient_type='GLOBAL',  orient_matrix_type='GLOBAL', constraint_axis=(True, True, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=2.0, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)

                break
                
    #**********************************************
    for c in range(0, 50):
        # print( 'se:', c )
        bpy.ops.mesh.select_all(action='DESELECT')
        #crown_bvh = createObjectBVH(crownObj)
        crown_bvh = mathutils.bvhtree.BVHTree.FromBMesh(crown_bm)
        move_faces = []
        move_faces_dist = []
        move_faces_dir = []
        inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)
        for f in crown_bm.faces:
            origin =  f.calc_center_median()
            direction = f.normal
            loc, normal, idx, dist = inlay_bvh.ray_cast( origin, direction )
            if loc != None:

                
                bBoundary = isNearBoundary(inlay_boundary_kd, origin)
                if not bBoundary and dist < 2:
                    #~ f.select_set(True)
                    ray_dir =f.normal
                    s_loc, s_nor, s_idx, s_dist = crown_bvh.ray_cast( origin + ray_dir*0.1, ray_dir )
                    if s_loc == None:
                  
                        move_faces.append( f.index )
                        move_faces_dist.append( dist )
                        move_faces_dir.append( direction )
                    

        if bDebug:            
                
            for i in move_faces:
                crown_bm.faces[i].select_set(True)
                
            return
        if len(move_faces) == 0 :
            break
        #~ return
                    #debug_view.drawPoint( n_loc )
                    
        mf_dist, mf, mf_dir = sort_arrays_based_on_first_desc( move_faces_dist, move_faces, move_faces_dir )
        bpy.ops.mesh.select_all(action='DESELECT')
        for i, f in enumerate(mf):
            #~ crown_bm.faces[f].select_set(True)
                #~ bpy.ops.mesh.select_all(action='DESELECT')
            crown_bm.faces[f].select_set(True)
            
            # print( '***', mf_dist[0] )
            bpy.ops.transform.translate(value=(0,0.0,0.2), orient_type='NORMAL',  orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=2.0, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)

            break
                    
    #~ inlayAdjust_obj = InlayAdjust(inlayObj, crownObj)
    #~ inlayAdjust_obj.check()
    #~ inlayAdjust_obj.check()
    
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.editmode_toggle()          
def testFirst2(crownObj, inlayObj):
    #crownObj = firstFindCrownObj()
    #inlayObj = firstFindInlayObj()
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.editmode_toggle()    
    selectObj(crownObj)
    
    inlay_bm = bmesh.new()
    inlay_bm.from_mesh(inlayObj.data)
    
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.editmode_toggle()
    
    crown_bm = bmesh.from_edit_mesh( crownObj.data)
    
    crown_bm.faces.ensure_lookup_table()
    
    #~ crown_bm = bmesh.new()
    #~ crown_bm.from_mesh( crownObj.data )
    
    boundary_count = 0
    #~ debug_view.close()
    #~ debug_view.open()
    
    for i in inlay_bm.verts:
        if i.is_boundary:
            boundary_count += 1
    
    inlay_boundary_kd = mathutils.kdtree.KDTree(boundary_count)
    for i in inlay_bm.verts:
        if i.is_boundary:
            inlay_boundary_kd.insert(i.co, i.index)
    inlay_boundary_kd.balance()
    
    
    bDebug = False
    
    for c in range(0, 80):
        # print( c )
        bpy.ops.mesh.select_all(action='DESELECT')
        #crown_bvh = createObjectBVH(crownObj)
        crown_bvh = mathutils.bvhtree.BVHTree.FromBMesh(crown_bm)
        move_faces = []
        move_faces_dist = []
        move_faces_dir = []
        inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)
        for f in crown_bm.faces:
            origin =  f.calc_center_median()
            direction = f.normal
            loc, normal, idx, dist = inlay_bvh.ray_cast( origin, direction )
            if loc != None and direction.dot(normal) > 0.7:
                
                #~ f.select_set(True)
                
                bBoundary = isNearBoundary(inlay_boundary_kd, origin)
                if not bBoundary and dist < 2:
                    
                    s_loc, s_nor, s_idx, s_dist = crown_bvh.ray_cast( origin + direction*0.1, direction )
                    if s_loc == None:
                  
                        move_faces.append( f.index )
                        move_faces_dist.append( dist )
                        move_faces_dir.append( direction )
                    
                    
        if bDebug:            
                
            for i in move_faces:
                crown_bm.faces[i].select_set(True)
                
            return
        if len(move_faces) == 0 :
            break
        #~ return
                    #debug_view.drawPoint( n_loc )
                    
        mf_dist, mf, mf_dir = sort_arrays_based_on_first_desc( move_faces_dist, move_faces, move_faces_dir )
        bpy.ops.mesh.select_all(action='DESELECT')
        for i, f in enumerate(mf):
            crown_bm.faces[f].select_set(True)
            upDir = mathutils.Vector((0,1,0))
            if crown_bm.faces[f].normal.dot( upDir ) > 0.8:
            
            #center = crown_bm.faces[f].calc_center_median()
                # print( '******', mf_dist[i] )
            #print( '***', center )
                bpy.ops.transform.translate(value=(0,0, mf_dist[i]), orient_type='NORMAL',  orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=2.0, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)

                break
                
    
    #~ inlayAdjust_obj = InlayAdjust(inlayObj, crownObj)
    #~ inlayAdjust_obj.check()
    #~ inlayAdjust_obj.check()
    
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.editmode_toggle()      
    
    
def build_crown( verts_filename, faces_filename ):
    hFile = open( verts_filename, 'rt' )
    bLoop = True
    verts = []
    while (bLoop):
        theLine = hFile.readline()
        if theLine == '':
            bLoop = False
            break;

        values = theLine.split(' ')
        verts.append( ( float(values[0]),  float(values[1]),  float(values[2]) ) )
        
        
    hFile.close()

    hFile = open( faces_filename, 'rt' )
    bLoop = True
    faces = []
    while (bLoop):
        theLine = hFile.readline()
        if theLine == '':
            bLoop = False
            break;

        values = theLine.split(' ')
        faces.append( ( int(values[0]),  int(values[1]),  int(values[2]) ) )
        
        
    hFile.close()
    
    return [ verts, faces ]
    
    
def get_oc_verts( oc_filename ):
    hFile = open( oc_filename, 'rt' )
    bLoop = True
    verts = []
    while (bLoop):
        theLine = hFile.readline()
        if theLine == '':
            bLoop = False
            break;

        
        verts.append( int(theLine)  )
        
        
    hFile.close()

    return verts
    
    
def is_slender_triangle(face, angle_threshold=5):
 
    if len(face.verts) != 3:
        return False

    # Get the three vertices of the triangle
    v0, v1, v2 = face.verts

    # Calculate the vectors of the edges
    edge0 = v1.co - v0.co
    edge1 = v2.co - v1.co
    edge2 = v0.co - v2.co

    # Normalize the edge vectors
    edge0.normalize()
    edge1.normalize()
    edge2.normalize()

    # Calculate the angles using the dot product
    angle0 = math.degrees(math.acos(max(-1.0, min(1.0, edge0.dot(-edge2)))))
    angle1 = math.degrees(math.acos(max(-1.0, min(1.0, edge1.dot(-edge0)))))
    angle2 = math.degrees(math.acos(max(-1.0, min(1.0, edge2.dot(-edge1)))))

    # Check if any angle is smaller than the threshold
    return any(angle < angle_threshold for angle in (angle0, angle1, angle2))
    
    
def get_proportional(kd, center):
    co, index, dist = kd.find(center)
    return dist
    
    
def sort_arrays_based_on_first(arr1, arr2, arr3):
    indices = list(range(len(arr1)))
    indices.sort(key=lambda i: arr1[i])
    sorted_arr1 = [arr1[i] for i in indices]
    sorted_arr2 = [arr2[i] for i in indices]
    sorted_arr3 = [arr3[i] for i in indices]
    return sorted_arr1, sorted_arr2, sorted_arr3    
    
    
def sort_arrays_based_on_first_desc(arr1, arr2, arr3):
    combined = list(zip(arr1, arr2, arr3))
    combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
    sorted_arr1 = [item[0] for item in combined_sorted]
    sorted_arr2 = [item[1] for item in combined_sorted]
    sorted_arr3 = [item[2] for item in combined_sorted]
    return sorted_arr1, sorted_arr2, sorted_arr3    
    
def loopFixSelfIntersection(inlay_bm, inlay_surface_array, boundary_kd, debug_cc):
    
    # print( '**************', 'loopFixSelfIntersection')

    inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm) 
    bLoop = True
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.print3d_check_intersect()
    bpy.ops.mesh.print3d_select_report()
    
    
    selectCount = 0
    
    for i in inlay_bm.faces:
        if i.select:
            selectCount += 1
            
    if selectCount == 0:
        return True
    # print( 'selectCount:', selectCount )
    _need_move_faces = [] 
    _need_center_faces = []        
    _need_dist_faces = [] 

    for i in inlay_bm.faces:
        if i.select:
            if i.index in inlay_surface_array:
                f_center = i.calc_center_median()
                _need_move_faces.append( i.index )
                _need_center_faces.append( f_center )
                _need_dist_faces.append( 0.1 )
                
                
                
                
   
    bpy.ops.mesh.select_all(action='DESELECT')


    for f in  inlay_surface_array:
        
        direction = inlay_bm.faces[f].normal
        origin = inlay_bm.faces[f].calc_center_median() + direction * 0.01
        loc, nor, idx, dist = inlay_bvh.ray_cast(origin, direction)
        if loc != None:
            if not is_slender_triangle(inlay_bm.faces[f]):
                # inlay_bm.faces[f].select_set(True)
                _need_move_faces.append( f )
                f_center = inlay_bm.faces[f].calc_center_median()
                _need_center_faces.append( f_center )
                _need_dist_faces.append( dist+0.05 )
                
    
    
    s_dist_faces, s_center_faces, s_move_faces = sort_arrays_based_on_first_desc(_need_dist_faces,  _need_center_faces, _need_move_faces )

        
    if debug_cc == 1:
        for i,f_id in enumerate(s_move_faces):
            inlay_bm.faces[f_id].select_set(True)
        return True

    
                
    # print( '_need_move_faces:', len(_need_move_faces) ) 
    bpy.ops.mesh.select_all(action='DESELECT')
    for i,f_id in enumerate(s_move_faces):
        inlay_bm.faces[f_id].select_set(True)
        f_center = inlay_bm.faces[f_id].calc_center_median()
        if (_need_center_faces[i] - f_center).length < 0.1:
            rrr = get_proportional( boundary_kd, f_center )
            dddd = inlay_bm.faces[f_id].normal.dot( mathutils.Vector((0,1,0) ))
            # print( dddd )
            if dddd > 0.6:
                bpy.ops.transform.translate(value=(0,0, s_dist_faces[i]), orient_type='NORMAL',  orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=1.0, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)
                
        inlay_bm.faces[f_id].select_set(False)     
        bpy.ops.mesh.select_all(action='DESELECT')
        
        
    return False
    
        
def findSurfaceObj():
    for i in bpy.data.objects:
        if i.name.find('final_inlay_outer' ) >= 0:
            return i
            
def findInlayObj():
    for i in bpy.data.objects:
        if i.name.find('stitched_inlay' ) >= 0:
            return i    
            
def findThicknessShell():
    for i in bpy.data.objects:
        if i.name.find('thickness_shell') >= 0:
            return i        
            
bDebugSelfIntersection = True        

def inlayPost22(surfaceObj, inlayObj, bSmooth=True, separate=True ):
    bpy.ops.preferences.addon_enable(module="object_print3d_utils")
    exist_selfIntersection = True

    if bpy.context.mode == 'EDIT_MESH':
        bpy.ops.object.editmode_toggle()    
    bpy.ops.object.select_all(action='DESELECT')


    selectObj(inlayObj)
    surfaceObj.select_set(True)

    
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.editmode_toggle()
        


    surface_bm = bmesh.from_edit_mesh(surfaceObj.data)
    inlay_bm = bmesh.from_edit_mesh(inlayObj.data)
    inlay_bm.verts.ensure_lookup_table()
    inlay_bm.faces.ensure_lookup_table()
    bpy.ops.mesh.select_all(action='DESELECT')

    surface_bvh = mathutils.bvhtree.BVHTree.FromBMesh(surface_bm)
    inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)

    inlay_surface_array = []
    inlay_back_array = []
    for f in surface_bm.faces:
        f_center = f.calc_center_median()
        loc,nor,idx,dist= inlay_bvh.find_nearest( f_center )
        if dist < 0.5:
            inlay_bm.faces[idx].select_set(True)
            inlay_surface_array.append( idx )
        
    for f in inlay_bm.faces:
        if not f.select:
            inlay_back_array.append(f.index)
        

    fixed_co_verts = []
    fixed_id_verts = []
    _need_move_faces = [] 

    for i in inlay_bm.verts:
        fixed_co_verts.append( (i.co[0], i.co[1], i.co[2]) )
        
        #~ if not flexibility:
        if i.select == False:
            fixed_id_verts.append( i.index )
                
    #~ print('***', len(fixed_id_verts))
    #~ return    
    boundary_count = 0
    for i in surface_bm.verts:
        if i.is_boundary:
            boundary_count += 1
    boundary_kd = mathutils.kdtree.KDTree(boundary_count)
    for v in surface_bm.verts:
        if v.is_boundary:
            boundary_kd.insert(v.co, v.index)
    boundary_kd.balance()
    bb_id_verts = []
    for i in inlay_bm.verts:
        loc, index, dist = boundary_kd.find( i.co )
        
        if loc != None and dist < 0.01:
            bb_id_verts.append( i.index )
            # fixed_co_verts.append( (i.co[0], i.co[1], i.co[2]) )
            
            fixed_id_verts.append( i.index )
    #~ bpy.ops.mesh.select_all(action='DESELECT')
  
    #~ bpy.ops.mesh.select_mode(use_expand=True, type='VERT')        
    #~ for i in fixed_id_verts:
        #~ inlay_bm.verts[i].select_set(True)
        
     
    bLoop = True
    testSelfIntersect = True
    
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.print3d_check_intersect()
    bpy.ops.mesh.print3d_select_report()

        
    _need_move_faces = [] 
    _need_center_faces = []        
    _need_dist_faces = [] 

    for i in inlay_bm.faces:
        if i.select:
            if i.index in inlay_surface_array:
                f_center = i.calc_center_median()
                
                b_co, b_idx, b_dist = boundary_kd.find(f_center)
                
                #print(b_co, b_idx, b_dist  )
                if b_dist > 0.5:
                    _need_move_faces.append( i.index )
                    _need_center_faces.append( f_center )
                    _need_dist_faces.append( 0.1 )
            
   
    bpy.ops.mesh.select_all(action='DESELECT')
    
    for i in _need_move_faces:
        inlay_bm.faces[i].select_set(True)
        

def inlayPost(surfaceObj, inlayObj, shellObj=None, bSmooth=True, separate=True ):
    bpy.ops.preferences.addon_enable(module="object_print3d_utils")
    exist_selfIntersection = True

    if bpy.context.mode == 'EDIT_MESH':
        bpy.ops.object.editmode_toggle()    
    bpy.ops.object.select_all(action='DESELECT')


    selectObj(inlayObj)
    surfaceObj.select_set(True)

    
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.editmode_toggle()
        


    surface_bm = bmesh.from_edit_mesh(surfaceObj.data)
    surface_bm.faces.ensure_lookup_table()
    inlay_bm = bmesh.from_edit_mesh(inlayObj.data)
    inlay_bm.verts.ensure_lookup_table()
    inlay_bm.faces.ensure_lookup_table()
    bpy.ops.mesh.select_all(action='DESELECT')

    surface_bvh = mathutils.bvhtree.BVHTree.FromBMesh(surface_bm)
    
    
    s_o = surface_bvh.overlap(surface_bvh)
    surface_exist_overlap = False
    if( len(s_o) > 0 ):
        surface_exist_overlap = True
    
    s_o_kd = None
    if surface_exist_overlap:
        s_o_kd = mathutils.kdtree.KDTree( len(s_o) )
        for i,j in s_o:
            center = surface_bm.faces[i].calc_center_median()
            s_o_kd.insert( center , i )
        s_o_kd.balance()
    
    inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)

    inlay_surface_array = []
    inlay_back_array = []
    for f in surface_bm.faces:
        f_center = f.calc_center_median()
        loc,nor,idx,dist= inlay_bvh.find_nearest( f_center )
        if dist < 0.5:
            inlay_bm.faces[idx].select_set(True)
            inlay_surface_array.append( idx )
        
    for f in inlay_bm.faces:
        if not f.select:
            inlay_back_array.append(f.index)
        

    fixed_co_verts = []
    fixed_id_verts = []
    _need_move_faces = [] 

    for i in inlay_bm.verts:
        fixed_co_verts.append( (i.co[0], i.co[1], i.co[2]) )
        
        #~ if not flexibility:
        if i.select == False:
            fixed_id_verts.append( i.index )
            
    
                
    #~ print('***', len(fixed_id_verts))
    #~ return    
    boundary_count = 0
    for i in surface_bm.verts:
        if i.is_boundary:
            boundary_count += 1
    boundary_kd = mathutils.kdtree.KDTree(boundary_count)
    for v in surface_bm.verts:
        if v.is_boundary:
            boundary_kd.insert(v.co, v.index)
    boundary_kd.balance()
    bb_id_verts = []
    for i in inlay_bm.verts:
        loc, index, dist = boundary_kd.find( i.co )
        
        if loc != None and dist < 0.01:
            bb_id_verts.append( i.index )
            # fixed_co_verts.append( (i.co[0], i.co[1], i.co[2]) )
            
            fixed_id_verts.append( i.index )
    #~ bpy.ops.mesh.select_all(action='DESELECT')
  
    #~ bpy.ops.mesh.select_mode(use_expand=True, type='VERT')        
    #~ for i in fixed_id_verts:
        #~ inlay_bm.verts[i].select_set(True)
        
     
    bLoop = True
    testSelfIntersect = False
    count = 0
    while( bLoop ):
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.print3d_check_intersect()
        bpy.ops.mesh.print3d_select_report()
        
        
        if testSelfIntersect:
            bLoop = False
            break
        
        
        
        _need_move_faces = [] 
        _need_center_faces = []        
        _need_dist_faces = [] 

        for i in inlay_bm.faces:
            if i.select:
                if i.index in inlay_surface_array:
                    f_center = i.calc_center_median()
                    
                    bFilter = False
                    if surface_exist_overlap:
                        co,idx,dist = s_o_kd.find( f_center )
                        # print( co, idx, dist )
                        if dist < 0.2:
                            bFilter = True
                    
                    if not bFilter:
                        b_co, b_idx, b_dist = boundary_kd.find(f_center)
                        
                        #print(b_co, b_idx, b_dist  )
                        #~ if b_dist > 0.5:
                        _need_move_faces.append( i.index )
                        _need_center_faces.append( f_center )
                        _need_dist_faces.append( 0.1 )
                        
       
        bpy.ops.mesh.select_all(action='DESELECT')
        
        if( len(_need_move_faces) == 0 ):
            break
        count += 1
        
        
        for i,f_id in enumerate(_need_move_faces):
            inlay_bm.faces[f_id].select_set(True)   
            
            bpy.ops.transform.translate(value=(0, 0, 0.1), orient_type='NORMAL',  orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=0.5, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)

            bpy.ops.mesh.select_all(action='DESELECT')
            break
            
        if count >= 200:
            break
            
    if testSelfIntersect:
        return
        
    bbbb = False
    if bbbb:
    
        global bDebugSelfIntersection
        if bDebugSelfIntersection:
            for i,f_id in enumerate(_need_move_faces):
                inlay_bm.faces[f_id].select_set(True)   
            return       
        if len(_need_move_faces) == 0:
            exist_selfIntersection = False
            
        for i in inlay_bm.verts:
            if i.select:
                points.append( i.co )
                
        #print( 'EXIST', exist_selfIntersection )
                
        if exist_selfIntersection:

            for f in  inlay_surface_array:
                
                direction = inlay_bm.faces[f].normal
                origin = inlay_bm.faces[f].calc_center_median() + direction * 0.01
                loc, nor, idx, dist = inlay_bvh.ray_cast(origin, direction)
                if loc != None:
                    if not is_slender_triangle(inlay_bm.faces[f]):
                        # inlay_bm.faces[f].select_set(True)
                        _need_move_faces.append( f )
                        f_center = inlay_bm.faces[f].calc_center_median()
                        _need_center_faces.append( f_center )
                        _need_dist_faces.append( dist+0.05 )
                    

        bpy.ops.mesh.select_all(action='DESELECT')
        for i,f_id in enumerate(_need_move_faces):
            inlay_bm.faces[f_id].select_set(True)
            f_center = inlay_bm.faces[f_id].calc_center_median()
            if (_need_center_faces[i] - f_center).length < 0.1:
                rrr = get_proportional( boundary_kd, f_center )

                bpy.ops.transform.translate(value=(0,0, _need_dist_faces[i]*0.5), orient_type='NORMAL',  orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=1.0, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)
                    
            inlay_bm.faces[f_id].select_set(False)     
            bpy.ops.mesh.select_all(action='DESELECT')
        # for k in range(0, 1 ):
        
        for i in range(0,10):
            bStop = loopFixSelfIntersection( inlay_bm, inlay_surface_array, boundary_kd, 0 )
            
            for j in fixed_id_verts:
                inlay_bm.verts[j].co[0] = fixed_co_verts[j][0]
                inlay_bm.verts[j].co[1] = fixed_co_verts[j][1]
                inlay_bm.verts[j].co[2] = fixed_co_verts[j][2]        
            
            bmesh.update_edit_mesh(inlayObj.data)
            if(bStop):
                break
    
    
    
    
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.data.scenes["Scene"].print_3d.thickness_min = 0.8
    thick_ops = bpy.ops.mesh.print3d_check_thick()
    bpy.ops.mesh.print3d_select_report()  
    need_move_faces = []
    need_center_faces = []
    for i in inlay_bm.faces:
        if i.select:
            if i.index in inlay_surface_array:
                f_center = i.calc_center_median()
                need_move_faces.append( i.index )
                need_center_faces.append( f_center )
                
    
                
    bpy.ops.mesh.select_all(action='DESELECT')
    
    for i, f_id in enumerate(need_move_faces):
        inlay_bm.faces[f_id].select_set(True)
    
    
    bCheckThick = True
    if bCheckThick:
        for j in range(0,10):
            last_dist = 0.0
            select_face = -1
            for i, f_id in enumerate(need_move_faces):
                f_center = inlay_bm.faces[f_id].calc_center_median()
                if (need_center_faces[i] - f_center).length < 0.1:
                    face_center = inlay_bm.faces[f_id].calc_center_median()
                    
                    #print( 'face_center', face_center,  boundary_kd )
                    loc, idx,dist = boundary_kd.find(face_center)
                    
                    #print( loc, idx, dist )
                    
                    if dist > last_dist and dist > 0.1:
                        last_dist = dist 
                        select_face = f_id
                        
            #print( 'select_face:', select_face, last_dist )
            if select_face != -1:
                inlay_bm.faces[select_face].select_set(True)
                bpy.ops.transform.translate(value=(0,0,0.1), orient_type='NORMAL',  orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=last_dist, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)
            bpy.ops.mesh.select_all(action='DESELECT')

        
    
        
    #~ bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=2)
   
    

        #~ for i in fixed_id_verts:
            #~ inlay_bm.verts[i].co[0] = fixed_co_verts[i][0]
            #~ inlay_bm.verts[i].co[1] = fixed_co_verts[i][1]
            #~ inlay_bm.verts[i].co[2] = fixed_co_verts[i][2]        
        
        #~ bpy.ops.mesh.select_mode(use_expand=True, type='VERT')
        #~ for i in bb_id_verts:
            #~ inlay_bm.verts[i].select_set(True)
           
            #~ offset = mathutils.Vector( fixed_co_verts[i] )  - mathutils.Vector( inlay_bm.verts[i].co )

            #~ bpy.ops.transform.translate(value=offset, orient_type='GLOBAL',  orient_matrix_type='NORMAL',  mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=0.5, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)
            #~ bpy.ops.mesh.select_all(action='DESELECT')

    for i in fixed_id_verts:
        inlay_bm.verts[i].co[0] = fixed_co_verts[i][0]
        inlay_bm.verts[i].co[1] = fixed_co_verts[i][1]
        inlay_bm.verts[i].co[2] = fixed_co_verts[i][2]
    for f in inlay_surface_array:
        inlay_bm.faces[f].select_set(True)       
    
    surfaceObj.select_set(False)
    
    if separate:

        bpy.ops.mesh.separate()
        
        inlayObj.select_set(False)
        bpy.context.view_layer.objects.active = None
        for i in bpy.data.objects:
            # print( i.name.find( inlayObj.name + '.') )
            if i.name.find( inlayObj.name + '.') >= 0:
                #print( '*********************', i.name, bSmooth )
                if bSmooth:
                    i.name = 'v2'
                else:
                    i.name = 'v1'  
                break
            #~ if ( i != inlayObj ):
                #~ if i.name.find(inlayObj.name):
                    
                    #~ print( '****************',i.name )
                    #~ bpy.context.view_layer.objects.active = i

                    #~ break
            
    if bSmooth:
        selectObj( bpy.data.objects['v2'] )
        
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_mode(type='VERT')    
        
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.select_all(action='INVERT')
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=3)   
        bpy.ops.object.editmode_toggle()        
        
    else:
        selectObj( bpy.data.objects['v1'] )
        
        
        
    ############################# NEW 2015/7/1 ###########################

    shellObj.select_set(True)

    bpy.ops.object.editmode_toggle()  
    

    inlay_bm = bmesh.from_edit_mesh( bpy.context.active_object.data )
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    if shellObj != None:    
        shell_bm = bmesh.from_edit_mesh( shellObj.data )
        bpy.ops.mesh.select_all(action='DESELECT')
        shell_bvh = mathutils.bvhtree.BVHTree.FromBMesh(shell_bm)   
        inlay_bm.faces.ensure_lookup_table()
        for i in range(0,50):
            inlay_bvh = mathutils.bvhtree.BVHTree.FromBMesh(inlay_bm)
            
            
            theList = inlay_bvh.overlap( shell_bvh)
            
            if len(theList) == 0:
                break
            for f, o_f in theList:
                inlay_bm.faces[f].select_set(True)
                bpy.ops.transform.translate(value=(0,0,0.01), constraint_axis=(False, False, True),  orient_type='NORMAL',  orient_matrix_type='NORMAL',  mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=0.5, use_proportional_connected=True, use_proportional_projected=False, release_confirm=True)
                inlay_bm.faces[f].select_set(False)
                break
    
    bExport = True
    
    if bExport:
        v_array = [ v.co for v in inlay_bm.verts if v.select ]
        out_verts = np.array(v_array)
        

        f_array = [ (f.verts[0].index, f.verts[1].index, f.verts[2].index ) for f in inlay_bm.faces if f.select ]
        
        # print(len(v_array), len( f_array ))
        out_faces = np.array( f_array, dtype="object")  
        
        bpy.ops.object.editmode_toggle()        
    tri_mesh = blender_mesh_to_trimesh(bpy.context.active_object.data)
    return tri_mesh
    #~ return out_verts, out_faces
    # bmesh.update_edit_mesh(inlayObj.data)
def inlayPostWarp(surface_verts, surface_faces,inlay_verts, inlay_faces, shell_verts, shell_faces ):

        

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    bpy.ops.outliner.orphans_purge()
    convertMesh( 'shell', shell_verts, shell_faces )
    shellObj = bpy.context.active_object
    
    convertMesh( 'surface', surface_verts, surface_faces )
    surfaceObj = bpy.context.active_object
    convertMesh( 'inlay', inlay_verts, inlay_faces )
    
    inlayObj = bpy.context.active_object
    tri_mesh = inlayPost(surfaceObj, inlayObj, shellObj=shellObj)  
    if bpy.context.mode == 'EDIT_MESH':
        bpy.ops.object.editmode_toggle()
    return tri_mesh
    
    
test_first_thickness = False
if __name__ == '__main__':
    if test_first_thickness:
        crownObj = firstFindCrownObj()
        inlayObj = firstFindInlayObj()
        
        
        testFirst(crownObj, inlayObj)
        
    else:
        
        # surfaceObj = bpy.data.objects['11 Final Inlay Outer 27']
        # inlayObj= bpy.data.objects['4_dilation_0.04-0.08_25']
        #surface_verts, surface_faces = loadStl( '/mnt/e/work/inlay_proj/test/11_final_inlay_outer_27.stl' )
        # inlay_verts, inlay_faces = loadStl( '/mnt/e/work/inlay_proj/test/12_stitched_inlay_27.stl' )  
        # inlayPostWarp(surface_verts, surface_faces, inlay_verts, inlay_faces )
        # crownObj = bpy.data.objects['5.5_inflated_outer_25']

      
        # bpy.ops.wm.save_as_mainfile(filepath='./inlay_test2.blend')
        # inlayAdjust_obj = InlayAdjust(inlayObj, crownObj)
        
        
        #inlayAdjust_obj.check()
        # inlayAdjust_fn( inlayObj, crownObj, p_size=1.0  )    
        # inlayAdjust_fn( inlayObj, crownObj, p_size=0.2  )    
        # inlayAdjust_obj = InlayAdjust(inlayObj, crownObj)
        # inlayAdjust_obj.onlyCheck()
        bBatch = False
        if bBatch:
            a = [True, False]
            homeDir = 'E:/work/inlay_data/INLAY_DATA'
            for b in a:
                bSmooth = b
                surfaceObj = None
                inlayObj = None
                
                for jj in os.listdir(homeDir):
                    
                    for jjj in os.listdir( homeDir+'/'+ jj  ):
                            
                        theDir = homeDir+'/'+ jj + '/' + jjj
                    
                        # print( theDir )
                        try:
                            for i in os.listdir(theDir):
                                if i.find( 'final_inlay_outer' ) > 0:
                                    bpy.ops.import_mesh.stl(filepath=theDir+'/' + i )
                                    surfaceObj = bpy.context.active_object
                                
                                if i.find( 'stitched_inlay' ) > 0:
                                    #print( '***', theDir+'/' + i )
                                    bpy.ops.import_mesh.stl(filepath=theDir+'/' + i )
                                    inlayObj = bpy.context.active_object
                            
                            # print( surfaceObj, inlayObj )
                            
                            
                            inlayPost(surfaceObj, inlayObj, bSmooth=bSmooth, separate=True)  
                            if bSmooth:
                                bpy.ops.export_mesh.stl(filepath=theDir+'/v2.stl', use_selection=True  )
                            else:
                                bpy.ops.export_mesh.stl(filepath=theDir+'/v1.stl', use_selection=True  )
                            
                            bpy.ops.object.select_all(action='SELECT')

                            bpy.ops.object.delete(use_global=False, confirm=False)
                        except:
                            pass
        else:
            
            surfaceObj = findSurfaceObj()
            inlayObj = findInlayObj()
            shellObj = findThicknessShell()

            inlayPost(surfaceObj, inlayObj, shellObj=shellObj, bSmooth=True, separate=True)  
