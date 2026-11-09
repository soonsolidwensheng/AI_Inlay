import bpy
import bmesh
import mathutils
from mathutils import Vector
from mathutils.bvhtree import BVHTree
# try:
#     import debug_view
#     debug_view_exist = True
# except:
#     debug_view_exist = False
print("bpy import success")
    
    

import numpy as np
g_boundary_kd = None

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
        print( result )
        if result[0] != None:
            
            offset =  result[3] 
            print('offset', offset )
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
                    bpy.ops.transform.translate(value=(0.0, 0.0, offset), orient_type='NORMAL', orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=1.5, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)



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
        from tqdm import tqdm
        for f in tqdm(aSelFaces):
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
            # if count > 0:
                # print( 'count', count )

   
                
                
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
            print(f)
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
    
    def set_oc_verts(self, verts_idx, crownObj):
        crownObj.select_set(True)
        bpy.context.view_layer.objects.active = crownObj
        if bpy.context.mode != 'EDIT_MESH':
            selectObj(crownObj)
            bpy.ops.object.editmode_toggle()
        # bpy.ops.object.editmode_toggle()   
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
        
    def check(self, thickness=0.8, boundary_protect_range_offset=0):
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
            print( '****************' )
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
        
        obj_name = bpy.context.active_object.name
        bpy.ops.mesh.duplicate()

        def isSelectVerts(bm): # 如果除了边缘附近没有需要增厚的点，就退出
            selected_verts = list(filter(lambda v: v.select, bm.verts))
            if len(selected_verts) > 2 :
                return True
            return False
        
        if not isSelectVerts( crown_bm ):
            return
        
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
        
        
        group_faces = []
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
        group_faces_center = []   
        group_faces_dir = []
        # debug_view.open()
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
        # debug_view.drawPoints( test )  


            
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
        # 临时offset一下max_proportional_size，以防止边缘太薄，下个版本更新自适应性proportional_size
        max_proportional_size += boundary_protect_range_offset
        bpy.ops.transform.translate(value=(0,0,move_max_thickness), orient_type='NORMAL', orient_matrix_type='NORMAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=max_proportional_size, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

                
        bpy.ops.object.editmode_toggle() 
        bpy.ops.object.select_all(action='DESELECT')
        
       
        for i in ref_objs:
            i.select_set(True)
        
        bpy.context.view_layer.objects.active = ref_objs[0]

        bpy.ops.object.delete()
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
    
def inlayAdjust(crown_verts, crown_faces, inlay_verts, inlay_faces, oc_verts = None,
                thickness=0.8,
                boundary_protect_range_offset=-0.2):   
    inlayObj, crownObj = to_objects( crown_verts, crown_faces, inlay_verts, inlay_faces )
 


    inlayAdjust_obj = InlayAdjust(inlayObj, crownObj)

    
    if oc_verts != None:
        inlayAdjust_obj.set_oc_verts(oc_verts, crownObj)
    inlayAdjust_obj.check(thickness=thickness, boundary_protect_range_offset=boundary_protect_range_offset)
    inlayAdjust_fn( inlayObj, crownObj, p_size=1.0  )    
    inlayAdjust_fn( inlayObj, crownObj, p_size=0.2  )    

    
    out_verts = np.array([ v.co for v in crownObj.data.vertices ])
    out_faces = np.array([tuple(polygon.vertices) for polygon in crownObj.data.polygons[:]], dtype="object")

    bpy.ops.outliner.orphans_purge()
    
    return out_verts, out_faces
    
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
            break

        
        verts.append( int(theLine)  )
        
        
    hFile.close()

    return verts
 
if __name__ == '__main__':
    inlayObj = bpy.data.objects['inlay.001']
    crownObj = bpy.data.objects['crown.001']
        
         

    inlayAdjust_obj = InlayAdjust(inlayObj, crownObj)
    
    
    #inlayAdjust_obj.set_oc_verts(oc_verts)
    inlayAdjust_obj.check()
    inlayAdjust_fn( inlayObj, crownObj, p_size=1.0  )    
    inlayAdjust_fn( inlayObj, crownObj, p_size=0.2  )    
    inlayAdjust_obj = InlayAdjust(inlayObj, crownObj)
    inlayAdjust_obj.onlyCheck()