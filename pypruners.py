import trimesh
import pruners
import numpy as np

def pruner(v, f, interest_verts, offset_1st = 1.0, offset_2nd = -2.5, offset_3rd = 1.5):
	# interest_verts = np.loadtxt("index.xyz").astype(np.int32)
	# f = np.loadtxt('F.xyz').astype(np.int32)
	# v = np.loadtxt('V.xyz').astype(np.float64)

	############## new interface with more parameters #############################
	# vsource_mark : marked/source indices of verts
	# offset_1st : first offset distance, > 0 (< 0 is not recommended but available)
	# offset_2nd : second offset distance, < or == 0 as usual
	# offset_3rd : third offset distance, > or == 0 as usual
	# reserve_hem : boundary extended-hem will be reserved if true
	# missingteeth_preprocess : true, missing-teeth-proc will be reprocessed.
	# liftingedge_preprocess : true, lifting-edge-proc will be reprocessed.
	# collapse_degenerates_preprocess : true, collapse-degenerates-proc will be reprocessed.
	# missingteeth_proc__xxx : sub-proc parameters, see following annotations
	# liftingedge_proc__xxx : sub-proc parameters, see following annotations
	# offset_1st = 1.0
	# offset_2nd = -2.5
	# offset_3rd = 1.5
	reserve_hem = False
	missingteeth_preprocess = False
	liftingedge_preprocess = False
	collapse_degenerates_preprocess = True
	missingteeth_proc__componentdetect_geodesic_distance = 1.0   # regard as ~ gingival-line
	missingteeth_proc__componentfilter_min_area = 80.0           # 40% * ( 200mm^2 ~= single crown )
	missingteeth_proc__shortestpath_extend_rounds = 8            # extend f/v around the connected path (width)
	liftingedge_proc__vmark_presmoothing_geodesic_distance = 3.0 # pre-smoothing vsource_mark (for slit of teeth)
	liftingedge_proc__protect_geodesic_distance = 1.5            # protect geodesic distance
	liftingedge_proc__curvature_island_threshold = -0.5          # curvature(vi) < #t is regarded as island (should be excluded)
	liftingedge_proc__curvature_region_upperthre = -0.1          # region flooding limit (upper) of island
	liftingedge_proc__boundaries_erode_dilate_rounds = 10        # erode == dilate rounds for main-companent boundaries
	liftingedge_proc__smallboundaries_max_collapse_rounds = 50   # small inner boundaries (excluded before) recover rounds


	(success, v, f) = pruners.pruning(v, f, interest_verts, offset_1st, offset_2nd, offset_3rd,
						reserve_hem,
						missingteeth_preprocess,
						liftingedge_preprocess,
						collapse_degenerates_preprocess,
						missingteeth_proc__componentdetect_geodesic_distance, 
						missingteeth_proc__componentfilter_min_area,         
						missingteeth_proc__shortestpath_extend_rounds,          
						liftingedge_proc__vmark_presmoothing_geodesic_distance,
						liftingedge_proc__protect_geodesic_distance,          
						liftingedge_proc__curvature_island_threshold,      
						liftingedge_proc__curvature_region_upperthre,        
						liftingedge_proc__boundaries_erode_dilate_rounds,    
						liftingedge_proc__smallboundaries_max_collapse_rounds
	)
	return trimesh.Trimesh(v, f, process=False)








