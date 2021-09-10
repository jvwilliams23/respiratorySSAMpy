'''
evaluate airway reconstruction by comparing landmark to landmark distance.
'''

import numpy as np
import pandas as pd 
import vedo as v
import userUtils as utils
from userUtils import GradientCurvature
import matplotlib.pyplot as plt
from glob import glob
import argparse
import networkx as nx
from sys import exit
from scipy import interpolate
from distutils.util import strtobool

from pylab import cross,dot,inv


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--caseID', '-c',
                    default='8684', 
                    type=str,#, required=True,
                    help='training data case'
                    )
parser.add_argument('--filename', '-f',
                    default='reconstruction_case', 
                    type=str,#, required=True,
                    help='string for filename [default is reconstruction_case]'
                    )
parser.add_argument('--visualise', '-v',
                    default=str(False), 
                    type=strtobool,#, required=True,
                    help='visualise error'
                    )
parser.add_argument('--write', '-w',
                    default=str(True), 
                    type=strtobool,#, required=True,
                    help='write errors to text files'
                    )
args = parser.parse_args()

def getBranchLength(lgraph, vgraph, landmarks, nTot=10):
  '''
  Get midpoints of a branch by fitting a cubic spline to the coordinates
  along the branch. 
  The spline is then sub-divided to get desired number of midpoints.
  params:
  lgraph (nx.DiGraph): Skeleton filtered only to landmarked nodes 
                        (first few bifurcations).
  vgraph (nx.DiGraph): Skeleton with coordinates from all landmarks along branch.
  landmarks (np.ndarray, N,3): corresponding landmarks for each sample
  nTot (int, optional): number of midpoints along branch to extract
  returns:
  landmarks: array with new interpolated points inserted at distal node index+1
  lgraph: new landmark graph which includes midpoints. 
  '''
  lgraph0 = lgraph.copy()
  root = list(nx.topological_sort(lgraph0))[0]
  for edge in lgraph0.edges:
    # if edge1 is above edge 0 (in hierarchy), then edge1 is proximal one 
    if edge[1] in nx.ancestors(vgraph, edge[0]):
      proximalNode, distalNode = edge[::-1]
    else:
      proximalNode, distalNode = edge
    # get nodes in voxel graph along path between landmarks
    e_path = list(nx.all_simple_paths(vgraph, proximalNode, distalNode) )[0]
    chunks = np.linspace(0, len(e_path)-1, nTot) # chunk size to split tree segment
    inter_pts = []
    dist = 0
    for i, c in enumerate(chunks):
      # if using a graph nodes, find the closest nth node and get its position
      ind = e_path[int(round(c))]
      inter_pts.append(vgraph.nodes[ind]['pos'])
      if i != 0:
        dist += utils.euclideanDist(inter_pts[i], inter_pts[i-1])
    lgraph0.edges[edge]['length'] = dist
    lgraph0.edges[edge]['generation'] = nx.shortest_path_length(lgraph0, 
                                                                root,
                                                                proximalNode)
  return lgraph0

def getBranchDiameter(lgraph, vgraph, landmarks, nTot=10):
  '''
  Get midpoints of a branch by fitting a cubic spline to the coordinates
  along the branch. 
  The spline is then sub-divided to get desired number of midpoints.
  params:
  lgraph (nx.DiGraph): Skeleton filtered only to landmarked nodes 
                        (first few bifurcations).
  vgraph (nx.DiGraph): Skeleton with coordinates from all landmarks along branch.
  landmarks (np.ndarray, N,3): corresponding landmarks for each sample
  nTot (int, optional): number of midpoints along branch to extract
  returns:
  landmarks: array with new interpolated points inserted at distal node index+1
  lgraph: new landmark graph which includes midpoints. 
  '''
  lgraph0 = lgraph.copy()
  root = list(nx.topological_sort(lgraph0))[0]
  for edge in lgraph0.edges:
    # if edge1 is above edge 0 (in hierarchy), then edge1 is proximal one 
    if edge[1] in nx.ancestors(vgraph, edge[0]):
      proximalNode, distalNode = edge[::-1]
    else:
      proximalNode, distalNode = edge
    # get nodes in voxel graph along path between landmarks
    e_path = list(nx.all_simple_paths(vgraph, proximalNode, distalNode) )[0]
    chunks = np.linspace(0, len(e_path)-1, nTot) # chunk size to split tree segment
    inter_pts_diameter = []
    dist = 0
    for i, c in enumerate(chunks):
      # if using a graph nodes, find the closest nth node and get its position
      ind = e_path[int(round(c))]
      inter_pts_diameter.append(vgraph.nodes[ind]['diameter'])
    lgraph0.edges[edge]['diameter'] = np.mean(inter_pts_diameter)
    lgraph0.edges[edge]['generation'] = nx.shortest_path_length(lgraph0, 
                                                                root,
                                                                proximalNode)
  return lgraph0

def getBranchAngle(graph):
  root = list(nx.topological_sort(graph))[0]
  for node in graph:
    children = list(graph.successors(node))
    if len(children) >= 2:
      pos_parent = graph.nodes[node]['pos']
      pos_child0 = graph.nodes[children[0]]['pos']
      pos_child1 = graph.nodes[children[1]]['pos']
      vec0 = pos_parent - pos_child0
      vec1 = pos_parent - pos_child1
      branch_angle = np.degrees(
                                np.arccos(np.dot(vec0, vec1)
                                          /(np.linalg.norm(vec0)*np.linalg.norm(vec1))
                                          )
                                )
      graph.nodes[node]['branch_angle'] = branch_angle
      graph.nodes[node]['generation'] = nx.shortest_path_length(graph, 
                                                                  root,
                                                                  node)
  return graph

def graph_to_spline(graph, nodes, npoints=20):
  """
  Convert graph nodes to coordinates along a spline

  Parameters
  ----------
  graph (nx.Graph): graph of landmarks
  nodes (list): nodes along the current branch segment to fit spline to
  npoints (int): number of points to sample spline with

  Returns
  -------
  spline_path (np.ndarray, npoints, 3): spline fitted to landmarks
  """
  pos_list = []
  for node in nodes:
    pos_list.append(graph.nodes[node]['pos'])
  pos_list = np.array(pos_list)
  indexes = np.unique(pos_list, return_index=True, axis=0)[1]
  pos_list = np.array([pos_list[index] for index in sorted(indexes)])
  x,y,z = pos_list[:,0], pos_list[:,1], pos_list[:,2]

  spl_xticks, _ = interpolate.splprep([x,y,z], 
                                          k=3, s=10.0)
  # generate the new interpolated dataset. sample spline to 100 points
  spline_path = interpolate.splev(np.linspace(0,1,npoints), spl_xticks, der=0)
  spline_path = np.vstack(spline_path).T
  return spline_path

def spline_curvature(points):
  """
  Get curvature at each point along a curve

  Parameters
  ----------
  points (np.ndarray, N,2): x-y points in space of a 2D curve

  Returns
  -------
  curv_abs (np.array, N): curvature in 1/mm at each point on curve. 
    take absolute value to ignore sign
  """
  points = list(zip(points[:,0], points[:,1]))
  curv = GradientCurvature(trace=points)
  curv.calculate_curvature()
  curv_abs = abs(curv.curvature)
  return curv_abs

def rotation_matrix_between_two_vectors(U,V):
  """ Find the rotation matrix that aligns vec1 to vec2
  :param vec1: A 3d "source" vector
  :param vec2: A 3d "destination" vector
  :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
  """
  W=cross(U,V)
  A=np.array([U,W,cross(U,W)]).T
  B=np.array([V,W,cross(V,W)]).T
  if np.allclose(U,V):
    return 'aligned'
  else:
    return dot(B,inv(A))

def lateral_dist_from_axis(curve):
  """
  Find max lateral distance from a curve by aligning to origin, rotating
  so the principal direction is aligned with an axis. Then can get max dist
  from central axis.

  Parameters
  ----------
  curve (np.ndarray, N, 3): coordinates along a curve

  Returns
  -------
  maximum lateral distance (float) from the central axis of the curve
  """
  # centre curve at origin
  curve -= curve[-1]
  # get normal vector
  curve_vec = curve[-1] - curve[0]
  curve_vec /= np.sum(curve_vec**2)**0.5
  # find which axis is the principal one (main direction of transport)
  main_axis = np.argmax(abs(curve_vec))
  align_vec = np.zeros(3)
  align_vec[main_axis] += 1.
  # rotate curve to be aligned with main/principal axis
  rot = rotation_matrix_between_two_vectors(curve_vec, align_vec)
  curve = rot.dot(curve.T).T
  # delete main axis so distance only includes lateral components
  new_curve = np.delete(curve, main_axis, axis=1)
  lateral_dist = utils.euclideanDist(new_curve, np.zeros(2))
  return lateral_dist.max() 

lm_index_file = 'allLandmarks/landmarkIndexAirway.txt'
airway_lm_index = np.loadtxt(lm_index_file).astype(int)
out_dir = 'outputLandmarks/{filename}{caseID}_{key}.csv'
# out_dir = 'outputLandmarks/reconstruction_case{}_{}.csv'
gt_dir = 'allLandmarks/allLandmarks{}.csv'
gt_mesh_dir = "/home/josh/project/imaging/airwaySSAMpy/segmentations/airwaysForRadiologistWSurf/{}/*.stl" 

out_lm = np.loadtxt(out_dir.format(filename=args.filename, caseID=args.caseID, key='ALL'), 
                    delimiter=',', skiprows=1)[airway_lm_index]
gt_lm = np.loadtxt(gt_dir.format(args.caseID), 
                    delimiter=',', skiprows=1)[airway_lm_index]

# center landmarks
out_lm -= out_lm.mean(axis=0)
offset_file = 'landmarks/manual-jw/landmarks{}.csv'.format(args.caseID)
carina = np.loadtxt(offset_file, skiprows=1, delimiter=',', usecols=[1,2,3])[1]
gt_offset =  carina + gt_lm.mean(axis=0)
gt_lm -= gt_lm.mean(axis=0)
gt_mesh = v.load(glob(gt_mesh_dir.format(args.caseID))[0]).pos(-gt_offset)
# gt_mesh.alignTo(v.Points(gt_lm), rigid=True)

dist = utils.euclideanDist(out_lm, gt_lm)
print('max dist is', dist.max())
print('distance statistics are')
print(pd.DataFrame(dist).describe())
# gt_pts = v.Points(gt_lm, r=4).c('black')
out_pts = v.Points(out_lm, r=4).cmap('hot', dist)
outliers = v.Points(out_lm[dist>10], r=10).c('black')
if args.visualise:
  v.show(gt_mesh.alpha(0.2), out_pts, outliers)
# plt.hist(dist, bins=100)
# plt.xlabel('landmark-to-landmark distance [mm]')
# plt.show()

def assignNewPositionsToTemplateGraph(template_graph, landmarks):
  graph_out = template_graph.copy()
  for node in template_graph:
    npID = template_graph.nodes[node]['npID']
    graph_out.nodes[node]['pos'] = landmarks[npID]
  return graph_out

skel_ids = np.loadtxt('allLandmarks/landmarkIndexSkeleton.txt').astype(int)
diameter_ids = ~np.isin(np.arange(0, len(gt_lm)), skel_ids)

num_diameter_pts = 14 # hard code 14 points representing diameter of airways


# get graph of all skeleton landmarks and graph of only branch/end point landmarks
template_graph = nx.read_gpickle('skelGraphs/nxGraphLandmarkMean.pickle')
template_bgraph = nx.read_gpickle('skelGraphs/nxGraphLandmarkMeanBranchesOnly.pickle') 
gt_skel_pts = gt_lm[skel_ids]
out_skel_pts = out_lm[skel_ids]
out_landmark_graph = assignNewPositionsToTemplateGraph(template_graph, out_lm)
gt_landmark_graph = assignNewPositionsToTemplateGraph(template_graph, gt_lm)
out_branch_graph = assignNewPositionsToTemplateGraph(template_bgraph, out_lm)
gt_branch_graph = assignNewPositionsToTemplateGraph(template_bgraph, gt_lm)

cols = ['black', 'blue', 'green', 'red', 'pink', 'yellow']
vp = v.Plotter()
# compute diameter from circumference at cross-sections
diameter_pts = gt_lm[diameter_ids][:num_diameter_pts]
diameter = []
for node in gt_landmark_graph.nodes:
  npID = gt_landmark_graph.nodes[node]['npID']
  gt_diameter_pts = gt_lm[npID+1:npID+1+num_diameter_pts]
  circumference = 0
  for i, pt in enumerate(gt_diameter_pts[1:]):
    # print(utils.euclideanDist(pt, diameter_pts[i]))
    circumference += utils.euclideanDist(pt, gt_diameter_pts[i])
  diameter = circumference/np.pi
  gt_landmark_graph.nodes[node]['diameter'] = diameter

  vp+=v.Points(gt_diameter_pts, r=3).c(cols[np.random.randint(len(cols))])

  out_diameter_pts = out_lm[npID+1:npID+1+num_diameter_pts]
  circumference = 0
  for i, pt in enumerate(out_diameter_pts[1:]):
    circumference += utils.euclideanDist(pt, out_diameter_pts[i])
  diameter = circumference/np.pi
  out_landmark_graph.nodes[node]['diameter'] = diameter
# vp.show()
# exit()

out_graph_w_diameter = getBranchDiameter(template_bgraph, out_landmark_graph, out_lm)
gt_graph_w_diameter = getBranchDiameter(template_bgraph, gt_landmark_graph, gt_lm)

# v.show(v.Points(gt_lm[diameter_ids][:14]))
# exit()


out_graph_w_lengths = getBranchLength(template_bgraph, out_landmark_graph, out_lm)
gt_graph_w_lengths = getBranchLength(template_bgraph, gt_landmark_graph, gt_lm)

print()
print('length stats below')
print('gt    output   diff   rel error')
length_gt_list = []
length_out_list = []
length_diff_list = []
length_diff_percent_list = []
gen_list = []
for edge in out_graph_w_lengths.edges:
  length_out = round(out_graph_w_lengths.edges[edge]['length'], 4)
  length_gt = round(gt_graph_w_lengths.edges[edge]['length'], 4)
  diff = round(length_out-length_gt, 4)
  diff_pct = round(diff/length_gt*100, 4)
  print(length_gt, length_out, diff, round(diff/length_gt*100, 4),'%')
  # print()
  length_gt_list.append(length_gt)
  length_out_list.append(length_out)
  length_diff_list.append(diff)
  length_diff_percent_list.append(diff_pct)
  gen_list.append(out_graph_w_lengths.edges[edge]['generation'])
if args.write:
  np.savetxt("morphologicalAnalysis/lengthStats{}.txt".format(args.caseID),
            np.c_[gen_list, length_gt_list, length_out_list, 
                  length_diff_list, length_diff_percent_list],
            header="bifurcation level\tground truth\treconstruction\tdifference [mm]\tdifference [%]",
            fmt="%4f")

print()
print('diameter stats below')
print('gt    output   diff   rel error')
diameter_gt_list = []
diameter_out_list = []
diameter_diff_list = []
diameter_diff_percent_list = []
gen_list = []
for edge in out_graph_w_lengths.edges:
  diameter_out = round(out_graph_w_diameter.edges[edge]['diameter'], 4)
  diameter_gt = round(gt_graph_w_diameter.edges[edge]['diameter'], 4)
  diff = round(diameter_out-diameter_gt, 4)
  diff_pct = round(diff/diameter_gt*100, 4)
  print(diameter_gt, diameter_out, diff, round(diff/diameter_gt*100, 4),'%')
  # print()
  diameter_gt_list.append(diameter_gt)
  diameter_out_list.append(diameter_out)
  diameter_diff_list.append(diff)
  diameter_diff_percent_list.append(diff_pct)
  gen_list.append(gt_graph_w_diameter.edges[edge]['generation'])
if args.write:
  np.savetxt("morphologicalAnalysis/diameterStats{}.txt".format(args.caseID),
            np.c_[gen_list, diameter_gt_list, diameter_out_list, 
                  diameter_diff_list, diameter_diff_percent_list],
            header="bifurcation level\tground truth\treconstruction\tdifference [mm]\tdifference [%]",
            fmt="%4f")

out_branch_graph = getBranchAngle(out_branch_graph)
gt_branch_graph = getBranchAngle(gt_branch_graph)

print()
print('branch angle stats below')
# once have this data for few patients can make boxplot of GT and XR, 
# compare statistical significance
angle_gt_list = []
angle_out_list = []
angle_diff_list = []
angle_diff_percent_list = []
gen_list = []
for node in out_branch_graph.nodes:
  if 'branch_angle' in out_branch_graph.nodes[node].keys():
    angle_out = round(out_branch_graph.nodes[node]['branch_angle']/2, 4)
    angle_gt = round(gt_branch_graph.nodes[node]['branch_angle']/2, 4)
    diff =  angle_out - angle_gt
    diff_pct = round(diff/angle_gt*100, 4)

    print(angle_gt, angle_out, diff, round(diff/angle_gt*100, 4),'%')
    angle_gt_list.append(angle_gt)
    angle_out_list.append(angle_out)
    angle_diff_list.append(diff)
    angle_diff_percent_list.append(diff_pct)
    gen_list.append(out_branch_graph.nodes[node]['generation'])
if args.write:
  np.savetxt("morphologicalAnalysis/angleStats{}.txt".format(args.caseID),
            np.c_[gen_list, angle_gt_list, angle_out_list, angle_diff_list, angle_diff_percent_list],
            header="bifurcation level\tground truth\treconstruction\tdifference [degrees]\tdifference [%]",
            fmt="%4f")

# print()
# print('curvature stats below')
curv_gt_list = []
curv_out_list = []
curv_diff_list = []
curv_diff_percent_list = []
gen_list = []
for edge in out_branch_graph.edges:
  # get all landmarks along branch segment
  nodes = list(nx.all_simple_paths(out_landmark_graph, edge[0], edge[1]))[0]
  # fit splines to landmarks to smooth
  out_spline = graph_to_spline(out_landmark_graph, nodes)
  gt_spline = graph_to_spline(gt_landmark_graph, nodes)
  # get mean curvature per segment
  curv_out = round(spline_curvature(out_spline[:,[0,2]]).mean(), 4)
  curv_gt = round(spline_curvature(gt_spline[:,[0,2]]).mean(), 4)
  diff = round(curv_out-curv_gt, 4)
  # save curvature data
  curv_gt_list.append(curv_gt)
  curv_out_list.append(curv_out)
  curv_diff_list.append(diff)
  curv_diff_percent_list.append(diff_pct)
  gen_list.append(out_graph_w_lengths.edges[edge]['generation'])

if args.write:
  np.savetxt("morphologicalAnalysis/curvatureStats{}.txt".format(args.caseID),
            np.c_[gen_list, curv_gt_list, curv_out_list, curv_diff_list, curv_diff_percent_list],
            header="bifurcation level\tground truth\treconstruction\tdifference [1/mm]\tdifference [%]",
            fmt="%4f")

print()
print('lateral distance stats below')
# initialise lists to save output data to
lateral_dist_gt_list = []
lateral_dist_out_list = []
lateral_dist_diff_list = []
lateral_dist_diff_percent_list = []
gen_list = []
for edge in out_branch_graph.edges:
  # get all landmarks along branch segment
  nodes = list(nx.all_simple_paths(out_landmark_graph, edge[0], edge[1]))[0]
  # fit splines to landmarks to smooth
  out_spline = graph_to_spline(out_landmark_graph, nodes)
  gt_spline = graph_to_spline(gt_landmark_graph, nodes)
  # find max lateral dist from central axis
  lateral_dist_out = round(lateral_dist_from_axis(out_spline), 4)
  lateral_dist_gt = round(lateral_dist_from_axis(gt_spline), 4)
  # get difference in mm and percent for terminal
  diff =  round(lateral_dist_out - lateral_dist_gt, 4)
  diff_pct = round(diff/lateral_dist_gt*100, 4)

  print(lateral_dist_gt, lateral_dist_out, diff, round(diff/lateral_dist_gt*100, 4),'%')
  # save results to list for writing
  lateral_dist_gt_list.append(lateral_dist_gt)
  lateral_dist_out_list.append(lateral_dist_out)
  lateral_dist_diff_list.append(diff)
  lateral_dist_diff_percent_list.append(diff_pct)
  gen_list.append(out_graph_w_lengths.edges[edge]['generation'])

if args.write:
  np.savetxt("morphologicalAnalysis/lateralDistanceStats{}.txt".format(args.caseID),
            np.c_[gen_list, lateral_dist_gt_list, lateral_dist_out_list, lateral_dist_diff_list, lateral_dist_diff_percent_list],
            header="bifurcation level\tground truth\treconstruction\tdifference [mm]\tdifference [%]",
            fmt="%4f")


# vp = v.Plotter()
# for edge in out_branch_graph.edges:
#   p1 = out_branch_graph.nodes[edge[0]]['pos']
#   p2 = out_branch_graph.nodes[edge[1]]['pos']
#   vp += v.Line(p1,p2, c='black')
# for edge in out_landmark_graph.edges:
#   p1 = out_landmark_graph.nodes[edge[0]]['pos']
#   p2 = out_landmark_graph.nodes[edge[1]]['pos']
#   vp += v.Line(p1,p2, c='black', lw=2)
  
# for edge in gt_branch_graph.edges:
#   p1 = gt_branch_graph.nodes[edge[0]]['pos']
#   p2 = gt_branch_graph.nodes[edge[1]]['pos']
#   vp += v.Line(p1,p2, c='blue')
# for edge in gt_landmark_graph.edges:
#   p1 = gt_landmark_graph.nodes[edge[0]]['pos']
#   p2 = gt_landmark_graph.nodes[edge[1]]['pos']
#   vp += v.Line(p1,p2, c='blue', lw=2)
# vp.show()


