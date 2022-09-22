'''
evaluate airway reconstruction by comparing landmark to landmark distance.
'''

from copy import copy
import numpy as np
import pandas as pd 
import vedo as v
import userUtils as utils
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
parser.add_argument('--out_name_tag', '-ot',
                    default='', 
                    type=str,#, required=True,
                    help='string to append to output graph file name'
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

def getBranchDiameter(lgraph, vgraph, landmarks, nTot=10, debug=False):
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
      if i <= 1 or i == len(chunks)-2:
        continue
      # if using a graph nodes, find the closest nth node and get its position
      ind = e_path[int(round(c))]
      inter_pts_diameter.append(vgraph.nodes[ind]['diameter'])
    lgraph0.edges[edge]['diameter'] = np.mean(inter_pts_diameter)
    if debug:
      pass
      # print("mean diameter is", np.mean(inter_pts_diameter))
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

def assignNewPositionsToTemplateGraph(template_graph, landmarks):
  graph_out = template_graph.copy()
  for node in template_graph:
    npID = template_graph.nodes[node]['npID']
    graph_out.nodes[node]['pos'] = landmarks[npID]
  return graph_out

def graphToCoords(graph):
  coords = []
  for node in graph.nodes:
    coords.append(graph.nodes[node]["pos"])
  return coords


lm_index_file = 'allLandmarks/landmarkIndexAirway.txt'
airway_lm_index = np.loadtxt(lm_index_file).astype(int)
out_dir = 'outputLandmarks/{filename}{caseID}_{key}.csv'

out_lm = np.loadtxt(out_dir.format(filename=args.filename, caseID=args.caseID, key='ALL'), 
                    delimiter=',', skiprows=1)[airway_lm_index]
# gt_lm = np.loadtxt(gt_dir.format(args.caseID), 
#                     delimiter=',', skiprows=1)[airway_lm_index]


# center landmarks
# out_lm -= out_lm.mean(axis=0)
# offset_file = 'landmarks/manual-jw/landmarks{}.csv'.format(args.caseID)
# carina = np.loadtxt(offset_file, skiprows=1, delimiter=',', usecols=[1,2,3])[1]

skel_ids = np.loadtxt(
  "allLandmarks_noFissures/landmarkIndexSkeleton.txt"
).astype(int)
diameter_ids = ~np.isin(np.arange(0, len(out_lm)), skel_ids)

num_diameter_pts = 10 # hard code 14 points representing diameter of airways


# get graph of all skeleton landmarks and graph of only branch/end point landmarks
template_graph = nx.read_gpickle('skelGraphs/nxGraphLandmarkMean.pickle')
template_bgraph = nx.read_gpickle('skelGraphs/nxGraphLandmarkMeanBranchesOnly.pickle') 
out_skel_pts = out_lm[skel_ids]
out_landmark_graph = assignNewPositionsToTemplateGraph(template_graph, out_lm)
out_branch_graph = assignNewPositionsToTemplateGraph(template_bgraph, out_lm)

cols = ['black', 'blue', 'green', 'red', 'pink', 'yellow']
# vp = v.Plotter()
# compute diameter from circumference at cross-sections
diameter_pts = out_lm[diameter_ids][:num_diameter_pts]
diameter = []
for node in out_landmark_graph.nodes:
  npID = out_landmark_graph.nodes[node]['npID']
  out_diameter_pts = out_lm[npID+1:npID+num_diameter_pts]
  circumference = 0
  for i, pt in enumerate(out_diameter_pts[1:]):
    circumference += utils.euclideanDist(pt, out_diameter_pts[i])
  diameter = circumference/np.pi
  out_landmark_graph.nodes[node]['diameter'] = copy(diameter)
  # print(diameter)

if args.out_name_tag != "":
  args.out_name_tag = "-" + args.out_name_tag


out_graph_w_diameter = getBranchDiameter(out_branch_graph, out_landmark_graph, out_lm)
# exit()
out_graph_w_lengths = getBranchLength(out_branch_graph, out_landmark_graph, out_lm)
out_bgraph = getBranchDiameter(out_branch_graph, out_landmark_graph, out_lm, debug=True)
out_bgraph = getBranchLength(out_bgraph, out_landmark_graph, out_lm)
nx.write_gpickle(out_bgraph, f"outputGraphs/nxGraph{args.caseID}{args.out_name_tag}BranchGraph.pickle")
out_lmgraph = getBranchDiameter(out_landmark_graph, out_landmark_graph, out_lm)
out_lmgraph = getBranchLength(out_lmgraph, out_landmark_graph, out_lm)
nx.write_gpickle(out_lmgraph, f"outputGraphs/nxGraph{args.caseID}{args.out_name_tag}VoxelGraph.pickle")
