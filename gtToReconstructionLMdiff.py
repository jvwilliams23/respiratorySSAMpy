'''
evaluate airway reconstruction by comparing landmark to landmark distance.
'''

import numpy as np
import pandas as pd 
import vedo as v
import userUtils as utils
import matplotlib.pyplot as plt
from glob import glob
import argparse
import networkx as nx
from sys import exit
from distutils.util import strtobool


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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', '-c',
                    default='8684', 
                    type=str,#, required=True,
                    help='training data case'
                    )
parser.add_argument('--visualise', '-v',
                    default=str(False), 
                    type=strtobool,#, required=True,
                    help='visualise error'
                    )
args = parser.parse_args()
caseID = args.case


# caseID = '3948'

lm_index_file = 'allLandmarks/landmarkIndexAirway.txt'
airway_lm_index = np.loadtxt(lm_index_file).astype(int)
out_dir = 'outputLandmarks/reconstruction_case{}_{}.csv'
gt_dir = 'allLandmarks/allLandmarks{}.csv'
gt_mesh_dir = "/home/josh/project/imaging/airwaySSAMpy/segmentations/airwaysForRadiologistWSurf/{}/*.stl" 

out_lm = np.loadtxt(out_dir.format(caseID, 'ALL'), 
                    delimiter=',', skiprows=1)[airway_lm_index]
gt_lm = np.loadtxt(gt_dir.format(caseID), 
                    delimiter=',', skiprows=1)[airway_lm_index]

# center landmarks
out_lm -= out_lm.mean(axis=0)
offset_file = 'landmarks/manual-jw/landmarks{}.csv'.format(caseID)
carina = np.loadtxt(offset_file, skiprows=1, delimiter=',', usecols=[1,2,3])[1]
gt_offset =  carina + gt_lm.mean(axis=0)
gt_lm -= gt_lm.mean(axis=0)
gt_mesh = v.load(glob(gt_mesh_dir.format(caseID))[0]).pos(-gt_offset)
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

print('TODO - get curvature per branch!')

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

print('length stats below')
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
  print('gt    output   diff   rel error')
  print(length_gt, length_out, diff, round(diff/length_gt*100, 4),'%')
  print()
  length_gt_list.append(length_gt)
  length_out_list.append(length_out)
  length_diff_list.append(diff)
  length_diff_percent_list.append(diff_pct)
  gen_list.append(out_graph_w_lengths.edges[edge]['generation'])
np.savetxt("morphologicalAnalysis/lengthStats{}.txt".format(caseID),
          np.c_[gen_list, length_gt_list, length_out_list, 
                length_diff_list, length_diff_percent_list],
          header="bifurcation level\tground truth\treconstruction\tdifference [mm]\tdifference [%]",
          fmt="%4f")

print('diameter stats')
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
  print('gt    output   diff   rel error')
  print(diameter_gt, diameter_out, diff, round(diff/diameter_gt*100, 4),'%')
  print()
  diameter_gt_list.append(diameter_gt)
  diameter_out_list.append(diameter_out)
  diameter_diff_list.append(diff)
  diameter_diff_percent_list.append(diff_pct)
  gen_list.append(gt_graph_w_diameter.edges[edge]['generation'])
np.savetxt("morphologicalAnalysis/diameterStats{}.txt".format(caseID),
          np.c_[gen_list, diameter_gt_list, diameter_out_list, 
                diameter_diff_list, diameter_diff_percent_list],
          header="bifurcation level\tground truth\treconstruction\tdifference [mm]\tdifference [%]",
          fmt="%4f")

out_branch_graph = getBranchAngle(out_branch_graph)
gt_branch_graph = getBranchAngle(gt_branch_graph)


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
np.savetxt("morphologicalAnalysis/angleStats{}.txt".format(caseID),
          np.c_[gen_list, angle_gt_list, angle_out_list, angle_diff_list, angle_diff_percent_list],
          header="bifurcation level\tground truth\treconstruction\tdifference [degrees]\tdifference [%]",
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


