##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import copy
import numpy as np
import os, torch

class Video_Meta():

  def __init__(self, method_name, vis_point=None):
    self.reset()
    self.method_name = method_name
    self.vis_point = vis_point

  def reset(self):
    self.predictions = []
    self.image_lists = []
    self.trackers = []

  def __len__(self):
    return len(self.image_lists)

  def append(self, _pred, image_path):
    assert _pred.shape[0] == 3 and len(_pred.shape) == 2, '_pred shape : {}'.format(_pred.shape)
    if (not self.predictions) == False:
      assert _pred.shape == self.predictions[-1].shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, _ground.shape)
    self.predictions.append(_pred)
    self.image_lists.append(image_path)

  def append_tracker(self, tracker):
    self.trackers.append(tracker)

  def save(self, filename):
    meta = { 'predictions': self.predictions, 
             'trackers':    self.trackers,
             'image_lists': self.image_lists,
             'method_name': self.method_name,
             'vis_point':   self.vis_point}
    torch.save(meta, filename)
    print ('save Video_Meta into {}'.format(filename))

  def get_vis_point(self):
    return copy.deepcopy(self.vis_point)

  def __getitem__(self, index):
    return copy.deepcopy(self.image_lists[index])

  def get_tracker(self, iframe, idx):
    assert iframe >= 0 and iframe < len(self.trackers), 'index of frame illegal : {}'.format(iframe)
    assert idx >=0 and idx < len(self.vis_point), 'index of the point : {}'.format(idx)
    det, track = self.trackers[iframe]
    return copy.deepcopy(det[:,idx]), copy.deepcopy(track[:,idx])

  def load(self, filename):
    assert os.path.isfile(filename), '{} is not a file'.format(filename)
    checkpoint       = torch.load(filename)
    self.predictions = checkpoint['predictions']
    self.image_lists = checkpoint['image_lists']
    self.method_name = checkpoint['method_name']
    self.vis_point   = checkpoint['vis_point']
    self.trackers    = checkpoint['trackers']
    assert len(self.trackers) == len(self.image_lists)
    assert len(self.predictions) == len(self.image_lists)

  def get_squence(self, point):
    x, y, t = [], [], []
    for i, points in enumerate(self.predictions):
      if bool(points[2, point]):
        x.append ( points[0, point] )
        y.append ( points[1, point] )
        t.append ( i )
    return np.array(x), np.array(y), np.array(t)

  def get_track_squence(self, point):
    assert point in self.vis_point, 'Point {} not in {}'.format(point, self.vis_point)
    point = self.vis_point.index( point )
    x, y, t = [], [], []
    for i, (det, track) in enumerate(self.trackers):
      if bool(track[2, point]):
        x.append ( track[0, point] )
        y.append ( track[1, point] )
        t.append ( i )
      else:
        x.append ( det[0, point] )
        y.append ( det[1, point] )
        t.append ( i )
        
    return np.array(x), np.array(y), np.array(t)
