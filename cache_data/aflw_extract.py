import sqlite3
import os, math
import os.path as osp
import numpy as np
import init_path
import datasets

#Change this paths according to your directories
this_dir = osp.dirname(os.path.abspath(__file__))
SAVE_DIR = osp.join(this_dir, 'lists', 'AFLW')
HOME_STR = 'DOME_HOME'
if HOME_STR not in os.environ: HOME_STR = 'HOME'
assert HOME_STR in os.environ, 'Doest not find the HOME dir : {}'.format(HOME_STR)
print ('This dir : {}, HOME : [{}] : {}'.format(this_dir, HOME_STR, os.environ[HOME_STR]))
if not osp.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)

image_dir = osp.join(os.environ[HOME_STR], 'datasets', 'aflw', 'data', 'flickr')
annot_dir = osp.join(os.environ[HOME_STR], 'datasets', 'aflw', 'data', 'annotation')
sqlit_path = osp.join(os.environ[HOME_STR], 'datasets', 'aflw', 'data', 'aflw.sqlite')

USE_BOX = 1
SUFFIXES = ['GTL', 'GTB']
SUFFIX = SUFFIXES[USE_BOX]
print ( SUFFIX )

def return_box(pts_path, boxes):
  if USE_BOX == 0:
    box_str = datasets.dataset_utils.for_generate_box_str(pts_path, 19, 0)
  elif USE_BOX == 1:
    box_str = '{:.4f} {:.4f} {:.4f} {:.4f}'.format(boxes[0], boxes[1], boxes[2], boxes[3])
  else:
    assert False, 'The box indicator not find : {}'.format(USE_BOX)
  return box_str

class AFLWFace():
  def __init__(self, query):
    assert isinstance(query, tuple), 'The type of query is not right : {}'.format(type(query))
    self.image_path = query[0]
    self.face_id = query[1]
    self.face_size = math.sqrt(float(query[4]) * float(query[5]))
    self.face_rect = [float(query[2]), float(query[3]), float(query[4]), float(query[5])]
    self.face_rect[2] = self.face_rect[0] + self.face_rect[2]
    self.face_rect[3] = self.face_rect[1] + self.face_rect[3]
    """
    self.roll = query[2]
    self.pitch = query[3]
    self.yaw = query[4]
    self.face_x = query[5]
    self.face_y = query[6]
    self.face_w = query[7]
    self.face_h = query[8]
    """
    self.landmarks = np.zeros((21, 3))

  def set_landmarks(self, landmarks):
    for landmark in landmarks:
      face_id = landmark[0]
      point_id = landmark[1]
      point_x, point_y = landmark[2], landmark[3]
      annotype = landmark[4]
      assert annotype == 0
      assert point_id >= 1 and point_id <= 21
      self.landmarks[point_id-1, 0] = point_x
      self.landmarks[point_id-1, 1] = point_y
      self.landmarks[point_id-1, 2] = 1

  def check_front(self):
    xs = self.landmarks[:, 2]
    oks = 0
    for x in xs:
      if bool(x): oks = oks + 1
    return oks == 19
    

  def from21to19(self):
    ids = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,18,19,20,21]
    assert len(ids) == 19, 'The length of ids is not right'
    ids = np.array( ids ) - 1
    self.landmarks = self.landmarks[ids,:]

  def __repr__(self):
    return ('{name}(path={image_path}, face-id={face_id})'.format(name=self.__class__.__name__, **self.__dict__))

EXPAND_RATIO = 0.0

def main():
  #Open the sqlite database
  conn = sqlite3.connect(sqlit_path)
  c = conn.cursor()

  #Creating the query string for retriving: roll, pitch, yaw and faces position
  #Change it according to what you want to retrieve
  select_string = "faceimages.filepath, faces.face_id, facerect.x, facerect.y, facerect.w, facerect.h"
  from_string = "faceimages, faces, facerect"
  where_string = "faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
  query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

  tables_string = "SELECT name FROM sqlite_master WHERE type='table'"
  table_list = list( c.execute( tables_string ))
  answers = list( c.execute(query_string) )
  all_faces = []
  for query in c.execute(query_string):
    all_faces.append( AFLWFace(query) )
  print ('Finish load all faces : {}'.format(len(all_faces)))

  for face in all_faces:
    faceid = face.face_id
    select_string = "featurecoords.face_id, featurecoords.feature_id, featurecoords.x, featurecoords.y, featurecoords.annot_type_id"
    from_string = "featurecoords"
    where_string = "featurecoords.face_id = {}".format(faceid)
    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string
    landmarks = list( c.execute(query_string) )
    face.set_landmarks(landmarks)
  c.close()

  print ('Finish load all facial landmarks : {}'.format(len(all_faces)))
  front_faces, other_faces = [], []
  for index, face in enumerate(all_faces):
    image_path = face.image_path
    image_path = osp.join(image_dir, image_path)
    face.from21to19()
    if osp.isfile(image_path):
      if face.check_front():
        front_faces.append( face )
      else:
        other_faces.append( face )
  print ('Front faces : {}, Other faces : {}'.format(len(front_faces), len(other_faces)))
  front_length = 1165
  train_length, test_length = 20000, 4396

  np.random.seed( 10 )
  random_indexes = np.random.permutation( len(front_faces) )
  # Get the testing frontal faces
  indicators = np.zeros(len(front_faces), dtype='bool')
  indicators[ random_indexes[:front_length] ] = True
  test_fron_faces, others  = [], []
  for index, face in enumerate(front_faces):
    if indicators[index]:
      test_fron_faces.append( face )
    else:
      others.append( face )
  new_test_length = len(others) + len(other_faces) - train_length
  np.random.seed( 19 )
  random_indexes = np.random.permutation( len(other_faces) )
  indicators = np.zeros(len(other_faces), dtype='bool')
  indicators[ random_indexes[:new_test_length] ] = True

  train_faces, test_other_faces = [], []
  for index, face in enumerate(other_faces):
    if indicators[index] == False:
      train_faces.append( face )
    else:
      test_other_faces.append( face )
  train_faces = train_faces + others
  print ('Train : {}, test front : {}, others : {}'.format(len(train_faces), len(test_fron_faces), len(test_other_faces)))
  np.random.seed( 10 )

  all_face_sizes = []
  
  train_save_list = open(osp.join(SAVE_DIR, 'train.'+SUFFIX), 'w')
  for face in train_faces:
    image_path = face.image_path
    sub_dir, base_name = image_path.split('/')
    cannot_dir = osp.join(annot_dir, sub_dir)
    cannot_path = osp.join(cannot_dir, base_name.split('.')[0] + '-{}.pts'.format(face.face_id))
    if not osp.isdir(cannot_dir): os.makedirs(cannot_dir)
    image_path = osp.join(image_dir, image_path)
    assert osp.isfile(image_path), 'The image [{}/{}] {} does not exsit'.format(index, len(all_faces), image_path)

    pts_str = datasets.PTSconvert2str( face.landmarks.T )
    pts_file = open(cannot_path, 'w')
    pts_file.write('{}'.format(pts_str))
    pts_file.close()

    box_str = return_box(cannot_path, face.face_rect)

    train_save_list.write('{} {} {} {}\n'.format(image_path, cannot_path, box_str, face.face_size))
    all_face_sizes.append( face.face_size )
  train_save_list.close()

  test_save_list = open(osp.join(SAVE_DIR, 'test.front.'+SUFFIX), 'w')
  for face in test_fron_faces:
    image_path = face.image_path
    sub_dir, base_name = image_path.split('/')
    cannot_dir = osp.join(annot_dir, sub_dir)
    cannot_path = osp.join(cannot_dir, base_name.split('.')[0] + '-{}.pts'.format(face.face_id))
    if not osp.isdir(cannot_dir): os.makedirs(cannot_dir)
    image_path = osp.join(image_dir, image_path)
    assert osp.isfile(image_path), 'The image [{}/{}] {} does not exsit'.format(index, len(all_faces), image_path)

    pts_str = datasets.PTSconvert2str( face.landmarks.T )
    pts_file = open(cannot_path, 'w')
    pts_file.write('{}'.format(pts_str))
    pts_file.close()

    box_str = return_box(cannot_path, face.face_rect)

    test_save_list.write('{} {} {} {}\n'.format(image_path, cannot_path, box_str, face.face_size))
    all_face_sizes.append( face.face_size )
  test_save_list.close()

  test_other_faces = test_other_faces + test_fron_faces
  test_save_list = open(osp.join(SAVE_DIR, 'test.'+SUFFIX), 'w')
  for face in test_other_faces:
    image_path = face.image_path
    sub_dir, base_name = image_path.split('/')
    cannot_dir = osp.join(annot_dir, sub_dir)
    cannot_path = osp.join(cannot_dir, base_name.split('.')[0] + '-{}.pts'.format(face.face_id))
    if not osp.isdir(cannot_dir): os.makedirs(cannot_dir)
    image_path = osp.join(image_dir, image_path)
    assert osp.isfile(image_path), 'The image [{}/{}] {} does not exsit'.format(index, len(all_faces), image_path)

    pts_str = datasets.PTSconvert2str( face.landmarks.T )
    pts_file = open(cannot_path, 'w')
    pts_file.write('{}'.format(pts_str))
    pts_file.close()

    box_str = return_box(cannot_path, face.face_rect)

    test_save_list.write('{} {} {} {}\n'.format(image_path, cannot_path, box_str, face.face_size))
    all_face_sizes.append( face.face_size )
  test_save_list.close()

  all_faces = np.array( all_face_sizes )
  print ('all faces : mean={}, std={}'.format(all_faces.mean(), all_faces.std()))

if __name__ == "__main__":
  main()
