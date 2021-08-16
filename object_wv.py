from gensim.models import KeyedVectors
import numpy as np

# Load vectors directly from the file
model = KeyedVectors.load_word2vec_format('/Users/Kellan/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
# Access vectors for specific words with a keyed lookup:
object_names3 = ['arrow', 'bow', 'bracer']
object_names8 = ['basketball', 'hoop', 'backboard']
object_names26 = ['pool', 'water', 'springboard']
object_names11 = ['bicycle', 'helmet', 'wheel']
object_names73 = ['raft', 'river', 'paddle']
object_names85 = ['soccer', 'goal', 'pitch']

vectors3 = np.empty((0,))

for x in object_names3:
  vec = np.array(model[x])
  vectors3 = np.append(vectors3, vec, axis=0)

vectors8 = np.empty((0,))

for x in object_names8:
  vec = np.array(model[x])
  vectors8 = np.append(vectors8, vec, axis=0)


vectors26 = np.empty((0,))

for x in object_names26:
  vec = np.array(model[x])
  vectors26 = np.append(vectors26, vec, axis=0)


vectors11 = np.empty((0,))

for x in object_names11:
  vec = np.array(model[x])
  vectors11 = np.append(vectors11, vec, axis=0)


vectors73 = np.empty((0,))

for x in object_names73:
  vec = np.array(model[x])
  vectors73 = np.append(vectors73, vec, axis=0)


vectors85 = np.empty((0,))

for x in object_names85:
  vec = np.array(model[x])
  vectors85 = np.append(vectors85, vec, axis=0)


att_obj = np.zeros((900, 101))
att_obj[:,2] = vectors3
att_obj[:,7] = vectors8
att_obj[:,25] = vectors26

att_obj[:,10] = vectors11
att_obj[:,72] = vectors73
att_obj[:,84] = vectors85

from scipy.io import savemat
savemat('att_obj.mat', {'att_obj': att_obj})