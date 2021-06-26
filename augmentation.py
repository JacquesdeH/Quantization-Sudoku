import os
import numpy as np
import megengine as mge
import megengine.module as M
import megengine.functional as F
import megengine.data.transform as T
import pandas as pd
import copy
from tqdm import tqdm
from typing import Tuple
from megengine import tensor
from sklearn.model_selection import train_test_split
from megengine.data import DataLoader, RandomSampler
from megengine.data.dataset import Dataset
from megengine.data.transform import Transform, PseudoTransform
from megengine.tensor import Parameter
from megengine.optimizer import Adam
from megengine.autodiff import GradManager
from typing import Tuple, Sequence
import random

# CONFIG
class Config:
    _taskname = 'augmentation'
    root = './'
    data_root = os.path.join(root, 'data')
    ckpt_root = os.path.join(os.path.join(root, 'ckpt'), _taskname)
    sudoku_file = os.path.join(data_root, 'sudoku.csv')
    ckpt_file = os.path.join(ckpt_root, 'model.mge')

    lr = 1e-3
    lowshot = 0.1
    permutation_cnt = 0
    augmentations = ['RAW', 'FLIP-LR', 'FLIP-UD', 'ROT90', 'ROT180', 'ROT270']
    # augmentations = ['RAW']

    def __init__(self):
        if not os.path.exists(os.path.join(self.root, 'ckpt')):
            os.mkdir(os.path.join(self.root, 'ckpt'))
        if not os.path.exists(self.ckpt_root):
            os.mkdir(self.ckpt_root)


config = Config()


# MODEL
class Net(M.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = M.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu0 = M.ReLU()
        self.bn0 = M.BatchNorm2d(64)
        self.conv1 = M.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = M.ReLU()
        self.bn1 = M.BatchNorm2d(64)
        self.conv2 = M.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = M.ReLU()
        self.bn2 = M.BatchNorm2d(64)
        self.conv3 = M.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = M.ReLU()
        self.bn3 = M.BatchNorm2d(64)
        self.conv4 = M.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.relu4 = M.ReLU()
        self.fc = M.Linear(10368, 81*9)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = F.flatten(x, 1)
        x = self.fc(x)
        x = F.reshape(x, (-1, 81, 9))
        return x


model = Net()

def _augment(x: np.ndarray, type_: str, perm=None):
    original = x.copy()
    shape = x.shape
    x = x.reshape(1, 9, 9)

    if type_ == 'RAW':
        x = x
    elif type_ == 'FLIP-LR':
        x = np.flip(x, axis=2)
    elif type_ == 'FLIP-UD':
        x = np.flip(x, axis=1)
    elif type_ == 'ROT90':
        x = np.rot90(x, k=1, axes=(1, 2))
    elif type_ == 'ROT180':
        x = np.rot90(x, k=2, axes=(1, 2))
    elif type_ == 'ROT270':
        x = np.rot90(x, k=3, axes=(1, 2))

    # import pdb
    # pdb.set_trace()
    if perm is not None:
        for index, value in np.ndenumerate(x):
            if value != 0:
                x[index] = perm[value]
    # import pdb
    # pdb.set_trace()
    x = x.reshape(shape)
    return x

def augments(x: np.ndarray, types: Sequence[str], perm=None):
    for _type in types:
        x = _augment(x, type_=_type, perm=perm)
    return x

def gen_perm() -> dict:
    map_list = [i for i in range(1, 10)]
    random.shuffle(map_list)
    ret = {}
    for idx, num in enumerate(map_list):
        ret[idx + 1] = num
    return ret

# DATASET
def get_data(file, lowshot=config.lowshot, permutation_cnt=config.permutation_cnt):
    data = pd.read_csv(file)

    # select partially
    print(f'>>> Full dataset has {len(data)} entries.')
    low_shot_len = int(np.round(len(data) * lowshot))
    data = data[:low_shot_len]
    print(f'>>> Selected {len(data)} entries with ratio loashot={lowshot} from raw dataset.')

    feat_raw = data['quizzes']
    label_raw = data['solutions']
    feat = []
    label = []

    use_augmentations = config.augmentations
    print(f'>>> Using augmentation list @ {use_augmentations}')
    print(f'>>> Using new permutation count @ {permutation_cnt}')

    for base_augmentation in use_augmentations:
        for i in feat_raw:
            x = np.array([int(j) for j in i]).reshape((1, 9, 9))
            feat.append(x)
        for i in label_raw:
            x = np.array([int(j) for j in i]).reshape((1, 81))
            label.append(x)
        # permuted new data
        for _ in range(permutation_cnt):
            _permutation = gen_perm()
            for i in feat_raw:
                x = np.array([int(j) for j in i]).reshape((1, 9, 9))
                x = augments(x, types=[base_augmentation], perm=_permutation)
                feat.append(x)
            
            for i in label_raw:
                x = np.array([int(j) for j in i]).reshape((1, 81))
                x = augments(x, types=[base_augmentation], perm=_permutation)
                label.append(x)

    del(feat_raw)
    del(label_raw)

    feat = np.array(feat)
    feat = feat/9
    feat -= .5

    label = np.array(label) - 1

    print(f'>>> Done augmentation with total data entries of {len(label)} as augmented dataset.')

    x_train, x_test, y_train, y_test = \
        train_test_split(feat, label, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = \
    get_data(config.sudoku_file)


class TrainDataset(Dataset):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.data_x = x_train
        self.data_y = y_train

    def __getitem__(self, index: int) -> Tuple:
        return self.data_x[index], self.data_y[index]

    def __len__(self) -> int:
        return len(self.data_x)


train_dataset = TrainDataset(x_train, y_train)
train_sampler = RandomSampler(dataset=train_dataset, batch_size=32)
train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler)

opt = Adam(model.parameters(), lr=config.lr)


# TRAIN
epochs = 2
for epoch in range(epochs):
    iter = 0
    with tqdm(total=train_dataloader.__len__() / 100,
              desc="epoch {}:training process".format(epoch)) as tq:
        for i, (feat, label) in enumerate(train_dataloader):

            gm = GradManager().attach(model.parameters())
            with gm:
                logits = model(tensor(feat))
                label = label.reshape(32, 81)
                loss = F.loss.cross_entropy(logits, label, axis=2)

                iter += 1
                if iter % 100 == 0:
                    tq.set_postfix({
                        "loss": "{0:1.5f}".format(loss.numpy().item()),
                    })
                    tq.update(1)

                gm.backward(loss)
            opt.step()
            opt.clear_grad()

mge.save(model, config.ckpt_file)
print("saved")


# EVALUATION
model = Net()
model = mge.load(config.ckpt_file)
model.eval()

eval_dataset = TrainDataset(x_test, y_test)
eval_sampler = RandomSampler(dataset=eval_dataset, batch_size=32)
eval_dataloader = DataLoader(dataset=eval_dataset, sampler=eval_sampler)

average_loss = 0
iter = 0
with tqdm(total=eval_dataloader.__len__() / 100,
          desc="evaluating process") as tq:
    for i, (feat, label) in enumerate(eval_dataloader):

        logits = model(tensor(feat))
        label = label.reshape(32, 81)
        loss = F.loss.cross_entropy(logits, label, axis=2)

        iter += 1
        if iter % 100 == 0:
            tq.set_postfix({
                "loss": "{0:1.5f}".format(loss.numpy().item()),
            })
            tq.update(1)
        average_loss += loss.numpy().item()

print(average_loss / eval_dataloader.__len__())


# MAIN
model = Net()
model = mge.load(config.ckpt_file)
model.eval()


def norm(x):
    return (x/9)-.5


def denorm(x):
    return (x+.5)*9


def inference_sudoku(sample):
    feat = copy.copy(sample)

    while(1):
        out = F.softmax(model(tensor(feat.reshape((1, 1, 9, 9)))), axis=2)
        out = out.reshape(81, 9)

        pred = F.argmax(out, axis=1).reshape((9, 9))+1
        prob = np.around(F.max(out, axis=1).reshape((9, 9)), 2)
        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)
        if mask.sum() == 0:
            break

        prob_new = prob * mask
        ind = F.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y].numpy()
        feat[x][y] = int(val)
        feat = norm(feat)

    return feat


def solve_sudoku(game):
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9, 9, 1))
    game = norm(game)
    game = inference_sudoku(game)
    return game


game = '''
          0 8 0 0 3 2 0 0 1
          7 0 3 0 8 0 0 0 2
          5 0 0 0 0 7 0 3 0
          0 5 0 0 0 1 9 7 0
          6 0 0 7 0 9 0 0 8
          0 4 7 2 0 0 0 5 0
          0 2 0 6 0 0 0 0 9
          8 0 0 0 9 0 3 0 5
          3 0 0 8 2 0 0 1 0
      '''

game = solve_sudoku(game)

print('Solved puzzle:\n')
print(game.astype("uint8"))
