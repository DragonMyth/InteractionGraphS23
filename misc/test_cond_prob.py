import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch_models
import torch.nn.functional as F

import ray
from ray import tune

import os
import pickle
import gzip

def gen_random_sequential_data_2D(num_points, x_start, x_min, x_max, scale_factor=0.1):
    assert 2 == x_start.shape[0]
    assert 2 == x_min.shape[0]
    assert 2 == x_max.shape[0]
    scale = scale_factor * (x_max-x_min)
    x1, x2 = [], []
    x = x_start
    for i in range(num_points):
        x1.append(x[0])
        x2.append(x[1])
        x = np.random.normal(loc=x, scale=scale)
        x = np.clip(x, x_min, x_max)
    return x1, x2

class LineDrawer:
    def __init__(self, ax, edit, max_length, **kwargs):
        self.ax = ax
        line, = self.ax.plot([], [], **kwargs)
        self.line = line
        self.start = None
        self.x_cur = 0
        self.y_cur = 0
        self.xs = []
        self.ys = []
        self.edit = edit
        self.max_length = max_length
        self.cid1 = line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid2 = line.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
    def length(self):
        return len(self.xs)
    def draw(self):
        self.ax.figure.canvas.draw()
    def on_press(self, event):
        if not self.edit: return
        if event.inaxes!=self.line.axes: return
        self.xs.append(self.x_cur)
        self.ys.append(self.y_cur)
        if self.max_length is not None and self.length() > self.max_length:
            self.xs.pop(0)
            self.ys.pop(0)
        if self.start is not None:
            self.start.remove()
        self.start = self.ax.scatter(self.xs[0], self.ys[0], color='k', s=100)
        self.line.set_data(self.xs+[self.x_cur], self.ys+[self.y_cur])
        self.draw()
    def on_move(self, event):
        if not self.edit: return
        if event.inaxes!=self.line.axes: return
        self.x_cur = event.xdata
        self.y_cur = event.ydata
        self.line.set_data(self.xs+[self.x_cur], self.ys+[self.y_cur])
        self.draw()
    def clear(self, draw=False):
        if self.start is not None:
            self.start.remove()
            self.start = None
        self.xs = []
        self.ys = []
        self.line.set_data([], [])
        if draw:
            self.draw()
    def set_data(self, xs, ys):
        self.clear(False)
        self.xs = xs
        self.ys = ys
        self.start = self.ax.scatter(self.xs[0], self.ys[0], color='k', s=100)
        self.line.set_data(self.xs, self.ys)

class TestSeqData2DBase:
    def __init__(
        self, 
        id=0, 
        title='PlotSeqData2D', 
        xlabel='x', 
        ylabel='y',
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        ):
        self.fig = plt.figure(id, figsize=(12, 10), dpi=100)
        self.fig.canvas.set_window_title(title)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.patch.set_facecolor('white')
        self.linedrawer_save = []
        self.linedrawer_cur = \
            LineDrawer(self.ax, edit=True, max_length=None, marker='o', color='k')
        self.linedrawer_eval = \
            LineDrawer(self.ax, edit=False, max_length=self.get_window_size(), marker='+', color='g')
        self.id_key_release = \
            self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.id_mouse_release = \
            self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.id_mouse_press = \
            self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.mode = 'data'
        self.trainer = None
        self.dataset_train = None
    def get_window_size(self):
        raise NotImplementedError
    def get_model(self):
        raise NotImplementedError
    def setup_eval(self):
        raise NotImplementedError
    def reset_eval(self):
        raise NotImplementedError
    def get_trainer_config(self):
        raise NotImplementedError
    def load_dataset(self):
        raise NotImplementedError
    def draw(self):
        self.fig.canvas.draw()
    def change_mode(self, mode=None):
        if mode is None:
            self.mode = 'eval' if self.mode == 'data' else 'data'
        else:
            assert mode in ['eval', 'data']
            self.mode = mode
        if self.mode == 'data':
            self.linedrawer_cur.edit = True
            self.linedrawer_eval.clear()
            self.linedrawer_eval.edit = False
        else:
            self.linedrawer_cur.clear()
            self.linedrawer_cur.edit = False
            self.linedrawer_eval.edit = True
    def finalize_line_cur(self):
        if self.linedrawer_cur.length() < self.get_window_size()+1: 
            print('The current line is too short')
            return
        self.linedrawer_cur.edit = False
        self.linedrawer_save.append(self.linedrawer_cur)
        self.linedrawer_cur = \
            LineDrawer(self.ax, edit=True, max_length=None, marker='o', color='k')
    def on_mouse_press(self, event):
        if self.mode == 'data':
            if event.dblclick:
                self.finalize_line_cur()
    def on_mouse_release(self, event):
        if self.mode == 'eval' and self.trainer is not None:
            if self.linedrawer_eval.length() < self.get_window_size():
                return
            self.setup_eval()
            self.draw()
    def reset(self):
        for line in self.linedrawer_save:
            line.clear()
        del self.linedrawer_save
        self.linedrawer_save = []
        self.linedrawer_cur.clear()
        self.linedrawer_eval.clear()
        self.reset_eval()
        self.change_mode('data')
    def on_key_release(self, event):
        key = event.key
        if key=='escape':
            exit(0)
        elif key=='r':
            if self.mode == 'data':
                self.linedrawer_cur.clear()
            else:
                self.reset_eval()
                self.linedrawer_eval.clear()
        elif key=='R':
            self.reset()
        elif key=='l':
            self.reset()
            file_name = input("Enter zipped data file: ")
            with open(file_name, "rb") as file:
                data = pickle.load(file)
                for episode_data in data:
                    z_task_list = np.array(episode_data['z_task'])
                    z1_task = z_task_list[:,0]
                    z2_task = z_task_list[:,1]
                    line = LineDrawer(
                        self.ax, edit=True, max_length=None, marker='o', color='k')
                    line.set_data(z1_task, z2_task)
                    line.edit = False
                    self.linedrawer_save.append(line)
            self.draw()
        elif key=='a':
            if self.mode != 'data':
                print('The current mode is not *data*')
                return
            xs, ys = gen_random_sequential_data_2D(
                num_points=10, 
                x_start=np.random.uniform(-1,1,size=2), 
                x_min=-np.ones(2), 
                x_max=np.ones(2), 
                scale_factor=0.1)
            self.linedrawer_cur.set_data(xs, ys)
            self.finalize_line_cur()
        elif key=='S':
            if len(self.linedrawer_save) > 0:
                data = []
                for l in self.linedrawer_save:
                    data.append(
                        [np.array([x, y]) for x, y in zip(l.xs, l.ys)]
                    )
                filename = 'data/temp/contrastive_test.data'
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
                print('Saved:', filename)
            if self.dataset_train is not None:
                filename = 'data/temp/contrastive_test.dataset'
                with open(filename, 'wb') as f:
                    pickle.dump(self.dataset_train, f)
                print('Saved:', filename)
            if self.trainer is not None:
                filename = 'data/temp/contrastive_test.model'
                torch.save(self.trainer.model, 'data/temp/contrastive_test.model')
                print('Saved:', filename)
        elif key=='t':
            if self.mode != 'data':
                print('The current mode is not *data*')
                return
            if len(self.linedrawer_save) < 1:
                print('No data points')
                return
            for i, line in enumerate(self.linedrawer_save):
                if len(line.xs) < self.get_window_size()+1:
                    print('%d-th line is too short'%(i))
                    return

            ''' Train a model given dataset '''
            trainer_config = self.get_trainer_config()

            ray.init(num_cpus=1)

            analysis = tune.run(
                torch_models.TrainModel,
                # scheduler=sched,
                stop={
                    # "mean_accuracy": 0.95,
                    "training_iteration": 200,
                },
                resources_per_trial={
                    "cpu": 1,
                    "gpu": 0,
                },
                num_samples=1,
                checkpoint_at_end=True,
                checkpoint_freq=50,
                config=trainer_config)

            logdir = analysis.get_best_logdir(metric='training_iteration', mode='max')
            checkpoint = analysis.get_trial_checkpoints_paths(logdir, metric='training_iteration')

            trainer = torch_models.TrainModel(trainer_config)
            trainer.restore(checkpoint[0][0])
            self.trainer = trainer

            torch.save(
                self.trainer.model.state_dict(), 
                'data/temp/task_embedding/model.pt'
            )

            ray.shutdown()

            self.change_mode('eval')          
        elif key=='m':
            self.change_mode()
            print("Current Mode:", self.mode)

        self.draw()

class TestSeqData2DPrediction(TestSeqData2DBase):
    def __init__(self):
        super(TestSeqData2DPrediction, self).__init__()
        self.target = None
    def get_window_size(self):
        return 3
    def get_trainer_config(self):
        dataset_train = self.load_dataset(
            [lb.xs for lb in self.linedrawer_save], 
            [lb.ys for lb in self.linedrawer_save],
            self.get_window_size())
        dataset_test = self.load_dataset(
            [lb.xs for lb in self.linedrawer_save], 
            [lb.ys for lb in self.linedrawer_save],
            self.get_window_size())
        model = torch_models.FCNN(
            size_in=dataset_train.X.shape[1],
            size_out=dataset_train.Y.shape[1],
            hiddens=[128, 128],
            activations=["relu", "relu", "linear"],
            init_weights=[1.0, 1.0, 1.0],
            init_bias=[0.0, 0.0, 0.0])
        trainer_config = {
            "model": model,
            "lr": 0.001,
            "dataset_train": dataset_train,
            "dataset_test": dataset_test,
            "use_gpu": False,
            "loss": "MSE",
        }
        return trainer_config
    def setup_eval(self):
        x = []
        for i in reversed(range(1, self.get_window_size()+1)):
            x.append(np.array(
                [self.linedrawer_eval.xs[-i], self.linedrawer_eval.ys[-i]]))
        x = np.hstack(x)
        x = self.trainer.dataset_train.preprocess_x(x)
        x = torch.Tensor(x)
        y = self.trainer.model(x).detach().numpy()
        y = self.trainer.dataset_train.postprocess_y(y)
        if self.target is not None:
            self.target.remove()
        self.target = self.ax.scatter(y[0], y[1], color='g')
    def reset_eval(self):
        if self.target is not None:
            self.target.remove()
            self.target = None
    def load_dataset(self, lines_x, lines_y, window=3):
        X, Y = [], []
        for i in range(len(lines_x)):
            if len(lines_x[i]) < window + 1:
                continue
            for j in range(len(lines_x[i])-window):
                x = [np.array([lines_x[i][j+k], lines_y[i][j+k]]) for k in range(window)]
                X.append(np.hstack(x))
                Y.append(np.array([lines_x[i][j+window], lines_y[i][j+window]]))
        return torch_models.DatasetBase(np.array(X), np.array(Y))
    
class TestSeqData2DContrastiveCrossEntropy(TestSeqData2DBase):
    def __init__(
        self,
        id=0, 
        title='PlotSeqData2D', 
        xlabel='x', 
        ylabel='y',
        xlim=(-10.0, 10.0),
        ylim=(-10.0, 10.0),
        ):
        super().__init__(
            id, title, xlabel, ylabel, xlim, ylim)
        # uniform_data = np.random.rand(10, 10)
        # self.prob = sns.heatmap(uniform_data, linewidth=0.5)
        self.n1, self.n2 = 20, 20
        self.x1, self.x2 = np.meshgrid(
            np.linspace(xlim[0], xlim[1], self.n1),
            np.linspace(ylim[0], ylim[1], self.n2))
        self.prob = np.zeros_like(self.x1)
        self.prob_im = self.ax.pcolormesh(
            self.x1, self.x2, self.prob, shading='gouraud', vmin=0.0, vmax=1.0, cmap='YlOrRd')
        self.fig.colorbar(self.prob_im, ax=self.ax)
    def get_window_size(self):
        return 3
    def get_trainer_config(self):
        dataset_train = self.load_dataset(
            [lb.xs for lb in self.linedrawer_save], 
            [lb.ys for lb in self.linedrawer_save],
            self.get_window_size())
        dataset_test = self.load_dataset(
            [lb.xs for lb in self.linedrawer_save], 
            [lb.ys for lb in self.linedrawer_save],
            self.get_window_size())
        model = torch_models.Classifier(
            torch_models.FCNN,
            size_in=dataset_train.X.shape[1],
            size_out=2,
            hiddens=[128, 128],
            activations=["relu", "relu", "linear"],
            init_weights=[1.0, 1.0, 1.0],
            init_bias=[0.0, 0.0, 0.0])
        trainer_config = {
            "model": model,
            "lr": 0.001,
            "dataset_train": dataset_train,
            "dataset_test": dataset_test,
            "use_gpu": False,
            "loss": "CrossEntropy",
        }
        return trainer_config
    def setup_eval(self):
        ''' Grid Evaluation '''
        x_prev = []
        for i in reversed(range(1, self.get_window_size()+1)):
            x_prev.append(np.array(
                [self.linedrawer_eval.xs[-i], self.linedrawer_eval.ys[-i]]))
        x_prev = np.hstack(x_prev)
        
        for i in range(self.n1):
            for j in range(self.n2):
                x = np.hstack([x_prev, np.array([self.x1[i,j], self.x2[i,j]])])
                x = self.trainer.dataset_train.preprocess_x(x)
                x = torch.Tensor(x)
                y = self.trainer.model(x)
                y = F.softmax(y, dim=0).detach().numpy()
                self.prob[i, j] = y[1]
        self.prob_im.set_array(self.prob.ravel())
        self.draw()
    def reset_eval(self):
        self.prob_im.set_array(np.zeros_like(self.prob.ravel()))
    def load_dataset(self, lines_x, lines_y, window=3):
        X, Y = [], []
        for i in range(len(lines_x)):
            if len(lines_x[i]) < window + 1:
                continue
            for j in range(len(lines_x[i])-window):
                x_prev = np.hstack(
                    [np.array([lines_x[i][j+k], lines_y[i][j+k]]) for k in range(window)])
                x_cur = np.array([lines_x[i][j+window], lines_y[i][j+window]])
                # Positive samples
                X.append(np.hstack([x_prev, x_cur]))
                Y.append(np.array([1]))
                num_added = 0
                while num_added < 10:
                    noise = np.random.uniform(-1.0, 1.0, size=2)
                    if np.linalg.norm(noise) > 0.5: continue
                    X.append(np.hstack([x_prev, x_cur+noise]))
                    Y.append(np.array([1]))
                    num_added += 1
                # Negative samples
                num_added = 0
                while num_added < 30:
                    noise = np.random.uniform(-10.0, 10.0, size=2)
                    if np.linalg.norm(noise) < 0.1: continue
                    X.append(np.hstack([x_prev, x_cur+noise]))
                    Y.append(np.array([0]))
                    num_added += 1
        return torch_models.DatasetClassifier(np.array(X), np.array(Y), normalize_y=False)


class TestSeqData2DContrastiveMSE(TestSeqData2DBase):
    def __init__(
        self,
        id=0, 
        title='TestSeqData2DProbabilityMSE', 
        xlabel='x', 
        ylabel='y',
        xlim=(-10.0, 10.0),
        ylim=(-10.0, 10.0),
        ):
        super().__init__(
            id, title, xlabel, ylabel, xlim, ylim)
        # uniform_data = np.random.rand(10, 10)
        # self.prob = sns.heatmap(uniform_data, linewidth=0.5)
        self.n1, self.n2 = 20, 20
        self.x1, self.x2 = np.meshgrid(
            np.linspace(xlim[0], xlim[1], self.n1),
            np.linspace(ylim[0], ylim[1], self.n2))
        self.prob = np.zeros_like(self.x1)
        self.prob_im = self.ax.pcolormesh(
            self.x1, self.x2, self.prob, shading='gouraud', vmin=0.0, vmax=1.0, cmap='YlOrRd')
        self.fig.colorbar(self.prob_im, ax=self.ax)
    def get_window_size(self):
        return 3
    def get_trainer_config(self):
        dataset_train = self.load_dataset(
            [lb.xs for lb in self.linedrawer_save], 
            [lb.ys for lb in self.linedrawer_save],
            self.get_window_size())
        dataset_test = self.load_dataset(
            [lb.xs for lb in self.linedrawer_save], 
            [lb.ys for lb in self.linedrawer_save],
            self.get_window_size())
        model = torch_models.FCNN(
            size_in=dataset_train.X.shape[1],
            size_out=1,
            hiddens=[128, 128],
            activations=["relu", "relu", "sigmoid"],
            init_weights=[1.0, 1.0, 1.0],
            init_bias=[0.0, 0.0, 0.0])
        trainer_config = {
            "model": model,
            "lr": 0.001,
            "dataset_train": dataset_train,
            "dataset_test": dataset_test,
            "use_gpu": False,
            "loss": "MSE",
        }
        self.dataset_train = dataset_train
        return trainer_config
    def setup_eval(self):
        ''' Grid Evaluation '''
        x_prev = []
        for i in reversed(range(1, self.get_window_size()+1)):
            x_prev.append(np.array(
                [self.linedrawer_eval.xs[-i], self.linedrawer_eval.ys[-i]]))
        x_prev = np.hstack(x_prev)
        for i in range(self.n1):
            for j in range(self.n2):
                x = np.hstack([x_prev, np.array([self.x1[i,j], self.x2[i,j]])])
                x = self.trainer.dataset_train.preprocess_x(x)
                x = torch.Tensor(x)
                y = self.trainer.model(x).detach().numpy()
                self.prob[i, j] = y
        self.prob_im.set_array(self.prob.ravel())
        self.draw()
    def reset_eval(self):
        self.prob_im.set_array(np.zeros_like(self.prob.ravel()))
    def load_dataset(self, lines_x, lines_y, window=3):
        X, Y = [], []
        for i in range(len(lines_x)):
            if len(lines_x[i]) < window + 1:
                continue
            for j in range(len(lines_x[i])-window):
                x_prev = np.hstack(
                    [np.array([lines_x[i][j+k], lines_y[i][j+k]]) for k in range(window)])
                x_pos = np.array([lines_x[i][j+window], lines_y[i][j+window]])
                # Positive samples
                X.append(np.hstack([x_prev, x_pos]))
                Y.append(np.array([1]))
                num_added = 0
                while num_added < 10:
                    noise = np.random.uniform(-1.0, 1.0, size=2)
                    if np.linalg.norm(noise) > 0.5: continue
                    X.append(np.hstack([x_prev, x_pos+noise]))
                    Y.append(np.array([1]))
                    num_added += 1
                # Negative samples
                num_added = 0
                while num_added < 30:
                    x_neg = np.random.uniform(-10.0, 10.0, size=2)
                    if np.linalg.norm(x_pos-x_neg) < 0.1: continue
                    X.append(np.hstack([x_prev, x_neg]))
                    Y.append(np.array([0]))
                    num_added += 1
        self.X = X
        self.Y = Y
        return torch_models.DatasetBase(np.array(X), np.array(Y), normalize_y=False)

# class TestSeqData2DProbability2(TestSeqData2DBase):
#     def __init__(
#         self,
#         id=0, 
#         title='PlotSeqData2D', 
#         xlabel='x', 
#         ylabel='y',
#         xlim=(-10.0, 10.0),
#         ylim=(-10.0, 10.0),
#         ):
#         super(TestSeqData2DProbability2, self).__init__(
#             id, title, xlabel, ylabel, xlim, ylim)
#         # uniform_data = np.random.rand(10, 10)
#         # self.prob = sns.heatmap(uniform_data, linewidth=0.5)
#         self.n1, self.n2 = 20, 20
#         self.x1, self.x2 = np.meshgrid(
#             np.linspace(-1.0, 1.0, self.n1),
#             np.linspace(-1.0, 1.0, self.n2))
#         self.prob = np.zeros_like(self.x1)
#         self.prob_im = self.ax.pcolormesh(
#             self.x1, self.x2, self.prob, shading='gouraud', vmin=0.0, vmax=1.0, cmap='YlOrRd')
#         self.ax.axis([-1.1, 1.1, -1.1, 1.1])
#         self.fig.colorbar(self.prob_im, ax=self.ax)
#     def get_window_size(self):
#         return 3
#     def get_trainer_config(self):
#         dataset_train = self.load_dataset(
#             [lb.xs for lb in self.linedrawer_save], 
#             [lb.ys for lb in self.linedrawer_save],
#             self.get_window_size())
#         dataset_test = self.load_dataset(
#             [lb.xs for lb in self.linedrawer_save], 
#             [lb.ys for lb in self.linedrawer_save],
#             self.get_window_size())
#         model = torch_models.FCNN(
#             size_in=dataset_train.X.shape[1],
#             size_out=1,
#             hiddens=[128, 128],
#             activations=["relu", "relu", "sigmoid"],
#             init_weights=[1.0, 1.0, 1.0],
#             init_bias=[0.0, 0.0, 0.0])
#         trainer_config = {
#             "model": model,
#             "lr": 0.001,
#             "dataset_train": dataset_train,
#             "dataset_test": dataset_test,
#             "use_gpu": False,
#             "loss": "MSE",
#         }
#         return trainer_config
#     def setup_eval(self):
#         ''' Grid Evaluation '''
#         x_prev = []
#         for i in reversed(range(1, self.get_window_size()+1)):
#             x_prev.append(np.array(
#                 [self.linedrawer_eval.xs[-i], self.linedrawer_eval.ys[-i]]))
#         x_prev = np.hstack(x_prev)
#         for i in range(self.n1):
#             for j in range(self.n2):
#                 x = np.hstack([x_prev, np.array([self.x1[i,j], self.x2[i,j]])])
#                 x = self.trainer.dataset_train.preprocess_x(x)
#                 x = torch.Tensor(x)
#                 y = self.trainer.model(x).detach().numpy()
#                 self.prob[i, j] = y
#         self.prob_im.set_array(self.prob.ravel())
#         self.draw()
#     def reset_eval(self):
#         self.prob_im.set_array(np.zeros_like(self.prob.ravel()))
#     def load_dataset(self, lines_x, lines_y, window=3):
#         X, Y = [], []
#         for i in range(len(lines_x)):
#             if len(lines_x[i]) < window + 1:
#                 continue
#             for j in range(len(lines_x[i])-window):
#                 x_prev = np.hstack(
#                     [np.array([lines_x[i][j+k], lines_y[i][j+k]]) for k in range(window)])
#                 x_pos = np.array([lines_x[i][j+window], lines_y[i][j+window]])
#                 # Positive samples
#                 X.append(np.hstack([x_prev, x_pos]))
#                 Y.append(np.array([1]))
#                 num_added = 0
#                 while num_added < 10:
#                     noise = np.random.uniform(-0.1, 0.1, size=2)
#                     if np.linalg.norm(noise) > 0.05: continue
#                     X.append(np.hstack([x_prev, x_pos+noise]))
#                     Y.append(np.array([1]))
#                     num_added += 1
#                 # Negative samples
#                 num_added = 0
#                 while num_added < 10:
#                     x_neg = np.random.uniform(-1.0, 1.0, size=2)
#                     if np.linalg.norm(x_pos-x_neg) < 0.1: continue
#                     X.append(np.hstack([x_prev, x_neg]))
#                     Y.append(np.array([0]))
#                     num_added += 1
#         return torch_models.DatasetBase(np.array(X), np.array(Y), normalize_y=False)


if __name__ == '__main__':

    # test = TestSeqData2DPrediction()
    test = TestSeqData2DContrastiveCrossEntropy()
    # test = TestSeqData2DContrastiveMSE()
    plt.show()
