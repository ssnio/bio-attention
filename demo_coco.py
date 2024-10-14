from PIL import Image as PILImage
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
from torch.nn.functional import one_hot
from src.model import AttentionModel
from prelude import load_dicts, get_device
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton
# from matplotlib.backend_tools import Cursors


class DemoImages:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.h, self.w = 256, 256
        self.k = 3
        self.fixation = 2
        self.n_iter = 3
        self.tensorfy = transforms.ToTensor()
        self.resize = transforms.Resize((self.h, self.w), interpolation=PILImage.BILINEAR, antialias=True)
        self.sfiles = [x for x in os.listdir(image_path) if x.endswith('_s.jpg')]
        self.gfiles = [x for x in os.listdir(image_path) if x.endswith('_g.jpg')]
        self.singles = [self.read_image(x) for x in self.sfiles]
        self.grids = [self.read_image(x) for x in self.gfiles]

    def _crop_square(self, x: torch.Tensor):
        _, x_h, x_w = x.shape
        hw = min(x_h, x_w)
        top = torch.randint(0, x_h - hw, (1, )).item() if x_h > hw else 0
        left = torch.randint(0, x_w - hw, (1, )).item() if x_w > hw else 0
        return x[:, top:top+hw, left:left+hw]

    def read_image(self, filename: str):
        x = PILImage.open(os.path.join(self.image_path, filename))
        x = self.tensorfy(x)
        if x.size(1) != x.size(2):
            x = self._crop_square(x)
        x = self.resize(x)
        x = torch.clamp(x, 0.0, 1.0)[None, ...]
        x = x.expand(self.n_iter, -1, -1, -1)
        return x

    def get_point(self, point: tuple):
        i, j = point
        i, j = int(i), int(j)
        x = torch.zeros(3, self.h, self.w)
        x[:, i-self.k:i+self.k, j-self.k:j+self.k] = 1.0
        return x

    def __len__(self):
        return len(self.singles)

    def __getitem__(self, idx: int, task: int, point: tuple = None):
        if task == 0:
            x = self.singles[idx]
        if task == 1:
            x = self.singles[idx]
            if point is not None:
                point = self.get_point(point)[None, ...]
                point = point.expand(self.fixation, -1, -1, -1)
                x = torch.cat([point, x], dim=0)
        elif task == 2:
        #     x = self.singles[idx]
        # elif task == 3:
            x = self.grids[idx]
        return x

# set up
start_folder = r"./pretrained/coco"
model_params = load_dicts(start_folder, "model_params")
train_params = load_dicts(start_folder, "train_params")
tasks = load_dicts(start_folder, "tasks")
DeVice, num_workers, pin_memory = get_device()

# model and optimizer...
model = AttentionModel(**model_params)
model_dir = os.path.join(start_folder, "model" + ".pth")
assert os.path.exists(model_dir), "Could not find the model.pth in the given dir!"
model.load_state_dict(torch.load(model_dir, map_location=DeVice))


demo_ds = DemoImages(r"./demo")
model_img = PILImage.open(r"./demo/model.jpg")


class DemoClass:
    def __init__(self, ds, model):
        self.ds = ds
        self.model = model
        self.class_names = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        self.n_classes = len(self.class_names)
        self.class_ids = torch.arange(self.n_classes).int()
        self.task_id = 0
        self.im_id = 0

    def do_it(self, point_pos: tuple = None, hot_labels: torch.Tensor = None):
        if self.task_id in (0, 2, 3):
            point_pos = None
        composites = demo_ds.__getitem__(self.im_id, self.task_id, point_pos)[None, ...]
        with torch.no_grad():
            model.eval()
            model.to(DeVice)
            composites, hot_labels = composites.to(DeVice), (hot_labels.to(DeVice) if hot_labels is not None else None)
            p_masks, p_labels, *_ = model(composites, self.task_id, hot_labels)
            p_labels = torch.softmax(p_labels[:, :, -1], dim=-1)[0]
            composites, p_masks, p_labels = composites.cpu(), p_masks.cpu(), p_labels.cpu()
        return composites, p_masks, p_labels

    def get_im(self):
        return self.ds.__getitem__(self.im_id, self.task_id)

    def next_im(self):
        self.im_id += 1
        if self.task_id == 2:
            self.im_id %= len(self.ds.grids)
        else:
            self.im_id %= len(self.ds.singles)

    def set_task(self, task_id: int):
        self.task_id = task_id

    def start_plot(self):
        fig, ax_main = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(top=0.85, right=0.75, left=0.3)
        cool_cm = plt.get_cmap("viridis")
        cax_main = ax_main.imshow(model_img)
        ax_main.axis('off')

        # Task buttons
        button_width = 0.08
        button_height = 0.03
        button_left = 0.39
        button_top = 0.62
        button_gap = 0.02
        ax_left = plt.axes([button_left, button_top, button_width, button_height])
        ax_mid = plt.axes([button_left + button_width + button_gap, button_top, button_width, button_height])
        ax_right = plt.axes([button_left + 2 * (button_width + button_gap), button_top, button_width, button_height])
        btn_recog = Button(ax_left, 'Recognition')
        btn_group = Button(ax_mid, 'Grouping')
        btn_search = Button(ax_right, 'Search')
        def_color = btn_search.color
        def set_rec(event):
            self.set_task(0)
            composites, p_masks, p_labels = self.do_it()
            draw_it(composites, p_masks, p_labels)
            btn_recog.color = 'pink'
            btn_group.color = def_color
            btn_search.color = def_color
        def set_group(event):
            self.set_task(1)
            btn_recog.color = def_color
            btn_group.color = 'pink'
            btn_search.color = def_color
        def set_search(event):
            self.set_task(2)
            btn_recog.color = def_color
            btn_group.color = def_color
            btn_search.color = 'pink'

        btn_recog.on_clicked(set_rec)
        btn_group.on_clicked(set_group)
        btn_search.on_clicked(set_search)

        # Class buttons
        def set_hot_labels(hot: int):
            if self.task_id == 2:
                hot_labels = one_hot(torch.tensor([hot]), self.n_classes).float().expand(3, -1)[None, ...]
                composites, p_masks, p_labels = self.do_it(hot_labels=hot_labels)
                draw_it(composites, p_masks, p_labels)

        label_axes = []
        label_buts = []
        label_funs = []
        but_width = 0.08
        but_height = 0.03
        but_left = 0.76
        but_top = 0.65
        but_gap = 0.01
        for i, cn in enumerate(self.class_names):
            label_axes.append(plt.axes([but_left, but_top - i * (but_height + but_gap), but_width, but_height]))
            label_buts.append(Button(label_axes[-1], cn))
            label_funs.append(lambda event, i=i: set_hot_labels(i))
            label_buts[-1].on_clicked(label_funs[-1])
            # label_axes[-1].cursor_to_use = Cursors.HAND
        ax_right = plt.axes([0.85, 0.27, 0.1, 1.07 * (but_height + but_gap) * self.n_classes])
        class_txt = plt.axes([0.85, 0.7, 0.1, 0.03])
        class_txt.text(0.5, 0.5, "Class probability ", horizontalalignment='center', verticalalignment='center')
        class_txt.axis('off')

        # image and attention
        im = self.get_im()
        ax_left1 = plt.axes([0.0, 0.52, 0.32, 0.42])
        ax_left2 = plt.axes([0.0, 0.01, 0.32, 0.42])
        ax_left1.imshow(torch.rand(256, 256), cmap='gray')
        ax_left2.imshow(im[-1].permute(1, 2, 0))
        ax_left1.axis('off')
        ax_left2.axis('off')
        # ax_left1.cursor_to_use = Cursors.HAND
        # ax_left2.cursor_to_use = Cursors.HAND

        y_pos = torch.linspace(0, 11, self.n_classes)
        ax_right.barh(y_pos, torch.zeros(self.n_classes), align='center', color='b', height=0.9)
        ax_right.invert_yaxis()  # labels read top-to-bottom
        ax_right.set_xlim(0.0, 1.0)
        ax_right.axis('off')

        # Next buttons
        ax_next = plt.axes([0.31, 0.01, 0.03, 0.03])
        btn_next = Button(ax_next, '>>')
        def next_im(event):
            self.next_im()
            im = self.get_im()
            ax_right.clear()
            ax_right.barh(y_pos, torch.zeros(self.n_classes), align='center', color='b', height=0.9)
            ax_right.invert_yaxis()  # labels read top-to-bottom
            ax_right.set_xlim(0.0, 1.0)
            ax_right.axis('off')

            ax_left1.imshow(torch.rand(256, 256), cmap='gray')
            ax_left2.imshow(im[-1].permute(1, 2, 0))
            ax_left1.axis('off')
            ax_left2.axis('off')
            plt.draw()
        btn_next.on_clicked(next_im)

        def draw_it(composites, p_masks, p_labels):
            ax_right.clear()
            bar_colors = [cool_cm.colors[i] for i in (p_labels * 256).int()]
            ax_right.barh(y_pos, p_labels, align='center', color=bar_colors, height=0.9)
            # ax_right.set_yticks(y_pos, labels=self.class_names)
            ax_right.invert_yaxis()  # labels read top-to-bottom
            ax_right.set_xlim(0.0, 1.0)
            ax_right.axis('off')

            ax_left1.clear()
            ax_left2.clear()
            ax_left1.imshow(p_masks[0, -1, 0], cmap='plasma')
            ax_left2.imshow(composites[0, -1].permute(1, 2, 0))
            ax_left1.axis('off')
            ax_left2.axis('off')
            plt.draw()

        def on_click(event):
            if self.task_id == 1:
                if event.button is MouseButton.LEFT:
                    if not (event.ydata is None or event.xdata is None or event.ydata > 256 or event.xdata > 256):
                        point_pos = (event.ydata, event.xdata)
                        composites, p_masks, p_labels = self.do_it(point_pos)
                        draw_it(composites, p_masks, p_labels)
            else:
                pass
        plt.connect('button_press_event', on_click)

        plt.show()


DemoClass(demo_ds, model).start_plot()
