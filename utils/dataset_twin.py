import os
from typing import List, Union
import json
import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from copy import deepcopy
import functools
from skimage.measure import regionprops
from shapely.geometry import Polygon
from skimage.draw import polygon
from skimage.filters import gaussian
import matplotlib.pyplot as plt

from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
# from .augmentation import DataAugmentor

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    },
    'cocostuff': {
        "train": 965042,
        'val': 42095,
        'val-test': 42095
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

class GraspTransforms:
    # Class for converting cv2-like rectangle formats and generate grasp-quality-angle-width masks

    def __init__(self, width_factor=100, width=640, height=480):
        self.width_factor = width_factor
        self.width = width
        self.height = height

    def __call__(self, grasp_rectangles, target):
        # grasp_rectangles: (M, 4, 2)
        M = grasp_rectangles.shape[0]
        p1, p2, p3, p4 = np.split(grasp_rectangles, 4, axis=1)

        center_x = (p1[..., 0] + p3[..., 0]) / 2
        center_y = (p1[..., 1] + p3[..., 1]) / 2

        width  = np.sqrt((p1[..., 0] - p4[..., 0]) * (p1[..., 0] - p4[..., 0]) + (p1[..., 1] - p4[..., 1]) * (p1[..., 1] - p4[..., 1]))
        height = np.sqrt((p1[..., 0] - p2[..., 0]) * (p1[..., 0] - p2[..., 0]) + (p1[..., 1] - p2[..., 1]) * (p1[..., 1] - p2[..., 1]))

        theta = np.arctan2(p4[..., 0] - p1[..., 0], p4[..., 1] - p1[..., 1]) * 180 / np.pi
        theta = np.where(theta > 0, theta - 90, theta + 90)

        target = np.tile(np.array([[target]]), (M,1))

        return np.concatenate([center_x, center_y, width, height, theta, target], axis=1)

    def inverse(self, grasp_rectangles):
        boxes = []
        for rect in grasp_rectangles:
            center_x, center_y, width, height, theta = rect[:5]
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.intp(box)
            boxes.append(box)
        return boxes

    def generate_masks(self, grasp_rectangles):
        pos_out = np.zeros((self.height, self.width))
        ang_out = np.zeros((self.height, self.width))
        wid_out = np.zeros((self.height, self.width))
        for rect in grasp_rectangles:
            center_x, center_y, w_rect, h_rect, theta = rect[:5]

            # Get 4 corners of rotated rect
            # Convert from our angle represent to opencv's
            r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
            box = cv2.boxPoints(r_rect)
            box = np.intp(box)

            rr, cc = polygon(box[:, 0], box[:,1])

            mask_rr = rr < self.width
            rr = rr[mask_rr]
            cc = cc[mask_rr]

            mask_cc = cc < self.height
            cc = cc[mask_cc]
            rr = rr[mask_cc]
            pos_out[cc, rr] = 1.0
            if theta < 0:
                ang_out[cc, rr] = int(theta + 180)
            else:
                ang_out[cc, rr] = int(theta)
            # Adopt width normalize accoding to class
            wid_out[cc, rr] = np.clip(w_rect, 0.0, self.width_factor) / self.width_factor

        qua_out = (gaussian(pos_out, 3, preserve_range=True) * 255).astype(np.uint8)
        pos_out = (pos_out * 255).astype(np.uint8)
        ang_out = ang_out.astype(np.uint8)
        wid_out = (gaussian(wid_out, 3, preserve_range=True) * 255).astype(np.uint8)


        return {'pos': pos_out,
                'qua': qua_out,
                'ang': ang_out,
                'wid': wid_out}

class OCIDVLGTWINDataset(Dataset):
    """ OCID-Vision-Language-Grasping dataset with referring expressions and grasps """

    def __init__(self,
                 root_dir,
                 split,
                 transform_img=None,
                 transform_grasp=GraspTransforms(),
                 input_size=416,
                 word_length=20,
                 with_depth=True,
                 with_segm_mask=True,
                 with_grasp_masks=True,
                 version="multiple",
                 twin=False,
                 args = False
                 ):
        super(OCIDVLGTWINDataset, self).__init__()
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, "data_split")
        self.split_map = {'train': 'train_expressions.json',
                          'val': 'val_expressions.json',
                          'test': 'test_expressions.json'
                          }
        self.split = split
        self.refer_dir = os.path.join(root_dir, "refer", version)
        self.transform_img = transform_img
        self.transform_grasp = transform_grasp
        self.with_depth = with_depth
        self.with_segm_mask = with_segm_mask
        self.with_grasp_masks = with_grasp_masks
        # assert (self.transform_grasp and self.with_grasp_masks) or (not self.transform_grasp and not self.with_grasp_masks)
        self.args = args
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.twin = twin
        self._load_dicts()
        self._load_split()

    def _load_dicts(self):
        cwd = os.getcwd()
        os.chdir(self.root_dir)
        # from .OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
        cnames_inv = {int(v): k for k, v in cnames.items()}
        subnames_inv = {v: k for k, v in subnames.items()}
        self.class_names = cnames
        self.idx_to_class = cnames_inv
        self.class_instance_names = subnames
        self.idx_to_class_instance = subnames_inv
        self.instance_idx_to_class_idx = sub_to_class
        os.chdir(cwd)

    def _load_split(self):
        refer_data = json.load(open(os.path.join(self.refer_dir, self.split_map[self.split])))
        self.seq_paths, self.img_names, self.scene_ids = [], [], []
        self.bboxes, self.grasps = [], []
        self.sent_to_index, self.sent_indices = {}, []
        self.rgb_paths, self.rgb_twin_paths, self.depth_paths, self.mask_paths = [], [], [], []
        self.targets, self.sentences, self.semantics, self.objIDs = [], [], [], []
        n = 0
        for item in refer_data['data']:
            seq_path, im_name = item['image_filename'].split(',')
            self.seq_paths.append(seq_path)
            self.img_names.append(im_name)
            self.scene_ids.append(item['image_filename'])
            self.bboxes.append(item['box'])  # 目标框
            self.grasps.append(item['grasps'])  # 5个抓取位置，包含4个抓取坐标点
            self.objIDs.append(item['answer'])
            self.targets.append(item['target'])  # 实例
            self.sentences.append(item['question'])  # 任务描述
            self.semantics.append(item['program'])
            self.rgb_paths.append(os.path.join(seq_path, "rgb", im_name))
            if self.args.rgb_type == 1:
                self.rgb_twin_paths.append(os.path.join(seq_path, "rgb-r1", im_name))
            elif self.args.rgb_type == 2:
                self.rgb_twin_paths.append(os.path.join(seq_path, "rgb-r2", im_name))
            self.depth_paths.append(os.path.join(seq_path, "depth", im_name))
            self.mask_paths.append(os.path.join(seq_path, "seg_mask_instances_combi", im_name))
            self.sent_indices.append(item['question_index'])  # 任务序列号
            self.sent_to_index[item['question_index']] = n

            # #-------------------------------------------------------------------------
            # #learn dataset-set(lb)
            # img = cv2.imread(os.path.join('data/OCID-VLG',seq_path,'rgb',im_name))
            # img = cv2.imread(os.path.join('data/OCID-VLG', seq_path, 'rgb', im_name))
            # box_x = item['box'][0]
            # box_y = item['box'][1]
            # box_width = item['box'][2]
            # box_height = item['box'][3]
            #
            # for i in range(len(item['grasps'])):
            #     for j in range(len(item['grasps'][i])):
            #
            #         cv2.circle(img, (int(item['grasps'][i][j][0]), int(item['grasps'][i][j][1])), 2, (255,0,0), -1)
            # # cv2.circle(img, (int(item['grasps'][i][0][0]), int(item['grasps'][i][0][1])), 2, (255, 0, 0), -1)
            # # cv2.circle(img, (int(item['grasps'][i][1][0]), int(item['grasps'][i][1][1])), 2, (255, 0, 0), -1)
            # # cv2.circle(img, (int(item['grasps'][i][2][0]), int(item['grasps'][i][2][1])), 2, (255, 0, 0), -1)
            # # cv2.circle(img, (int(item['grasps'][i][3][0]), int(item['grasps'][i][3][1])), 2, (255, 0, 0), -1)
            #
            # cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 255), 3)
            #
            # cv2.imshow("cls_show", img)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     exit()
            #     # -------------------------------------------------------------------------

            n += 1

    def get_index_from_sent(self, sent_id):
        return self.sent_to_index[sent_id]

    def get_sent_from_index(self, n):
        return self.sent_indices[n]

    def _load_sent(self, sent_id):
        n = self.get_index_from_sent(sent_id)

        scene_id = self.scene_ids[n]

        img_path = os.path.join(self.root_dir, self.rgb_paths[n])
        img = self.get_image_from_path(img_path)
        img_twin_path = os.path.join(self.root_dir, self.rgb_twin_paths[n])
        img_twin = self.get_image_twin_from_path(img_twin_path)

        x, y, w, h = self.bboxes[n]
        bbox = np.asarray([x, y, x + w, y + h])

        sent = self.sentences[n]

        target = self.targets[n]
        target_idx = self.class_instance_names[target]
        objID = self.objIDs[n]

        grasps = np.asarray(self.grasps[n])

        result = {'img': self.transform_img(img) if self.transform_img else img,
                  'img_twin':self.transform_img(img_twin) if self.transform_img else img_twin,
                  'grasps': self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
                  'grasp_rects': self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
                  'sentence': sent,
                  'target': target,
                  'objID': objID,
                  'bbox': bbox,
                  'target_idx': target_idx,
                  'sent_id': sent_id,
                  'scene_id': scene_id,
                  'img_path': img_path
                  }

        if self.with_depth:
            depth_path = os.path.join(self.root_dir, self.depth_paths[n])
            depth = self.get_depth_from_path(depth_path)
            result = {**result, 'depth': torch.from_numpy(depth) if self.transform_img else depth}

        if self.with_segm_mask:
            mask_path = os.path.join(self.root_dir, self.mask_paths[n])
            msk_full = self.get_mask_from_path(mask_path)
            msk = np.where(msk_full == objID, True, False)
            result = {**result, 'mask': torch.from_numpy(msk) if self.transform_img else msk}

        if self.with_grasp_masks:
            grasp_masks = self.transform_grasp.generate_masks(result['grasp_rects'])
            result = {**result, 'grasp_masks': grasp_masks}

        result = self.preprocess(result)

        return result

    def get_transform_mat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def preprocess(self, data):
        img = data["img"]
        img_twin = data["img_twin"]
        sent = data["sentence"]
        if np.max(data["mask"]) <= 1.0:
            ins_mask = (data["mask"] * 255).astype(np.uint8)
        else:
            ins_mask = data["mask"]

        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_ang_mask = data["grasp_masks"]["ang"]
        grasp_wid_mask = data["grasp_masks"]["wid"]

        img_size = img.shape[:2]
        mat, mat_inv = self.get_transform_mat(img_size, True)

        img = cv2.warpAffine(
            img, mat, self.input_size, flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        )
        img_twin = cv2.warpAffine(
            img_twin, mat, self.input_size, flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        )

        img = torch.from_numpy(img.transpose((2, 0, 1)))
        img_twin = torch.from_numpy(img_twin.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
            img_twin = img_twin.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        img_twin.div_(255.).sub_(self.mean).div_(self.std)

        ins_mask = cv2.warpAffine(ins_mask,
                                  mat,
                                  self.input_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderValue=0.)
        grasp_qua_mask = cv2.warpAffine(grasp_qua_mask,
                                        mat,
                                        self.input_size,
                                        flags=cv2.INTER_LINEAR,
                                        borderValue=0.)

        grasp_ang_mask = cv2.warpAffine(grasp_ang_mask,
                                        mat,
                                        self.input_size,
                                        flags=cv2.INTER_LINEAR,
                                        borderValue=0.)

        grasp_wid_mask = cv2.warpAffine(grasp_wid_mask,
                                        mat,
                                        self.input_size,
                                        flags=cv2.INTER_LINEAR,
                                        borderValue=0.)

        ins_mask = ins_mask / 255.
        grasp_qua_mask = grasp_qua_mask / 255.
        grasp_ang_mask = grasp_ang_mask * np.pi / 180.
        grasp_wid_mask = grasp_wid_mask / 255.
        grasp_sin_mask = np.sin(2 * grasp_ang_mask)
        grasp_cos_mask = np.cos(2 * grasp_ang_mask)

        word_vec = tokenize(sent, self.word_length, True).squeeze(0)

        data["img"] = img
        data["img_twin"] = img_twin
        data["mask"] = ins_mask
        data["grasp_masks"]["qua"] = grasp_qua_mask
        data["grasp_masks"]["ang"] = grasp_ang_mask
        data["grasp_masks"]["wid"] = grasp_wid_mask
        data["grasp_masks"]["sin"] = grasp_sin_mask
        data["grasp_masks"]["cos"] = grasp_cos_mask
        data["word_vec"] = word_vec
        data["inverse"] = mat_inv
        data["ori_size"] = np.array(img_size)

        # del data["sentence"]

        return data

    def __len__(self):
        return len(self.sent_indices)

    def __getitem__(self, n):
        sent_id = self.get_sent_from_index(n)
        data = self._load_sent(sent_id)

        return data

    @staticmethod
    def transform_grasp_inv(grasp_pt):
        pass

    # @functools.lru_cache(maxsize=None)
    def get_image_from_path(self, path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(os.path.exists(path))
            print(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img
    def get_image_twin_from_path(self, path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(os.path.exists(path))
            print(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img

    # @functools.lru_cache(maxsize=None)
    def get_mask_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # @functools.lru_cache(maxsize=None)
    def get_depth_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.  # mm -> m

    def get_image(self, n):
        img_path = os.path.join(self.root_dir, self.imgs[n])
        return self.get_image_from_path(img_path)

    def get_annotated_image(self, n, text=True):
        sample = self.__getitem__(n)

        img, sent, grasps, bbox = sample['img'], sample['sentence'], sample['grasp_rects'], sample['bbox']
        if isinstance(img, torch.FloatTensor):
            img = img.permute(1, 2, 0)
            img = (img.cpu().numpy() * 255).astype(np.uint8)
        if self.transform_img:
            img = np.asarray(tfn.to_pil_image(img))
        if self.transform_grasp:
            # grasps = list(map(self.transform_grasp_inv, list(grasps)))
            grasps = self.transform_grasp.inverse(grasps)

        tmp = img.copy()
        for entry in grasps:
            ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
            tmp = cv2.line(tmp, ptA, ptB, (0, 0, 0xff), 2)
            tmp = cv2.line(tmp, ptD, ptC, (0, 0, 0xff), 2)
            tmp = cv2.line(tmp, ptB, ptC, (0xff, 0, 0), 2)
            tmp = cv2.line(tmp, ptA, ptD, (0xff, 0, 0), 2)

        tmp = cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if text:
            tmp = cv2.putText(tmp, sent, (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return tmp

    def visualization(self, n, save_path):
        s = self.__getitem__(n)

        rgb = s['img']
        if isinstance(rgb, torch.FloatTensor):
            rgb = rgb.permute(1, 2, 0)
            rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
        depth = (0xff * s['depth'] / 3).astype(np.uint8)
        ii = self.get_annotated_image(n, text=False)
        sentence = s['sentence']
        msk = s['mask'].astype(np.uint8) / 255
        # msk_img = (rgb * 0.3).astype(np.uint8).copy()
        # msk_img[msk, 0] = 255

        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(rgb)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(depth, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(msk)
        ax.set_title('Segm Mask')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 4)
        ax.imshow(ii)
        ax.set_title('Box & Grasp')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 5)
        plot = ax.imshow(s['grasp_masks']['qua'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Grasp quality')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 6)
        plot = ax.imshow(s['grasp_masks']['sin'], cmap='rainbow', vmin=-1, vmax=1)
        ax.set_title('Angle-cosine')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 7)
        plot = ax.imshow(s['grasp_masks']['cos'], cmap='rainbow', vmin=-1, vmax=1)
        ax.set_title('Angle-sine')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 8)
        plot = ax.imshow(s['grasp_masks']['wid'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Width')
        ax.axis('off')
        plt.colorbar(plot)

        plt.suptitle(f"{sentence}", fontsize=20)
        plt.tight_layout()
        print("save")
        plt.savefig(os.path.join(save_path, f"sample_{n}.png"))

    @staticmethod
    def collate_fn(batch):
        # for x in batch:
        #     # print(x["img"])
        #     print(torch.stack(x["img"]))
        return {
            "img": torch.stack([x["img"] for x in batch]),
            "img_twin": torch.stack([x["img_twin"] for x in batch]),
            "depth": torch.stack([torch.from_numpy(x["depth"]) for x in batch]),
            "mask": torch.stack([torch.from_numpy(x["mask"]).float() for x in batch]),
            "grasp_masks": {
                "qua": torch.stack([torch.from_numpy(x["grasp_masks"]["qua"]).float() for x in batch]),
                "sin": torch.stack([torch.from_numpy(x["grasp_masks"]["sin"]).float() for x in batch]),
                "cos": torch.stack([torch.from_numpy(x["grasp_masks"]["cos"]).float() for x in batch]),
                "wid": torch.stack([torch.from_numpy(x["grasp_masks"]["wid"]).float() for x in batch])
            },
            "word_vec": torch.stack([x["word_vec"].long() for x in batch]),
            "grasps": [x["grasps"] for x in batch],
            "target": [x["target"] for x in batch],
            "sentence": [x["sentence"] for x in batch],
            "bbox": [x["bbox"] for x in batch],
            "target_idx": [x["target_idx"] for x in batch],
            "sent_id": [x["sent_id"] for x in batch],
            "scene_id": [x["scene_id"] for x in batch],
            "inverse": [x["inverse"] for x in batch],
            "ori_size": [x["ori_size"] for x in batch],
            "img_path": [x["img_path"] for x in batch]
        }

if __name__ == '__main__':
    from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
    root_dir = '/home/lbycdy/work/datasets/OCID-VLG'  # 数据集根目录
    split = 'test'  # 使用验证集
    input_size = 416  # 图像输入大小
    import argparse
    import utils.config as config

    def get_parser():
        parser = argparse.ArgumentParser(
            description='Pytorch Referring Expression Segmentation')
        parser.add_argument('--config',
                            default='/home/lbycdy/work/crog_lb/config/OCID-VLG/test.yaml',
                            type=str,
                            help='config file')
        parser.add_argument('--opts',
                            default=None,
                            nargs=argparse.REMAINDER,
                            help='override some settings in the config.')

        args = parser.parse_args()
        assert args.config is not None
        cfg = config.load_cfg_from_cfg_file(args.config)

        cfg.save_dir = os.path.splitext(os.path.split(args.config)[-1])[0]

        if args.opts is not None:
            cfg = config.merge_cfg_from_list(cfg, args.opts)
        return cfg
    args = get_parser()
    # 创建数据集实例
    dataset = OCIDVLGTWINDataset(root_dir=root_dir,
                                 split=split,
                                 input_size=input_size,
                                 with_depth=True,
                                 with_segm_mask=True,
                                 with_grasp_masks=True,args=args
                                 )  # 不启用抓取掩码

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 读取一个数据项
    for data in dataloader:
        # 获取图像和掩码
        img = data['img'].numpy().squeeze().transpose(1, 2, 0)  # 转换为HWC格式
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 将图像从RGB转换为BGR格式
        mask = data['mask'].numpy().squeeze()  # 提取掩码

        # 使用 OpenCV 显示图像和掩码
        img_with_mask = img.copy()
        mask = mask.astype(bool)
        # 将掩码叠加到图像上（将掩码部分显示为绿色）
        img_with_mask[mask] = [0, 255, 0]  # OpenCV使用BGR格式

        # 使用 OpenCV 显示图像和掩码
        cv2.imshow('Original Image', img)
        cv2.imshow('Mask', mask.astype(np.uint8) * 255)  # 掩码转换为可显示的图像
        cv2.imshow('Image with Mask', img_with_mask)

        # 等待按键事件
        cv2.waitKey(0)
        cv2.destroyAllWindows()

