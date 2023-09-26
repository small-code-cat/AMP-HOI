"""
HICO-DET dataset utils
"""
import os
import json
import collections
import torch
import torch.utils.data
from torchvision.datasets import CocoDetection
import datasets.transforms as T
import random
import re
from PIL import Image
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS, ZERO_SHOT_INTERACTION_IDS, NON_INTERACTION_IDS
from utils.sampler import repeat_factors_from_category_frequency, get_dataset_indices
from .bounding_box import BoxList


# NOTE: Replace the path to your file
HICO_TRAIN_ROOT = "/home/xkj/hico_20160224_det/images/train2015"
HICO_TRAIN_ANNO = "/home/xkj/hico_20160224_det/annotations/trainval_hico_ann.json"
HICO_VAL_ROOT = "/home/xkj/hico_20160224_det/images/test2015"
HICO_VAL_ANNO = "/home/xkj/hico_20160224_det/annotations/test_hico_ann.json"


class HICO(CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        image_set,
        zero_shot_exp,
        repeat_factor_sampling,
        ignore_non_interaction
    ):
        """
        Args:
            json_file (str): full path to the json file in HOI instances annotation format.
            image_root (str or path-like): the directory where the images in this json file exists.
            transforms (class): composition of image transforms.
            image_set (str): 'train', 'val', or 'test'.
            repeat_factor_sampling (bool): resampling training data to increase the rate of tail
                categories to be observed by oversampling the images that contain them.
            zero_shot_exp (bool): if true, see the last 120 rare HOI categories as zero-shot,
                excluding them from the training data. For the selected rare HOI categories, please
                refer to `<datasets/hico_categories.py>: ZERO_SHOT_INTERACTION_IDS`.
            ignore_non_interaction (bool): Ignore non-interaction categories, since they tend to
                confuse the models with the meaning of true interactions.
        """
        self.root = img_folder
        self.transforms = transforms
        # Text description of human-object interactions
        dataset_texts, text_mapper = prepare_dataset_text()
        self.dataset_texts = dataset_texts
        self.text_mapper = text_mapper # text to contiguous ids for evaluation
        object_to_related_hois, action_to_related_hois = prepare_related_hois()
        self.object_to_related_hois = object_to_related_hois
        self.action_to_related_hois = action_to_related_hois
        # Load dataset
        repeat_factor_sampling = repeat_factor_sampling and image_set == "train"
        zero_shot_exp = zero_shot_exp and image_set == "train"
        self.dataset_dicts = load_hico_json(
            json_file=ann_file,
            image_root=img_folder,
            zero_shot_exp=zero_shot_exp,
            repeat_factor_sampling=repeat_factor_sampling,
            ignore_non_interaction=ignore_non_interaction)
        self.prepare = ConvertCocoPolysToMask(False, True, tokenizer=None, max_query_len=256)

    def generate_sentence_from_hico_objects(self):
        label_to_positions = {}
        pheso_caption = ''
        for index, i in enumerate(HICO_OBJECTS):
            start_index = len(pheso_caption)
            pheso_caption += self.clean_name(i['name'])
            end_index = len(pheso_caption)
            label_to_positions[i['id']] = [start_index, end_index]
            if index != len(HICO_OBJECTS)-1:
                pheso_caption += '. '
        return label_to_positions, pheso_caption

    def get_related_hois(self, hois):
        action_set = {i['text'][0] for i in hois}
        object_set = {i['text'][1] for i in hois}
        related_hois = []
        label_to_positions = {}
        pheso_caption = ''
        for i in object_set:
            related_hois.extend(self.object_to_related_hois[i])
        # related_hois = [' '.join(j['text']) for i in action_set for j in self.action_to_related_hois[i]]
        related_hois = sorted(related_hois, key=lambda x: x['hoi_id'])
        for index, i in enumerate(related_hois):
            start_index = len(pheso_caption)
            pheso_caption += self.clean_name(' '.join(i['text']))
            end_index = len(pheso_caption)
            label_to_positions[i['hoi_id']] = [start_index, end_index]
            if index != len(related_hois)-1:
                pheso_caption += '. '
        return label_to_positions, pheso_caption

    def convert_od_to_grounding_simple(self, hois, target, image_id):
        label_to_positions, pheso_caption = self.generate_sentence_from_hico_objects()
        areas = target.area()
        greenlight_span_for_masked_lm_objective = []
        new_target = []
        for i in range(len(target)):
            new_target_i = {}
            new_target_i["area"] = areas[i]
            new_target_i["iscrowd"] = 0
            new_target_i["image_id"] = image_id
            new_target_i["category_id"] = target.extra_fields["labels"][i].item()
            new_target_i["id"] = None
            new_target_i['bbox'] = target.bbox[i].numpy().tolist()

            label_i = target.extra_fields["labels"][i].item()

            if label_i in label_to_positions:  # NOTE: Only add those that actually appear in the final caption
                new_target_i["tokens_positive"] = [label_to_positions[label_i]]
                new_target.append(new_target_i)
                greenlight_span_for_masked_lm_objective.append(label_to_positions[label_i])

        return new_target, pheso_caption, greenlight_span_for_masked_lm_objective

    def __getitem__(self, idx: int):

        filename = self.dataset_dicts[idx]["file_name"]
        image = Image.open(filename).convert("RGB")

        w, h = image.size
        assert w == self.dataset_dicts[idx]["width"], "image shape is not consistent."
        assert h == self.dataset_dicts[idx]["height"], "image shape is not consistent."

        image_id = self.dataset_dicts[idx]["image_id"]
        annos = self.dataset_dicts[idx]["annotations"]

        boxes = torch.as_tensor(annos["boxes"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        target = BoxList(boxes, image.size)

        classes = torch.tensor(annos["classes"], dtype=torch.int64)
        target.add_field('labels', classes)

        annotations, caption, greenlight_span_for_masked_lm_objective = self.convert_od_to_grounding_simple(annos['hois'], target, image_id)
        anno = {"image_id": image_id, "annotations": annotations, "caption": caption}
        anno["greenlight_span_for_masked_lm_objective"] = greenlight_span_for_masked_lm_objective
        img, anno = self.prepare(image, anno, box_format="xyxy")

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # add additional property
        for ann in anno:
            target.add_field(ann, anno[ann])

        sanity_check_target_after_processing(target)

        return img, target, idx

    def clean_name(self, name):
        name = re.sub(r"\(.*\)", "", name)
        name = re.sub(r"_", " ", name)
        name = re.sub(r"  ", " ", name)
        return name

    def __len__(self):
        return len(self.dataset_dicts)


def load_hico_json(
    json_file,
    image_root,
    zero_shot_exp=True,
    repeat_factor_sampling=False,
    ignore_non_interaction=True,
):
    """
    Load a json file with HOI's instances annotation.

    Args:
        json_file (str): full path to the json file in HOI instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        repeat_factor_sampling (bool): resampling training data to increase the rate of tail
            categories to be observed by oversampling the images that contain them.
        zero_shot_exp (bool): if true, see the last 120 rare HOI categories as zero-shot,
            excluding them from the training data. For the selected rare HOI categories, please
            refer to `<datasets/hico_categories.py>: ZERO_SHOT_INTERACTION_IDS`.
        ignore_non_interaction (bool): Ignore non-interaction categories, since they tend to
            confuse the models with the meaning of true interactions.
    Returns:
        list[dict]: a list of dicts in the following format.
        {
            'file_name': path-like str to load image,
            'height': 480,
            'width': 640,
            'image_id': 222,
            'annotations': {
                'boxes': list[list[int]], # n x 4, bounding box annotations
                'classes': list[int], # n, object category annotation of the bounding boxes
                'hois': [
                    {
                        'subject_id': 0,  # person box id (corresponding to the list of boxes above)
                        'object_id': 1,   # object box id (corresponding to the list of boxes above)
                        'action_id', 76,  # person action category
                        'hoi_id', 459,    # interaction category
                        'text': ('ride', 'skateboard') # text description of human action and object
                    }
                ]
            }
        }
    """
    imgs_anns = json.load(open(json_file, "r"))

    id_to_contiguous_id_map = {x["id"]: i for i, x in enumerate(HICO_OBJECTS)}
    action_object_to_hoi_id = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

    dataset_dicts = []
    images_without_valid_annotations = []
    for anno_dict in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, anno_dict["file_name"])
        record["height"] = anno_dict["height"]
        record["width"] = anno_dict["width"]
        record["image_id"] = anno_dict["img_id"]

        ignore_flag = False
        if len(anno_dict["annotations"]) == 0 or len(anno_dict["hoi_annotation"]) == 0:
            images_without_valid_annotations.append(anno_dict)
            continue

        boxes = [obj["bbox"] for obj in anno_dict["annotations"]]
        classes = [obj["category_id"] for obj in anno_dict["annotations"]]
        hoi_annotations = []
        for hoi in anno_dict["hoi_annotation"]:
            action_id = hoi["category_id"] - 1 # Starting from 1
            target_id = hoi["object_id"]
            object_id = id_to_contiguous_id_map[classes[target_id]]
            text = (HICO_ACTIONS[action_id]["name"], HICO_OBJECTS[object_id]["name"])
            hoi_id = action_object_to_hoi_id[text]

            # Ignore this annotation if we conduct zero-shot simulation experiments
            if zero_shot_exp and (hoi_id in ZERO_SHOT_INTERACTION_IDS):
                ignore_flag = True
                continue

            # Ignore non-interactions
            if ignore_non_interaction and action_id == 57:
                continue

            hoi_annotations.append({
                "subject_id": hoi["subject_id"],
                "object_id": hoi["object_id"],
                "action_id": action_id,
                "hoi_id": hoi_id,
                "text": text
            })

        if len(hoi_annotations) == 0 or ignore_flag:
            continue

        targets = {
            "boxes": boxes,
            "classes": classes,
            "hois": hoi_annotations,
        }

        record["annotations"] = targets
        dataset_dicts.append(record)

    if repeat_factor_sampling:
        repeat_factors = repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh=0.003)
        dataset_indices = get_dataset_indices(repeat_factors)
        dataset_dicts = [dataset_dicts[i] for i in dataset_indices]

    return dataset_dicts


def prepare_dataset_text():
    texts = []
    text_mapper = {}
    for i, hoi in enumerate(HICO_INTERACTIONS):
        action_name = " ".join(hoi["action"].split("_"))
        object_name = hoi["object"]
        s = [action_name, object_name]
        text_mapper[len(texts)] = i
        texts.append(s)
    return texts, text_mapper


def prepare_related_hois():
    ''' Gather related hois based on object names and action names
    Returns:
        object_to_related_hois (dict): {
            object_text (e.g., chair): [
                {'hoi_id': 86, 'text': ['carry', 'chair']},
                {'hoi_id': 87, 'text': ['hold', 'chair']},
                ...
            ]
        }

        action_to_relatedhois (dict): {
            action_text (e.g., carry): [
                {'hoi_id': 10, 'text': ['carry', 'bicycle']},
                {'hoi_id': 46, 'text': ['carry', 'bottle']},
                ...
            ]
        }
    '''
    object_to_related_hois = collections.defaultdict(list)
    action_to_related_hois = collections.defaultdict(list)

    for x in HICO_INTERACTIONS:
        action_text = x['action']
        object_text = x['object']
        hoi_id = x['interaction_id']
        if hoi_id in ZERO_SHOT_INTERACTION_IDS or hoi_id in NON_INTERACTION_IDS:
            continue
        hoi_text = [action_text, object_text]

        object_to_related_hois[object_text].append({'hoi_id': hoi_id, 'text': hoi_text})
        action_to_related_hois[action_text].append({'hoi_id': hoi_id, 'text': hoi_text})

    return object_to_related_hois, action_to_related_hois

def make_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])

    scales = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
            T.RandomSelect(
                T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                T.Compose([
                    T.RandomCrop_InteractionConstraint((0.75, 0.75), 0.8),
                    T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                ])
            ),
            normalize,
        ])

    if image_set == "val":
        return T.Compose([
            T.RandomResize([args.eval_size], max_size=args.eval_size * 1333 // 800),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')

    """ deprecated (Fixed image resolution + random cropping + centering)
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
            T.RandomSelect(
                T.ResizeAndCenterCrop(224),
                T.Compose([
                    T.RandomCrop_InteractionConstraint((0.7, 0.7), 0.9),
                    T.ResizeAndCenterCrop(224)
                ]),
            ),
            normalize
        ])
    if image_set == "val":
        return T.Compose([
            T.ResizeAndCenterCrop(224),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')
    """


def build(image_set, args):
    # NOTE: Replace the path to your file
    PATHS = {
        "train": (HICO_TRAIN_ROOT, HICO_TRAIN_ANNO),
        "val": (HICO_VAL_ROOT, HICO_VAL_ANNO),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = HICO(
        img_folder,
        ann_file,
        transforms=make_transforms(image_set, args),
        image_set=image_set,
        zero_shot_exp=args.zero_shot_exp,
        repeat_factor_sampling=args.repeat_factor_sampling,
        ignore_non_interaction=args.ignore_non_interaction
    )

    return dataset

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, return_tokens=False, tokenizer=None, max_query_len=256):
        self.return_masks = return_masks
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len

    def get_box_mask(self, rect, img_size, mode="poly"):
        assert mode=="poly", "Only support poly mask right now!"
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        return [[x1, y1, x1, y2, x2, y2, x2, y1]]

    def __call__(self, image, target, ignore_box_screen=False, box_format="xywh"):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None
        label_to_positions = target.get("label_to_positions", {})

        greenlight_span_for_masked_lm_objective = target.get("greenlight_span_for_masked_lm_objective", None)

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if box_format == "xywh":
            boxes[:, 2:] += boxes[:, :2] - 1  # TO_REMOVE = 1
            boxes[:, 0::2].clamp_(min=0, max=w-1)  # TO_REMOVE = 1
            boxes[:, 1::2].clamp_(min=0, max=h-1)  # TO_REMOVE = 1

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            masks = []
            is_box_mask = []
            for obj, bbox in zip(anno, boxes):
                if "segmentation" in obj:
                    masks.append(obj["segmentation"])
                    is_box_mask.append(0)
                else:
                    masks.append(self.get_box_mask(bbox, image.size, mode='poly'))
                    is_box_mask.append(1)
            masks = SegmentationMask(masks, image.size, mode='poly')
            is_box_mask = torch.tensor(is_box_mask)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        isfinal = None
        if anno and "isfinal" in anno[0]:
            isfinal = torch.as_tensor([obj["isfinal"] for obj in anno], dtype=torch.float)

        tokens_positive = [] if self.return_tokens else None
        if self.return_tokens and anno and "tokens" in anno[0]:
            tokens_positive = [obj["tokens"] for obj in anno]
        elif self.return_tokens and anno and "tokens_positive" in anno[0]:
            tokens_positive = [obj["tokens_positive"] for obj in anno]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
            is_box_mask = is_box_mask[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
            target["is_box_mask"] = is_box_mask
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if tokens_positive is not None:
            target["tokens_positive"] = []

            for i, k in enumerate(keep):
                if k or ignore_box_screen:
                    target["tokens_positive"].append(tokens_positive[i])

        if isfinal is not None:
            target["isfinal"] = isfinal

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.return_tokens and self.tokenizer is not None:
            if not ignore_box_screen:
                assert len(target["boxes"]) == len(target["tokens_positive"])
            tokenized = self.tokenizer(caption, return_tensors="pt",
                max_length=self.max_query_len,
                truncation=True)
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"])
            target['greenlight_map'] = create_greenlight_map(greenlight_span_for_masked_lm_objective,tokenized)
            target["positive_map_for_od_labels"] = create_positive_map_for_od_labels(tokenized, label_to_positions)

        original_od_label = []
        for obj in anno:
            original_od_label.append(
                obj.get("original_od_label", -10))  # NOTE: The padding value has to be not the same as -1 or -100
        target["original_od_label"] = torch.as_tensor(original_od_label)

        return image, target