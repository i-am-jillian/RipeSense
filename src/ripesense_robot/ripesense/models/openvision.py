"""Local CLIP image/text inference from pinned, quantized ONNX artifacts.

This is general visual similarity, not a trained ripeness or food-safety model.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image


ITEMS = ["apple","avocado","banana","tomato","orange","lime","lemon","grape",
         "guava","jujube","pomegranate","strawberry","kiwi","persimmon","papaya",
         "mango","peach","pear","plum","apricot","cherry","blueberry","raspberry",
         "blackberry","pineapple","watermelon","cantaloupe","dragon fruit","passion fruit",
         "fig","date fruit","cucumber","pepper","eggplant","carrot","broccoli","potato"]
NON_PRODUCE = ["a computer keyboard", "a laptop computer", "a person or human face",
               "a human hand", "a plastic bottle", "an empty room", "an empty tabletop",
               "a plate of cooked food", "a package of food", "a toy", "a phone", "a blank image"]


def normalize_rows(array):
    return array / np.maximum(np.linalg.norm(array,axis=-1,keepdims=True),1e-9)


class OpenVision:
    def __init__(self, root):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        root = Path(root)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.vision = ort.InferenceSession(str(root/"vision_model_quantized.onnx"),
                                           sess_options=options,providers=["CPUExecutionProvider"])
        self.text = ort.InferenceSession(str(root/"text_model_quantized.onnx"),
                                         sess_options=options,providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(root/"tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=77)
        self.tokenizer.enable_padding(pad_id=49407,pad_token="<|endoftext|>")
        self.text_lock = threading.Lock()
        config = json.loads((root/"preprocessor_config.json").read_text())
        self.mean = np.array(config["image_mean"],np.float32)
        self.std = np.array(config["image_std"],np.float32)
        self.catalog_labels = ITEMS + NON_PRODUCE
        self.catalog = self.encode_text(tuple(f"a photo of {item}" for item in self.catalog_labels))

    @lru_cache(maxsize=48)
    def encode_text(self, captions):
        # The tokenizer's batch padding configuration is shared between calls.
        with self.text_lock:
            encoded = self.tokenizer.encode_batch(list(captions))
        ids = np.array([e.ids for e in encoded],np.int64)
        mask = np.array([e.attention_mask for e in encoded],np.int64)
        available = {"input_ids":ids,"attention_mask":mask}
        inputs = {i.name:available[i.name] for i in self.text.get_inputs()}
        embeddings = self.text.run(["text_embeds"],inputs)[0]
        return normalize_rows(embeddings)

    def encode_image(self,bgr):
        image = Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        w,h = image.size
        size = (224,int(h*224/w)) if w<h else (int(w*224/h),224)
        image = image.resize(size,Image.Resampling.BICUBIC)
        x,y = (size[0]-224)//2,(size[1]-224)//2
        image = image.crop((x,y,x+224,y+224))
        array = (np.asarray(image).astype(np.float32)/255-self.mean)/self.std
        array = array.transpose(2,0,1)[None]
        vector = self.vision.run(["image_embeds"],{"pixel_values":array})[0]
        return normalize_rows(vector)[0]

    def identify(self,vector):
        similarities = self.catalog @ vector
        indices = np.argsort(similarities)[::-1]
        best = int(indices[0])
        return {"item":self.catalog_labels[best], "is_produce":best<len(ITEMS),
                "similarity":float(similarities[best]),
                "margin":float(similarities[best]-similarities[indices[1]]),
                "candidates":[{"item":self.catalog_labels[int(i)],"similarity":float(similarities[i])}
                              for i in indices[:5]]}

    def assess_description(self,vector,item):
        stages = ["unripe","ripe","overripe","visible_deterioration"]
        captions = (f"a photo of an unripe {item}",f"a photo of a ripe fresh {item}",
                    f"a photo of an overripe {item}",f"a photo of a rotten moldy {item}")
        similarities = self.encode_text(captions) @ vector
        order = np.argsort(similarities)[::-1]
        # These percentages are relative matches among these four captions.
        # They are explicitly not probabilities of ripeness or correctness.
        weights = np.exp(20*(similarities-similarities.max()))
        weights /= weights.sum()
        return {"stage":stages[int(order[0])],
                "margin":float(similarities[order[0]]-similarities[order[1]]),
                "score":float(weights[order[0]]),
                "probabilities":dict(zip(stages,map(float,weights))),
                "caption_similarities":dict(zip(stages,map(float,similarities)))}
