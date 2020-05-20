import io
import json
import logging
import os
import tarfile
import urllib.request
from urllib.error import HTTPError

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .image_encoders import InceptionResNetEncoder
from .preprocess import Tokenizer, LabelEncoder

logger = logging.getLogger(__name__)


class PersonalityCaptions:
    def __init__(self, data_dir):
        self.main_dir = os.path.join(data_dir, "main")
        self.cache_dir = os.path.join(data_dir, "cache")
        self.image_dir = os.path.join(data_dir, "images")
        self.image_url_prefix = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images"
        self.captions_url = "http://parl.ai/downloads/personality_captions/personality_captions.tgz"
        self.train_file, self.val_file, self.test_file = "train.json", "val.json", "test.json"
        self.valid_images_file = "valid_images.json"
        self.dataset_files = {"train": self.train_file, "val": self.val_file, "test": self.test_file}
        self.metadata_files = ["personalities.json", "personalities.txt"]

    def download(self):
        os.makedirs(self.main_dir, exist_ok=True)
        if not all([cf in os.listdir(self.main_dir) for cf in self.metadata_files + list(self.dataset_files.values())]):
            response = urllib.request.urlopen(self.captions_url)
            tar = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
            tar.extractall(path=self.main_dir)
            tar.close()
        hashes = []
        for file_name in self.dataset_files.values():
            with open(os.path.join(self.main_dir, file_name), "r") as f:
                data = json.load(f)
                hashes += [d["image_hash"] for d in data]
        os.makedirs(self.image_dir, exist_ok=True)
        downloaded_images = set(os.listdir(self.image_dir))
        for hash_ in tqdm(hashes, unit="img"):
            image_file_name = f"{hash_}.jpg"
            if image_file_name in downloaded_images:
                continue
            image_url = f"{self.image_url_prefix}/{hash_[:3]}/{hash_[3:6]}/{image_file_name}"
            try:
                response = urllib.request.urlopen(image_url)
                with open(os.path.join(self.image_dir, image_file_name), "wb") as f:
                    f.write(response.read())
                    downloaded_images.add(image_file_name)
            except HTTPError as e:
                print(f"HTTP Error {e.code} - {image_url}")
                continue
        with open(os.path.join(self.main_dir, self.valid_images_file), "w") as f:
            json.dump(list(downloaded_images), f)

    def load(self, split):
        file_to_load = self.dataset_files[split]
        with open(os.path.join(self.main_dir, self.valid_images_file), "r") as f:
            downloaded_images = set(json.load(f))
        with open(os.path.join(self.main_dir, file_to_load), "r") as f:
            data = json.load(f)
        data = filter(lambda d: f"{d['image_hash']}.jpg" in downloaded_images, data)
        data = map(lambda d: {
            "style": d["personality"],
            "caption": d["comment"],
            "additional_captions": d["additional_comments"] if "additional_comments" in d else [],
            "image_path": os.path.join(self.image_dir, f"{d['image_hash']}.jpg")
        }, data)
        return list(data)


class DatasetManager:
    def __init__(self, dataset, max_seq_len):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer()
        self.style_encoder = LabelEncoder()
        self.tokenizer.fit_on_texts([d["caption"] for d in self.dataset.load("train")])
        self.style_encoder.fit_on_labels([d["style"] for d in self.dataset.load("train")])

    def load_generator_dataset(self, split, batch_size, repeat):
        tf_dataset = self._load_cached_dataset(split)
        tf_dataset = tf_dataset.shuffle(buffer_size=1000)
        if split == "test":
            tf_dataset = tf_dataset.map(lambda i, c, s, ac: (
                i, tf.py_function(self.tokenizer.text_to_sequence, (c, self.max_seq_len), tf.int64),
                tf.py_function(self.style_encoder.transform, (s,), tf.int32),
                tf.py_function(self.tokenizer.texts_to_sequences, (ac, self.max_seq_len), tf.int64)
            ), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=(InceptionResNetEncoder.IMAGE_FEATURE_SHAPE,
                                                                            [None], [], [None, None]))
        else:
            tf_dataset = tf_dataset.map(lambda i, c, s, ac: (
                i, tf.py_function(self.tokenizer.text_to_sequence, (c, self.max_seq_len), tf.int64),
                tf.py_function(self.style_encoder.transform, (s,), tf.int32)
            ), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=(InceptionResNetEncoder.IMAGE_FEATURE_SHAPE,
                                                                            [None], []))
        tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tf_dataset = tf_dataset.repeat(repeat)
        return tf_dataset

    def load_discriminator_dataset(self, split, batch_size, repeat, label, sample_weight, randomize_captions):
        tf_dataset = self._load_cached_dataset(split)
        tf_dataset = tf_dataset.map(lambda i, c, s, ac: (i, c, s), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if randomize_captions:
            captions = [d["caption"] for d in self.dataset.load(split)]
            randomized_captions_dataset = tf.data.Dataset.from_tensor_slices((
                tf.convert_to_tensor(captions, dtype=tf.string)
            )).shuffle(buffer_size=len(captions))
            tf_dataset = tf.data.Dataset.zip((tf_dataset, randomized_captions_dataset))
            tf_dataset = tf_dataset.map(lambda i_c_s, rc: (i_c_s[0], rc, i_c_s[2]))
        tf_dataset = tf_dataset.map(lambda i, c, s: (
            i, tf.py_function(self.tokenizer.text_to_sequence, (c, self.max_seq_len), tf.int64),
            tf.constant(label, dtype=tf.int32, shape=(1,)), tf.constant(sample_weight, dtype=tf.float32, shape=(1,)),
            tf.py_function(self.style_encoder.transform, (s,), tf.int32)
        ), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tf_dataset = tf_dataset.shuffle(buffer_size=1000)
        tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=(InceptionResNetEncoder.IMAGE_FEATURE_SHAPE,
                                                                        [None], [None], [None], []))
        tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tf_dataset = tf_dataset.repeat(repeat)
        return tf_dataset

    def cache_dataset(self, split, batch_size, num_batches_per_shard):
        logger.info(f"-- Caching {split} split")
        data = self.dataset.load(split)
        tf_dataset = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor([d["image_path"] for d in data], dtype=tf.string),
            tf.convert_to_tensor([d["caption"] for d in data], dtype=tf.string),
            tf.convert_to_tensor([d["style"] for d in data], dtype=tf.string),
            tf.convert_to_tensor([d["additional_captions"] for d in data], dtype=tf.string)
        ))
        tf_dataset = tf_dataset.map(
            lambda i, c, s, ac: (tf.py_function(self.load_image, (i,), tf.float32), c, s, ac),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        encoder = InceptionResNetEncoder()
        num_batches = tf.data.experimental.cardinality(tf_dataset).numpy()
        num_shards = np.ceil(num_batches / num_batches_per_shard).astype(np.int64)
        for shard_id in tqdm(range(num_shards), desc="Writing TFRecord shards", unit="shard"):
            shard_dataset = tf_dataset.take(num_batches_per_shard)
            tf_dataset = tf_dataset.skip(num_batches_per_shard)
            shard_dataset = shard_dataset.map(
                lambda i, c, s, ac: (self._encode_image(encoder, i), c, s, ac),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            shard_dataset = shard_dataset.unbatch()
            shard_dataset = shard_dataset.map(lambda i, c, s, ac: tf.py_function(
                self._serialize_example, (i, c, s, ac), tf.string
            ), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            tf.data.experimental.TFRecordWriter(
                os.path.join(self.dataset.cache_dir, f"{split}-{shard_id}.tfrecord"), compression_type="ZLIB"
            ).write(shard_dataset)

    def _load_cached_dataset(self, split):
        filenames = tf.data.Dataset.list_files(os.path.join(self.dataset.cache_dir, f"{split}*.tfrecord"))
        dataset = tf.data.TFRecordDataset(filenames, compression_type="ZLIB")
        dataset = dataset.map(self._deserialize_example)
        return dataset

    @staticmethod
    @tf.function
    def _encode_image(encoder, image_batch):
        return encoder(image_batch)

    @staticmethod
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        width, height = img.shape[0], img.shape[1]
        if width > height:
            left = int((width - height) / 2)
            right = width - left
            top = 0
            bottom = height
        else:
            top = int((height - width) / 2)
            bottom = height - top
            left = 0
            right = width
        img = img[left:right, top:bottom]
        return tf.image.resize(img, [350, 350], method="bilinear", antialias=True)

    @staticmethod
    def _serialize_example(image, caption, style, additional_captions):
        def _bytes_feature(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        return tf.train.Example(features=tf.train.Features(feature={
            "image": _bytes_feature(tf.io.serialize_tensor(image)),
            "caption": _bytes_feature(caption),
            "style": _bytes_feature(style),
            "additional_captions": _bytes_feature(tf.io.serialize_tensor(additional_captions))
        })).SerializeToString()

    @staticmethod
    def _deserialize_example(example_proto):
        example = tf.io.parse_single_example(example_proto, features={
            "image": tf.io.FixedLenFeature([], tf.string),
            "caption": tf.io.FixedLenFeature([], tf.string),
            "style": tf.io.FixedLenFeature([], tf.string),
            "additional_captions": tf.io.FixedLenFeature([], tf.string)
        })
        return (
            tf.io.parse_tensor(example["image"], out_type=tf.float32),
            example["caption"],
            example["style"],
            tf.io.parse_tensor(example["additional_captions"], out_type=tf.string)
        )
