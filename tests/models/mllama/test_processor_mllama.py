# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import tempfile
import unittest
from typing import Optional

import numpy as np

from transformers import MllamaProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class MllamaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MllamaProcessor

    def setUp(self):
        self.checkpoint = "hf-internal-testing/mllama-11b"
        processor = MllamaProcessor.from_pretrained(self.checkpoint)
        self.image1 = Image.new("RGB", (224, 220))
        self.image2 = Image.new("RGB", (512, 128))
        self.image_token = processor.image_token
        self.image_token_id = processor.image_token_id
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.bos_token = processor.bos_token
        self.bos_token_id = processor.tokenizer.bos_token_id
        self.tmpdirname = tempfile.mkdtemp()
        processor.save_pretrained(self.tmpdirname)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "What do these images show?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The first image shows the statue of Liberty in New York."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And who is that?"},
                ],
            },
        ]
        processor = MllamaProcessor.from_pretrained(self.tmpdirname)
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        expected_rendered = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "<|image|><|image|>What do these images show?"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "The first image shows the statue of Liberty in New York."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "And who is that?"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        self.assertEqual(rendered, expected_rendered)

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "This is a test sentence."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a response."},
                ],
            },
        ]
        input_ids = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        expected_ids = [
            [
                128000,  # <|begin_of_text|>
                128006,  # <|start_header_id|>
                9125,  # "system"
                128007,  # <|end_of_header|>
                271,  # "\n\n"
                2028,
                374,
                264,
                1296,
                11914,
                13,  # "This is a test sentence."
                128009,  # <|eot_id|>
                128006,  # <|start_header_id|>
                882,  # "user"
                128007,  # <|end_of_header|>
                271,  # "\n\n"
                2028,
                374,
                264,
                2077,
                13,  # "This is a response.",
                128009,  # <|eot_id|>
                128006,  # <|start_header_id|>
                78191,  # "assistant"
                128007,  # <|end_of_header|>
                271,  # "\n\n"
            ]
        ]

        self.assertEqual(input_ids, expected_ids)

        # test image in multiple locations
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in two sentences"},
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": " Test sentence   "},
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "ok\n"},
                ],
            }
        ]

        rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        expected_rendered = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Describe this image in two sentences<|image|> Test sentence   <|image|>ok\n<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        self.assertEqual(rendered, expected_rendered)

        input_ids = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        # fmt: off
        expected_ids = [[
            128000, 128006, 882, 128007, 271, 75885, 420, 2217, 304, 1403, 23719, 128256,
            3475, 11914, 262, 128256, 564, 198, 128009, 128006, 78191, 128007, 271,
        ]]
        # fmt: on
        self.assertEqual(input_ids, expected_ids)

        # text format for content
        messages_list = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in two sentences"},
                ],
            }
        ]
        messages_str = [
            {
                "role": "user",
                "content": "<|image|>Describe this image in two sentences",
            }
        ]

        rendered_list = processor.apply_chat_template(messages_list, add_generation_prompt=True, tokenize=False)
        rendered_str = processor.apply_chat_template(messages_str, add_generation_prompt=True, tokenize=False)
        self.assertEqual(rendered_list, rendered_str)

    def test_process_interleaved_images_prompts_image_splitting(self):
        processor = MllamaProcessor.from_pretrained(self.tmpdirname)
        # Test that a single image is processed correctly
        inputs = processor(images=self.image2, size={"width": 224, "height": 224})
        self.assertEqual(inputs["pixel_values"].shape, (1, 1, 4, 3, 224, 224))

        # Test that text is processed correctly
        text = "<|begin_of_text|>This is a test sentence.<|end_of_text|>"
        inputs = processor(text=text)
        expected_ids = [128000, 2028, 374, 264, 1296, 11914, 13, 128001]
        self.assertEqual(inputs["input_ids"][0], expected_ids)
        self.assertEqual(inputs["attention_mask"][0], [1] * len(expected_ids))
        self.assertEqual(inputs.get("cross_attention_mask"), None)

        # Test a single sample with image and text
        image_str = "<|image|>"
        text_str = "This is a test sentence."
        text = image_str + text_str
        inputs = processor(
            text=text,
            images=self.image1,
            size={"width": 128, "height": 128},
        )
        expected_ids = [self.image_token_id, self.bos_token_id] + [2028, 374, 264, 1296, 11914, 13]

        self.assertEqual(inputs["pixel_values"].shape, (1, 1, 4, 3, 128, 128))
        self.assertEqual(inputs["input_ids"][0], expected_ids)
        self.assertEqual(inputs["attention_mask"][0], [1] * len(expected_ids))
        cross_attention_mask = inputs["cross_attention_mask"]
        self.assertEqual(cross_attention_mask.shape, (1, 8, 1, 4))
        self.assertTrue(
            np.all(cross_attention_mask == 1), f"Cross attention mask is not all ones: {cross_attention_mask}"
        )

        # Test batch
        text = [
            "<|image|>This is a test sentence.",
            "This is a test sentence.<|image|><|image|>This is a test sentence.",
        ]
        # fmt: off
        expected_ids = [
            [self.image_token_id, self.bos_token_id, 2028, 374, 264, 1296, 11914, 13],
            [self.bos_token_id, 2028, 374, 264, 1296, 11914, 13, self.image_token_id, self.image_token_id, 2028, 374, 264, 1296, 11914, 13],
        ]
        # fmt: onn
        images = [[self.image1], [self.image1, self.image2]]
        inputs = processor(text=text, images=images, padding=True, size={"width": 256, "height": 256})

        self.assertEqual(inputs["pixel_values"].shape, (2, 2, 4, 3, 256, 256))
        for input_ids_i, attention_mask_i, expected_ids_i in zip(inputs["input_ids"], inputs["attention_mask"], expected_ids):
            pad_ids = [id for id, m in zip(input_ids_i, attention_mask_i) if m == 0]
            input_ids = [id for id, m in zip(input_ids_i, attention_mask_i) if m == 1]
            self.assertEqual(input_ids, expected_ids_i)
            self.assertEqual(pad_ids, [self.pad_token_id] * len(pad_ids))

        cross_attention_mask = inputs["cross_attention_mask"]
        self.assertEqual(cross_attention_mask.shape, (2, 15, 2, 4))

        # Check that only first tile of first sample is attended to all text tokens
        first_sample_mask = cross_attention_mask[0].copy()
        first_image_first_tile_attention = first_sample_mask[:, :1, :1]  # text tokens, images, tiles
        self.assertTrue(np.all(first_image_first_tile_attention == 1), f"Cross attention mask is not all ones: {first_image_first_tile_attention}")

        # zero out first tile of first image
        first_image_first_tile_attention[:, :1, :1] = 0
        self.assertTrue(np.all(first_image_first_tile_attention == 0), f"Cross attention mask is not all zeros: {first_image_first_tile_attention}")

        # second sample
        second_sample_mask = cross_attention_mask[1].copy()
        first_image_first_tile_attention = second_sample_mask[7:, :1, :1]  # text tokens, images, tiles
        self.assertTrue(np.all(first_image_first_tile_attention == 1), f"Cross attention mask is not all ones: {first_image_first_tile_attention}")

        second_image_two_tiles_attention = second_sample_mask[8:, 1:2, :2]  # text tokens, images, tiles
        self.assertTrue(np.all(second_image_two_tiles_attention == 1), f"Cross attention mask is not all ones: {second_image_two_tiles_attention}")

        # zero out both images masks
        second_sample_mask[7:, :1, :1] = 0
        second_sample_mask[8:, 1:2, :2] = 0
        self.assertTrue(np.all(second_sample_mask == 0), f"Cross attention mask is not all zeros: {second_sample_mask}")

    def test_process_interleaved_images_prompts_image_error(self):
        text = [
            "This is a test sentence.",
            "In this other sentence we try some good things",
        ]
        processor = MllamaProcessor.from_pretrained(self.tmpdirname)
        inputs = processor(text=text, images=None, padding=True)
        self.assertIsNotNone(inputs["input_ids"])

        text = [
            "This is a test sentence.<|image|>",
            "In this other sentence we try some good things",
        ]
        with self.assertRaises(ValueError):
            processor(text=text, images=None, padding=True)

        images = [[self.image1], []]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)

        text = [
            "This is a test sentence.<|image|>",
            "In this other sentence we try some good things<|image|>",
        ]
        with self.assertRaises(ValueError):
            processor(text=text, images=None, padding=True)

        text = [
            "This is a test sentence.<|image|>",
            "In this other sentence we try some good things<|image|>",
        ]
        images = [[self.image1], [self.image2]]
        inputs = processor(text=text, images=images, padding=True)

        images = [[self.image1, self.image2], []]
        with self.assertRaises(ValueError):
            processor(text=text, images=None, padding=True)

    # Override as MllamaProcessor needs image tokens in prompts
    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer <|image|>"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer <|image|>"]
        return ["lower newer <|image|>", "<|image|> upper older longer string"] + ["<|image|> lower newer"] * (
            batch_size - 2
        )
