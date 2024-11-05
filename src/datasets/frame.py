import json
import random
from collections import defaultdict
from collections.abc import Callable
from csv import DictReader
from pathlib import Path
from typing import Any, Dict
import pandas as pd

import torch
from pytorchvideo.data.video import VideoPathHandler
from torch.utils.data import Dataset


class FrameDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        model_args,
        frames_dir: str,
        annotation_file: str | None = None,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        data_filter: Callable[[dict[str, Any]], bool] | None = None,
        return_frames: bool = True,
    ) -> None:
        """
        :param frames_dir: path to dir that contains extracted frames.
            Optionally, this directory may contain narrated_actions.csv
            for annotations.
        :param annotation_file: path to annotation file. If frames_dir contains
            narrated_actions.csv, this is optional.
        :param transform: transform function to be called for each datapoint
        :param data_filter: function to be used to filter datapoints
        :param return_frames: whether to return frame data for each datapoint or not
        """
        self.frames_dir = Path(frames_dir)
        self.return_frames = return_frames
        self.data: list[dict] = []
        self.dict_data: dict[str, dict] = {}

        self.annotation_file_path = Path(annotation_file)
        assert self.annotation_file_path.exists()

        # Read the CSV file
        self.df = pd.read_csv(self.annotation_file_path, sep='\t', usecols=["text", "cmd_ad_filename"])

        # Convert each text entry into a list of dictionaries
        #self.df['captions'] = self.df['text'].apply(lambda x: [{"captions": x}])
        self.df['captions'] = self.df['text'].apply(lambda x: [{x}])

        # Filter the data if a filter is provided
        if data_filter is not None:
            self.df = self.df[self.df.apply(data_filter, axis=1)]

        # Convert the DataFrame to a list of dictionaries
        self.data = self.df.to_dict('records')
        self.dict_data = {str(row['cmd_ad_filename']): row for row in self.data}

        # Read the frame features TSV file manually
        frame_data = []
        with open(self.frames_dir, 'r') as f:
            current_id = None
            current_frames = []
    
            for line in f:
                parts = line.strip().split('\t')
                id_str = parts[0]
                frames = [json.loads(frame) for frame in parts[1:]]

                # Check if we're still on the same ID or a new one
                if id_str != current_id:
                    # If we've reached the specified number of frames, save the previous ID's data
                    if current_id is not None and len(current_frames) == model_args.num_subsample_frames:
                        frame_data.append({"id": current_id, "frames": current_frames})
            
                    # Reset for the new ID
                    current_id = id_str
                    current_frames = []

                # Append frames according to num_subsample_frames setting
                if model_args.num_subsample_frames == 16:
                    # Load all frames up to 16
                    if len(current_frames) < 16:
                        current_frames.extend(frames[:16 - len(current_frames)])
                elif model_args.num_subsample_frames == 8:
                    # Load every other frame, selecting 8 in total
                    if len(current_frames) < 8:
                        current_frames.extend(frames[::2][:8])

            # Save the last ID's data
            if len(current_frames) == model_args.num_subsample_frames:
                frame_data.append({"id": current_id, "frames": current_frames})

        self.df_frames = pd.DataFrame(frame_data)
        
        # Convert the DataFrame to a list of dictionaries
        self.data_frames = self.df_frames.to_dict('records')
        self.dict_data_frames = {str(row['id']): row for row in self.data_frames}

        self._transform = transform

    def __getitem__(self, index: int | str) -> Dict[str, Any]:
        if isinstance(index, int):
            datapoint = self.data[index]
        else:
            datapoint = self.dict_data[str(index)]
        item = {**datapoint}

        if self.return_frames:
            frame_datapoint = self.dict_data_frames.get(str(datapoint['id']), None)
            if frame_datapoint is None:
                raise FileNotFoundError(f"Frame data not found for ID: {datapoint['id']}")
            frames = torch.tensor(frame_datapoint['frames'])  # Assuming frames are saved as JSON-encoded lists
            item["frames"] = frames

        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self.data)


class FrameInterleavedDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        frames_dir: str,
        annotation_file: str,
        in_context_example_frames_dir: str | None = None,
        in_context_example_annotation_file: str | None = None,
        num_in_context_examples_per_sample: int = 4,
        transform: Callable[[Dict], Any] | None = None,
        return_frames: bool = True,
        random_in_context_examples: bool = False,
        target_dataset_len: int | None = None,
    ) -> None:
        """
        :param frames_dir: path to dir that contains extracted frames.
        :param annotation_file: path to annotation file (TSV format).
        :param in_context_example_frames_dir: path to dir that contains
            extracted frames for in-context examples.
        :param in_context_example_annotation_file: path to annotation file for
            in-context examples (TSV format).
        :param num_in_context_examples_per_sample: number of in-context examples to
            sample per datapoint
        :param transform: transform function to be called for each datapoint
        :param return_frames: whether to return frame data for each datapoint or not
        :param random_in_context_examples: whether to sample random in-context examples
            or not
        :param target_dataset_len: if given, we upsample datapoints to match target_dataset_len
        """
        self.num_in_context_examples_per_sample = num_in_context_examples_per_sample
        self.return_frames = return_frames
        self.random_in_context_examples = random_in_context_examples
        self.target_dataset_len = target_dataset_len
        self._dataset = FrameDataset(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            return_frames=return_frames,
        )
        if self.target_dataset_len is not None and self.target_dataset_len > len(
            self._dataset
        ):
            num_to_sample = self.target_dataset_len - len(self._dataset)
            sampled_data = random.choices(self._dataset.data, k=num_to_sample)
            self._dataset.data.extend(sampled_data)
            self._dataset.dict_data.update({d['id']: d for d in sampled_data})
        
        if in_context_example_frames_dir is None:
            self.in_context_examples_from_main_dataset = True
            self._in_context_dataset = self._dataset
        else:
            self.in_context_examples_from_main_dataset = False
            self._in_context_dataset = FrameDataset(
                in_context_example_frames_dir,
                annotation_file=in_context_example_annotation_file,
                return_frames=return_frames,
            )

        self._transform = transform

    def _sample_in_context_examples(self, index: int) -> set[int]:
        all_indices = set(range(len(self._in_context_dataset)))
        if self.in_context_examples_from_main_dataset:
            all_indices.remove(index)
        return set(random.sample(all_indices, self.num_in_context_examples_per_sample))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        datapoint = self._dataset[index]

        if self.random_in_context_examples:
            examples = self._sample_in_context_examples(index)
        else:
            examples = self._sample_in_context_examples(index)  # Fallback to random if no specific method

        item = {
            "items": [self._in_context_dataset[i] for i in examples] + [datapoint]
        }

        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self._dataset)


class FrameInterleavedPresampledDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        frames_dir: str,
        in_context_query_map_file_path: str,
        in_context_example_frames_dir: str,
        annotation_file: str | None = None,
        in_context_example_annotation_file: str | None = None,
        transform: Callable[[Dict], Any] | None = None,
        return_frames: bool = True,
        shuffle_in_context_example_frames: bool = False,
    ) -> None:
        """
        :param frames_dir: path to dir that contains extracted frames.
        :param in_context_query_map_file_path: path to file that specifies
            the mapping between in-context examples and queries.
        :param in_context_example_frames_dir: path to dir that contains
            extracted frames for in-context examples.
        :param annotation_file: path to annotation file (TSV format).
        :param in_context_example_annotation_file: path to annotation file for
            in-context examples (TSV format).
        :param transform: transform function to be called for each datapoint
        :param return_frames: whether to return frame data for each datapoint or not
        :param shuffle_in_context_example_frames: shuffle video frames of in-context
            examples. This option actually generates "permutations with no fixed points"
            or "derangements" (https://en.wikipedia.org/wiki/Derangement).
            Useful for ablation studies.
        """
        self.return_frames = return_frames
        self.shuffle_in_context_example_frames = shuffle_in_context_example_frames
        self._transform = transform
        self._dataset = FrameDataset(
            frames_dir, annotation_file=annotation_file, return_frames=return_frames
        )
        self._in_context_dataset = FrameDataset(
            in_context_example_frames_dir,
            annotation_file=in_context_example_annotation_file,
            return_frames=return_frames,
        )
        self._in_context_query_map: list[Dict[str, Any]] = []
        with open(in_context_query_map_file_path) as f:
            for line in f:
                self._in_context_query_map.append(json.loads(line))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        in_context_query = self._in_context_query_map[index]
        in_context_examples = [
            self._in_context_dataset[in_context_example]
            for in_context_example in in_context_query["context"]
        ]
        if self.shuffle_in_context_example_frames:
            video_idx = list(range(len(in_context_examples)))
            shuffled_video_idx = video_idx[:]
            while True:
                random.shuffle(shuffled_video_idx)
                for a, b in zip(video_idx, shuffled_video_idx):
                    if a == b:
                        break
                else:
                    break
            shuffled_videos = [
                in_context_examples[idx]["frame"] for idx in shuffled_video_idx
            ]
            for example, frames in zip(in_context_examples, shuffled_videos):
                example["frame"] = frames
        item = {
            "items": in_context_examples + [self._dataset[in_context_query["query"]]]
        }
        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self._in_context_query_map)