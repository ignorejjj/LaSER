import random
import os
from typing import List, Tuple

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from PIL import Image

from tevatron.retriever.arguments import DataArguments

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TrainDataset(Dataset):
    """
    Dataset for training which handles both query and passage data.
    Loads dataset and optional corpus from the provided paths/configurations.
    """

    def __init__(self,
                 data_args: DataArguments,
                 trainer=None,
                 dataset_name=None,
                 corpus_name=None,
                 dataset_path=None,
                 corpus_path=None,
                 corpus_assets_path=None):
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        # self.train_data = load_dataset(
        #     self.data_args.dataset_name if dataset_name is None else dataset_name,
        #     self.data_args.dataset_config,
        #     data_files=self.data_args.dataset_path if dataset_path is None else dataset_path,
        #     split=self.data_args.dataset_split,
        #     cache_dir=self.data_args.dataset_cache_dir,
        #     num_proc=self.data_args.num_proc,
        # )

        self.train_data = load_from_disk(self.data_args.dataset_name)
        # Load corpus if provided
        if self.data_args.corpus_name is None and corpus_name is None:
            self.corpus = None
        else:
            self.corpus = load_dataset(
                self.data_args.corpus_name if corpus_name is None else corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path if corpus_path is None else corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
                num_proc=self.data_args.num_proc,
            )
        
        # for video we use assets_path to load the video
        self.corpus_assets_path = corpus_assets_path if corpus_assets_path is not None else self.data_args.assets_path

        # create a map between docid and index
        self.docid_to_index = {}
        if self.corpus is not None:
            corpus_ids = self.corpus.select_columns(['docid'])
            docids = corpus_ids['docid']
            self.docid_to_index = {docid: index for index, docid in enumerate(tqdm(docids))}


    def set_trainer(self, trainer):
        """Sets the trainer for the dataset."""
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def _get_info_from_docid(self, docid, prefix):
        """
        Retrieves document information from the corpus given a docid.
        Returns:
            tuple: (formatted_text, image, video, audio)
        """
        document_info = self.corpus[self.docid_to_index[docid]]
        assert document_info['docid'] == docid
        image = document_info.get('image', None)

        video = document_info.get('video', None)
        if video is not None:
            video = os.path.join(self.corpus_assets_path, video)

        audio = document_info.get('audio', None)
        if audio is not None: # either an dict with 'array' key or a string .mp3 path
            if isinstance(audio, dict) and 'array' in audio:
                audio = audio['array']
            else:
                assert isinstance(audio, str) and audio.endswith('.mp3')
                audio = os.path.join(self.corpus_assets_path, audio)

        text = document_info.get('text', '')

        if not self.data_args.encode_text:
            text = None
        if not self.data_args.encode_image:
            image = None
        if not self.data_args.encode_video:
            video = None
        if not self.data_args.encode_audio:
            audio = None
        text = '' if text is None else text

        return prefix + text, image, video, audio

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        # Handling the legacy format with 'positive_passages'
        if 'positive_passages' in group:
            query_text = group['query']
            query_image = query_video = query_audio = None
            formatted_query = (self.data_args.query_prefix + query_text,
                               query_image, query_video, query_audio)
            formatted_documents = []
            # Select positive document
            #selected_positive = group['positive_passages'][(_hashed_seed + epoch) % len(group['positive_passages'])]
            # 收集3个
            num_thinking_steps = 2
            num_thinking_steps += 1
            psg_docs = group['positive_passages'][:num_thinking_steps]
            if len(psg_docs) < num_thinking_steps:
                psg_docs = psg_docs + [psg_docs[-1]] * (num_thinking_steps - len(psg_docs))
 
            for psg_doc in psg_docs:
                positive_text = psg_doc
                formatted_documents.append(self.data_args.passage_prefix + positive_text)
            # Select negative documents
            negative_size = self.data_args.train_group_size - 1
            if len(group['negative_passages']) >= negative_size:
                selected_negatives = random.choices(group['negative_passages'], k=negative_size)
            elif self.data_args.train_group_size == 1:
                selected_negatives = []
            else:
                offset = epoch * negative_size % len(group['negative_passages'])
                selected_negatives = list(group['negative_passages'])
                random.Random(_hashed_seed).shuffle(selected_negatives)
                selected_negatives = selected_negatives * 20
                selected_negatives = selected_negatives[offset: offset + negative_size]

            for negative in selected_negatives:
                negative_text = negative
                # negative_text = (negative['title'] + ' ' + negative['text']
                #                  if 'title' in negative else negative['text'])
                formatted_documents.append(self.data_args.passage_prefix + negative_text)

            return formatted_query, formatted_documents

        # Handling the new format
        query_id = group['query_id']
        query_text = group.get('query_text', '') or ''
        query_image = group.get('query_image', None)
        query_video = group.get('query_video', None)
        query_audio = group.get('query_audio', None)
        formatted_query = (self.data_args.query_prefix + query_text,
                           query_image, query_video, query_audio)

        formatted_documents = []
        positive_document_ids = group['positive_document_ids']
        negative_document_ids = group['negative_document_ids']

        # Select positive document id
        selected_positive_docid = positive_document_ids[(_hashed_seed + epoch) % len(positive_document_ids)]
        formatted_documents.append(
            self._get_info_from_docid(selected_positive_docid, self.data_args.passage_prefix)
        )

        # Select negative document ids
        negative_size = self.data_args.train_group_size - 1
        if len(negative_document_ids) < negative_size:
            selected_negative_docids = random.choices(negative_document_ids, k=negative_size)
        elif self.data_args.train_group_size == 1:
            selected_negative_docids = []
        else:
            offset = epoch * negative_size % len(negative_document_ids)
            selected_negative_docids = list(negative_document_ids)
            random.Random(_hashed_seed).shuffle(selected_negative_docids)
            selected_negative_docids = selected_negative_docids * 2
            selected_negative_docids = selected_negative_docids[offset: offset + negative_size]

        for neg_docid in selected_negative_docids:
            formatted_documents.append(
                self._get_info_from_docid(neg_docid, self.data_args.passage_prefix)
            )

        return formatted_query, formatted_documents


class COCOV1Dataset(Dataset):
    """
    Dataset for training which handles both query and passage data.
    Loads dataset and optional corpus from the provided paths/configurations.
    """

    def __init__(self,
                 data_args: DataArguments,
                 trainer=None,
                 dataset_name=None,
                 corpus_name=None,
                 dataset_path=None,
                 corpus_path=None,
                 corpus_assets_path=None):
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        # self.train_data = load_dataset(
        #     self.data_args.dataset_name if dataset_name is None else dataset_name,
        #     self.data_args.dataset_config,
        #     data_files=self.data_args.dataset_path if dataset_path is None else dataset_path,
        #     split=self.data_args.dataset_split,
        #     cache_dir=self.data_args.dataset_cache_dir,
        #     num_proc=self.data_args.num_proc,
        # )

        self.train_data = load_from_disk(self.data_args.dataset_name)
        # Load corpus if provided
        if self.data_args.corpus_name is None and corpus_name is None:
            self.corpus = None
        else:
            self.corpus = load_dataset(
                self.data_args.corpus_name if corpus_name is None else corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path if corpus_path is None else corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
                num_proc=self.data_args.num_proc,
            )
        
        # for video we use assets_path to load the video
        self.corpus_assets_path = corpus_assets_path if corpus_assets_path is not None else self.data_args.assets_path

        # create a map between docid and index
        self.docid_to_index = {}
        if self.corpus is not None:
            corpus_ids = self.corpus.select_columns(['docid'])
            docids = corpus_ids['docid']
            self.docid_to_index = {docid: index for index, docid in enumerate(tqdm(docids))}


    def set_trainer(self, trainer):
        """Sets the trainer for the dataset."""
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def _get_info_from_docid(self, docid, prefix):
        """
        Retrieves document information from the corpus given a docid.
        Returns:
            tuple: (formatted_text, image, video, audio)
        """
        document_info = self.corpus[self.docid_to_index[docid]]
        assert document_info['docid'] == docid
        image = document_info.get('image', None)

        video = document_info.get('video', None)
        if video is not None:
            video = os.path.join(self.corpus_assets_path, video)

        audio = document_info.get('audio', None)
        if audio is not None: # either an dict with 'array' key or a string .mp3 path
            if isinstance(audio, dict) and 'array' in audio:
                audio = audio['array']
            else:
                assert isinstance(audio, str) and audio.endswith('.mp3')
                audio = os.path.join(self.corpus_assets_path, audio)

        text = document_info.get('text', '')

        if not self.data_args.encode_text:
            text = None
        if not self.data_args.encode_image:
            image = None
        if not self.data_args.encode_video:
            video = None
        if not self.data_args.encode_audio:
            audio = None
        text = '' if text is None else text

        return prefix + text, image, video, audio

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        # Handling the legacy format with 'positive_passages'
        query_text = group['query']
        query_image = query_video = query_audio = None
        formatted_query = (self.data_args.query_prefix + query_text,
                            query_image, query_video, query_audio)
        formatted_documents = []
        # Select positive document

        psg_doc = group['positive_passages'][0]
        formatted_documents.append(self.data_args.passage_prefix + psg_doc)
        # Select negative documents
        negative_size = self.data_args.train_group_size - 1
        if len(group['negative_passages']) < negative_size:
            selected_negatives = random.choices(group['negative_passages'], k=negative_size)
        elif self.data_args.train_group_size == 1:
            selected_negatives = []
        else:
            offset = epoch * negative_size % len(group['negative_passages'])
            selected_negatives = list(group['negative_passages'])
            random.Random(_hashed_seed).shuffle(selected_negatives)
            selected_negatives = selected_negatives * 20
            selected_negatives = selected_negatives[offset: offset + negative_size]

        for negative in selected_negatives:
            negative_text = negative
            # negative_text = (negative['title'] + ' ' + negative['text']
            #                  if 'title' in negative else negative['text'])
            formatted_documents.append(self.data_args.passage_prefix + negative_text)

        return formatted_query, formatted_documents

        
