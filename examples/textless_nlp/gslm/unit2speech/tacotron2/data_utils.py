import random
import numpy as np
import torch
import torch.utils.data
import itertools
import os

from . import layers
from .utils import load_wav_to_torch, load_filepaths_and_text
from .text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextIntMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        audiopaths_file, labels_file = audiopaths_and_text
        audiopaths, labels = [], []
        with open(audiopaths_file, 'r') as f:
            root_dir = f.readline().strip()
            for l in f:
                audiopath, _ = l.strip().split('\t')
                audiopaths.append(os.path.join(root_dir, audiopath))

        with open(labels_file, 'r') as f:
            
            for l in f:
                # label = list(map( int, l.strip().split() ))
                labels.append(l.strip())
        assert len(labels) == len(audiopaths), f'audiopaths: {len(audiopaths)} != labels: {len(labels)}'
        self.audiopaths_and_text = list(zip(audiopaths, labels))

        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

        self.SOI = torch.IntTensor([hparams.vocab_size])
        self.EOI = torch.IntTensor([hparams.vocab_size+1])

        self.base_chunk_size = hparams.base_chunk_size
        self.extra_chunk_size_per_epoch = hparams.extra_chunk_size_per_epoch
        self.feature_size = hparams.feature_size
        self.epoch = 0
        self.chunk_size = self.base_chunk_size
        self.is_curriculum = hparams.is_curriculum
        # self.set_epoch(0)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        # text = self.get_text(text)
        if self.is_curriculum:
            text, start_idx, end_idx = self.get_chunked_text_and_offset(text)
            mel = self.get_chunked_mel(audiopath, start_idx, end_idx)
        else:
            text = self.get_text(text)
            mel = self.get_mel(audiopath)
        # mel = self.get_mel(audiopath)
        return (text, mel)

    def get_chunked_mel(self, filename, start_idx, end_idx):
        offset = float(start_idx * self.feature_size)
        duration = float( (end_idx - start_idx) * self.feature_size )

        audio, sampling_rate = load_wav_to_torch(filename, offset=offset, duration=duration)
        # print('sampling_rate, self.stft.sampling_rate', sampling_rate, self.stft.sampling_rate)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec
    
    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        # print('sampling_rate, self.stft.sampling_rate', sampling_rate, self.stft.sampling_rate)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.is_curriculum:
            self.chunk_size = self.base_chunk_size + self.extra_chunk_size_per_epoch * self.epoch
            print(f'Epoch: {self.epoch}, chunk_size: {self.chunk_size}')
        else:
            print(f'Epoch: {self.epoch}, full input')

    def get_chunked_text_and_offset(self, text_dup): # text contains duplicates
        text_dup = text_dup.strip().split()

        if len(text_dup) > self.chunk_size:
            start_max = len(text_dup) - self.chunk_size
            start_idx = torch.randint(0, start_max+1, size=(1,)) # 0 to start_max
            end_idx = start_idx+self.chunk_size
            text_dup_chunk = text_dup[start_idx: end_idx]
            assert len(text_dup_chunk) == self.chunk_size
        else:
            start_idx = 0
            end_idx = len(text_dup)
            text_dup_chunk = text_dup

        text_nodup = list(i for i, x in itertools.groupby(text_dup_chunk))
        text = torch.IntTensor( list(map( int, text_nodup )) )

        text = torch.cat([self.SOI, text, self.EOI])
        return text, start_idx, end_idx

    def get_text(self, text_dup):
        text_dup = text_dup.strip().split()
        text_nodup = list(i for i, x in itertools.groupby(text_dup))
        text = torch.IntTensor( list(map( int, text_nodup )) )
        text = torch.cat([self.SOI, text, self.EOI])
        return text

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
        
        obs_labels = torch.LongTensor([0])

        return text_padded, input_lengths, obs_labels, mel_padded, gate_padded, \
            output_lengths
