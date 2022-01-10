# import tensorflow as tf
# from .text import symbols
from easydict import EasyDict as edict


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    # hparams = tf.contrib.training.HParams(
    hparams = edict(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=1000,
        seed=6789,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=True,
        dist_backend="nccl",
        dist_url="env://",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        base_chunk_size=50,
        extra_chunk_size_per_epoch=5,

        feature_size=0.02,
        vocab_size=None,
        dataloader='TextMelLoader',

        training_audiopaths=None,
        training_labels=None,
        validation_audiopaths=None,
        validation_labels=None,
        text_cleaners=['english_cleaners'],
        num_workers=6,
        add_sos=True,
        add_eos=True,
        code_dict='',
        collapse_code=True,
        text_or_code='code',
        is_curriculum=True,

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        n_symbols=None,
        symbols_embedding_path=None,

        ################################
        # Model Parameters             #
        ################################
        # n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        lat_kernel_size=3,
        lat_n_convolutions=2,
        lat_n_filters=512,
        lat_n_blstms=2,
        lat_dim=0,
        obs_n_class=1,
        obs_dim=0,


        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,


        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        for hp in hparams_string.split(','):
            k, v = hp.strip().split('=')
            if k in ['feature_size']:
                hparams[k] = float(v)
            elif k in ['vocab_size', 'base_chunk_size', 'extra_chunk_size_per_epoch', 'n_symbols', 'num_workers', 'world_size', 'rank']:
                hparams[k] = int(v)
            else:
                hparams[k] = v
        # tf.logging.info('Parsing command line hparams: %s', hparams_string)
        # hparams.parse(hparams_string)
    
    # if verbose:
        # tf.compat.v1.logging.info(f'Final parsed hparams: {hparams}')

    return hparams
