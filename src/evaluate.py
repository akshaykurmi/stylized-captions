import logging

import tensorflow as tf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from tqdm import tqdm

from .models import Encoder, Generator
from .transformer import TransformerGenerator
from .utils import MultiCheckpointManager

logger = logging.getLogger(__name__)


def generate_captions_for_image(args, dataset_manager, checkpoint_number):
    encoder = Encoder()
    generator = Generator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                          style_vocab_size=dataset_manager.style_encoder.num_classes,
                          style_embedding_units=args.generator_style_embedding_units,
                          token_embedding_units=args.generator_token_embedding_units,
                          lstm_units=args.generator_lstm_units,
                          lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units,
                          encoder_units=args.generator_encoder_units,
                          z_units=args.generator_z_units, stylize=args.stylize)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
        "generator": {"generator": generator}
    })
    checkpoint_manager.restore({"generator": checkpoint_number})

    image_path = input("Enter image path : ")
    style = input("Enter style: ")

    image = dataset_manager.load_image(image_path)
    encoder_output = encoder(tf.expand_dims(image, axis=0))
    style = tf.constant(dataset_manager.style_encoder.label_to_index[style], dtype=tf.int32, shape=(1,))

    sequences, sequences_logits = generator.beam_search(encoder_output, style, sequence_length=args.max_seq_len,
                                                        beam_size=5, sos=dataset_manager.tokenizer.start_id,
                                                        eos=dataset_manager.tokenizer.end_id)
    logger.info("-- Beam Search")
    for seq, logit in zip(sequences.numpy()[0], sequences_logits.numpy()[0]):
        logger.info(f"Logit: {logit:0.5f} | Seq: {_seq_to_text(dataset_manager, seq)}")

    logger.info("-- Random Sampling")
    initial_sequence = tf.ones((1, 1), dtype=tf.int64) * dataset_manager.tokenizer.start_id
    sequences = generator.sample(encoder_output, initial_sequence, style,
                                 sequence_length=args.max_seq_len, mode="stochastic", n_samples=5,
                                 training=False, sos=dataset_manager.tokenizer.start_id,
                                 eos=dataset_manager.tokenizer.end_id)[0]
    for seq in sequences:
        logger.info(f"Seq: {_seq_to_text(dataset_manager, seq.numpy()[0])}")


def score_on_test_set(args, dataset_manager, checkpoint_numbers):
    for checkpoint_number in checkpoint_numbers:
        logger.info(f"-- Evaluating checkpoint {checkpoint_number}")
        generator = TransformerGenerator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                                         style_vocab_size=dataset_manager.style_encoder.num_classes,
                                         model_dim=512, style_dim=64, pffn_dim=2048, z_dim=512,
                                         encoder_blocks=2, decoder_blocks=6, num_attention_heads=8, max_pe=64,
                                         dropout=0.1, stylize=True)
        checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
            "generator": {"generator": generator}
        })
        checkpoint_manager.restore({"generator": checkpoint_number})

        num_test_samples = 10000
        batch_size = 32
        test_dataset = dataset_manager.load_generator_dataset("test", batch_size, 1)

        ground_truths = {}
        predictions = {}
        sample_id = 0
        logger.info("-- Generating predictions")
        for batch in tqdm(test_dataset, desc="Batch", unit="batch", total=int(num_test_samples / batch_size) + 1):
            encoder_output, caption, style, additional_captions = batch
            sequences, sequences_logits = _run_beam_search(generator, encoder_output, style, 5, args.max_seq_len,
                                                           dataset_manager.tokenizer.start_id,
                                                           dataset_manager.tokenizer.end_id)
            for s, c, acs in zip(sequences.numpy(), caption.numpy(), additional_captions.numpy()):
                pred = _seq_to_text(dataset_manager, s[0])
                gts = [_seq_to_text(dataset_manager, ac) for ac in acs]
                gts.append(_seq_to_text(dataset_manager, c))
                predictions[sample_id] = [pred]
                ground_truths[sample_id] = gts
                sample_id += 1

        scores = _compute_scores(ground_truths, predictions)
        logger.info("========================================")
        logger.info(f"-- Run ID: {args.run_id} | Checkpoint Number: {checkpoint_number}")
        for metric, score in scores.items():
            logger.info(f"-- {metric}: {score:0.5f}")
        logger.info("========================================")


def human_performance_on_test_set(dataset_manager):
    num_test_samples = 10000
    batch_size = 32
    test_dataset = dataset_manager.load_generator_dataset("test", batch_size, 1)

    ground_truths = {}
    predictions = {}
    sample_id = 0
    logger.info("-- Generating predictions")
    for batch in tqdm(test_dataset, desc="Batch", unit="batch", total=int(num_test_samples / batch_size) + 1):
        encoder_output, caption, style, additional_captions = batch
        for c, acs in zip(caption.numpy(), additional_captions.numpy()):
            gts = [_seq_to_text(dataset_manager, ac) for ac in acs]
            pred = _seq_to_text(dataset_manager, c)
            predictions[sample_id] = [pred]
            ground_truths[sample_id] = gts
            sample_id += 1

    scores = _compute_scores(ground_truths, predictions)
    logger.info("========================================")
    logger.info(f"-- Human Performance")
    for metric, score in scores.items():
        logger.info(f"-- {metric}: {score:0.5f}")
    logger.info("========================================")


def _compute_scores(ground_truths, predictions):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]
    evaluation = {}
    for scorer, method in scorers:
        logger.info(f"-- Computing {scorer.method()} score")
        score, scores = scorer.compute_score(ground_truths, predictions)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                evaluation[m] = sc
        else:
            evaluation[method] = score
    return evaluation


@tf.function
def _run_beam_search(generator, encoder_output, style, beam_size, sequence_length, sos, eos):
    return generator.beam_search(encoder_output, style, sequence_length=sequence_length,
                                 beam_size=beam_size, sos=sos, eos=eos)


def _seq_to_text(dataset_manager, seq):
    text = dataset_manager.tokenizer.sequences_to_texts(seq[seq > 0]).numpy()
    return " ".join([t.decode("utf-8") for t in text][1:-1])
