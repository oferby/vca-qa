import threading

import qa.bert_model.tokenization as tokenization
import tensorflow as tf
import qa.bert_model.optimization as optimization
import qa.bert_model.modeling as modeling

BERT_BASE_DIR = './bert_model/data/uncased_L-12_H-768_A-12/'
CONFIG_FILE = BERT_BASE_DIR + 'bert_config.json'
INIT_CHECKPOINT = BERT_BASE_DIR + 'bert_model.ckpt'
LEARNING_RATE = 2e-5
WARMUP_PROPORTION = 0.1
# OUTPUT_DIR = '/tmp/vca-qa/'
OUTPUT_DIR = '/tmp/bert2/'

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class Predictor(threading.Thread):
    def __init__(self, main_q, thread_q):
        self.thread_q = thread_q
        self.main_q = main_q
        self.bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
        self.label_list = ['0', '1']
        self.tokenizer = tokenization.FullTokenizer(BERT_BASE_DIR + 'vocab.txt')
        super(Predictor, self).__init__()

    def run(self):
        self.predict()

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @staticmethod
    def convert_single_example(example, label_list, max_seq_length,
                               tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            Predictor._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.label is not None:
            label_id = label_map[example.label]
        else:
            label_id = label_map['0']
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature

    def input_fn_builder(self, max_seq_len):
        def gen():
            while True:
                request = self.main_q.get(True)
                question = request.question
                label_id_list = []
                input_ids_list = []
                input_mask_list = []
                segment_ids_list = []
                for paragraph in request.paragraphs:
                    example = InputExample(paragraph, question)
                    features = Predictor.convert_single_example(example, self.label_list, max_seq_len, self.tokenizer)
                    label_id_list.append(features.label_id)
                    input_ids_list.append(features.input_ids)
                    input_mask_list.append(features.input_mask)
                    segment_ids_list.append(features.segment_ids)

                yield {'label_id': label_id_list,
                       'input_ids': input_ids_list,
                       'input_mask': input_mask_list,
                       'segment_ids': segment_ids_list}

        def input_fn(params):

            return (tf.data.Dataset.from_generator(gen,
                                                   output_types={'input_ids': tf.int32, 'input_mask': tf.int32,
                                                                 'segment_ids': tf.int32, 'label_id': tf.int32},
                                                   output_shapes={
                                                       'label_id': (4),
                                                       'input_ids': (4, max_seq_len),
                                                       'input_mask': (4, max_seq_len),
                                                       'segment_ids': (4, max_seq_len)}
                                                   ).prefetch(10))

        return input_fn

    @staticmethod
    def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities)

    @staticmethod
    def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps, use_tpu,
                         use_one_hot_embeddings):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_id"]
            if "is_real_example" in features:
                is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
            else:
                is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, probabilities) = Predictor.create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None
            if init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                if use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    accuracy = tf.metrics.accuracy(
                        labels=label_ids, predictions=predictions, weights=is_real_example)
                    loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                    return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                    }

                eval_metrics = (metric_fn,
                                [per_example_loss, label_ids, logits, is_real_example])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"probabilities": probabilities},
                    scaffold_fn=scaffold_fn)
            return output_spec

        return model_fn

    def get_estimator(self, num_train_steps=None, num_warmup_steps=None):
        model_fn = Predictor.model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(self.label_list),
            init_checkpoint=INIT_CHECKPOINT,
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False,
            use_one_hot_embeddings=False)

        run_config = tf.contrib.tpu.RunConfig(
            model_dir=OUTPUT_DIR,
            save_checkpoints_steps=True)

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=FLAGS.predict_batch_size)

        return estimator

    def predict(self):
        estimator = self.get_estimator()
        predict_input_fn = self.input_fn_builder(FLAGS.max_seq_length)
        for result in estimator.predict(input_fn=predict_input_fn, yield_single_examples=False):
            print(result)
            self.thread_q.put(result)
