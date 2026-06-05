#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
# 数据集 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414

import json
import numpy as np
import os

os.environ.setdefault('TF_KERAS', '1')
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

if not hasattr(np, '_no_nep50_warning'):
    def _no_nep50_warning(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    np._no_nep50_warning = _no_nep50_warning

from bert4keras.backend import keras, K, tf
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.models import Model
from tqdm import tqdm

Adam = keras.optimizers.Adam

maxlen = 256
epochs = 10
batch_size = 16
eval_batch_size = int(os.environ.get('GP_EVAL_BATCH_SIZE', '16'))
experiment_mode = int(os.environ.get('GP_EXPERIMENT_MODE', '2'))
if experiment_mode not in (1, 2, 3):
    raise ValueError('GP_EXPERIMENT_MODE must be 1, 2, or 3.')

if experiment_mode == 1:
    experiment_name = 'original_globalpointer'
    sparse_max_span_len = 0
    sparse_topk = 0
    sparse_loss_mask = False
elif experiment_mode == 3:
    experiment_name = 'graph_globalpointer'
    sparse_max_span_len = 0
    sparse_topk = 0
    sparse_loss_mask = False
else:
    experiment_name = 'sparse_globalpointer'
    sparse_max_span_len = int(os.environ.get('GP_SPARSE_MAX_SPAN_LEN', '128'))
    sparse_topk = int(os.environ.get('GP_SPARSE_TOPK', '512'))
    sparse_loss_mask = os.environ.get('GP_SPARSE_LOSS_MASK', '1') != '0'
graph_topk = int(os.environ.get('GP_GRAPH_TOPK', '256'))
graph_lambda = float(os.environ.get('GP_GRAPH_LAMBDA', '0.2'))
graph_isolated_penalty = float(os.environ.get('GP_GRAPH_ISOLATED_PENALTY', '0.5'))
learning_rate = 2e-5
categories = set()
TQDM_KWARGS = dict(ncols=100, mininterval=2, leave=False)

# bert配置
INPUT_DIR = os.environ.get(
    'GP_INPUT_DIR',
    '/kaggle/input/datasets/megafangyangpiy/gpgpgpgpgp'
)
OUTPUT_DIR = os.environ.get('GP_OUTPUT_DIR', '/kaggle/working')
PRETRAINED_DIR = os.path.join(
    INPUT_DIR, 'pretrained_model', 'chinese_L-12_H-768_A-12'
)
DATA_DIR = os.path.join(INPUT_DIR, 'data', 'CMeEE')
BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR, 'best_model_cmeee_%s.weights' % experiment_name
)
PREDICT_PATH = os.path.join(OUTPUT_DIR, 'CMeEE_test_%s.json' % experiment_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_path = os.path.join(PRETRAINED_DIR, 'bert_config.json')
checkpoint_path = os.path.join(PRETRAINED_DIR, 'bert_model.ckpt')
dict_path = os.path.join(PRETRAINED_DIR, 'vocab.txt')

print('experiment_mode: %s (%s)' % (experiment_mode, experiment_name))
print(
    'sparse_config: max_span_len=%s, topk=%s, loss_mask=%s' %
    (sparse_max_span_len, sparse_topk, sparse_loss_mask)
)
print(
    'graph_config: topk=%s, lambda=%s, isolated_penalty=%s' %
    (graph_topk, graph_lambda, graph_isolated_penalty)
)
print('best_model_path: %s' % BEST_MODEL_PATH)


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    for d in json.load(open(filename)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, label))
            categories.add(label)
    return D


# 标注数据
train_data = load_data(os.path.join(DATA_DIR, 'CMeEE_train.json'))
valid_data = load_data(os.path.join(DATA_DIR, 'CMeEE_dev.json'))
categories = list(sorted(categories))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def build_sparse_span_mask(seq_len):
    idxs = tf.range(seq_len)
    start = tf.expand_dims(idxs, 1)
    end = tf.expand_dims(idxs, 0)
    span_len = end - start + 1
    valid_mask = tf.cast(span_len >= 1, K.floatx())
    if sparse_max_span_len > 0:
        valid_mask *= tf.cast(span_len <= sparse_max_span_len, K.floatx())
    return tf.reshape(valid_mask, (1, 1, seq_len, seq_len))


def apply_sparse_train_mask(y_true, y_pred):
    if not sparse_loss_mask:
        return y_pred
    seq_len = tf.shape(y_pred)[-1]
    valid_mask = build_sparse_span_mask(seq_len)
    gold_mask = tf.cast(y_true > 0, K.floatx())
    valid_mask = tf.maximum(valid_mask, gold_mask)
    return y_pred - (1.0 - valid_mask) * 1e12


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    y_pred = apply_sparse_train_mask(y_true, y_pred)
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = apply_sparse_train_mask(y_true, y_pred)
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)


model = build_transformer_model(config_path, checkpoint_path)
output = GlobalPointer(len(categories), 64)(model.output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=global_pointer_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=[global_pointer_f1_score]
)


def apply_sparse_decode(scores):
    if experiment_mode == 1:
        return scores
    seq_len = scores.shape[-1]
    start = np.arange(seq_len)[:, None]
    end = np.arange(seq_len)[None, :]
    valid_mask = end >= start
    if sparse_max_span_len > 0:
        valid_mask &= (end - start + 1) <= sparse_max_span_len
    scores[:, ~valid_mask] = -np.inf
    return scores


def select_sparse_entities(scores, mapping, threshold=0):
    if experiment_mode == 3:
        return select_graph_entities(scores, mapping, threshold)

    scores = apply_sparse_decode(scores)
    scores[:, [0, -1]] = -np.inf
    scores[:, :, [0, -1]] = -np.inf
    labels, starts, ends = np.where(scores > threshold)

    if sparse_topk > 0 and len(labels) > sparse_topk:
        values = scores[labels, starts, ends]
        top_indices = np.argpartition(values, -sparse_topk)[-sparse_topk:]
        top_indices = top_indices[np.argsort(values[top_indices])[::-1]]
        labels = labels[top_indices]
        starts = starts[top_indices]
        ends = ends[top_indices]

    entities = []
    for label, start, end in zip(labels, starts, ends):
        entities.append(
            (mapping[start][0], mapping[end][-1], categories[label])
        )
    return entities


def select_graph_entities(scores, mapping, threshold=0):
    scores = scores.copy()
    seq_len = scores.shape[-1]
    start_grid = np.arange(seq_len)[:, None]
    end_grid = np.arange(seq_len)[None, :]
    valid_mask = end_grid >= start_grid
    scores[:, ~valid_mask] = -np.inf
    scores[:, [0, -1]] = -np.inf
    scores[:, :, [0, -1]] = -np.inf

    labels, starts, ends = np.where(np.isfinite(scores))
    if len(labels) == 0:
        return []

    values = scores[labels, starts, ends]
    if graph_topk > 0 and len(values) > graph_topk:
        top_indices = np.argpartition(values, -graph_topk)[-graph_topk:]
        labels = labels[top_indices]
        starts = starts[top_indices]
        ends = ends[top_indices]
        values = values[top_indices]

    order = np.argsort(values)[::-1]
    labels = labels[order]
    starts = starts[order]
    ends = ends[order]
    values = values[order]
    refined_values = values + graph_lambda * span_graph_residual(
        starts, ends, values
    )

    entities = []
    for label, start, end, value in zip(labels, starts, ends, refined_values):
        if value > threshold:
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[label])
            )
    return entities


def span_graph_residual(starts, ends, values):
    if len(values) <= 1 or graph_lambda == 0:
        return np.zeros_like(values)

    s_i = starts[:, None]
    e_i = ends[:, None]
    s_j = starts[None, :]
    e_j = ends[None, :]

    same = (s_i == s_j) & (e_i == e_j)
    contains = (s_i <= s_j) & (e_i >= e_j) & ~same
    inside = (s_i >= s_j) & (e_i <= e_j) & ~same
    overlaps = (s_i <= e_j) & (s_j <= e_i) & ~(contains | inside | same)
    same_start = (s_i == s_j) & ~same
    same_end = (e_i == e_j) & ~same

    relation = (
        0.35 * contains.astype('float32') +
        0.35 * inside.astype('float32') +
        0.20 * overlaps.astype('float32') +
        0.10 * same_start.astype('float32') +
        0.10 * same_end.astype('float32')
    )
    np.fill_diagonal(relation, 0)

    denom = relation.sum(axis=1)
    neighbor_signal = relation.dot(np.tanh(values))
    residual = np.zeros_like(values)
    valid = denom > 0
    bounded_values = np.tanh(values)
    residual[valid] = neighbor_signal[valid] / denom[valid] - bounded_values[valid]
    residual[~valid] = -graph_isolated_penalty * bounded_values[~valid]
    return residual


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        return self.recognize_batch([text], threshold=threshold)[0]

    def recognize_batch(self, texts, threshold=0):
        batch_token_ids, batch_segment_ids = [], []
        mappings, token_lengths = [], []
        for text in texts:
            tokens = tokenizer.tokenize(text, maxlen=512)
            mapping = tokenizer.rematch(text, tokens)
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            mappings.append(mapping)
            token_lengths.append(len(token_ids))

        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        batch_scores = model.predict(
            [batch_token_ids, batch_segment_ids],
            verbose=0
        )

        batch_entities = []
        for scores, mapping, token_length in zip(
            batch_scores, mappings, token_lengths
        ):
            scores = scores[:, :token_length, :token_length]
            batch_entities.append(
                select_sparse_entities(scores, mapping, threshold)
            )
        return batch_entities


NER = NamedEntityRecognizer()


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for i in tqdm(range(0, len(data), eval_batch_size), **TQDM_KWARGS):
        batch_data = data[i:i + eval_batch_size]
        batch_entities = NER.recognize_batch([d[0] for d in batch_data])
        for d, entities in zip(batch_data, batch_entities):
            R = set(entities)
            T = set([tuple(i) for i in d[1:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(BEST_MODEL_PATH)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    可以提交到 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
    """
    data = json.load(open(in_file))
    for d in tqdm(data, **TQDM_KWARGS):
        d['entities'] = []
        entities = NER.recognize(d['text'])
        for e in entities:
            d['entities'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2]
            })
    json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights(BEST_MODEL_PATH)
    # predict_to_file(os.path.join(DATA_DIR, 'CMeEE_test.json'), PREDICT_PATH)
