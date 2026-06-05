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

from bert4keras.backend import keras, K
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
    OUTPUT_DIR, 'best_model_cmeee_globalpointer.weights'
)
PREDICT_PATH = os.path.join(OUTPUT_DIR, 'CMeEE_test.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_path = os.path.join(PRETRAINED_DIR, 'bert_config.json')
checkpoint_path = os.path.join(PRETRAINED_DIR, 'bert_model.ckpt')
dict_path = os.path.join(PRETRAINED_DIR, 'vocab.txt')


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


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
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
            scores[:, [0, -1]] -= np.inf
            scores[:, :, [0, -1]] -= np.inf
            entities = []
            for l, start, end in zip(*np.where(scores > threshold)):
                entities.append(
                    (mapping[start][0], mapping[end][-1], categories[l])
                )
            batch_entities.append(entities)
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
