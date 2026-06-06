#! -*- coding: utf-8 -*-
# 鐢℅lobalPointer鍋氫腑鏂囧懡鍚嶅疄浣撹瘑鍒?# 鏁版嵁闆?https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414

import atexit
import builtins
import datetime
import json
import numpy as np
import os
import sys

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


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


RUN_NAME = os.environ.get('GP_RUN_NAME', 'baseline').strip() or 'baseline'

# bert閰嶇疆
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
RUN_START_TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_LOG_PATH = os.environ.get(
    'GP_RUN_LOG_PATH',
    os.path.join(OUTPUT_DIR, '%s_run_%s.log' % (RUN_NAME, RUN_START_TIME))
)
METRICS_LOG_PATH = os.environ.get(
    'GP_METRICS_LOG_PATH',
    os.path.join(OUTPUT_DIR, '%s_metrics_%s.jsonl' % (
        RUN_NAME, RUN_START_TIME
    ))
)
SAVE_FILE_LOGS = not env_flag('GP_DISABLE_FILE_LOG', False)


class TeeStream(object):
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], 'encoding', 'utf-8')

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            if data.endswith('\n'):
                stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            getattr(stream, 'isatty', lambda: False)()
            for stream in self.streams
        )


def setup_file_logging():
    if not SAVE_FILE_LOGS:
        return None
    log_dir = os.path.dirname(RUN_LOG_PATH)
    metrics_dir = os.path.dirname(METRICS_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    log_file = builtins.open(
        RUN_LOG_PATH,
        'a',
        encoding='utf-8',
        buffering=1
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def close_file_logging():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(close_file_logging)
    print('run_log_path: %s' % RUN_LOG_PATH)
    print('metrics_log_path: %s' % METRICS_LOG_PATH)
    return log_file


RUN_LOG_FILE = setup_file_logging()

config_path = os.path.join(PRETRAINED_DIR, 'bert_config.json')
checkpoint_path = os.path.join(PRETRAINED_DIR, 'bert_model.ckpt')
dict_path = os.path.join(PRETRAINED_DIR, 'vocab.txt')


def load_data(filename):
    """鍔犺浇鏁版嵁
    鍗曟潯鏍煎紡锛歔text, (start, end, label), (start, end, label), ...]锛?              鎰忓懗鐫€text[start:end + 1]鏄被鍨嬩负label鐨勫疄浣撱€?    """
    D = []
    for d in json.load(open(filename)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, label))
            categories.add(label)
    return D


# 鏍囨敞鏁版嵁
train_data = load_data(os.path.join(DATA_DIR, 'CMeEE_train.json'))
valid_data = load_data(os.path.join(DATA_DIR, 'CMeEE_dev.json'))
categories = list(sorted(categories))
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """鏁版嵁鐢熸垚鍣?    """
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
    """缁橤lobalPointer璁捐鐨勪氦鍙夌喌
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """缁橤lobalPointer璁捐鐨凢1
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
    """鍛藉悕瀹炰綋璇嗗埆鍣?    """
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


def to_jsonable(value):
    if isinstance(value, dict):
        return dict((key, to_jsonable(val)) for key, val in value.items())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_metrics_log(epoch, train_logs, metrics, best_val_f1):
    if not SAVE_FILE_LOGS:
        return
    record = {
        'epoch': epoch + 1,
        'run_name': RUN_NAME,
        'train': to_jsonable(train_logs or {}),
        'valid': to_jsonable(metrics),
        'best_val_f1': best_val_f1,
        'run_log_path': RUN_LOG_PATH,
        'metrics_log_path': METRICS_LOG_PATH
    }
    with builtins.open(METRICS_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def evaluate(data):
    """璇勬祴鍑芥暟
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
    return {
        'overall': (f1, precision, recall)
    }


class Evaluator(keras.callbacks.Callback):
    """璇勪及涓庝繚瀛?    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(valid_data)
        f1, precision, recall = metrics['overall']
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(BEST_MODEL_PATH)
        append_metrics_log(epoch, logs, metrics, self.best_val_f1)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """棰勬祴鍒版枃浠?    鍙互鎻愪氦鍒?https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
    """
    data = json.load(open(in_file))
    for i in tqdm(range(0, len(data), eval_batch_size), **TQDM_KWARGS):
        batch_data = data[i:i + eval_batch_size]
        batch_entities = NER.recognize_batch([d['text'] for d in batch_data])
        for d, entities in zip(batch_data, batch_entities):
            d['entities'] = []
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
