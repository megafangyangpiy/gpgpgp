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


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_int(name, default):
    return int(os.environ.get(name, default))


# Ablation presets:
#   baseline              original GlobalPointer behavior
#   no_relation           no span-pair structural support in decoding
#   no_consistency        no nested-positive consistency loss
#   no_structure_decoding original threshold decoding, keep training changes
#   full                  all structure-aware components enabled
ABLATION_MODE = os.environ.get('GP_ABLATION', 'full').strip().lower()
USE_SPAN_PAIR_RELATION = env_flag(
    'GP_USE_SPAN_PAIR_RELATION',
    ABLATION_MODE not in {'baseline', 'no_relation'}
)
USE_CONSISTENCY_LOSS = env_flag(
    'GP_USE_CONSISTENCY_LOSS',
    ABLATION_MODE not in {'baseline', 'no_consistency'}
)
USE_STRUCTURE_DECODING = env_flag(
    'GP_USE_STRUCTURE_DECODING',
    ABLATION_MODE not in {'baseline', 'no_structure_decoding'}
)
STRUCTURE_TOPK = env_int('GP_STRUCTURE_TOPK', 512)
STRUCTURE_CANDIDATE_MARGIN = env_float('GP_STRUCTURE_CANDIDATE_MARGIN', 1.0)
INNER_OUTER_LAMBDA = env_float('GP_INNER_OUTER_LAMBDA', 0.35)
SHARED_BOUNDARY_LAMBDA = env_float('GP_SHARED_BOUNDARY_LAMBDA', 0.15)
CROSSING_PENALTY_LAMBDA = env_float('GP_CROSSING_PENALTY_LAMBDA', 0.45)
CONSISTENCY_LOSS_WEIGHT = env_float('GP_CONSISTENCY_LOSS_WEIGHT', 0.1)
PRUNE_CROSSING = env_flag('GP_PRUNE_CROSSING', True)

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
category2id = dict((c, i) for i, c in enumerate(categories))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def span_relation(a_start, a_end, b_start, b_end):
    """Return the token/char span relation used by nested-entity modules."""
    if a_start == b_start and a_end == b_end:
        return 'exact'
    if a_start <= b_start and b_end <= a_end:
        return 'contain'
    if b_start <= a_start and a_end <= b_end:
        return 'inside'
    if a_end < b_start or b_end < a_start:
        return 'disjoint'
    return 'crossing'


def spans_share_boundary(a_start, a_end, b_start, b_end):
    return a_start == b_start or a_end == b_end


def find_nested_token_keys(mapped_entities):
    """Find token-level entities participating in a nested relation."""
    nested_keys = set()
    for i, entity in enumerate(mapped_entities):
        _, start, end = entity
        for j, other in enumerate(mapped_entities):
            if i == j:
                continue
            _, other_start, other_end = other
            relation = span_relation(start, end, other_start, other_end)
            if relation in {'contain', 'inside'}:
                nested_keys.add(entity)
                break
    return nested_keys


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
            mapped_entities = []
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = category2id[label]
                    if start <= end:
                        mapped_entities.append((label, start, end))
            nested_keys = find_nested_token_keys(mapped_entities)
            for label, start, end in mapped_entities:
                labels[label, start, end] = (
                    2 if (label, start, end) in nested_keys else 1
                )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def stable_softplus(x):
    return K.maximum(x, 0) + K.log(1 + K.exp(-K.abs(x)))


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true_binary = K.cast(K.greater(y_true, 0), K.floatx())
    nested_mask = K.cast(K.greater(y_true, 1), K.floatx())
    y_true_binary = K.reshape(y_true_binary, (bh, -1))
    nested_mask = K.reshape(nested_mask, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    loss = K.mean(multilabel_categorical_crossentropy(y_true_binary, y_pred))
    if USE_CONSISTENCY_LOSS:
        nested_loss = K.sum(nested_mask * stable_softplus(-y_pred))
        nested_loss = nested_loss / (K.sum(nested_mask) + K.epsilon())
        loss = loss + CONSISTENCY_LOSS_WEIGHT * nested_loss
    return loss


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_true = K.cast(K.greater(y_true, 0), K.floatx())
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / (
        K.sum(y_true + y_pred) + K.epsilon()
    )


model = build_transformer_model(config_path, checkpoint_path)
output = GlobalPointer(len(categories), 64)(model.output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=global_pointer_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=[global_pointer_f1_score]
)

print(
    'ablation: %s, span_pair_relation: %s, consistency_loss: %s, '
    'structure_decoding: %s, structure_topk: %d, candidate_margin: %.2f' % (
        ABLATION_MODE,
        USE_SPAN_PAIR_RELATION,
        USE_CONSISTENCY_LOSS,
        USE_STRUCTURE_DECODING,
        STRUCTURE_TOPK,
        STRUCTURE_CANDIDATE_MARGIN
    )
)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def collect_span_candidates(scores, threshold, topk, margin=0.0):
    pre_threshold = threshold - margin
    candidates = []
    for label, start, end in zip(*np.where(scores > pre_threshold)):
        if start <= end:
            candidates.append((label, start, end, float(scores[label, start, end])))
    candidates.sort(key=lambda item: item[3], reverse=True)
    if topk and topk > 0 and len(candidates) > topk:
        candidates = candidates[:topk]
    return candidates


def candidate_structure_delta(candidate, candidates):
    if not USE_SPAN_PAIR_RELATION:
        return 0.0

    _, start, end, _ = candidate
    inner_outer_support = 0.0
    shared_boundary_support = 0.0
    crossing_penalty = 0.0
    for other in candidates:
        if other[:3] == candidate[:3]:
            continue
        _, other_start, other_end, other_score = other
        relation = span_relation(start, end, other_start, other_end)
        confidence = sigmoid(other_score)
        if relation in {'contain', 'inside'}:
            inner_outer_support = max(inner_outer_support, confidence)
        elif relation == 'crossing':
            crossing_penalty = max(crossing_penalty, confidence)
        if relation != 'exact' and spans_share_boundary(
            start, end, other_start, other_end
        ):
            shared_boundary_support = max(shared_boundary_support, confidence)

    return (
        INNER_OUTER_LAMBDA * inner_outer_support
        + SHARED_BOUNDARY_LAMBDA * shared_boundary_support
        - CROSSING_PENALTY_LAMBDA * crossing_penalty
    )


def decode_candidate(candidate, mapping):
    label, start, end, _ = candidate
    if start >= len(mapping) or end >= len(mapping):
        return None
    if not mapping[start] or not mapping[end]:
        return None
    return (mapping[start][0], mapping[end][-1], categories[label])


def threshold_decode(scores, mapping, threshold):
    entities = []
    for label, start, end in zip(*np.where(scores > threshold)):
        if start > end:
            continue
        entity = decode_candidate((label, start, end, scores[label, start, end]), mapping)
        if entity is not None:
            entities.append(entity)
    return entities


def structure_aware_decode(scores, mapping, threshold):
    candidates = collect_span_candidates(
        scores,
        threshold,
        STRUCTURE_TOPK,
        margin=STRUCTURE_CANDIDATE_MARGIN
    )
    rescored = []
    for candidate in candidates:
        final_score = candidate[3] + candidate_structure_delta(
            candidate, candidates
        )
        if final_score > threshold:
            rescored.append((candidate, final_score))
    rescored.sort(key=lambda item: item[1], reverse=True)

    selected = []
    for candidate, final_score in rescored:
        _, start, end, _ = candidate
        if PRUNE_CROSSING:
            has_crossing = False
            for kept, _ in selected:
                if span_relation(start, end, kept[1], kept[2]) == 'crossing':
                    has_crossing = True
                    break
            if has_crossing:
                continue
        selected.append((candidate, final_score))

    entities = []
    for candidate, _ in selected:
        entity = decode_candidate(candidate, mapping)
        if entity is not None:
            entities.append(entity)
    return entities


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
            if USE_STRUCTURE_DECODING:
                entities = structure_aware_decode(scores, mapping, threshold)
            else:
                entities = threshold_decode(scores, mapping, threshold)
            batch_entities.append(entities)
        return batch_entities


NER = NamedEntityRecognizer()


def update_prf_counts(counts, predicted, gold):
    predicted = set(predicted)
    gold = set(gold)
    counts['hit'] += len(predicted & gold)
    counts['pred'] += len(predicted)
    counts['gold'] += len(gold)


def counts_to_prf(counts):
    hit, pred, gold = counts['hit'], counts['pred'], counts['gold']
    precision = hit / pred if pred else 0
    recall = hit / gold if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return f1, precision, recall


def entity_roles(entities):
    entities = set(entities)
    roles = dict((entity, set()) for entity in entities)
    entity_list = list(entities)
    for i, entity in enumerate(entity_list):
        for j, other in enumerate(entity_list):
            if i == j:
                continue
            relation = span_relation(entity[0], entity[1], other[0], other[1])
            if relation == 'inside':
                roles[entity].add('inner')
            elif relation == 'contain':
                roles[entity].add('outer')
    for entity, role_set in roles.items():
        if 'inner' in role_set or 'outer' in role_set:
            role_set.add('nested')
        else:
            role_set.add('flat')
    return roles


def select_role_predictions(predicted, predicted_roles, gold_roles, role):
    selected = set()
    for entity in predicted:
        if role in predicted_roles.get(entity, set()):
            selected.add(entity)
        elif role in gold_roles.get(entity, set()):
            selected.add(entity)
    return selected


def select_role_gold(gold, gold_roles, role):
    return set(entity for entity in gold if role in gold_roles.get(entity, set()))


def count_crossing_pairs(entities):
    entities = list(set(entities))
    count = 0
    for i, entity in enumerate(entities):
        for other in entities[i + 1:]:
            if span_relation(entity[0], entity[1], other[0], other[1]) == 'crossing':
                count += 1
    return count


def entities_overlap(a, b):
    return a[2] == b[2] and not (a[1] < b[0] or b[1] < a[0])


def count_boundary_errors(predicted, gold):
    predicted = set(predicted)
    gold = set(gold)
    errors = 0
    for entity in predicted - gold:
        if any(entities_overlap(entity, target) for target in gold):
            errors += 1
    return errors


def evaluate(data):
    """评测函数
    """
    prf_counts = dict(
        overall={'hit': 0, 'pred': 0, 'gold': 0},
        nested_context={'hit': 0, 'pred': 0, 'gold': 0},
        nested={'hit': 0, 'pred': 0, 'gold': 0},
        inner={'hit': 0, 'pred': 0, 'gold': 0},
        outer={'hit': 0, 'pred': 0, 'gold': 0},
        flat={'hit': 0, 'pred': 0, 'gold': 0}
    )
    crossing_pairs = 0
    total_pred_pairs = 0
    boundary_errors = 0
    total_predictions = 0

    for i in tqdm(range(0, len(data), eval_batch_size), **TQDM_KWARGS):
        batch_data = data[i:i + eval_batch_size]
        batch_entities = NER.recognize_batch([d[0] for d in batch_data])
        for d, entities in zip(batch_data, batch_entities):
            predicted = set(entities)
            gold = set(tuple(item) for item in d[1:])
            predicted_roles = entity_roles(predicted)
            gold_roles = entity_roles(gold)

            update_prf_counts(prf_counts['overall'], predicted, gold)
            if any('nested' in roles for roles in gold_roles.values()):
                update_prf_counts(
                    prf_counts['nested_context'], predicted, gold
                )

            for role in ['nested', 'inner', 'outer', 'flat']:
                role_predicted = select_role_predictions(
                    predicted, predicted_roles, gold_roles, role
                )
                role_gold = select_role_gold(gold, gold_roles, role)
                update_prf_counts(prf_counts[role], role_predicted, role_gold)

            crossing_pairs += count_crossing_pairs(predicted)
            total_pred_pairs += len(predicted) * (len(predicted) - 1) / 2
            boundary_errors += count_boundary_errors(predicted, gold)
            total_predictions += len(predicted)

    metrics = dict(
        (name, counts_to_prf(counts)) for name, counts in prf_counts.items()
    )
    metrics['crossing_error_rate'] = (
        crossing_pairs / total_pred_pairs if total_pred_pairs else 0
    )
    metrics['boundary_error_rate'] = (
        boundary_errors / total_predictions if total_predictions else 0
    )
    return metrics


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(valid_data)
        f1, precision, recall = metrics['overall']
        nested_context_f1 = metrics['nested_context'][0]
        nested_f1 = metrics['nested'][0]
        inner_f1 = metrics['inner'][0]
        outer_f1 = metrics['outer'][0]
        flat_f1 = metrics['flat'][0]
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(BEST_MODEL_PATH)
        print(
            'valid: f1: %.5f, precision: %.5f, recall: %.5f, '
            'nested_context_f1: %.5f, nested_f1: %.5f, '
            'inner_f1: %.5f, outer_f1: %.5f, flat_f1: %.5f, '
            'crossing_error: %.5f, boundary_error: %.5f, '
            'best f1: %.5f\n' % (
                f1,
                precision,
                recall,
                nested_context_f1,
                nested_f1,
                inner_f1,
                outer_f1,
                flat_f1,
                metrics['crossing_error_rate'],
                metrics['boundary_error_rate'],
                self.best_val_f1
            )
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    可以提交到 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
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
