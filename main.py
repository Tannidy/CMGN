# -*- coding: utf-8 -*-
# @Author: Yiding Tan
# @Last Modified by:   Yiding Tan,    Contact: 20210240297@fudan.edu.cn
import argparse
import json

from stanfordcorenlp import StanfordCoreNLP
import torch
from fastNLP import GradientClipCallback, EvaluateCallback, Trainer, SpanFPreRecMetric, BucketSampler, DataSetIter, \
    SequentialSampler, Sampler, WarmupCallback, EngChar2DPadder
from fastNLP.embeddings import StackEmbedding, StaticEmbedding, LSTMCharEmbedding, CNNCharEmbedding
from fastNLP.io import Conll2003NERPipe
from torch import optim
import numpy as np

from CMG import CMG
from utils import sentence_pairing, pair_list_to_dict


def deTokenize(sent_list):
    str = ""
    for word in sent_list:
        if str == "":
            str = word
        else:
            str += " " + word
    return str

def load_data():
    # paths = {'test': "../data/conll2003/test.txt",
    #          'train': "../data/conll2003/train.txt",
    #          'dev': "../data/conll2003/dev.txt"}
    paths = {'test': args.test,
             'train': args.train,
             'dev': args.dev}
    data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    char_embed = None
    if char_type == 'cnn':
        char_embed = CNNCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, filter_nums=[30],
                                      kernel_sizes=[3], word_dropout=0, dropout=0.3, pool_method='max'
                                      , include_word_start_end=False, min_char_freq=2)
    elif char_type == 'lstm':
        char_embed = LSTMCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                                       dropout=0.3, hidden_size=100, pool_method='max', activation='relu',
                                       min_char_freq=2, bidirectional=True, requires_grad=True,
                                       include_word_start_end=False)
    word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                 model_dir_or_name='en-glove-6b-100d',
                                 requires_grad=True, lower=True, word_dropout=0, dropout=0.5,
                                 only_norm_found_vector=normalize_embed)
    if char_embed is not None:
        embed = StackEmbedding([word_embed, char_embed], dropout=0, word_dropout=0.02)
    else:
        word_embed.word_drop = 0.02
        embed = word_embed

    data.rename_field('words', 'chars')
    return data, embed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training set.', default='data/conll2003/train.txt')
    parser.add_argument('--dev', help='Developing set.', default='data/conll2003/dev.txt')
    parser.add_argument('--test', help='Testing set.', default='data/conll2003/test.txt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', help='Output results for decoding.')
    parser.add_argument('--saved_model', help='Dir Path of saved model.', default="saved_model")
    parser.add_argument('--stanford_nlp_path', help='Path of stanfordnlp package.',
                        default=r'/home/ydtan/stanford-corenlp-full-2018-10-05')
    parser.add_argument('--char_emb', help='Path of character embedding file.',
                        default="data/gigaword_chn.all.a2b.uni.ite50.vec")
    parser.add_argument('--word_emb', help='Path of word embedding file.', default="data/ctb.50d.vec")

    parser.add_argument('--batch_size', help='Batch size.', default=32, type=int)
    parser.add_argument('--num_epochs', default=100, type=int, help="Epoch number.")
    parser.add_argument('--num_layers', default=4, type=int, help='The number of Graph iterations.')
    parser.add_argument('--hidden_dim', default=50, type=int, help='Hidden state size.')
    parser.add_argument('--num_head', default=14, type=int, help='Number of transformer head.')
    parser.add_argument('--head_dim', default=128, type=int, help='Head dimension of transformer.')
    parser.add_argument('--warm_up_steps', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.0007)

    args = parser.parse_args()

    # status = args.status.lower()

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    output_file = args.output
    saved_model_path = args.saved_model

    char_type = 'cnn'
    encoding_type = 'bio'
    normalize_embed = True
    warmup_steps = 0.01
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    device = args.gpu

    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_heads = args.num_head
    head_dim = args.head_dim
    d_model = num_heads * head_dim
    d_label = 10
    lr = args.lr

    nlp = StanfordCoreNLP(args.stanford_nlp_path)

    dependency_label_vocab = {'self': 1, 'left_context': 2, 'right_context': 3, 'depen': 4, 'rev_depen': 5}

    task_list = ['train', 'dev', 'test']
    data_bundle, embed = load_data()
    print(data_bundle)
    for task in task_list:
        # mask_list = []
        # dependency_list = []
        # for instance in data_bundle.get_dataset(task):
        #     seq_len = instance['seq_len']
        #     raw_words = instance['raw_words']
        #     raw_words = deTokenize(raw_words)
        #     tokenized_words = nlp.word_tokenize(raw_words)
        #     dependency_tree = nlp.dependency_parse(raw_words)
        #     set_dict = pair_list_to_dict(sentence_pairing(instance['raw_words'], tokenized_words))
        #     cur_mask = np.zeros(shape=(seq_len, seq_len), dtype=np.int)
        #     cur_dependency = np.zeros(shape=(seq_len, seq_len), dtype=np.int)
        #     for i in range(seq_len):  # add context edge
        #         cur_dependency[i, i] = dependency_label_vocab['self']
        #         cur_mask[i, i] = 1
        #         if i > 0:
        #             cur_dependency[i, i - 1] = dependency_label_vocab['left_context']
        #             cur_mask[i, i - 1] = 1
        #         if i < seq_len - 1:
        #             cur_dependency[i, i + 1] = dependency_label_vocab['right_context']
        #             cur_mask[i, i + 1] = 1
        #     for relation in dependency_tree:  # add dependency relationship edge
        #         if relation[0] == 'ROOT':
        #             continue
        #         head_list = set_dict[relation[1]-1]
        #         tail_list = set_dict[relation[2]-1]
        #         for head in head_list:
        #             for tail in tail_list:
        #                 cur_dependency[tail, head] = dependency_label_vocab['depen']
        #                 cur_dependency[head, tail] = dependency_label_vocab['rev_depen']
        #                 cur_mask[tail, head] = 1
        #                 cur_mask[head, tail] = 1
        #     mask_list.append(cur_mask.tolist())
        #     dependency_list.append(cur_dependency.tolist())
        # with open("mask_list_" + task + ".json", "w") as f:
        #     json.dump(mask_list, f)
        # with open("dependency_list" + task + ".json", "w") as f:
        #     json.dump(dependency_list, f)
        with open("mask_list_" + task + ".json", "r") as f:
            mask_list = json.load(f)
        with open("dependency_list_" + task + ".json", "r") as f:
            dependency_list = json.load(f)
        data_bundle.get_dataset(task).add_field('attn_mask', mask_list, padder=EngChar2DPadder(), is_input=True)
        data_bundle.get_dataset(task).add_field('attn_category', dependency_list, padder=EngChar2DPadder(), is_input=True)

    data_bundle.get_dataset('test').print_field_meta()
    # print(data_bundle.get_dataset('test').field_arrays['raw_words'])
    for instance in data_bundle.get_dataset('test'):
        print(instance['raw_words'])
        print(instance['target'])
        print(instance['seq_len'])
        print(instance['chars'])
        print(instance['attn_mask'])
        print(instance['attn_category'])
        break

    model = CMG(tag_vocab=data_bundle.get_vocab('target'), embed=embed,
                d_model=d_model, n_heads=num_heads, d_k=hidden_dim, d_v=head_dim, n_layers=num_layers,
                fc_dropout=0.4, gpu=device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    callbacks = []
    clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
    evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))

    if warmup_steps > 0:
        warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
        callbacks.append(warmup_callback)
    callbacks.extend([clip_callback, evaluate_callback])

    trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size,
                      sampler=BucketSampler(),
                      num_workers=2, n_epochs=num_epochs, dev_data=data_bundle.get_dataset('dev'),
                      metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                      dev_batch_size=batch_size * 5, callbacks=callbacks, device=device, test_use_tqdm=False,
                      use_tqdm=True, print_every=300, save_path=None)
    trainer.train(load_best_model=False)
