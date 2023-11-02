import codecs
import nltk
# import logging
import re
import numpy as np
import pickle as pk
import utils
url_replacer = '<url>'
logger = utils.get_logger("Loading data...")
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'
high_score = [0,10,5,3,3,3,3,24,48]

MAX_SENTLEN = 50
MAX_SENTNUM = 100

asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}


def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def get_score_range(prompt_id):
    return asap_ranges[prompt_id]
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def scoreSortClassification(scores_array):
    # print(scores_array)
    # [[10.]
    #  [7.]
    #  [8.]
    #      ...
    #  [8.]
    #  [10.]
    #  [9.]]
    # #上面是未排序的分数列表
    # print(sss)
    scores=sorted(scores_array)
    # print(scores)
    # print(sss)
    #上面是排序的分数列表
    num,leftnum,rightnum=0.0,0.0,0.0
    average,averageleft,averageright=0.0,0.0,0.0
    left,right=0,0
    for i in scores:
        num+=i
    average=num/len(scores)
    for i in scores:
        if i<average:
            left+=1
            leftnum+=i
        else:
            right+=1
            rightnum+=i
    averageleft=leftnum/left
    averageright=rightnum/right
    score_array=[]
    for i in scores_array:
        if i<averageleft:
            score_array.append(0)
        elif averageleft<=i<average:
            score_array.append(1)
        elif average<=i<averageright:
            score_array.append(2)
        else:
            score_array.append(3)
    # print(score_array)
    #上面是分类之后的列表
    return score_array
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#   假设 y_train=[10.0,7.0]，train_prompts=[1,1]  分别是两篇文章对应的分数和主题标签
#     Y_train = get_model_friendly_scores(y_train, train_prompts)
def get_model_friendly_scores(scores_array, prompt_id_array):
    # print(prompt_id_array)-->[1,1]
    for k ,i in enumerate(prompt_id_array):
        # print(k)--->0(下标)
        # print(i)--->1（该下标对应的主题）
	#assert i in range(1, 9)
        if i == 1:
            minscore = 2
            maxscore = 12
        elif i == 2:
            minscore = 1
            maxscore = 6
        elif i in [3, 4]:
            minscore = 0
            maxscore = 3
        elif i in [5, 6]:
            minscore = 0
            maxscore = 4
        elif i == 7:
            minscore = 0
            maxscore = 30
        elif i == 8:
            minscore = 0
            maxscore = 60
        else:
            minscore = 1
            maxscore = 3
        scores_array[k] = (scores_array[k]-minscore) / (maxscore - minscore)
        if 0 <= scores_array[k] <0.4:
            scores_array[k] = 0
        elif 0.4 <= scores_array[k] < 0.6:
            scores_array[k]= 1
        elif 0.6 <= scores_array[k] <0.8:
            scores_array[k] = 2
        elif 0.8<=scores_array[k]<=1:
            scores_array[k]=3

    scores_array=scores_array.astype(np.int64)
    return scores_array


def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
    i = prompt_id_array
    if i == 1:
        minscore = 2
        maxscore = 12
    elif i == 2:
        minscore = 1
        maxscore = 6
    elif i in [3, 4]:
        minscore = 0
        maxscore = 3
    elif i in [5, 6]:
        minscore = 0
        maxscore = 4
    elif i == 7:
        minscore = 0
        maxscore = 30
    elif i == 8:
        minscore = 0
        maxscore = 60
    else:
        minscore = 1
        maxscore = 3

    for k ,i in enumerate(scores_array):
        scores_array[k] = scores_array[k]* (maxscore - minscore)+minscore
    return np.round(scores_array)


#num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
def is_number(token):
    return bool(num_regex.match(token))


def load_vocab(vocab_path):
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab


def create_vocab(file_path, source1_id,source2_id, source3_id,source4_id,source5_id,source6_id,source7_id,vocab_size, tokenize_text, to_lower):
    total_words, unique_words = 0, 0
    word_freqs = {}
    for path in file_path:
        print("Current file_path is:"+ path)
        with codecs.open(path, mode='r', encoding='UTF8') as input_file:
            input_file.readline()
            for line in input_file:
                tokens = line.strip().split('\t')
                essay_id = int(tokens[0])
                essay_set = int(tokens[1])
                content = tokens[2].strip()
                score = float(tokens[6])
                if essay_set in [source1_id,source2_id, source3_id,source4_id,source5_id,source6_id,source7_id]:
                    if tokenize_text:
                        content = text_tokenizer(content, True, True, True)
                    if to_lower:
                        content = [w.lower() for w in content]
                    for word in content:
                        try:
                            word_freqs[word] += 1
                        except KeyError:
                            unique_words += 1
                            word_freqs[word] = 1
                        total_words += 1

    logger.info('  %i total words, %i unique words' % (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    print("Vocab is OK！")
    return vocab


def read_essays(file_path, prompt_id):
    logger.info('Reading tsv from: ' + file_path)
    essays_list = []
    essays_ids = []
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        #input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            if int(tokens[1]) == prompt_id or prompt_id <= 0:
                essays_list.append(tokens[2].strip())
                essays_ids.append(int(tokens[0]))
    return essays_list, essays_ids

#re.sub，实现正则的替换---对于输入的一个字符串，利用正则表达式（的强大的字符串处理功能），去实现（相对复杂的）字符串替换处理，然后返回被替换后的字符串
#url_replacer = '<url>'
def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text

#text_tokenizer(content, True, True, True)
def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
        # print text
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
        # print text
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)
        # print text

    # TODO here
    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        # print sent_tokens
        # sys.exit(0)
        # if not create_vocab_flag:
        #     print "After processed and tokenized, sentence num = %s " % len(sent_tokens)
        return sent_tokens
    else:
        raise NotImplementedError



def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):

    # tokenize a long text to a list of sentences
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)

    # Note
    # add special preprocessing for abnormal sentence splitting
    # for example, sentence1 entangled with sentence2 because of period "." connect the end of sentence1 and the begin of sentence2
    # see example: "He is running.He likes the sky". This will be treated as one sentence, needs to be specially processed.
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            # print sent
            # print s
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        # print tokens
        return tokens

    # TODO here
    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    # if len(sent_tokens) > 90:
    #     print len(sent_tokens), sent_tokens
    return sent_tokens


def shorten_sentence(sent, max_sentlen):
    # handling extra long sentence, truncate to no more extra max_sentlen
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        # print len(tokens)
        # Step 1: split sentence based on keywords
        # split_keywords = ['because', 'but', 'so', 'then', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sentlen
            k_indexes = [(i+1)*max_sentlen for i in range(int(num))]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        # Step 2: split sentence to no more than max_sentlen
        # if there are still sentences whose length exceeds max_sentlen
        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                s_indexes = [(i+1)*max_sentlen for i in range(int(num))]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
            return [tokens]

    # print "Before processed sentences length = %d, after processed sentences num = %d " % (len(tokens), len(new_tokens))
    return new_tokens

#read_dataset(train_path, prompt_id, vocab, to_lower)
#read_dataset(source1_path, source1_id, vocab, to_lower)
def read_dataset(file_path, prompt_id, vocab, to_lower, score_index=6, char_level=False):
    logger.info('Reading dataset from: ' + file_path)
    data_x, data_y, prompt_ids,essay_ids= [], [], [],[]
    class_labels=[]
    num_hit, unk_hit, total = 0., 0., 0.
    max_sentnum = -1
    max_sentlen = -1
    with codecs.open(file_path, mode='r', encoding='utf-8-sig') as input_file:
        essay_idx=-1
        for line in input_file:
            tokens = line.strip().split('\t')
            # essay_id = int(float(tokens[0]))
            essay_idx=essay_idx+1
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id or prompt_id <= 0:
                # tokenize text into sentences
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
                if to_lower:
                    sent_tokens = [[w.lower() for w in s] for s in sent_tokens]
                if char_level:
                    raise NotImplementedError
                sent_indices = []
                indices = []
                # print(sent_tokens)---->[['did', 'you', '?'],['social', 'life', '.'],['you', 'can', '?']]   应该是代表整个essay里面的每一个sentent
                if char_level:
                    raise NotImplementedError
                else:
                    for sent in sent_tokens:
                        # print(sent)--->['did', 'you', '?']（每一个essay里面的每一个sentence）
                        length = len(sent)
                        # sum=sum+length
                        # num=num+1
                        if max_sentlen < length:
                            max_sentlen = length
                            #max_sentlen是最长的句子的单词数
                        for word in sent:
                            # print(word)---->did（每一段sentence里面的每一个word）
                            if is_number(word):
                                indices.append(vocab['<num>'])
                                num_hit += 1
                            elif word in vocab:
                                indices.append(vocab[word])
                                #如果当前单词如did在词汇表里面,查找当前单词在词汇表对应的值即vocab[word]，然后追加到列表indices中
                            else:
                                indices.append(vocab['<unk>'])
                                unk_hit += 1
                            total += 1
                        # print(sent)
                        # ['did', 'you', 'know', 'that', 'more', 'and', 'more', 'people', 'these', 'days', 'are',
                        #  'depending', 'on', 'computers', 'for', 'their', 'safety', ',', 'natural', 'education', ',',
                        #  'and', 'their', 'social', 'life', '?']
                        # print(indices)--->[177, 8, 69, 13, 45, 7, 45, 12, 112, 293, 18, 2312, 11, 10, 26, 46, 1273, 4, 855, 360, 4, 7, 46, 219, 92, 67]
                        # print("每一个sent的indices打印完成")
                        sent_indices.append(indices)
                        # print(sent_indices)-->sent_tokens中的第一个sent的indices打印[[177, 8, 69, 13, 45, 7, 45, 12, 112, 294, 18, 2372, 11, 10, 26, 46, 1260, 4, 845, 361, 4, 7, 46, 219, 92, 67]]
                        indices = []
                data_x.append(sent_indices)
            #例如[]表示一个句子，[12,34,22,1]表示一个句子（里面包含5个单词。[[],[]]表示一篇文章（里面有2个句子).[[[],[]],[[],[]],[[],[]]]表示该主题下有多少文章
                # print(sent_indices)--->[[22,34,23],[],[]]
                # print(data_x)
                #--->[[[],[]],[[],[],[]],[[],[]],[[],[]],[[],[]]]
                data_y.append(score)
                # print("data_y:")
                # print(data_y)
                #--->第一次打印，表示该主题下的[10.0]--》第一次追加的第一篇文章的分数
                #prompt_ids：当路径是来自训练集时，里面是源域的类标签
                prompt_ids.append(essay_set)
                essay_ids.append(essay_idx)
                # print(prompt_ids)---->第一次打印的话是第一篇文章的主题：[1]
                if max_sentnum < len(sent_indices):
                    max_sentnum = len(sent_indices)
                    #max_sentnum是最大的文章句子数（每个文章包含的句子个数不一样，选择其中最大的作为max_sentnum）
    logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    return data_x, data_y, prompt_ids,essay_ids, max_sentlen, max_sentnum


def get_data(paths,source1_id,source2_id,source3_id,source4_id,source5_id,source6_id,source7_id,devprompt_id, vocab, tokenize_text=True, to_lower=True, sort_by_len=False,  score_index=6):
    source1_path,source2_path,source3_path,source4_path,source5_path,source6_path,source7_path,dev_path,test_path= paths[0], paths[1],paths[2],paths[3],paths[4],paths[5],paths[6],paths[7],paths[8]
    source1_x, source1_y, source1_prompts,essay1_ids, source1_maxsentlen, source1_maxsentnum = read_dataset(source1_path, source1_id, vocab, to_lower)
    source2_x, source2_y, source2_prompts, essay2_ids,source2_maxsentlen, source2_maxsentnum = read_dataset(source2_path,source2_id, vocab,to_lower)
    source3_x, source3_y, source3_prompts,essay3_ids, source3_maxsentlen, source3_maxsentnum = read_dataset(source3_path, source3_id, vocab,to_lower)
    source4_x, source4_y, source4_prompts, essay4_ids,source4_maxsentlen, source4_maxsentnum = read_dataset(source4_path,source4_id, vocab,to_lower)
    source5_x, source5_y, source5_prompts, essay5_ids,source5_maxsentlen, source5_maxsentnum = read_dataset(source5_path,source5_id, vocab,to_lower)
    source6_x, source6_y, source6_prompts, essay6_ids,source6_maxsentlen, source6_maxsentnum = read_dataset(source6_path,source6_id, vocab,to_lower)
    source7_x, source7_y, source7_prompts, essay7_ids,source7_maxsentlen, source7_maxsentnum = read_dataset(source7_path,source7_id, vocab,to_lower)
    dev_x, dev_y, dev_prompts, essaydev_ids,dev_maxsentlen, dev_maxsentnum = read_dataset(dev_path, devprompt_id, vocab, to_lower)
    test_x, test_y, test_prompts, essaytest_ids,test_maxsentlen,test_maxsentnum = read_dataset(test_path, devprompt_id, vocab, to_lower)




    overal_maxlen = max(source1_maxsentlen,source2_maxsentlen,source3_maxsentlen,source4_maxsentlen,source5_maxsentlen,source6_maxsentlen,source7_maxsentlen, dev_maxsentlen,test_maxsentlen)
    #overal_maxlen：训练集开发集和测试集的句子最大单词数
    overal_maxnum = max(source1_maxsentnum, source2_maxsentnum,source3_maxsentnum,source4_maxsentnum,source5_maxsentnum,source6_maxsentnum,source7_maxsentnum,dev_maxsentnum,test_maxsentnum)
    #训练集开发集和测试集的文章最大句子数
    logger.info("Overall max sentence num = %s, max sentence length = %s" % (overal_maxnum, overal_maxlen))
    return (source1_x, source1_y, source1_prompts,essay1_ids), (source2_x, source2_y, source2_prompts,essay2_ids),(source3_x, source3_y, source3_prompts,essay3_ids),(source4_x, source4_y, source4_prompts,essay4_ids),(source5_x, source5_y, source5_prompts,essay5_ids),(source6_x, source6_y, source6_prompts,essay6_ids),(source7_x, source7_y, source7_prompts,essay7_ids),(dev_x, dev_y, dev_prompts,essaydev_ids),(test_x, test_y, test_prompts,essaytest_ids) ,overal_maxlen, overal_maxnum

def prompt(file_path, prompt,vocab):
    indices= []
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        for line in input_file:
            tokens = line.strip().split('\t')
            prompt_id = int(tokens[0])
            content = tokens[1].strip()
            if prompt_id  == prompt:
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
                sent_tokens = [[w.lower() for w in s] for s in sent_tokens]
            for sent in sent_tokens:
                for word in sent:
                        if is_number(word):
                            indices.append(vocab['<num>'])
                        elif word in vocab:
                            indices.append(vocab[word])
                        else:
                            indices.append(vocab['<unk>'])
    return  indices

def prepare_sentence_data(embedd_dim, source1_id,source2_id,source3_id,source4_id,source5_id,source6_id,source7_id,devprompt_id,datapaths, vocab,embedding_path=None, embedding='glove'):

    # assert len(datapaths) == 4, "data paths should include source1_path,source2_path,dev_path,test_path"
    (source1_x, source1_y, source1_prompts,essay1_ids), (source2_x, source2_y, source2_prompts,essay2_ids),(source3_x, source3_y, source3_prompts,essay3_ids),(source4_x, source4_y, source4_prompts,essay4_ids),(source5_x, source5_y, source5_prompts,essay5_ids),(source6_x, source6_y, source6_prompts,essay6_ids),(source7_x, source7_y, source7_prompts,essay7_ids),(dev_x, dev_y, dev_prompts,essaydev_ids),(test_x, test_y, test_prompts,essaytest_ids),overal_maxlen, overal_maxnum = \
        get_data(datapaths, source1_id,source2_id,source3_id,source4_id,source5_id,source6_id,source7_id,devprompt_id, vocab, tokenize_text=True, to_lower=True, sort_by_len=False,  score_index=6)


    X_source1, y_source1, mask_source1 = utils.padding_sentence_sequences(source1_x, source1_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_source2, y_source2, mask_source2 = utils.padding_sentence_sequences(source2_x, source2_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_source3, y_source3, mask_source3= utils.padding_sentence_sequences(source3_x, source3_y, overal_maxnum,
                                                                          overal_maxlen, post_padding=True)
    X_source4, y_source4, mask_source4 = utils.padding_sentence_sequences(source4_x, source4_y, overal_maxnum,
                                                                           overal_maxlen, post_padding=True)
    X_source5, y_source5, mask_source5 = utils.padding_sentence_sequences(source5_x, source5_y, overal_maxnum,
                                                                           overal_maxlen, post_padding=True)
    X_source6, y_source6, mask_source6 = utils.padding_sentence_sequences(source6_x, source6_y, overal_maxnum,
                                                                           overal_maxlen, post_padding=True)
    X_source7, y_source7, mask_source7 = utils.padding_sentence_sequences(source7_x, source7_y, overal_maxnum,
                                                                           overal_maxlen, post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overal_maxnum, overal_maxlen, post_padding=True)

    if source1_id:
        source1_pmt = np.array(source1_prompts, dtype='int32')
        essay1_ids = np.array(essay1_ids, dtype='int32')
    if source2_id:
        source2_pmt = np.array(source2_prompts, dtype='int32')
        essay2_ids = np.array(essay2_ids, dtype='int32')
    if source3_id:
        source3_pmt = np.array(source3_prompts, dtype='int32')
        essay3_ids = np.array(essay3_ids, dtype='int32')
    if source4_id:
        source4_pmt = np.array(source4_prompts, dtype='int32')
        essay4_ids = np.array(essay4_ids, dtype='int32')
    if source5_id:
        source5_pmt = np.array(source5_prompts, dtype='int32')
        essay5_ids = np.array(essay5_ids, dtype='int32')
    if source6_id:
        source6_pmt = np.array(source6_prompts, dtype='int32')
        essay6_ids = np.array(essay6_ids, dtype='int32')
    if source7_id:
        source7_pmt = np.array(source7_prompts, dtype='int32')
        essay7_ids = np.array(essay7_ids, dtype='int32')
    if devprompt_id:
        dev_pmt = np.array(dev_prompts, dtype='int32')
        test_pmt=np.array(test_prompts, dtype='int32')
        essaydev_ids = np.array(essaydev_ids, dtype='int32')
        essaytest_ids = np.array(essaytest_ids, dtype='int32')

    Y_source1 = get_model_friendly_scores(y_source1,source1_prompts)
    Y_source2 = get_model_friendly_scores(y_source2, source2_prompts)
    Y_source3 = get_model_friendly_scores(y_source3, source3_prompts)
    Y_source4 = get_model_friendly_scores(y_source4, source4_prompts)
    Y_source5 = get_model_friendly_scores(y_source5, source5_prompts)
    Y_source6 = get_model_friendly_scores(y_source6, source6_prompts)
    Y_source7 = get_model_friendly_scores(y_source7, source7_prompts)
    Y_dev = get_model_friendly_scores(y_dev,dev_prompts)
    Y_test = get_model_friendly_scores(y_test,test_prompts)

    Y_source1=Y_source1.astype(np.int64)
    Y_source2 = Y_source2.astype(np.int64)
    Y_source3 = Y_source3.astype(np.int64)
    Y_source4 = Y_source4.astype(np.int64)
    Y_source5 = Y_source5.astype(np.int64)
    Y_source6 = Y_source6.astype(np.int64)
    Y_source7 = Y_source7.astype(np.int64)
    Y_dev=Y_dev.astype(np.int64)
    Y_test = Y_test.astype(np.int64)
    if embedding_path:
        #logger = utils.get_logger("Loading data...")
        embedd_dict, embedd_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        #应该是创建词汇表对应的嵌入表（嵌入矩阵）embedd_matrixembedding
        embedd_matrix = utils.build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    return (X_source1, Y_source1, mask_source1,source1_pmt,essay1_ids),(X_source2, Y_source2, mask_source2,source2_pmt,essay2_ids), (X_source3, Y_source3, mask_source3, source3_pmt,essay3_ids), (X_source4, Y_source4, mask_source4, source4_pmt,essay4_ids),\
    (X_source5, Y_source5, mask_source5, source5_pmt,essay5_ids), (X_source6, Y_source6, mask_source6, source6_pmt,essay6_ids),\
    (X_source7, Y_source7, mask_source7, source7_pmt,essay7_ids),(X_dev, Y_dev, mask_dev,dev_pmt,essaydev_ids), (X_test, Y_test, mask_test,test_pmt,essaytest_ids),\
            embedd_matrix, overal_maxlen, overal_maxnum
