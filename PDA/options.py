import argparse

parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
parser.add_argument('--embedding', type=str, default='glove',
                        help='Word embedding type, word2vec, senna or glove')
parser.add_argument('--embedding_dict', type=str, default="newdata/glove.6B.50d.txt", help='Pretrained embedding path')
parser.add_argument('--embedding_dim', type=int, default=64,help='Only useful when embedding is randomly initialised')
parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs for training')
#20
parser.add_argument('--batch_size', type=int, default=32, help='Number of texts in each batch')
#32
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=5000,
                        help="Vocab size (default=5000)")
parser.add_argument('--oov', choices=['random', 'embedding'], help="Embedding for oov word", default='embedding')
parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop','adam'],
                        help='updating algorithm', default='sgd')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
#0.0001
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
parser.add_argument('--datapath3', type=str, default='newdatamfsan')
parser.add_argument('--datapath', type=str, default='newdata/fold_')
parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')
parser.add_argument('--devprompt_id', type=int, default=5, help='prompt id of essay devset')
parser.add_argument('--source1_id', type=int, default=1, help='source1 prompt id of essay set')
parser.add_argument('--source2_id', type=int, default=2, help='source2 prompt id of essay set')
parser.add_argument('--source3_id', type=int, default=3, help='source3 prompt id of essay set')
parser.add_argument('--source4_id', type=int, default=4, help='source4 prompt id of essay set')
parser.add_argument('--source5_id', type=int, default=6, help='source5 prompt id of essay set')
parser.add_argument('--source6_id', type=int, default=7, help='source6 prompt id of essay set')
parser.add_argument('--source7_id', type=int, default=8, help='source7 prompt id of essay set')
parser.add_argument('--domains', type=int, nargs='+', default=[])
parser.add_argument('--dev_domains', type=int, nargs='+', default=[])
args = parser.parse_args()