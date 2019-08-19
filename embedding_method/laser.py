from something.platform.static import require_static_directory
from something.datascience.transforms.embeddings.sentence_embeddings.laser.model.encoder import Encoder

from subprocess import run, check_output, DEVNULL
import tempfile
import os
import re
from collections import namedtuple

import sys

import concurrent.futures
import torch
import numpy as np
import time
import multiprocessing
from itertools import repeat
from tqdm import tqdm

# Path to static storage
STATIC_STORAGE_PATH = "pretrained_embeddings/laser"

TOKENIZER_TOOLS_PATH = "tools/tokenizer"

# Script names and args
MOSES_TOKENIZER_NAME = "tokenizer.perl"
MOSES_TOKENIZER_ARGS = " -q -no-escape -threads 20 -l"

DESCAPE_NAME = "deescape-special-chars.perl"

NORM_PUNC_NAME = "normalize-punctuation.perl"
NORM_PUNC_ARGS = ' -l'

REM_NON_PRINT_CHAR_NAME = "remove-non-printing-char.perl"

# bpe files
BPE_VOCAB_NAME = "93langs.vocab"
BPE_CODES_NAME = "93langs.codes"

FAST_BPE = "tools/fast"

# Model files
ENCODER_NAME = "encoder.pt"

Batch = namedtuple('Batch', 'srcs tokens lengths')


class LaserSentenceEmbeddings:

    def __init__(self, max_batch_size=500):

        # Download necessary files
        self.file_dir = require_static_directory(STATIC_STORAGE_PATH)

        self._setup_script_paths()
        self.embedder = self.LaserSentenceEmbedder(self.ENCODER_PATH, max_sentences=max_batch_size)

    def _setup_script_paths(self):
        tokenizer_bpath = os.path.join(self.file_dir, TOKENIZER_TOOLS_PATH)

        # Tokenize scripts
        self.MOSES_TOKENIZER = os.path.join(tokenizer_bpath, MOSES_TOKENIZER_NAME)
        self.set_file_permissions(self.MOSES_TOKENIZER)

        self.DESCAPE = os.path.join(tokenizer_bpath, DESCAPE_NAME)
        self.set_file_permissions(self.DESCAPE)

        self.NORM_PUNC = os.path.join(tokenizer_bpath, NORM_PUNC_NAME)
        self.set_file_permissions(self.NORM_PUNC)

        self.REM_NON_PRINT_CHAR = os.path.join(tokenizer_bpath, REM_NON_PRINT_CHAR_NAME)
        self.set_file_permissions(self.REM_NON_PRINT_CHAR)

        # FastBPE path
        self.FASTBPE = os.path.join(self.file_dir, FAST_BPE)
        self.set_file_permissions(self.FASTBPE)

        # Model paths
        self.BPE_VOCAB = os.path.join(self.file_dir, BPE_VOCAB_NAME)
        self.BPE_CODES = os.path.join(self.file_dir, BPE_CODES_NAME)
        self.ENCODER_PATH = os.path.join(self.file_dir, ENCODER_NAME)

    def set_file_permissions(self, file):
        run("chmod +x {}".format(file), shell=True)

    def vectorize_text(self, text, **kwargs):
        if not isinstance(text, (str, list)):
            raise TypeError("Should be string or list as input for Laser embeddings")
        method = kwargs.get('method', None)
        language = kwargs.get('language', 'en')
        split_method = kwargs.get('split_method', None)
        if method in ['bpe', 'laser_bpe']:
            return self.bpe_vectorization(text, language)
        elif method in ['sent', 'laser_sent', 'sentence', 'laser_sent_newline_split']:
            if split_method:
                embeddings = np.array([self.sent_vectorization(txt, language) for txt in split_method(text)])
                if len(embeddings.shape) > 2:
                    return np.squeeze(embeddings, axis=1)
                elif len(embeddings.shape) == 2 and embeddings.shape[0] == 1:
                    return embeddings
                else:
                    return None
            else:
                return self.sent_vectorization(text, language)
        else:
            raise ValueError("Invalid method {} provided".format(method))

    def _tokenize(self, text, language):
        if isinstance(text, str):
            text = [text]
        text = [x.lower().replace("\n", " ") for x in text]


        # TODO: Take each perl script used below and rewrite to python
        if len(text) < 10:
            tokenized = [self._call_perl_scripts(txt, language) for txt in text]
        else:
            tokenized = []
            pbar = tqdm(total=len(text), desc="tokenizing sentences",miniters=1000)

            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                future_to_tokenize = {executor.submit(self._call_perl_scripts, t, language): t for t in text}
                for future in concurrent.futures.as_completed(future_to_tokenize):
                    results = future_to_tokenize[future]
                    try:
                        tokenized.append(future.result())
                        pbar.update()
                    except Exception as exc:
                        tqdm.write('%r generated an exception: %s' % (results, exc))
                        pbar.update()


        return tokenized



    def _call_perl_scripts(self, text_string, language):
        #with suppress_stderr():
        tok = check_output(
            self.REM_NON_PRINT_CHAR
            + ' | ' + self.NORM_PUNC + NORM_PUNC_ARGS + " " + language
            + ' | ' + self.DESCAPE
            + ' | ' + self.MOSES_TOKENIZER + MOSES_TOKENIZER_ARGS + " " + language,
            input=text_string,
            encoding='UTF-8',
            shell=True)
        return tok.strip()

    def _bpe(self, line):
        with tempfile.TemporaryDirectory() as tmpdir:
            ifn = os.path.join(tmpdir, 'tok')
            ofn = os.path.join(tmpdir, 'bpe')
            with open(ifn, 'w') as f:
                if isinstance(line, list):
                    f.write("\n".join(line) + "\n")
                else:
                    f.write('{}\n'.format(line))
            run(self.FASTBPE + ' applybpe ' + ofn + ' ' + ifn
                + ' ' + self.BPE_CODES + ' ' + self.BPE_VOCAB,
                shell=True, stderr=DEVNULL)
            try:
                with open(ofn, 'r') as f:
                    bpe = f.readlines()
            except FileNotFoundError:
                sys.exit("FileNotFoundError: "
                         "If the file is not found, you might have to recompile fastbpe for your system: "
                         "g++ -std=c++11 -pthread -O3 fast.cc -o fast")
            assert len(bpe) == 1 or (isinstance(line, list) and len(line) == len(bpe)), 'ERROR: unexpected BPE output'
            if isinstance(line, list):
                return [x.strip() for x in bpe]
            else:
                return bpe[0].strip()

    def _create_sentence_embedding(self, bpe_tokens):
        if isinstance(bpe_tokens, str):
            bpe_tokens = [bpe_tokens]

        return self.embedder(bpe_tokens)

    def sent_vectorization(self, text, language):
        tokenized = self._tokenize(text, language)
        bpe_tokens = self._bpe(tokenized)
        sentence_embedding = self._create_sentence_embedding(bpe_tokens)
        return sentence_embedding

    def bpe_vectorization(self, text, language):
        tokenized = self._tokenize(text, language)
        bpe_tokens = self._bpe(tokenized)
        if isinstance(bpe_tokens, str):
            bpe_tokens = [bpe_tokens]

        token_ids = [self.embedder._tokenize(token)[:-1] for token in bpe_tokens]  # Skip eos token

        bpe_embeddings = [self.embedder.encoder.embed_tokens(t_id).detach().cpu().numpy() for t_id in token_ids]

        if isinstance(text, str):
            return bpe_embeddings[0]
        else:
            return bpe_embeddings

    def __call__(self, data, **kwargs):
        vecs = self.vectorize_text(data, **kwargs)
        return vecs

    class LaserSentenceEmbedder:

        SPACE_NORMALIZER = re.compile("\s+")

        def __init__(self, model_path, max_sentences=None, max_tokens=None, cpu=True, fp16=False):
            self.use_cuda = torch.cuda.is_available() and not cpu
            self.max_sentences = max_sentences
            self.max_tokens = max_tokens
            if self.max_tokens is None and self.max_sentences is None:
                self.max_sentences = 1

            state_dict = torch.load(model_path)
            self.encoder = Encoder(**state_dict['params'])

            self.encoder.load_state_dict(state_dict['model'])
            self.dictionary = state_dict['dictionary']
            self.pad_index = self.dictionary['<pad>']
            self.eos_index = self.dictionary['</s>']
            self.unk_index = self.dictionary['<unk>']
            if fp16:
                self.encoder.half()
            if self.use_cuda:
                self.encoder.cuda()
            self.sort_kind = 'quicksort'

        def _process_batch(self, batch):
            tokens = batch.tokens
            lengths = batch.lengths
            if self.use_cuda:
                tokens = tokens.cuda()
                lengths = lengths.cuda()
            self.encoder.eval()
            sent_embeddings = self.encoder(tokens, lengths)['sentemb']
            return sent_embeddings.detach().cpu().numpy()

        def _tokenize(self, line):
            tokens = self.SPACE_NORMALIZER.sub(" ", line).strip().split()
            ntokens = len(tokens)
            ids = torch.LongTensor(ntokens + 1)
            for i, token in enumerate(tokens):
                ids[i] = self.dictionary.get(token, self.unk_index)
            ids[ntokens] = self.eos_index
            return ids

        def _make_batches(self, lines):
            tokens = [self._tokenize(line) for line in lines]
            lengths = np.array([t.numel() for t in tokens])
            indices = np.argsort(-lengths, kind=self.sort_kind)

            def batch(tokens, lengths, indices):
                toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]), self.pad_index)
                for i in range(len(tokens)):
                    toks[i, -tokens[i].shape[0]:] = tokens[i]
                return Batch(
                    srcs=None,
                    tokens=toks,
                    lengths=torch.LongTensor(lengths)
                ), indices

            batch_tokens, batch_lengths, batch_indices = [], [], []
            ntokens = nsentences = 0
            for i in indices:
                if nsentences > 0 and ((self.max_tokens is not None and ntokens + lengths[i] > self.max_tokens) or
                                       (self.max_sentences is not None and nsentences == self.max_sentences)):
                    yield batch(batch_tokens, batch_lengths, batch_indices)
                    ntokens = nsentences = 0
                    batch_tokens, batch_lengths, batch_indices = [], [], []
                batch_tokens.append(tokens[i])
                batch_lengths.append(lengths[i])
                batch_indices.append(i)
                ntokens += tokens[i].shape[0]
                nsentences += 1
            if nsentences > 0:
                yield batch(batch_tokens, batch_lengths, batch_indices)

        def encode_sentences(self, sentences):
            indices = []
            results = []
            pbar = tqdm(total=int(len(sentences)/self.max_sentences),desc="embedding sentences")
            for batch, batch_indices in self._make_batches(sentences):
                indices.extend(batch_indices)
                results.append(self._process_batch(batch))
                pbar.update(1)
            return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]

        def __call__(self, *args, **kwargs):
            return self.encode_sentences(*args)


def download_files():
    laser = LaserSentenceEmbeddings()
    laser("test", method="bpe")


# Define a context manager to suppress stdout and stderr.
class suppress_stderr(object):
    """
    A context manager for doing a "deep suppression" of stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a null file
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        # Save the actual stderr (2) file descriptors.
        self.save_fd = os.dup(2)

    def __enter__(self):
        # Assign the null pointer to stderr.
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        # Re-assign the real stderr back to (2)
        os.dup2(self.save_fd, 2)
        # Close all file descriptors
        os.close(self.null_fd)
        os.close(self.save_fd)


if __name__ == '__main__':
    # Usage
    embedder = LaserSentenceEmbeddings(max_batch_size=100)  # Max batch size limits batchsize for laser model

    data_list = ["Jeg vil gerne have lavet den her sætning om til embeddings",
                 "Jeg vil også gerne have den her sætning lavet om, tak"] * 10
    # Create BPE embeddings - (bpe_len x 320) pr element
    bpe_embeddings_list = embedder(data_list, method='bpe', language='da')  # With list input
    bpe_embeddings_string = embedder(data_list[0], method='bpe', language='da')  # With string input
    # Create laser embeddings (1x1024) pr element
    laser_embeddings_list = embedder(data_list, method='sentence', language='da')  # With list input
    laser_embeddings_string = embedder(data_list[0], method='sentence', language='da')  # With string input

