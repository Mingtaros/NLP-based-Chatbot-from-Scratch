# Add NER to Indonesian Model
# NER = Named Entity Recognizer
# Reference: https://spacy.io/usage/training

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import re

spacy.prefer_gpu() # use GPU if available, else use CPU


def replace_with_entity(re_group):
    # replace every ENAMEX tag in sentence with the entity itself
    entity = re.findall(r'<ENAMEX TYPE=.*?>(.*?)</ENAMEX>', re_group)
    return entity[0]


def find_index_in_sentence(patterns, sentence):
    # find index of start and end of occured entity in sentence
    index_in_sentence = []
    for pattern in patterns:
        # add regex special characters in pattern with backslash
        pattern = re.sub('\(', '\\(', pattern)
        pattern = re.sub('\)', '\\)', pattern)
        pattern = re.sub('\+', '\\+', pattern)
        pattern = re.sub('\$', '\\$', pattern)

        index_in_sentence.append([(index.start(0), index.end(0)) for index in re.finditer(pattern, sentence)][0])

    return index_in_sentence


def read_enamex_file(filename):
    # open file with enamex XML Entities
    # convert it to standard SpaCy training dataset for NER
    with open(filename, 'r') as f:
        content = f.read()

    # remove \t...\n from the content data
    content_removed_endline = re.sub('\t.*?\n', '', content)

    # split the content to list of sentences
    before_sentences = content_removed_endline.split(".")
    train_data = []

    # converted the splitted content to list of tuple
    # showing where the entity occur and what's the entity
    for each_sentence in before_sentences:
        if each_sentence == "":
            continue
        # find enamex for every sentence
        entities = re.findall(r'<ENAMEX TYPE=.*?>(.*?)</ENAMEX>', each_sentence)
        types = re.findall(r'<ENAMEX TYPE=\"(.*?)\">.*?</ENAMEX>', each_sentence)

        # replace XML ENAMEX with entity
        new_sentence = re.sub(r'<ENAMEX TYPE=.*?>.*?</ENAMEX>', lambda enamex: replace_with_entity(enamex.group()), each_sentence)

        index_in_sentence = find_index_in_sentence(entities, new_sentence)

        # append to train data with format = (new_sentence, {"entities": [(start, end, type)]})
        entity_dictionary = {
            "entities": []
        }

        for entity_type, entity_index in zip(types, index_in_sentence):
            if (entity_type != "DATETIME"):
                continue

            # only append if entities don't overlap
            check_overlap = False
            for entity in entity_dictionary["entities"]:
                if ((entity_index[0] >= entity[0] and entity_index[0] <= entity[1]) or (entity_index[1] <= entity[1] and entity_index[1] >= entity[0])):
                    check_overlap = True

            if (not check_overlap):
                entity_dictionary["entities"].append((entity_index[0], entity_index[1], entity_type))

        train_data.append((new_sentence, entity_dictionary))

    return train_data


@plac.annotations(
    model = ("Model name. Defaults to blank 'id' model.", "option", "m", str),
    output_dir = ("Optional output directory", "option", "o", Path),
    n_iter = ("Number of training iterations", "option", "n", int))
def main(model = None, output_dir = None, n_iter = 100):
    # training the model and also saving it to output_dir.
    # returns list of NER Loss in every iteration
    try:
        assert model is not None
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    except:
        # use blank model if the model not found / not inputted
        nlp = spacy.blank('id')
        print("Created blank 'id' model")

    # pipeline component for NER
    if ("ner" not in nlp.pipe_names):
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last = True)
    else:
        ner = nlp.get_pipe("ner")

    # added Entity types to the model
    train_data = read_enamex_file("ner_data/ner_data.txt") + read_enamex_file("ner_data/ner_data_2.txt")

    # add entity type label to NER model
    for _, annotations in train_data:
        for entities in annotations["entities"]:
            ner.add_label(entities[2])

    ner_loss_ = []

    # do training (but only NER and some other exceptions)
    train_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    not_ner_pipes = [pipe for pipe in nlp.pipe_names if (pipe not in train_exceptions)]
    with nlp.disable_pipes(*not_ner_pipes) and warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category = UserWarning, module = 'spacy')

        if model is None:
            nlp.begin_training()

        # Start training for n_iter times
        for iteration in range(n_iter):
            print("===================Iteration:", iteration)
            random.shuffle(train_data)
            losses = {}

            # train in batches
            batches = minibatch(train_data, size = compounding(4.0, 32.0, 1.001))

            for each_batch in batches:
                text, annotations = zip(*each_batch)

                nlp.update(
                    text, # batch of text
                    annotations, # batch of annotations
                    drop = 0.5, # NN dropout rate
                    losses = losses
                )
            print("LOSSES:", losses, "\n")
            ner_loss_.append(losses['ner'])

    # save model to output_dir
    if (output_dir is not None):
        output_dir = Path(output_dir)
        if (not output_dir.exists()):
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Model saved at:", output_dir)

    return ner_loss_ # the loss result can then be plotted 


if __name__ == "__main__":
    ner = read_enamex_file("ner_data/ner_data.txt")

    for e in ner:
        if (e[1]['entities']):
            print(e)
