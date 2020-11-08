import re

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

        #append to train data with format = (new_sentence, {"entities": [(start, end, type)]})
        entity_dictionary = {
            "entities": []
        }

        for entity_type, entity_index in zip(types, index_in_sentence):
            # only append if entities don't overlap
            check_overlap = False
            for entity in entity_dictionary["entities"]:
                if ((entity_index[0] >= entity[0] and entity_index[0] <= entity[1]) or (entity_index[1] <= entity[1] and entity_index[1] >= entity[0])):
                    check_overlap = True

            if (not check_overlap and entity_type == 'DATETIME'):
                entity_dictionary["entities"].append((entity_index[0], entity_index[1], entity_type))
        train_data.append((new_sentence, entity_dictionary))

    return train_data