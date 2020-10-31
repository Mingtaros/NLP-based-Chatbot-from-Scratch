import re


def replace_with_enamex(re_group):
    entities = re.findall(r'<(.*?)>(.*?)</(.*?)>', re_group)
    return '<ENAMEX TYPE=%s>%s</ENAMEX>' % (entities[0][0], entities[0][1])


def convert_to_enamex(content):
    entity_types = re.sub(r'<.*?>.*?</.*?>', lambda entity: replace_with_enamex(entity.group()), content)
    print(entity_types)


if __name__ == "__main__":
    with open("dirty_ner_data_2.txt", 'r') as f:
        content = f.read()

    convert_to_enamex(content)
