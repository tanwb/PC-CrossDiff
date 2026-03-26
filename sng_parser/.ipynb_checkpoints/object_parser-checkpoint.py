from formatter import NullWriter

import spacy

# 加载预训练的 SpaCy 模型
nlp = spacy.load("/dsjxytest/can_not_remove/model/vg3d/diff_MCLN-main/sng_parser/en_core_web_sm-3.3.0/en_core_web_sm/en_core_web_sm-3.3.0")


def extract_descriptions(sentence):
    # 解析句子
    doc = nlp(sentence)
    descriptions = []

    # 遍历句子中的每个词语
    for token in doc:
        # 检查词语的依存标签，寻找描述关系
        # 'amod' 表示形容词修饰，'attr' 表示属性修饰
        if token.dep_ in ('amod', 'attr'):
            # 获取描述对象（通常是名词）
            head = token.head
            # 确保描述对象是名词
            if head.pos_ == 'NOUN':
                description = {
                    'object': head.text,
                    'description': token.text
                }
                descriptions.append(description)

    # 返回描述对象及其描述关系的列表
    a=""

    for i, desc in enumerate(descriptions, start=1):
        a=a+desc['description']+" "+desc['object']


    return a