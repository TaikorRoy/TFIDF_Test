import pynlpir


class tf_parser(object):
    def __init__(self, file):
        with open(file, 'r', encoding="utf-8") as f:
            content = f.read()
            pynlpir.open()
            result = pynlpir.segment(content, pos_tagging=False)
        self.PoS = result

    def __call__(self):
        vector_space = set(self.PoS)
        TF_vector = {element: 0 for element in vector_space}
        for pos in self.PoS:
            if pos in vector_space:
                TF_vector[pos] += 1
        return TF_vector

f_path = r"C:\workspace\pynlpir_TFIDF\test.txt"

parser = tf_parser(f_path)   #construct a parser
print(parser())
