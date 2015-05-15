import pynlpir
import os

def folder_parser(folder_path):
    file_names = os.listdir(folder_path)
    dirname = os.path.dirname(folder_path)
    files = list()
    for file_name in file_names:
        files.append(os.path.join(dirname, file_name))
    return files

class TfParser(object):
    def __init__(self, file):
        with open(file, 'r', encoding="utf-8") as f:
            content = f.read()
            pynlpir.open()
            result = pynlpir.segment(content, pos_tagging=False)
        self.PoS = result

    def __call__(self):
        vector_space = set(self.PoS)
        tf_vector = {element: 0 for element in vector_space}
        for pos in self.PoS:
            if pos in vector_space:
                tf_vector[pos] += 1
        return vector_space, tf_vector

f_path = r"C:\workspace\pynlpir_TFIDF\test.txt"


#parser = TfParser(f_path)   #construct a parser
#print(parser())


class BatchProcessor(object):
    def __init__(self, folder_path):
        tf_vector = list()
        global_pos = set()
        files = folder_parser(folder_path)
        print(files)
        """
        for file in files:
            parser = TfParser(file)   #construct a parser
            pos_buffer, vector_buffer = parser()
            tf_vector.append(vector_buffer)
            global_pos = global_pos | pos_buffer
        print(global_pos)
        """
folder_path = r'C:\workspace\Github_MasterRepository\TFIDF_Test\test_data'
bp = BatchProcessor(folder_path)

