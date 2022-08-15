# Generated content DO NOT EDIT
def eisner(scores, stn_length, remove_root=False):
    """
    Decode with Eisner's algorithm
    """
    pass

def get_entities(tags):
    """
    Convert Tags to Entities
    """
    pass

def viterbi_decode_postprocess(history, last_tags, stn_length, labels_num):
    """
    Viterbi Decode Postprocessing
    """
    pass

class Hook:
    def __init__(self):
        pass
    def add_word(self, word, freq=None):
        """
        add words to the hook, the freq can be zero
        """
        pass
    def hook(self, sentence, words):
        """
        hook to the new words
        """
        pass

class StnSplit:
    def __init__(self):
        pass
    def batch_split(self, batch_text, threads=8):
        """
        batch split to sentences
        """
        pass
    @property
    def bracket_as_entity(self):
        """
        Get the value of the bracket_as_entity option.
        """
        pass
    @property
    def en_quote_as_entity(self):
        """
        Get the value of the en_quote_as_entity option.
        """
        pass
    def split(self, text):
        """
        split to sentences
        """
        pass
    @property
    def use_en(self):
        """
        Get the value of the use_en option.
        """
        pass
    @property
    def use_zh(self):
        """
        Get the value of the use_zh option.
        """
        pass
    @property
    def zh_quote_as_entity(self):
        """
        Get the value of the zh_quote_as_entity option.
        """
        pass
