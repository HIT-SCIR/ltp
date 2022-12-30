# Generated content DO NOT EDIT
class Algorithm:
    """
    The perceptron algorithm.
    algorithm support "AP", "Pa", "PaI", "PaII"
    AP: average perceptron, param is the threads
    PA: parallel average perceptron, param is c(margin)
    """

    def __init__(self, algorithm, param=None):
        pass

class CWSModel:
    def __init__(self, path):
        pass
    def add_feature_rule(self, core, feature, s, b, m, e):
        """
        自定义新feature
        """
        pass
    def batch_predict(self, batch_text, parallelism=True):
        """
        Predict batched sentences
        """
        pass
    def disable_feature_rule(self, core, feature, s, b, m, e):
        """
        移除自定义新 feature
        """
        pass
    def disable_type_rule(self, a, b):
        """
        关闭连续不同类型之间的强制连接/切分
        """
        pass
    def disable_type_rule_d(self, a, b):
        """
        关闭连续不同类型之间的强制连接/切分(双向)
        """
        pass
    def enable_feature_rule(self, core, feature):
        """
        启用自定义新 feature
        """
        pass
    def enable_type_concat(self, a, b):
        """
        开启连续不同类型之间的强制连接
        """
        pass
    def enable_type_concat_d(self, a, b):
        """
        开启连续不同类型之间的强制连接(双向)
        """
        pass
    def enable_type_cut(self, a, b):
        """
        开启连续不同类型之间的强制切分
        """
        pass
    def enable_type_cut_d(self, a, b):
        """
        开启连续不同类型之间的强制切分(双向)
        """
        pass
    @staticmethod
    def load(path):
        """
        Load Model from a path
        """
        pass
    def predict(self, text):
        """
        Predict a sentence
        """
        pass
    def save(self, path):
        """
        Save Model to a path
        """
        pass

class CWSTrainer:
    def __init__(self):
        pass
    @property
    def algorithm(self):
        """
        Get the value of the algorithm parameter.
        """
        pass
    @property
    def compress(self):
        """
        Get the value of the compress parameter.
        """
        pass
    @property
    def epoch(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    def eval(self, model):
        """
        Eval a Segmentor model
        """
        pass
    @property
    def eval_threads(self):
        """
        Get the value of the eval_threads parameter.
        """
        pass
    def load_eval_data(self, path):
        """
        Load Eval Data from a path
        """
        pass
    def load_train_data(self, path):
        """
        Load Train Data from a path
        """
        pass
    @property
    def ratio(self):
        """
        Get the value of the ratio parameter.
        """
        pass
    @property
    def shuffle(self):
        """
        Get the value of the shuffle parameter.
        """
        pass
    @property
    def threshold(self):
        """
        Get the value of the threshold parameter.
        """
        pass
    def train(self):
        """
        Train a Segmentor model
        """
        pass
    @property
    def verbose(self):
        """
        Get the value of the verbose parameter.
        """
        pass

class CharacterType:
    """
    Digit: Digit character. (e.g. 0, 1, 2, ...)
    Roman: Roman character. (e.g. A, B, C, ...)
    Hiragana: Japanese Hiragana character. (e.g. あ, い, う, ...)
    Katakana: Japanese Katakana character. (e.g. ア, イ, ウ, ...)
    Kanji: Kanji (a.k.a. Hanzi or Hanja) character. (e.g. 漢, 字, ...)
    Other: Other character.
    """

class Model:
    def __init__(self, path, model_type=ModelType.Auto):
        pass
    def batch_predict(self, *args, parallelism=True):
        """
        Predict batched sentences
        """
        pass
    @staticmethod
    def load(path, model_type=ModelType.Auto):
        """
        Load Model from a path
        """
        pass
    def predict(self, *args):
        """
        Predict a sentence
        """
        pass
    def save(self, path):
        """
        Save Model to a path
        """
        pass
    def specialize(self):
        """
        Specialize the Model
        """
        pass

class ModelType:
    def __init__(self, model_type=None):
        pass

class NERModel:
    def __init__(self, path):
        pass
    def batch_predict(self, batch_words, batch_pos, parallelism=True):
        """
        Predict batched sentences
        """
        pass
    @staticmethod
    def load(path):
        """
        Load Model from a path
        """
        pass
    def predict(self, words, pos):
        """
        Predict a sentence
        """
        pass
    def save(self, path):
        """
        Save Model to a path
        """
        pass

class NERTrainer:
    def __init__(self, labels):
        pass
    @property
    def algorithm(self):
        """
        Get the value of the algorithm parameter.
        """
        pass
    @property
    def compress(self):
        """
        Get the value of the compress parameter.
        """
        pass
    @property
    def epoch(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    def eval(self, model):
        """
        Eval a Segmentor model
        """
        pass
    @property
    def eval_threads(self):
        """
        Get the value of the eval_threads parameter.
        """
        pass
    def load_eval_data(self, path):
        """
        Load Eval Data from a path
        """
        pass
    def load_train_data(self, path):
        """
        Load Train Data from a path
        """
        pass
    @property
    def ratio(self):
        """
        Get the value of the ratio parameter.
        """
        pass
    @property
    def shuffle(self):
        """
        Get the value of the shuffle parameter.
        """
        pass
    @property
    def threshold(self):
        """
        Get the value of the threshold parameter.
        """
        pass
    def train(self):
        """
        Train a Segmentor model
        """
        pass
    @property
    def verbose(self):
        """
        Get the value of the verbose parameter.
        """
        pass

class POSModel:
    def __init__(self, path):
        pass
    def batch_predict(self, batch_words, parallelism=True):
        """
        Predict batched sentences
        """
        pass
    @staticmethod
    def load(path):
        """
        Load Model from a path
        """
        pass
    def predict(self, words):
        """
        Predict a sentence
        """
        pass
    def save(self, path):
        """
        Save Model to a path
        """
        pass

class POSTrainer:
    def __init__(self, labels):
        pass
    @property
    def algorithm(self):
        """
        Get the value of the algorithm parameter.
        """
        pass
    @property
    def compress(self):
        """
        Get the value of the compress parameter.
        """
        pass
    @property
    def epoch(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    def eval(self, model):
        """
        Eval a Segmentor model
        """
        pass
    @property
    def eval_threads(self):
        """
        Get the value of the eval_threads parameter.
        """
        pass
    def load_eval_data(self, path):
        """
        Load Eval Data from a path
        """
        pass
    def load_train_data(self, path):
        """
        Load Train Data from a path
        """
        pass
    @property
    def ratio(self):
        """
        Get the value of the ratio parameter.
        """
        pass
    @property
    def shuffle(self):
        """
        Get the value of the shuffle parameter.
        """
        pass
    @property
    def threshold(self):
        """
        Get the value of the threshold parameter.
        """
        pass
    def train(self):
        """
        Train a Segmentor model
        """
        pass
    @property
    def verbose(self):
        """
        Get the value of the verbose parameter.
        """
        pass

class Trainer:
    def __init__(self, model_type=ModelType.Auto, labels=None):
        pass
    @property
    def algorithm(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    @property
    def compress(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    @property
    def epoch(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    def eval(self, model):
        """
        Eval a Segmentor model
        """
        pass
    @property
    def eval_threads(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    def load_eval_data(self, path):
        """
        Load Eval Data from a path
        """
        pass
    def load_train_data(self, path):
        """
        Load Train Data from a path
        """
        pass
    @property
    def ratio(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    @property
    def shuffle(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    @property
    def threshold(self):
        """
        Get the value of the epoch parameter.
        """
        pass
    def train(self):
        """
        Train a model
        """
        pass
    @property
    def verbose(self):
        """
        Get the value of the epoch parameter.
        """
        pass
