"""Helper functions for code sanity"""
import numpy as np
from tensorflow.keras.callbacks import Callback

def regular_encode(texts, tokenizer, maxlen=512):
    """
    This function is from the kernel: 
    https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta
    """
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])

# def build_model(transformer, max_len=512):
#     """
#     https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
#     """
#     input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     sequence_output = transformer(input_word_ids)[0]
#     cls_token = sequence_output[:, 0, :]
#     out = Dense(1, activation='sigmoid')(cls_token)
    
#     model = Model(inputs=input_word_ids, outputs=out)
#     model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
#     return model

def roc_auc(predictions,target):
    """
    ROC-AUC value for binary classification.
    From:
    https://www.kaggle.com/tanulsingh077/
    """   
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

class RocAucEvaluation(Callback):
    '''
    https://www.kaggle.com/tarunpaparaju/jigsaw-multilingual-toxicity-eda-models/output#Modeling-
    '''
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))