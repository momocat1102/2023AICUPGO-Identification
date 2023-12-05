from model.models import cnn_model, kata_cnn, TransFormer
from model.models_v2 import move_pred_mix_v2
from model.kata import kata_cnn_2

models = {
    "cnn_model": cnn_model,
    "kata_cnn"  : kata_cnn,
    "move_pred_mix_v2": move_pred_mix_v2,
    "kata_cnn_2": kata_cnn_2,
    "transformer": TransFormer
}
