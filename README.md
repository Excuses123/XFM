# XFM

Xgboost+FM for CTR predict.
Xgboost擅长处理数值特征，FM擅长类别特征，将数值特征放入Xgboost训练后把中间结果输出作为类别特征，拼上原来的特征，最后放入FM进行训练
（瞎写的，只是一个架子，效果可能不好，需要继续优化）

