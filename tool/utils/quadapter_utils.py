import copy
import numpy as np
import random

def real_round(x):
    if x < 0:
        return np.ceil(x - 0.5)
    elif x > 0:
        return np.floor(x + 0.5)
    else:
        return 0

def int_get_min_max(num_bits, frac_bits):
    num_value_bits = num_bits - 1
    min_value = -(2 ** num_value_bits) / (2 ** frac_bits)
    max_value = (2 ** num_value_bits - 1) / (2 ** frac_bits)
    return (min_value, max_value)


def quantize_int(float_value, num_bits, frac_bits):
    min_value, max_value = int_get_min_max(num_bits, frac_bits)
    float_value = np.clip(float_value, min_value, max_value)

    scaled = float_value * (2 ** frac_bits)
    quant = np.int32(scaled)
    if type(quant) == np.ndarray:
        incs = (scaled - quant) >= 0.5
        decs = (scaled - quant) <= -0.5
        quant[incs] += 1
        quant[decs] -= 1
    else:
        if scaled - quant >= 0.5:
            quant += 1
        elif scaled - quant <= -0.5:
            quant -= 1

    return np.int32(quant)


def forward_DNN(x, ilpModel):
    model = ilpModel.deep_model

    all_layers = copy.copy(ilpModel.dense_layers)
    all_layers.append(ilpModel.output_layer)
    for i, l in enumerate(all_layers):
        tf_layer = model.dense_layers[i]
        w_cont, b_cont = tf_layer.get_weights()
        out_x = []
        before_relu_x = []
        ifLast = (i == len(ilpModel.dense_layers))

        for out_index in range(l.layer_size):
            weight_row = np.float32(w_cont[:, out_index])
            bias = np.float32(b_cont[out_index])

            accumulator = np.float32(np.array(weight_row * x).sum() + bias)

            before_relu_x.append(accumulator)

            if not ifLast:
                if accumulator < 0:
                    accumulator = 0

            out_x.append(accumulator)

        l.set_realVal(before_relu_x)

        x = np.array(out_x)

    return x

def forward_DNN_multi(x_set, ilpModel):
    model = ilpModel.deep_model

    all_layers = copy.copy(ilpModel.dense_layers)
    all_layers.append(ilpModel.output_layer)

    for original_x in x_set:
        x = original_x
        for i, l in enumerate(all_layers):
            tf_layer = model.dense_layers[i]
            w_cont, b_cont = tf_layer.get_weights()
            out_x = []
            before_relu_x = []
            ifLast = (i == len(ilpModel.dense_layers))

            for out_index in range(l.layer_size):
                weight_row = np.float32(w_cont[:, out_index])
                bias = np.float32(b_cont[out_index])

                accumulator = np.float32(np.array(weight_row * x).sum() + bias)

                before_relu_x.append(accumulator)

                if not ifLast:
                    if accumulator < 0:
                        accumulator = 0

                out_x.append(accumulator)

            l.set_realVal_multi(before_relu_x)

            x = np.array(out_x)


# randomly pick K samples
def backdoor_random(x_test, y_test, K, originalCls, targetCls):

    sample_list = [i for i in range(len(x_test))]
    sample_list = random.sample(sample_list, K * 100)
    real_sample_ID = []
    real_sample_input = []
    sample_label = []
    real_sample_label = []

    if originalCls < 10:
        for x_index in sample_list:
            sample_label.append(y_test[x_index])
            if len(real_sample_input) >= K:
                break
            elif y_test[x_index] == originalCls:
                real_sample_ID.append(x_index)
                real_sample_input.append(x_test[x_index])
                real_sample_label.append(y_test[x_index])

            else:
                continue
    else:
        for x_index in sample_list:
            sample_label.append(y_test[x_index])
            if len(real_sample_input) >= K:
                break
            elif y_test[x_index] != targetCls:
                real_sample_ID.append(x_index)
                real_sample_input.append(x_test[x_index])
                real_sample_label.append(y_test[x_index])

            else:
                continue

    assert len(real_sample_label) == K

    return real_sample_input, real_sample_ID, real_sample_label
