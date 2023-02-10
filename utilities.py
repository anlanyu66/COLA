import numpy as np

def float_bin(number, resolution, num):
    if number == 0:
        return '0' * resolution
    else:
        tmp = round(number * 2 ** (resolution - num))
        if tmp >= 2 ** resolution:
            k = '1' * resolution
        else:
            k = bin(int(tmp)).lstrip("0b")
        return k


def bin_dec(bstr, resolution, num):
    length = len(bstr)
    tmp_o = float(2 ** (num - resolution))
    tmp = 0
    for i in range(length):
        tmp += int(bstr[length - 1 - i]) * tmp_o
        tmp_o *= 2

    return tmp

def quantized(weight, resolution, rate):

    for i, w in enumerate(weight):
        if i % 2 == 0:
            ma, mi = np.max(w), np.min(w)
            ma, mi = np.percentile(w, 99.8), np.percentile(w, 0.2)
            del_w = (ma - mi) / 2 ** resolution
            # print(del_w / np.linalg.norm(w))
            # print(np.max(w), np.min(w))
            w = np.clip(w, mi, ma)
            # print(np.max(w), np.min(w), ma, mi)

            level = np.round((w - mi) / del_w).astype(int)

            prob = np.random.uniform(0, 1, np.array(w).shape + (resolution,))
            prob = prob < rate

            s = prob.shape
            a = np.tile([2 ** a for a in reversed(range(resolution))], s[:len(s) - 1] + (1,))
            error_m = np.sum(prob * a, axis=-1)

            level ^= error_m

            qweight = mi + del_w * level

            weight[i] = qweight

    return weight


def weight_prosessing(weight, resolution, rate):
    pweight = []
    for w in weight:
        if len(w.shape) < 2:
            pweight.append(w)
            continue
        num = int(np.log2(np.max(abs(w))))
        if num < 0:
            num -= 1
        if len(w.shape) == 2:
            in_node, out_node = w.shape
            for i in range(in_node):
                for j in range(out_node):
                    if not w[i][j]:
                        continue
                    binstr = float_bin(abs(w[i][j]), resolution, num)
                    rand = np.random.uniform(0, 1, len(binstr))
                    for k in range(len(binstr)):
                        if rand[k] <= rate:
                            l = list(binstr)
                            l[k] = (int(binstr[k]) + 1) % 2
                            binstr = ''.join(l)
                    tmp = bin_dec(binstr, resolution, num) * np.sign(w[i][j])
                    # print(w[i][j], tmp)
                    w[i][j] = tmp
            pweight.append(w)
        else:
            dim1, dim2, dim3, dim4 = w.shape
            for i in range(dim1):
                for j in range(dim2):
                    for p in range(dim3):
                        for q in range(dim4):
                            if not w[i][j][p][q]:
                                continue
                            binstr = float_bin(abs(w[i][j][p][q]), resolution, num)
                            rand = np.random.uniform(0, 1, len(binstr))
                            for k in range(len(binstr)):
                                if rand[k] <= rate:
                                    l = list(binstr)
                                    l[k] = (int(binstr[k]) + 1) % 2
                                    binstr = ''.join(l)
                            tmp = bin_dec(binstr, resolution, num) * np.sign(w[i][j][p][q])
                            # print(w[i][j][p][q], tmp)
                            w[i][j][p][q] = tmp
            pweight.append(w)

    return pweight

def weight_perturbation(weights, stddev, type):
    for i, layer in enumerate(weights):
        if i % 2 == 0:
            noise = np.random.normal(loc=0.0, scale=stddev, size=layer.shape)
        else:
            continue
        if type == 'awgn':
            layer += noise
        elif type == 'lognormal':
            layer *= np.exp(noise)
    model.set_weights(weights)