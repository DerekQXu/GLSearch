from options import opt


def get_flags_with_prefix_as_list(prefix):
    rtn = []
    i_check = 1  # one-based
    d = vars(opt)

    for k in d.keys():
        v = d[k]
        sp = k.split(prefix)
        if len(sp) == 2 and sp[0] == '' and sp[1].startswith('_'):
            id = int(sp[1][1:])
            if i_check != id:
                raise ValueError('Wrong flag format {}={} '
                                 '(should start from _1'.format(k, v))
            rtn.append(v)
            i_check += 1
    return rtn
