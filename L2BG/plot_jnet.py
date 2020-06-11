def viz_one_net(n):
    res = ''
    for k, v in n.items():
        if v == 'new': res += ' *,'
        else:
            r = (v.split()[2]).split('_')[-1]
            res += f' {r},'
    return res


def viz_jnet(data):
    print('columns: ',  )
    res = ''
    cols = None
    for k, v in data.items():
        if not cols: cols = f'columns: {list(v.keys())}'
        nodes = viz_one_net(v)
        res += '{:>10s}:'.format(k) + f' {nodes} \n'

    res = f'{cols}\n{res}'
    return res



def test():
    data1 = {'task_2': {'conv2d_input1': 'new', 'conv2d_2': 'new', 'conv2d_3': 'conv2d_3 of task_1', 'linear1': 'linear1 of task_1', 'linear2': 'linear2 of task_1'}, 'task_3': {'conv2d_input1': 'new', 'conv2d_2': 'conv2d_2 of task_1', 'conv2d_3': 'new', 'linear1': 'linear1 of task_1', 'linear2': 'linear2 of task_1'}, 'task_4': {'conv2d_input1': 'new', 'conv2d_2': 'new', 'conv2d_3': 'conv2d_3 of task_1', 'linear1': 'linear1 of task_1', 'linear2': 'linear2 of task_1'}, 'task_5': {'conv2d_input1': 'new', 'conv2d_2': 'new', 'conv2d_3': 'conv2d_3 of task_3', 'linear1': 'linear1 of task_3', 'linear2': 'linear2 of task_3'}, 'task_6': {'conv2d_input1': 'new', 'conv2d_2': 'new', 'conv2d_3': 'conv2d_3 of task_4', 'linear1': 'linear1 of task_4', 'linear2': 'linear2 of task_4'}, 'task_7': {'conv2d_input1': 'new', 'conv2d_2': 'new', 'conv2d_3': 'conv2d_3 of task_5', 'linear1': 'linear1 of task_5', 'linear2': 'linear2 of task_5'}, 'task_8': {'conv2d_input1': 'new', 'conv2d_2': 'new', 'conv2d_3': 'conv2d_3 of task_2', 'linear1': 'linear1 of task_2', 'linear2': 'linear2 of task_2'}, 'task_9': {'conv2d_input1': 'new', 'conv2d_2': 'new', 'conv2d_3': 'conv2d_3 of task_3', 'linear1': 'linear1 of task_3', 'linear2': 'linear2 of task_3'}, 'task_10': {'conv2d_input1': 'new', 'conv2d_2': 'new', 'conv2d_3': 'conv2d_3 of task_1', 'linear1': 'linear1 of task_1', 'linear2': 'linear2 of task_1'}}
    viz_jnet(data1)