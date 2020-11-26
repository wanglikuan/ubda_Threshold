import torch

def trimmed_mean(g_list, workers, trimK):
    workers_num = len(workers)
    g_trimmed_mean = []
    for p_idx, g_layer in enumerate(g_list[0]):
        g_trimmed_mean_layer = torch.zeros_like(g_layer.data)
        g_layer_list = []
        for w in workers:
            g_layer_list.append(g_list[w - 1][p_idx])
        data_dim = g_layer_list[0].dim()
        tensor_max = torch.min(torch.topk(torch.stack(g_layer_list, data_dim), trimK)[0], -1)[0]
        tensor_min = -torch.min(torch.topk(-torch.stack(g_layer_list, data_dim), trimK)[0], -1)[0]

        for w in workers:
            max_mask = g_list[w - 1][p_idx].data >= tensor_max
            min_mask = g_list[w - 1][p_idx].data <= tensor_min

            tmp_layer = g_list[w - 1][p_idx].data + torch.zeros_like(g_list[w - 1][p_idx].data)
            tmp_layer[max_mask] = 0
            tmp_layer[min_mask] = 0

            g_list[w - 1][p_idx] = tmp_layer

            g_trimmed_mean_layer.data += g_list[w-1][p_idx].data / (workers_num - 2 * trimK)
        g_trimmed_mean.append(g_trimmed_mean_layer)
    return g_trimmed_mean

a = [[], [], []]
a[0] = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
a[1] = torch.tensor([[2, 1], [4, 3]], dtype=torch.float64)
a[2] = torch.tensor([[3, 4], [2, 1]], dtype=torch.float64)

g_list = [[], [], []]
g_list2 = [a[0], a[1], a[2]]
workers = [1, 2, 3]
for i in range(len(a)):
    for w_layer in a[i]:
        g_list[i].append(w_layer)


print(g_list)
# print(g_list2)
print(a)

trimmed_mean(g_list, workers, 1)
# trimmed_mean(g_list2, workers, 1)

print(g_list)
# print(g_list2)
print(a)

