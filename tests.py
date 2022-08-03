import torch

def batched_index_select(input, dim, index):
    views = [1 if i != dim else -1 for i in range(len(input.shape))]
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    # making the first dim of output be B
    return torch.cat(torch.chunk(torch.gather(input, dim, index), chunks=index.shape[0], dim=dim), dim=0)


mask = torch.randn([3, 100, 224, 224])
score = torch.randn(3 ,100)
_, keep = torch.topk(score, 20)

mask2 = mask.view(-1, 224, 224)
keep2 = keep.view(-1, 20)

a = mask2[keep2]

score = score.view(-1)
b = score[keep2]
print(a.shape)
print(b.shape)
# print(keep)
# print(mask)
