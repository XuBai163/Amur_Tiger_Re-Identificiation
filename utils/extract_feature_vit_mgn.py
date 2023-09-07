import torch


def extract_feature(model, loader):
    all_features = []
    
    for (inputs, labels) in loader:
        ff = torch.FloatTensor(inputs.size(0), 4096).zero_().to('cuda')
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')

            outputs = model(input_img)

            ff += outputs.data

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        all_features.append(ff.cpu())

    features = torch.cat(all_features, dim=0)
    return features
