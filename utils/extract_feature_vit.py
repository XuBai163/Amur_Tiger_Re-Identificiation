import torch


def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (inputs, labels) in loader:
        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        # ff = torch.FloatTensor(inputs.size(0), 512).zero_()
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')

            # print("Input size:", input_img.size())
            # input_img.size() = ([8, 3, 128, 288])
            outputs = model(input_img)
            # print("Output size:", outputs.size())
            # outputs.size() = ([8, 2048])

            f = outputs.data.cpu()
            # outputs.data.cpu() Change for Resnet + Vit model.
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features