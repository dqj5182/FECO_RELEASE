def get_contact_thres(backbone_type='hamer'):
    if backbone_type == 'vit-h-14':
        return 0.5
    elif backbone_type == 'vit-l-16':
        return 0.5
    elif backbone_type == 'vit-b-16':
        return 0.5
    elif backbone_type == 'vit-s-16':
        return 0.5
    elif backbone_type == 'resnet-152':
        return 0.5
    elif backbone_type == 'resnet-101':
        return 0.5
    elif backbone_type == 'resnet-50':
        return 0.5
    elif backbone_type == 'resnet-34':
        return 0.5
    elif backbone_type == 'resnet-18':
        return 0.5
    else:
        raise NotImplementedError