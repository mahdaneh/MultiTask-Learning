"""This version considers task's datasets have equal number of labeled samples
"""


import util
import MTL_pairwise as mtl

def main():
    """"options for criterion is wasserstien, h_divergence"""
    # criterion = ['wasserstien', 'h_divergence']
    itertn = 6

    for c3_value in [0.5, 0.2, 1]:
        for trial in range(1,itertn):
            args = {'img_size': 28,
                    'chnnl': 1,
                    'lr': 0.01,
                    'momentum': 0.9,
                    'epochs': 100,
                    'tr_smpl': 3000,
                    'test_smpl': 10000,
                    'tsk_list': ['mnist', 'svhn', 'm_mnist'],
                    'grad_weight': 1,
                    'Trials': trial,
                    'criterion': 'h_divergence',
                    'c3':c3_value}
            ft_extrctor_prp = {'layer1': {'conv': [1, 32, 5, 1, 2], 'relu': [], 'maxpool': [3, 2, 0]},
                               'layer2': {'conv': [32, 64, 5, 1, 2], 'relu': [], 'maxpool': [3, 2, 0]}}

            hypoth_prp = {
                'layer3': {'fc': [util.in_feature_size(ft_extrctor_prp, args['img_size']), 128], 'act_fn': 'relu'},
                'layer4': {'fc': [128, 10], 'act_fn': 'softmax'}}

            discrm_prp = {'reverse_gradient': {},
                          'layer3': {'fc': [util.in_feature_size(ft_extrctor_prp, args['img_size']), 128],
                                     'act_fn': 'relu'},
                          'layer4': {'fc': [128, 1], 'act_fn': 'sigm'}}

            mtl.MTL_pairwise(ft_extrctor_prp, hypoth_prp, discrm_prp, **args)



if __name__ == '__main__':
    main()



