import matplotlib.pyplot as plt
import torch as t

data=['colored_mnist/colored_mnist_bias_0.01_noise_0.1.pt',
        'biased_mnist/biased_mnist_bias_0.01_noise_0.0.pt',
        'corrupted_cifar/corrupted_cifar_bias_0.01_noise_0.01.pt',
        'bar/bar_noise_0.01.pt',
        'bffhq/bffhq_noise_0.01.pt']


for idx in range(len(data)):
    directory = '../dataset/'+data[idx]
    data_type = data[idx].split('/')[0]
    data_dict = t.load(directory)['train']
    _data = data_dict['data'].permute((0,2,3,1))
    _label = data_dict['label']
    _gtlabel = data_dict['gt_label']
    _blabel = data_dict['b_label']

    
    maj_clean = t.where( (_blabel == _gtlabel) & (_gtlabel == _label) )[0]
    maj_noisy = t.where( (_blabel == _gtlabel) & (_gtlabel != _label) )[0]
    min_clean = t.where( (_blabel != _gtlabel) & (_gtlabel == _label) )[0]
    min_noisy = t.where( (_blabel != _gtlabel) & (_gtlabel != _label) )[0]
    print("[Before Denoising] Clean_Major %d / Clean_Minor %d / Noisy_Major %d/ Noisy_Minor %d" %(len(maj_clean), len(min_clean), len(maj_noisy), len(min_noisy)))

    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(2,5,i+1)

        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title("%d"%(_label[i]))

        subplot.imshow(_data[i])

    plt.savefig('../log/plots/'+data_type+'.png')

