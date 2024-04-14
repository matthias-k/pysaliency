function [ ] = AIM_wrapper(filename, outname, convolve, filters)

    addpath('AIM');

    saliency_map = AIM(filename, 1.0, convolve, filters);
    save(outname, 'saliency_map', '-v6');
    
