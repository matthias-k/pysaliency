function [ ] = CovSal_wrapper(filename, outname, size, quantile, centerbias, modeltype)
    addpath('saliency')

    % options for saliency estimation
    options.size = size;
    options.quantile = quantile;
    options.centerBias = centerbias;
    options.modeltype = modeltype;
    saliency_map = saliencymap(filename, options);
    save(outname, 'saliency_map');
