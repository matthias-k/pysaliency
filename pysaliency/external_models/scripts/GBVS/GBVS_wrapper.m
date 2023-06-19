function [ ] = GBVS_rwapper(filename, outname)

    addpath('gbvs')
    addpath('gbvs/algsrc')
    addpath('gbvs/compile')
    addpath('gbvs/initcache')
    addpath('gbvs/saltoolbox')
    addpath(genpath('gbvs/util'))

    img = imread(filename);
    map = gbvs(img);
    saliency_map = map.master_map_resized;
    save(outname, 'saliency_map');
    
