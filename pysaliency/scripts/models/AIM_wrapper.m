function [ ] = BruceTsotso(filename, outname)
%function [ ] = BruceTsotso(filename, outname, filters)

    addpath('AIM')

    saliency_map = AIM(filename, 1.0, 1, '31jade950');
    save(outname, 'saliency_map');
    
