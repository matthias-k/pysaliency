function [ ] =  RARE2012(filename, outname)

    addpath('source/simplegabortb-v1.0.0')
    addpath('source/Rare2012')

	I = im2double(imread(filename));
    saliency_map = RARE2012(I);
    save(outname, 'saliency_map');
    
