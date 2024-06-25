function [ ] =  RARE2012(filename, outname)

    addpath('source/VisualAttention-Rare2012-55ba7414b971429e5e899ddfa574e4235fc806e6')

	I = im2double(imread(filename));
    saliency_map = rare2012(I);
    save(outname, 'saliency_map');

