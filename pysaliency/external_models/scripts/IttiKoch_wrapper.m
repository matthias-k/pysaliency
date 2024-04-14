function [ ] = IttiKoch(filename, outname)

    addpath('SaliencyToolbox')

    img = initializeImage(filename);
    params = defaultSaliencyParams;
    salmap = makeSaliencyMap(img, params);
    saliency_map = imresize(salmap.data,img.size(1:2));
    save(outname, 'saliency_map');
    
