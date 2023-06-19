function [ ] = ContextAwareSaliency(filename, outname)

    addpath('source')

	file_names{1} = filename;
	img = imread(filename);
	[nrows ncols cc] = size(img);
	MOV = saliency(file_names);
	saliency_map = MOV{1}.SaliencyMap;
    saliency_map = imresize(saliency_map, [nrows, ncols]);
    save(outname, 'saliency_map');
    
