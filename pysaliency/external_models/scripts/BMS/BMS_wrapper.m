function [ ] = BMS(filename, outname)

    addpath('source')

	directory = sprintf('tmp_%s', int2str(randi(1e15)));
	while exist(directory, 'dir')
		directory = sprintf('tmp_%s', int2str(randi(1e15)));
	end
	mkdir(directory);
	copyfile(filename, directory)
	
	output_directory = fullfile(directory, 'output');
	BMS(directory, output_directory, false);
    
	[path, name, ext] = fileparts(filename);
	outfile = fullfile(output_directory, sprintf('%s.png', name));
	saliency_map = imread(outfile);
    save(outname, 'saliency_map');

	rmdir(directory, 's');
