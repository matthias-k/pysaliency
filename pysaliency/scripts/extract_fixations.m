function [ ] = extract_fixations(filename, datafolder, outname)
	fprintf('Loading %s %s\n', datafolder, filename);
	addpath('DatabaseCode')
	datafile = strcat(filename(1:end-4), 'mat');
	load(fullfile(datafolder, datafile));
	stimFile = eval([datafile(1:end-4)]);
	eyeData = stimFile.DATA(1).eyeData;
	[eyeData Fix Sac] = checkFixations(eyeData);
	fixs = find(eyeData(:,3)==0); % these are the indices of the fixations in the eyeData for a given image and user
	fixations = Fix.medianXY;
	starts = Fix.start;
	save(outname, 'fixations', 'starts', '-v6')  % version 6 makes octave files compatible with scipy
