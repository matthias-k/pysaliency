function [ ] = Judd(filename, outname)

    addpath('source/FaceDetect')
    addpath('source/FaceDetect/src')
    addpath('source/LabelMeToolbox/features')
    addpath('source/LabelMeToolbox/imagemanipulation')
    addpath('source/matlabPyrTools')
    addpath('source/SaliencyToolbox')
    addpath('source/voc-release3.1')
    addpath('source/JuddSaliencyModel')

    saliency_map = saliency(filename);
    save(outname, 'saliency_map');
    
