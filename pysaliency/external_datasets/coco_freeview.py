import json
import os
import zipfile

import numpy as np
from tqdm import tqdm

from ..datasets import ScanpathFixations, Scanpaths, create_subset
from ..utils import TemporaryDirectory, atomic_directory_setup, download_and_check, filter_files
from .coco_search18 import _prepare_stimuli
from .utils import _load, create_stimuli

TEST_STIMULUS_INDICES = [
    1, 5, 10, 11, 12, 17, 24, 31, 35, 41,
    62, 65, 69, 71, 73, 77, 79, 83, 86, 102,
    103, 104, 105, 106, 110, 137, 140, 157, 164, 165,
    173, 181, 188, 201, 203, 206, 214, 216, 217, 226,
    231, 235, 236, 240, 241, 256, 262, 263, 267, 270,
    277, 279, 280, 283, 288, 289, 301, 302, 303, 308,
    322, 325, 329, 332, 337, 338, 339, 341, 343, 355,
    356, 364, 368, 373, 380, 382, 388, 398, 404, 409,
    413, 414, 415, 422, 426, 433, 435, 438, 441, 442,
    446, 451, 455, 469, 470, 482, 483, 486, 493, 495,
    498, 501, 505, 506, 508, 509, 518, 524, 525, 529,
    535, 537, 541, 543, 551, 553, 600, 601, 616, 618,
    621, 622, 623, 626, 629, 631, 634, 637, 640, 652,
    653, 655, 658, 662, 666, 667, 674, 680, 681, 693,
    701, 703, 707, 708, 716, 721, 725, 740, 753, 771,
    786, 789, 794, 806, 808, 812, 820, 826, 840, 841,
    842, 857, 904, 906, 907, 909, 910, 919, 923, 930,
    958, 960, 965, 977, 979, 989, 990, 997, 999, 1008,
    1013, 1037, 1042, 1045, 1046, 1047, 1048, 1050, 1060, 1061,
    1065, 1074, 1077, 1091, 1093, 1109, 1115, 1119, 1120, 1126,
    1131, 1132, 1137, 1139, 1142, 1156, 1158, 1172, 1174, 1175,
    1178, 1182, 1185, 1196, 1200, 1203, 1213, 1229, 1236, 1240,
    1246, 1248, 1249, 1253, 1255, 1262, 1273, 1274, 1277, 1285,
    1289, 1292, 1295, 1300, 1301, 1306, 1312, 1316, 1320, 1322,
    1324, 1328, 1336, 1342, 1351, 1352, 1355, 1364, 1370, 1371,
    1373, 1388, 1391, 1394, 1398, 1406, 1412, 1421, 1422, 1426,
    1430, 1436, 1443, 1444, 1447, 1449, 1456, 1465, 1466, 1467,
    1469, 1473, 1480, 1484, 1490, 1501, 1508, 1513, 1515, 1518,
    1524, 1532, 1536, 1540, 1543, 1546, 1552, 1569, 1574, 1577,
    1585, 1589, 1590, 1591, 1596, 1601, 1611, 1612, 1624, 1626,
    1628, 1646, 1651, 1674, 1676, 1684, 1686, 1691, 1698, 1701,
    1704, 1709, 1712, 1713, 1715, 1716, 1743, 1749, 1751, 1753,
    1764, 1767, 1768, 1774, 1779, 1782, 1784, 1785, 1790, 1791,
    1792, 1803, 1811, 1815, 1816, 1820, 1821, 1829, 1830, 1833,
    1851, 1855, 1859, 1869, 1884, 1888, 1893, 1902, 1903, 1905,
    1906, 1920, 1922, 1924, 1925, 1932, 1936, 1940, 1942, 1943,
    1944, 1954, 1955, 1956, 1959, 1962, 1973, 1975, 1978, 1980,
    1985, 1986, 1989, 1995, 1997, 2001, 2004, 2014, 2018, 2019,
    2020, 2025, 2029, 2032, 2033, 2040, 2044, 2048, 2053, 2054,
    2077, 2083, 2084, 2088, 2090, 2097, 2107, 2108, 2110, 2118,
    2119, 2125, 2129, 2133, 2134, 2143, 2176, 2181, 2192, 2193,
    2195, 2197, 2209, 2211, 2223, 2226, 2228, 2233, 2244, 2247,
    2251, 2254, 2257, 2260, 2269, 2277, 2282, 2284, 2289, 2291,
    2292, 2294, 2296, 2304, 2305, 2319, 2321, 2328, 2343, 2344,
    2349, 2351, 2353, 2355, 2357, 2366, 2370, 2374, 2376, 2386,
    2387, 2397, 2399, 2404, 2410, 2414, 2432, 2440, 2443, 2452,
    2454, 2455, 2456, 2457, 2464, 2465, 2480, 2488, 2491, 2499,
    2500, 2507, 2515, 2516, 2524, 2527, 2531, 2533, 2534, 2536,
    2540, 2549, 2557, 2578, 2580, 2587, 2590, 2591, 2601, 2612,
    2619, 2643, 2646, 2647, 2648, 2649, 2655, 2661, 2665, 2667,
    2672, 2674, 2676, 2683, 2689, 2696, 2697, 2701, 2702, 2712,
    2716, 2738, 2739, 2741, 2747, 2748, 2753, 2754, 2757, 2760,
    2764, 2765, 2776, 2781, 2784, 2786, 2789, 2797, 2798, 2810,
    2820, 2824, 2825, 2829, 2843, 2846, 2847, 2848, 2855, 2864,
    2867, 2869, 2874, 2879, 2883, 2885, 2888, 2891, 2898, 2904,
    2909, 2911, 2923, 2928, 2931, 2950, 2955, 2957, 2958, 2962,
    2967, 2968, 2973, 2979, 2981, 2990, 2995, 3007, 3043, 3054,
    3057, 3065, 3067, 3069, 3071, 3079, 3081, 3084, 3090, 3103,
    3105, 3115, 3122, 3126, 3130, 3134, 3138, 3148, 3153, 3169,
    3171, 3179, 3183, 3190, 3194, 3196, 3202, 3203, 3204, 3210,
    3215, 3220, 3224, 3233, 3235, 3239, 3242, 3244, 3245, 3248,
    3268, 3272, 3277, 3286, 3296, 3297, 3301, 3303, 3306, 3318,
    3324, 3327, 3329, 3330, 3331, 3336, 3337, 3340, 3345, 3346,
    3349, 3352, 3363, 3370, 3375, 3379, 3385, 3386, 3395, 3400,
    3406, 3409, 3411, 3423, 3428, 3437, 3440, 3446, 3447, 3452,
    3461, 3467, 3468, 3469, 3480, 3487, 3488, 3490, 3501, 3502,
    3511, 3518, 3520, 3530, 3554, 3559, 3564, 3573, 3578, 3579,
    3583, 3588, 3589, 3602, 3603, 3607, 3614, 3620, 3632, 3646,
    3655, 3662, 3664, 3667, 3675, 3683, 3689, 3698, 3712, 3719,
    3734, 3735, 3736, 3737, 3738, 3740, 3746, 3752, 3757, 3765,
    3769, 3770, 3775, 3779, 3781, 3783, 3784, 3791, 3809, 3810,
    3811, 3818, 3827, 3833, 3840, 3851, 3859, 3860, 3862, 3863,
    3876, 3890, 3891, 3902, 3903, 3904, 3908, 3911, 3912, 3916,
    3926, 3927, 3930, 3935, 3954, 3957, 3964, 3968, 3971, 3973,
    3994, 3997, 4001, 4003, 4004, 4009, 4011, 4012, 4013, 4014,
    4018, 4020, 4021, 4023, 4031, 4037, 4045, 4051, 4055, 4065,
    4066, 4067, 4068, 4071, 4073, 4078, 4080, 4085, 4104, 4108,
    4112, 4125, 4128, 4139, 4141, 4145, 4150, 4151, 4152, 4154,
    4156, 4174, 4175, 4183, 4189, 4199, 4211, 4231, 4236, 4239,
    4248, 4249, 4253, 4256, 4258, 4259, 4261, 4263, 4281, 4285,
    4290, 4309, 4318, 4320, 4322, 4325, 4334, 4336, 4338, 4341,
    4348, 4351, 4359, 4366, 4370, 4371, 4374, 4376, 4380, 4382,
    4390, 4392, 4407, 4412, 4416, 4418, 4424, 4428, 4429, 4445,
    4448, 4453, 4455, 4456, 4458, 4465, 4470, 4475, 4478, 4479,
    4492, 4497, 4498, 4501, 4502, 4506, 4509, 4511, 4512, 4513,
    4518, 4525, 4527, 4535, 4544, 4548, 4553, 4556, 4562, 4566,
    4570, 4574, 4579, 4583, 4588, 4605, 4623, 4626, 4628, 4635,
    4636, 4643, 4644, 4647, 4651, 4664, 4675, 4683, 4684, 4687,
    4689, 4690, 4694, 4695, 4699, 4701, 4702, 4708, 4709, 4717,
    4723, 4734, 4736, 4737, 4744, 4761, 4771, 4774, 4778, 4781,
    4792, 4799, 4806, 4813, 4819, 4820, 4824, 4828, 4833, 4837,
    4847, 4848, 4851, 4859, 4863, 4869, 4871, 4913, 4914, 4920,
    4923, 4929, 4931, 4934, 4939, 4940, 4944, 4946, 4956, 4966,
    4968, 4970, 4973, 4977, 4981, 5001, 5008, 5011, 5030, 5031,
    5041, 5049, 5056, 5060, 5061, 5062, 5063, 5071, 5073, 5087,
    5088, 5090, 5092, 5105, 5107, 5110, 5114, 5118, 5120, 5123,
    5132, 5152, 5157, 5165, 5170, 5174, 5181, 5188, 5189, 5191,
    5201, 5207, 5208, 5211, 5212, 5224, 5233, 5236, 5241, 5246,
    5252, 5253, 5255, 5256, 5258, 5259, 5263, 5269, 5271, 5272,
    5275, 5276, 5278, 5283, 5284, 5285, 5292, 5294, 5311, 5313,
]


def get_COCO_Freeview(location=None, test_data=None):
    """
    Loads or downloads and caches the COCO Freeview dataset.

    The dataset consists of about 5317 images from MS COCO with
    scanpath data from 10 observers doing freeviewing.

    The COCO images have been rescaled and padded to a size of
    1680x1050 pixels.

    The scanpaths come with attributes for
    - (fixation) duration in seconds

    @type  location: string, defaults to `None`
    @param location: If and where to cache the dataset. The dataset
                     will be stored in the subdirectory `COCO-Search18` of
                     location and read from there, if already present.
    @type test_data: string, defaults to `None`
    @parm test_data: filename of the test data, if you have access to it. If that's the case, also a
                     test data FixationTrains object will be created and saved, but not returned.

    @return: Training stimuli, training FixationTrains, validation Stimuli, validation FixationTrains

    .. seealso::

        Chen, Y., Yang, Z., Chakraborty, S., Mondal, S., Ahn, S., Samaras, D., Hoai, M., & Zelinsky, G. (2022).
        Characterizing Target-Absent Human Attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (pp. 5031-5040).

        Yang, Z., Mondal, S., Ahn, S., Zelinsky, G., Hoai, M., & Samaras, D. (2023).
        Predicting Human Attention using Computational Attention. arXiv preprint arXiv:2303.09383.
    """

    if location:
        location = os.path.join(location, 'COCO-Freeview')
        if os.path.exists(location):
            stimuli_train = _load(os.path.join(location, 'stimuli_train.hdf5'))
            fixations_train = _load(os.path.join(location, 'fixations_train.hdf5'))
            stimuli_validation = _load(os.path.join(location, 'stimuli_validation.hdf5'))
            fixations_validation = _load(os.path.join(location, 'fixations_validation.hdf5'))
            stimuli_test = _load(os.path.join(location, 'stimuli_test.hdf5'))

            return stimuli_train, fixations_train, stimuli_validation, fixations_validation, stimuli_test
        os.makedirs(location)

    with atomic_directory_setup(location):
        with TemporaryDirectory(cleanup=True) as temp_dir:
            download_and_check('http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip',
                               os.path.join(temp_dir, 'COCOSearch18-images-TP.zip'),
                               '4a815bb591cb463ab77e5ba0c68fedfb')

            download_and_check('http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TA.zip',
                               os.path.join(temp_dir, 'COCOSearch18-images-TA.zip'),
                               '85af7d74fa57c202320fa5e7d0dcc187')

            download_and_check('http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json',
                               os.path.join(temp_dir, 'COCOFreeView_fixations_trainval.json'),
                               'c7f2fbc92afbe55d4dedc445ac2063d3')


            # Stimuli
            print('Creating stimuli')
            f = zipfile.ZipFile(os.path.join(temp_dir, 'COCOSearch18-images-TP.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.svn', '__MACOSX', '.DS_Store'])
            f.extractall(temp_dir, namelist)

            f = zipfile.ZipFile(os.path.join(temp_dir, 'COCOSearch18-images-TA.zip'))
            namelist = f.namelist()
            namelist = filter_files(namelist, ['.svn', '__MACOSX', '.DS_Store'])
            f.extractall(temp_dir, namelist)

            # unifying images for different tasks

            stimulus_directory = os.path.join(temp_dir, 'stimuli')
            os.makedirs(stimulus_directory)

            filenames, stimulus_tasks = _prepare_stimuli(temp_dir, stimulus_directory, merge_tasks=True, unique_images=False)

            stimuli_src_location = os.path.join(temp_dir, 'stimuli')
            stimuli_target_location = os.path.join(location, 'stimuli') if location else None
            stimuli_filenames = filenames
            stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

            print('creating fixations')

            with open(os.path.join(temp_dir, 'COCOFreeView_fixations_trainval.json')) as fixation_file:
                json_data = json.load(fixation_file)

            all_scanpaths = _get_COCO_Freeview_fixations(json_data, filenames)

            scanpaths_train = all_scanpaths.filter_scanpaths(all_scanpaths.scanpaths.scanpath_attributes['split'] == 'train')
            scanpaths_validation = all_scanpaths.filter_scanpaths(all_scanpaths.scanpaths.scanpath_attributes['split'] == 'valid')

            del scanpaths_train.scanpaths.scanpath_attributes['split']
            del scanpaths_validation.scanpaths.scanpath_attributes['split']

            ns_train = sorted(set(scanpaths_train.n))
            stimuli_train, fixations_train = create_subset(stimuli, scanpaths_train, ns_train)

            ns_val = sorted(set(scanpaths_validation.n))
            stimuli_val, fixations_val = create_subset(stimuli, scanpaths_validation, ns_val)

            if test_data:
                with open(test_data) as f:
                    json_test_data = json.load(f)
                    scanpaths_test = _get_COCO_Freeview_fixations(json_test_data, filenames)
                    del scanpaths_test.scanpath_attributes['split']
                    ns_test = sorted(set(scanpaths_test.n))

                    assert len(ns_test) == len(TEST_STIMULUS_INDICES)
                    assert np.all(np.array(ns_test) == TEST_STIMULUS_INDICES)
                    _, fixations_test = create_subset(stimuli, scanpaths_test, ns_test)

            stimuli_test = stimuli[TEST_STIMULUS_INDICES]

        if location:
            stimuli_train.to_hdf5(os.path.join(location, 'stimuli_train.hdf5'))
            fixations_train.to_hdf5(os.path.join(location, 'fixations_train.hdf5'))
            stimuli_val.to_hdf5(os.path.join(location, 'stimuli_validation.hdf5'))
            fixations_val.to_hdf5(os.path.join(location, 'fixations_validation.hdf5'))
            stimuli_test.to_hdf5(os.path.join(location, 'stimuli_test.hdf5'))
            if test_data:
                fixations_test.to_hdf5(os.path.join(location, 'fixations_test.hdf5'))

    return stimuli_train, fixations_train, stimuli_val, fixations_val, stimuli_test


def get_COCO_Freeview_train(location=None):
    stimuli_train, fixations_train, stimuli_val, fixations_val, stimuli_test = get_COCO_Freeview(location=location)
    return stimuli_train, fixations_train


def get_COCO_Freeview_validation(location=None):
    stimuli_train, fixations_train, stimuli_val, fixations_val, stimuli_test = get_COCO_Freeview(location=location)
    return stimuli_val, fixations_val


def get_COCO_Freeview_test(location=None):
    stimuli_train, fixations_train, stimuli_val, fixations_val, stimuli_test = get_COCO_Freeview(location=location)
    return stimuli_test


def _get_COCO_Freeview_fixations(json_data, filenames):
    train_xs = []
    train_ys = []
    train_ts = []
    train_ns = []
    train_subjects = []
    train_durations = []
    split = []

    for item in tqdm(json_data):
        filename = item['name']
        n = filenames.index(filename)

        train_xs.append(item['X'])
        train_ys.append(item['Y'])
        train_ts.append(np.arange(item['length']))
        train_ns.append(n)
        train_subjects.append(item['subject'])
        train_durations.append(np.array(item['T']) / 1000)
        split.append(item['split'])

    scanpath_attributes = {
        'split': split,
    }
    fixation_attributes = {
        'durations': train_durations,
    }
    scanpath_attribute_mapping = {
        'durations': 'duration'
    }
    fixations = ScanpathFixations(Scanpaths(
        xs=train_xs,
        ys=train_ys,
        ts=train_ts,
        n=train_ns,
        subject=train_subjects,
        scanpath_attributes=scanpath_attributes,
        fixation_attributes=fixation_attributes,
        attribute_mapping=scanpath_attribute_mapping,
    ))

    return fixations
