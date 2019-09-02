# Project Nemo

Foraminifera (forams for short) classification via deep feature extraction.

## Image dataset

All models have been trained on a dataset of large, high-resolution images of
forams. The dataset has been produced by our research group, and will be made
publically available in the near future. Each of the source images consist of
a single class of forams. From these images, patches of _224x224_ pixels are
extracted using combinations of Gaussian smoothing, binary image generation
via thresholding, and connected components. The first step removes the metallic
border present in all source images, and the second step extracts candidate
patches. Each patch that passes a defined selection critera is extracted by
placing a _224x224_ crop at the centroid of the candidate region. The entire
process is automated in the `preprocess_data.py` script.

Once the source images have been preprocessed by extracting patches, datasets
for training, validation, and testing are generated automatically by using the
`build_datasets.py` script.

### Caveat regarding `raw-halves` source images

The `raw-halves` source images are slightly different in nature, and requires
that the `preprocess_data.py` script be invoked with `--border-threshold=50`.
Patches extracted from this dataset should be manually copied to the `train`
folder built by process outlined above. In the future, we should find a way to
fully automate this process as well.
