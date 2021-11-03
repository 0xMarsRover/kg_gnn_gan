# Combining Text and Image Knowledge with GANs for Zero-Shot ActionRecognition in videos
This work aims to produce enriched semantic embedding by including 
the knowledgesources from text and image for action classes.

### Image-based semantic embedding for action classes

1. Scraping images for each action class from Google Imageg Source. `Check codes in the google images folder`
2. Downloaed images according to the given keywords in the sub-folder `downloads`
3. Using a pre-defined RESNET101 as feature extractor. [Reference](https://github.com/akshitac8/Generative_MLZSL/tree/main/datasets/extract_features)
3. Averaging image representations for each action class

### Reference

[1] Narayan, S., Gupta, A., Khan, F. S., Snoek, C. G., & Shao, L. (2020). Latent embedding feedback and discriminative features for zero-shot classification. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXII 16 (pp. 479-495). Springer International Publishing.
    [github repo.](https://github.com/akshitac8/tfvaegan)