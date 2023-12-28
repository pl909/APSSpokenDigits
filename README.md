# APSSpokenDigits
Project: Recognizing Spoken Arabic Digits 

Using signals + statistical methods, including Mel-frequency cepstral coefficients, K-Means clustering, and Gaussian Mixture Models to recognize spoken arabic digits. 

Results: concatenating a time variable (value of 0 indicating frame at start of audio sample and 1 indicating end of audio sample) to the data produces a model w/ 5% (90->95) higher accuracy than all others.

Please see GMMPredictTime.py and confusionmatrixTimeGMM.png (time-aware gaussian mixture modeling classification) for the training and testing procedure+results with highest accuracy.

Credits to https://archive.ics.uci.edu/dataset/195/spoken+arabic+digit for dataset.

More detailed description in progress.
