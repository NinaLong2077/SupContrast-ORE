## To Do
Biased models:
1. Evaluate racial accuracy (Sat)
2. Get a linear prob dataset (Sat)
3. Architecture explore 
    - mobilenet  → test if it works on race data
    - Mosaic ML resnet → see performance + time + compute cost on race data (Fri)
4. SL biased models:
    - Mobilenet v3 + SwAV (Fri+Sat)

* Wed: Have all biased models 
* Thurs: biased model ready + CL set up explored




## Notes
* cifar data with 1 epoch has 20.xxx vs BUPT Balanced split into biased and balanced is 1.xxx
* A100 took 30 min to run one model
* So at most 2 hour per biased model
* cost a lot, 1 model used 1/8 of my compute 




- 
- train on linear prob using embedding we got from the model on the balanced dataset
  - Just the embedding
  - The test on balance test data set


- pick 2 or 3 ppl per race
- randomly from in training
  - Test set would have more images of those ppl right

- Overall performance
- Inrace performance: Same race accuracy
- Out of race performace: Other race accuracy


- For each image, you have a 1 or 0
  - Take the mean of just the sample of the race
  - Make a vector of right or wrong
    - This is if you are using a threshold
    - Pick the argmax and say if we got it right or wrong



1. We need to reach out to Yusef and Jahair
   - About 