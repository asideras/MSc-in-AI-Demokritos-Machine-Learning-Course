# Heart Murmur Detection using Machine Learning
#### MSc-in-AI-Demokritos-Machine-Learning-Course  
------------------------------------------------
### Authors
| Name | Registration number |
| ------ | ------ |
| Boura Tatiana | MTN2210 |
| Sideras Andreas | MTN2214 |


## Contents of this repository: 
- [the-circor-digiscope-phonocardiogram-dataset-1.0.3] - The dataset
- [The_CirCor_DigiScope_Dataset.pdf] - The dataset's paper
- [notebooks] - Folder that includes: 

    The Jupyter Notebooks,
    * **Feature_Extraction_&_Demonstration.ipynb**, where feature extraction and demonstration is performed.
    * **Feature_Selection.ipynb**, where the *train*, *validation*, *test* sets are created and feature selection is performed.
    * **murmur_classification.ipynb**, where we select the ML model, train and evaluate it.
    
    The Python modules:
    * **data_loader_ML.py**, that is used from *Feature_Extraction_&_Demonstration.ipynb* to load the dataset.
    * **feature_extraction_ML.py**, that is used from *Feature_Extraction_&_Demonstration.ipynb* to extract the audio features.

    The extracted dataset from *Feature_Extraction_&_Demonstration.ipynb*:
    * **murmor_dataset.csv**


- [train_val_test_datasets] - Folder that includes the the *train*, *validation*, *test* sets are stored from **Feature_Selection.ipynb**.

- [important_features] - Folder that includes *.txt* files where each feature selection method stores its most important features **Feature_Selection.ipynb**.

- [classifiers_results] - Folder that includes *.txt* files where **murmur_classification.ipynb** stores for each feature selection, the selected models' classifiers results are stored.

### Process
In order to run the whole process you should execute the notebooks,
>    1. Feature_Extraction_&_Demonstration.ipynb
>    2. Feature_Selection.ipynb
>    3. murmur_classification.ipynb

with the given order. 
    
However, every notebook **can also be executed separately**






[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[The_CirCor_DigiScope_Dataset.pdf]: <https://github.com/asideras/MSc-in-AI-Demokritos-Machine-Learning-Course/blob/master/The_CirCor_DigiScope_Dataset.pdf>
[the-circor-digiscope-phonocardiogram-dataset-1.0.3]: <https://github.com/asideras/MSc-in-AI-Demokritos-Machine-Learning-Course/tree/master/the-circor-digiscope-phonocardiogram-dataset-1.0.3>
[notebooks]:
<https://github.com/asideras/MSc-in-AI-Demokritos-Machine-Learning-Course/tree/master/notebooks>
[train_val_test_datasets]:
<https://github.com/asideras/MSc-in-AI-Demokritos-Machine-Learning-Course/tree/master/train_val_test_datasets>
[important_features]:
<https://github.com/asideras/MSc-in-AI-Demokritos-Machine-Learning-Course/tree/master/important_features>
[classifiers_results]:
<https://github.com/asideras/MSc-in-AI-Demokritos-Machine-Learning-Course/tree/master/classifiers_results>
   
