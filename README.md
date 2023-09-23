# Detect Autism Spectrum Disorder
The repo contains the implementation of a project for the Data Science Lab in Bioscences exam, completed with a rating of 30/30. It is an image recognition work in which, Using deep learning techniques, patients are classified as having or not having autism spectrum disorder.

Given the excellent results obtained in terms of accuracy, the second part of the project consisted of investigating how much these techniques are used as support of doctors in diagnosing asd, as well as investigating more generally, how research is moving in this area. The necessary data were acquired from pubmed and topic modeling and data visualization techniques were used to achieve this goal.

Link to the original project that inspired this work: https://github.com/Mahmoud-Elbattah/Predicting_ASD
Reference publications:
- Elbattah, M., Carette, R. ,Dequen, G., Guérin, J, & Cilia, F. (2019, July). Learning Clusters in Autism Spectrum Disorder: Image-Based Clustering of Eye-Tracking Scanpaths with Deep Autoencoder. In Proceedings of the 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE.
https://ieeexplore.ieee.org/document/8856904
- Carette, R., Elbattah, M., Dequen, G., Guérin, J, & Cilia, F. (2019, February). Learning to predict autism spectrum disorder based on the visual patterns of eye-tracking scanpaths. In Proceedings of the 12th International Conference on Health Informatics (HEALTHINF 2019).
https://www.researchgate.net/publication/331784416_Learning_to_Predict_Autism_Spectrum_Disorder_based_on_the_Visual_Patterns_of_Eye-tracking_Scanpaths
- Carette, R., Elbattah, M., Dequen, G., Guérin J.L., & Cilia F. (2018, September). Visualization of eye-tracking patterns in autism spectrum disorder: method and dataset. In Proceedings of the 13th International Conference on Digital Information Management (ICDIM 2018).IEEE.
https://ieeexplore.ieee.org/document/8846967

## CNN from scratch results
- 92% of accuracy on test set achieved in the last training epoch
- 95% of accuracy on test set achieved in the 19th of 20 training epochs
![Immagine 2023-09-23 101806](https://github.com/GiacomoPracucci/detect-asd/assets/94844087/bd877304-6ac3-41c1-b46d-653aff81f013)

## VGG16 results
- 93% of accuracy on test set achieved in the last epoch of training
- 95% of accuracy on test set achieved in the 19th of 20 training epochs
![Immagine 2023-09-23 101905](https://github.com/GiacomoPracucci/detect-asd/assets/94844087/4f91da21-53b4-4c7f-92f3-12edf2c23c35)

## berTopic results
The initial idea was to perform a very general query on PubMed, to obtain a large number of documents on which to do TopicModeling using the BERTopic algorithm.
The hope was that topics would be identified that represented various lines of research relating to the diagnosis on which we could then carry out our analysis.
However, the attempt failed, as the 11 topics identified by BERT did not group together potential research fields of ASD diagnosis
![Immagine 2023-09-23 102404](https://github.com/GiacomoPracucci/detect-asd/assets/94844087/6f567a6a-3461-4bb3-9580-951a89578f79)

## Research trends
Using as a reference what was cited in the reference papers, which lists alternative or complementary support methodologies to eye tracking analysis, we counted how many publications each year contained the keywords associated with the methodologies and their frequency out of the total articles
- Time series of the number of publications per year (3-year moving average)
![Immagine 2023-09-23 102834](https://github.com/GiacomoPracucci/detect-asd/assets/94844087/3caadbc2-85be-4045-b443-20c361399052)
- Percentage of publications on topics of interest compared to the total
![Immagine 2023-09-23 102924](https://github.com/GiacomoPracucci/detect-asd/assets/94844087/e20eb2ec-f8c2-4018-8d5d-03d0e31e6dd9)
- Number of publications per keyword before and after 2015
![Immagine 2023-09-23 103044](https://github.com/GiacomoPracucci/detect-asd/assets/94844087/8e1fcf8d-0648-4c2c-b0be-52eb0b893ba1)

## Conclusions
The results obtained from the classification models indicate that scan-path analysis can be a very useful method to support a doctor in the diagnosis of ASD.
Scientific research has begun to explore these new possibilities in recent years, but their use still remains limited compared to that of more "traditional" methodologies.
The results obtained suggest a significant potential in the diagnostic technique in question. It is therefore essential that research continues along this path, with the aim of further refining existing methodologies and developing practical solutions that can be used effectively by medical professionals.

