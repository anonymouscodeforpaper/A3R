# A<sup>3</sup>R


This is the implementation of our paper titled "A<sup>3</sup>R : Argumentative explanations for recommendations"  submitted to DSAA 2023

## Example of A<sup>3</sup>R

![A toy example in book recommendation showing how A<sup>3</sup>R works. The thickness of the arrow reflects the importance of attribute type. In this toy example, the user accords more importance to the genre of the book.](https://github.com/anonymouscodeforpaper/A3R/blob/main/framework.png)

A toy example in book recommendation showing how A<sup>3</sup>R works. The thickness of the arrow reflects the importance of attribute type. In this toy example, the user accords more importance to the genre of the book.

## Strusture of code
-- In data_loader.py, we preprocess the dataset and returns the training set, testset

-- utilities.py contains the function necessary for data processing and model training


To run the code, python3 main.py--name = 'book', this is to run on the Dbook2014 dataset, for CarMusic, python3 main.py--name = 'book', for the three movie datasets, python3 main.py--name = 'movie'


## Datasets:

-- The source for Carmusic: L. Baltrunas, M. Kaminskas, B. Ludwig, O. Moling, F. Ricci, A. Aydin,K.-H. L ̈uke, and R. Schwaiger, “Incarmusic: Context-aware music recommendations in a car,” in International conference on electronic commerce and web technologies. Springer, 2011, pp. 89–100.

-- The source for DBook 2014: https://2014.eswc-conferences.org/important-dates/call-RecSys.html to get the types of attributes (aka. authors and genres) of items, we crawl the DBpedia links provided in the original dataset. For the books whose information is missing, we manually complete them according to Goodreads (https://www.goodreads.com/)

-- The source for Netflix, MovieLens Development (Dev.) and the MovieLens 100K, we reuse the version provided in A. Rago, O. Cocarascu, C. Bechlivanidis, D. Lagnado, and F. Toni, “Argumentative explanations for interactive recommendations,” Artificial Intelligence, vol. 296, p. 103506, 2021.

## Other files

Notebook NETFLIX: This file shows how to cluster the users in the NETFLIX dataset with the model learned. 
model_0.01_32_1e-06_[256].pkl: This file contains the parameters of the best model.

