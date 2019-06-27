# knn-with-mnist-datasets
running knn on mnist dataset for digit recognition

# prerequisites for running this program
1. Jupyter Notebook
2. Python 3 or higher version
3. Python Libraries  <br />
  i. pandas  <br />
  ii. numpy  <br />
  iii. matplotlib<br/>
  iv. opencv (subject to Note see below)

# to run this program
1. open this in Jupyter Notebook and run each cell (shift + Enter)
2. "test" is list which has 1D array (784x1) containing test image data
3. pass this test data ie. test[x] to KNN along with training and label data (which is X and Y respectively in this case)
4. Now you will get the prediction

# Note 
In case if you want to perform this on your own test data which I've done in this notebook, you will to need install opencv to read the image input, but let me make one thing very clear, prediction on custom input will be horrible becuase MNIST dataset is very clean data KNN is a naive algorithm which does not do much for accuracy.  
