# 284-Labs-1-3
Training a single layer perceptron model on sparse data. 

###

<ol>
  <li>Lab 1 - just simple batch gradient descent.
    <ul>
      <li>Sequential - no multithreading, just sequential C++ code</li>
      <li>Vienna - multithreaded code using the BLAS library, <a href="http://viennacl.sourceforge.net/">ViennaCL</a></li>
  </li>
  <li>Lab 2 - <a href="https://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent">Hogwild!</a>, a multithreaded gradient descent algorithm without locks.</li>
  <li>Lab 3 - modified <a href="https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf">Stochatic Variance-reduced gradient descent</a> algorithm, with Hogwild! updates.</li>
</ol>

Lab dataset is the [w8a dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).

## Tools

<ul>
  <li>C++ 11</li>
  <li><a href="https://computing.llnl.gov/tutorials/openMP/">OpenMP</a></li>
  <li><a href="http://viennacl.sourceforge.net/">ViennaCL</a></li>
</ul>
