# 284-Labs-1-3
Training a single layer perceptron model on sparse data. 

###

<ol>
  <li><strong>Lab 1</strong> - just simple batch gradient descent.
    <ul>
      <li>Sequential - no multithreading, just sequential C++ code</li>
      <li>Vienna - multithreaded code using the BLAS library, <a href="http://viennacl.sourceforge.net/">ViennaCL</a>
        <br> If I were to do this again, I would use <a href="http://eigen.tuxfamily.org">Eigen</a>, which I seems to have better <a href="https://computing.llnl.gov/tutorials/openMP/">OpenMP</a> performance.
      </li>
    </ul>
  </li>
  <li><strong>Lab 2</strong> - <a href="https://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent">Hogwild!</a>, a multithreaded gradient descent algorithm without locks.</li>
  <li><strong>Lab 3</strong> - modified <a href="https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf">Stochatic Variance-reduced gradient descent</a> algorithm, with Hogwild! updates.</li>
</ol>

Lab dataset is the [w8a dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).

## Tools

<ul>
  <li>C++ 11</li>
  <li><a href="https://computing.llnl.gov/tutorials/openMP/">OpenMP</a></li>
  <li><a href="http://viennacl.sourceforge.net/">ViennaCL</a></li>
</ul>
