import java.util.Random;

/// This class may not be used
abstract class UnsupervisedLearner {
  Random random;

  UnsupervisedLearner(Random r) {
    random = r;
  }

  void train_unsupervised(Matrix x) {

    // k represents degrees of freedom
    int k = 10; // Some value of dimensions for V

    // V is used to represent state
    Matrix v = new Matrix(x.rows(), k);

    v.fill(0.0); // assume we dont know the state in any of the images
    double learning_rate = 0.1;

    for(int j = 0; j < 10; ++j) {
      for(int i = 0; i < 10000000; ++i) {
        int t = random.nextInt(x.rows()); // t is a random row
        Vec feature = v.row(t); // Vector of states
        Vec label = x.row(t); // Actual picture itself

        //predict(feature, pred);

        /// Compute the blame on the inputs
        //compute the blame terms for V[t]

        /// Refine the weights by gradient descent
        //use gradient descent to refine the weights and bias values

        /// Update the states by gradient descent
        //use gradient descent to update V[t]
      }


      learning_rate *= 0.75;
    }
  }

  // void train_with_images(Matrix x) {
  //   if(x.cols() % (width * height) != 0)
  //     throw new IllegalArgumentException("does not divide evenly");
  //
  //   int channels = x.cols() / (width * height);
  // }
}
