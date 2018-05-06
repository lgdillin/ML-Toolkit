import java.util.Random;


class LayerMaxPooling2D extends Layer {
  static final int[] pooling_dims = {2, 2}; // Emulating a nxn matrix as a vector
  static final int poolsize = 4; // multiply all elements of the pool together

  Vec maxes;

  Matrix pooling, input; // Pool matrix for pooling operation
  int width, depth, height, planeSize;

  int getNumberWeights() { return 0; }

  LayerMaxPooling2D(int width, int height, int depth) {
    super(width * height * depth, (width / pooling_dims[0]) * (height / pooling_dims[1]) * depth);
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.planeSize = width * height;

    // Pre-allocate matrices for use during computation
    input = new Matrix(height * depth, width);
    pooling = new Matrix(pooling_dims);

    maxes = new Vec(outputs);

    // Error checking to make sure we can pool over the planar dimensions
    if((width * height) % poolsize != 0)
      throw new IllegalArgumentException("WxH: " + width + " * " + height + " / " + poolsize + " not an integer");
  }

  /// Pool over each 2-tensor
  void activate(Vec weights, Vec x) {
    if(x.size() % poolsize != 0)
      throw new RuntimeException("input vector cannot be rastered evenly");

    // Push the input vector into a matrix
    int x_pos = 0;
    for(int i = 0; i < input.rows(); ++i) {
      for(int j = 0; j < input.cols(); ++j) {
        input.row(i).set(j, x.get(x_pos));
        ++x_pos;
      }
    }
    // Find the max for each pooling matrix
    int pos = 0; // iterate the activation vec
    for(int i = 0; i < (input.rows() / pooling.rows()); ++i) {
      for(int j = 0; j < (input.cols() / pooling.cols()); ++j) {
        // Copy a block into pooling for comparison
        pooling.copyBlock(0, 0, input, i * pooling.rows(), j * pooling.cols(), pooling.rows(), pooling.cols());

        // Find the maximum value given each pooling block
        int index = 0;
        int maxIndex = 0; // save the spot where the max is for backProp
        double max = pooling.row(0).get(0);
        for(int k = 0; k < pooling.rows(); ++k) {
          for(int l = 0; l < pooling.cols(); ++l) {

            if(pooling.row(k).get(l) > max) {
              max = pooling.row(k).get(l);
              maxIndex = index;
            }
            ++index;
          }
        }

        maxes.set(pos, maxIndex);
        activation.set(pos, max);
        ++pos;
      }
    }

  }

  Vec backProp(Vec weights, Vec prevBlame) {
    blame.fill(0.0);
    blame.add(prevBlame);

    // re-use the input matrix from activation, for backpropping blame
    input.fill(0.0);

    // re-use the pooling matrix
    pooling.fill(0.0);

    Vec nextBlame = new Vec(inputs);

    // Copy the blame back into a matrix of area input
    int in_row = 0;
    int in_col = 0;
    for(int i = 0; i < prevBlame.size(); ++i) {
      pooling.fill(0.0);

      double blame_i = prevBlame.get(i);
      double b_i = maxes.get(i);

      int pos = 0;
      for(int k = 0; k < pooling.rows(); ++k) {
        for(int l = 0; l < pooling.cols(); ++l) {
          if(pos == b_i)
            pooling.row(k).set(l, blame_i);
            ++pos;
        }
      }

      in_col = (i * pooling.cols()) % input.cols();
      if(in_col == 0 && i != 0)
       in_row = in_row + pooling.rows();

      input.copyBlock(in_row, in_col, pooling, 0, 0, pooling.rows(), pooling.cols());
    }

    // push the blame matrix back into a vector
    int pos = 0;
    for(int i = 0; i < input.rows(); ++i) {
      Vec v = new Vec(nextBlame, pos, input.cols());
      v.add(input.row(i));
      pos += input.cols();
    }

    return nextBlame;
  }

  void updateGradient(Vec x, Vec gradient) {
  } // LayerMaxPooling2D contains no weights so this is empty

  void initWeights(Vec weights, Random random) {
  } // LayerMaxPooling2D contains no weights so this is empty

  void debug() {
    System.out.println("---LayerMaxPooling2D---");
    System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    System.out.println(activation);
    System.out.println("blame:");
    System.out.println(blame);
  }
}
