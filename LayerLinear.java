import java.util.Arrays;
import java.util.Random;


public class LayerLinear extends Layer {

  int getNumberWeights() { return (outputs + (inputs * outputs)); }

  LayerLinear(int inputs, int outputs) {
    super(inputs, outputs);
  }

  void initWeights(Vec weights, Random random) {
    for(int i = 0; i < weights.size(); ++i) {
      weights.set(i, (Math.max(0.03, (1.0 / inputs)) * random.nextGaussian()));
    }
  }

  void activate(Vec weights, Vec x) {
    Vec b = new Vec(weights, 0, outputs);

    int pos = outputs;
    for(int i = 0; i < outputs; ++i) {
      Vec temp = new Vec(weights, pos, inputs);
      double newEntry = x.dotProduct(temp);
      activation.set(i, newEntry);
      pos += inputs;
    }
    activation.add(b);
  }

  Vec backProp(Vec weights, Vec prevBlame) {
    blame.fill(0.0);
    blame.add(prevBlame);

    int pos = outputs; // Ignore b
    Matrix mTranspose = new Matrix(outputs, inputs);

    /// Turn our section of weights into a Matrix
    for(int i = 0; i < mTranspose.rows(); ++i) {
      for(int j = 0; j < mTranspose.cols(); ++j) {
        mTranspose.row(i).set(j, weights.get(pos));
        ++pos;
      }
    }
    Matrix t = mTranspose.transpose();

    /// Create the blame on the preceding layer
    Vec nextBlame = new Vec(inputs);
    nextBlame.fill(0.0);
    for(int i = 0; i < nextBlame.size(); ++i) {
      double newEntry = prevBlame.dotProduct(t.row(i));
      nextBlame.set(i, newEntry);
    }

    return nextBlame;
  }

  void updateGradient(Vec x, Vec gradient) {

    // Remove b
    Vec b = new Vec(gradient, 0, outputs);

    // add the blame to our bias
    b.add(blame);

    // Compute the outer product as scaling copies of X by each element in b
    int pos = outputs;
    for(int i = 0; i < outputs; ++i) {
      double b_i = b.get(i);
      Vec temp = new Vec(gradient, pos, inputs);
      temp.addScaled(b_i, x);
      pos += inputs;
    }
  }


  void ordinary_least_squares(Matrix x, Matrix y, Vec weights) {
    /// x are features
    /// y are labels

    Matrix xCentroid = new Matrix();
    Matrix yCentroid = new Matrix();
    xCentroid.newColumns(x.cols());
    yCentroid.newColumns(y.cols());

    double[] xRow = new double[x.cols()];
    for(int i = 0; i < x.cols(); ++i) {
      double xMean = x.columnMean(i);
      xRow[i] = xMean;
      for(int j = 0; j < x.rows(); ++j) {
        double value = x.row(j).get(i) - xMean;
        x.row(j).set(i, value);
      }
    }
    xCentroid.takeRow(xRow);

    double[] yRow = new double[y.cols()];
    for(int i = 0; i < y.cols(); ++i) {
      double yMean = y.columnMean(i);
      yRow[i] = yMean;
      for(int j = 0; j < y.rows(); ++j) {
        double value = y.row(j).get(i) - yMean;
        y.row(j).set(i, value);
      }
    }
    yCentroid.takeRow(yRow);

    Matrix featuresCrossLabels = Matrix.multiply(y.transpose(), x, false, false); // heeeelp

    Matrix xTranspose = new Matrix(x.transpose());
    Matrix featuresCrossFeatures = Matrix.multiply(xTranspose, x, false, false);
    Matrix fcfInverse = featuresCrossFeatures.pseudoInverse();
    Matrix weightsMatrix = Matrix.multiply(featuresCrossLabels, fcfInverse, false, false);

    //
    // Calculate bias
    Matrix mx = Matrix.multiply(weightsMatrix, xCentroid.transpose(), false, false);
    yCentroid.addScaled(mx, -1);

    //
    // Push the bias Matrix (yCentroid) and weightsMatrix into one long vector
    int weightsIndex = 0;
    for(int i = 0; i < yCentroid.rows(); ++i) {
      Vec temp = yCentroid.row(i);
      for(int j = 0; j < yCentroid.cols(); ++j) {
        weights.set(weightsIndex, temp.get(j));
        ++weightsIndex;
      }
    }

    for(int i = 0; i < weightsMatrix.rows(); ++i) {
      Vec temp = weightsMatrix.row(i);
      for(int j = 0; j < weightsMatrix.cols(); ++j) {
        weights.set(weightsIndex, temp.get(j));
        ++weightsIndex;
      }
    }
  }

  void debug() {
    System.out.println("---LayerLinear---");
    System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    System.out.println(activation);
    System.out.println("blame:");
    System.out.println(blame);
  }

}
