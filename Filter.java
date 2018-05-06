import java.util.Random;


public class Filter extends SupervisedLearner {
  NeuralNet nn;

  // I want to change this later, but for now I'm keeping a set of
  // preprocessors for both features and labels
  // This makes it easier to pre and post process without doing a lot of
  // extra calculations and coding
  Imputer fim, lim;
  Normalizer fnorm, lnorm;
  NomCat fnm, lnm;


  Filter(Random r) {
    super(r);
    nn = new NeuralNet(r);

    fim = new Imputer();
    lim = new Imputer();
    fnorm = new Normalizer();
    lnorm = new Normalizer();
    fnm = new NomCat();
    lnm = new NomCat();
  }

  String name() { return ""; }

  /// process data into a format readily available for training
  void train(Matrix features, Matrix labels, int[] indices, int batch_size, double momentum) {
    features.copy(preProcess(features, fim, fnorm, fnm));
    labels.copy(preProcess(labels, lim, lnorm, lnm));
  }

  /// Train the NeuralNet
  void trainNeuralNet(Matrix features, Matrix labels, int[] indices, int batch_size,
    double momentum) {
    nn.train(features, labels, indices, batch_size, momentum);
  }

  Matrix preProcess(Matrix data, Imputer im, Normalizer norm, NomCat nm) {
    Matrix imputed = transform(im, data);
    Matrix normalized = transform(norm, imputed);
    Matrix nomcated = transform(nm, normalized);
    return nomcated;
  }

  /// produces a matrix of transformed data under the operation
  Matrix transform(PreprocessingOperation po, Matrix data) {
    po.train(data);
    Matrix output = po.outputTemplate();
    for(int i = 0; i < data.rows(); ++i) {
      double[] in = data.row(i).vals;
      double[] out = new double[output.cols()];
      po.transform(in, out);
      output.takeRow(out);
    }
    return output;
  }

  /// Untransformation process
  Vec postProcess(Vec prediction, NomCat nm, Normalizer norm, Imputer im) {
    Vec un_nomcated = untransform(nm, prediction);
    Vec un_normalized = untransform(norm, un_nomcated);
    Vec un_imputed = untransform(im, un_normalized);
    return un_imputed;
  }

  /// Transforms a vector back into its nominal representation
  Vec untransform(PreprocessingOperation po, Vec prediction) {
    double[] output = new double[1]; /// I HATE THIS HARDCODED VALUE
    po.untransform(prediction.vals, output);
    return new Vec(output);
  }

  /// Overloaded superclass definition for filtered data
  int countMisclassifications(Matrix features, Matrix labels) {
    if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");

    // PreProcess the features
    // Matrix processedTestFeatures = preProcess(features, new Imputer(),
    //   new Normalizer(), new NomCat());

    // So in this function we actually need a persistent set of preprocessors
    // That are trained on labels so we can re-transform them back into
    // their nominal representation for misclassification
    // Imputer im = new Imputer();
    // Normalizer norm = new Normalizer();
    // NomCat nm = new NomCat();
    // Matrix processedTestLabels = preProcess(labels, im, norm, nm);

    // predict a value and check if it is an accurate prediction
		int mis = 0;
		for(int i = 0; i < features.rows(); i++) {
			Vec feat = features.row(i);
      Vec lab = labels.row(i);

      Vec prediction = nn.predict(feat);
      Vec out = postProcess(prediction, lnm, lnorm, lim);

      // Component-wise comparison of nominal values
      for(int j = 0; j < lab.size(); ++j) {
        if(out.get(j) != lab.get(j))
					mis++;
      }

		}
		return mis;
  }

  /// overloaded SSE function for use with processed data
  double sum_squared_error(Matrix features, Matrix labels) {
    if(features.rows() != labels.rows())
      throw new IllegalArgumentException("Mismatching number of rows");

    // PreProcess the features
    // Matrix processedTestFeatures = preProcess(features, new Imputer(),
    //   new Normalizer(), new NomCat());

    // So in this function we actually need a persistent set of preprocessors
    // That are trained on labels so we can re-transform them back into
    // their nominal representation for misclassification
    // Imputer im = new Imputer();
    // Normalizer norm = new Normalizer();
    // NomCat nm = new NomCat();
    // Matrix processedTestLabels = preProcess(labels, im, norm, nm);

    // predict a value and check if it is an accurate prediction
    double mis = 0;
    for(int i = 0; i < features.rows(); i++) {
      Vec feat = features.row(i);
      Vec lab = labels.row(i);

      Vec prediction = nn.predict(feat);
      //Vec out = postProcess(prediction, lnm, lnorm, lim);

      // Component-wise comparison of nominal values
      for(int j = 0; j < lab.size(); ++j) {
        if(prediction.get(j) != lab.get(j)) {
          double blame = (lab.get(j) - prediction.get(j)) * (lab.get(j) - prediction.get(j));
                    mis = mis + blame;
        }
      }

    }

    return mis;
  }

  Vec predict(Vec in) {
    throw new RuntimeException("This class does not use predict!");
  }

}
