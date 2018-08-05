import java.util.Random;

public class DataProcessor {
  Random random;

  DataProcessor() {

  }

  /// If the data patterns are merged with the labels seperate them
	// Assumes labels are vectors
  static void splitLabels(Matrix data, Matrix features, Matrix labels) {
		// copy the features over
		features.setSize(data.rows(), data.cols()-1);
		features.copyBlock(0, 0, data, 0, 0, data.rows(), data.cols()-1);

		// This assumes labels are a vector
		labels.setSize(data.rows(), 1);
		labels.copyBlock(0, 0, data, 0, data.cols()-1, data.rows(), 1);
	}

  /// This splits Features/Labels into training and testing partitions
  // Supports cross-validation
  static void splitData(Matrix featureData, Matrix labelData, Matrix trainingFeatures, Matrix trainingLabels,
    Matrix testingFeatures, Matrix testingLabels, int partitions, int index) {

    // calculate indexes and ranges for
    int beginIndex = index * (featureData.rows() / partitions);
    int endIndex = (index + 1) * (featureData.rows() / partitions);

    // Create the partitioned matrices
    double splitRatio = 1.0 - (1.0 / partitions);
    int trainingSize = (int)Math.ceil(featureData.rows() * splitRatio);

    // Size the matrices
    trainingFeatures.setSize(trainingSize, featureData.cols());
    trainingLabels.setSize(trainingSize, labelData.cols());

    testingFeatures.setSize(featureData.rows() - trainingSize, featureData.cols());
    testingLabels.setSize(labelData.rows() - trainingSize, labelData.cols());

    // First Training block
    trainingFeatures.copyBlock(0, 0, featureData, 0, 0, beginIndex, featureData.cols());
    trainingLabels.copyBlock(0, 0, labelData, 0, 0, beginIndex, labelData.cols());

    // Test block
    testingFeatures.copyBlock(0, 0, featureData, beginIndex, 0, endIndex-beginIndex, featureData.cols());
    testingLabels.copyBlock(0, 0, labelData, beginIndex, 0, endIndex-beginIndex, labelData.cols());

    // 2nd Training block
    trainingFeatures.copyBlock(beginIndex, 0, featureData,
      beginIndex+1, 0, featureData.rows() - endIndex, featureData.cols());
    trainingLabels.copyBlock(beginIndex, 0, labelData,
      beginIndex+1, 0, featureData.rows() - endIndex, labelData.cols());

  }
}
