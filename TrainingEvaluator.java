import java.util.Random;

class TrainingEvaluator {


  TrainingEvaluator() {

  }

  // double sum_squared_error(Matrix features, Matrix labels, SupervisedLearner learner) {
  //   if(features.rows() != labels.rows())
  //     throw new IllegalArgumentException("Mistmatching number of rows");

  //   double mis = 0;
  //   for(int i = 0; i < features.rows(); i++) {
  //     Vec feat = features.row(i);
  //     Vec pred = learner.predict(feat);
  //     Vec lab = labels.row(i);
  //     for(int j = 0; j < lab.size(); j++) {
  //       double blame = (lab.get(j) - pred.get(j)) * (lab.get(j) - pred.get(j));
  //       mis = mis + blame;
  //     }
  //   }

  //   return mis;
  // }

  // /// Measures the misclassifications with the provided test data
  // int countMisclassifications(Matrix features, Matrix labels) {
  //   if(features.rows() != labels.rows())
  //     throw new IllegalArgumentException("Mismatching number of rows");
  //   int mis = 0;
  //   for(int i = 0; i < features.rows(); i++) {
  //     Vec feat = features.row(i);
  //     Vec pred = predict(feat);
  //     Vec lab = formatLabel((int)labels.row(i).get(0));
  //     if(poorClassification(pred, lab)) {
  //       mis++;
  //     }
  //   }
  //   return mis;
  // }
}
