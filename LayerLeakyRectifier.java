import java.util.Random;


class LayerLeakyRectifier extends Layer {
  double scale = 0.01;

  int getNumberWeights() { return 0; }

  LayerLeakyRectifier(int outputs) {
    super(outputs, outputs);
  }

  void activate(Vec weights, Vec x) {
    for(int i = 0; i < outputs; ++i) {
      double val = x.get(i);
      activation.set(i, val > 0 ? val : scale * val);
    }
  }

  Vec backProp(Vec weights, Vec prevBlame) {
    if(activation.size() != blame.size())
      throw new IllegalArgumentException("derivative problem, vector size mismatch");

    Vec nextBlame = new Vec(prevBlame.size());

    blame.fill(0.0);
    blame.add(prevBlame);

    for(int i = 0; i < inputs; ++i) {
      double act = activation.get(i);
      double val = blame.get(i);
      double derivative = (act > 0 ? val : scale * val);
      nextBlame.set(i, derivative);
    }

    return nextBlame;
  }

  void updateGradient(Vec x, Vec gradient) {
  } // Leaky Rectifier has no weights

  void initWeights(Vec weights, Random random) {
  } // Leaky Rectifier has no weights


  void debug() {
    System.out.println("---LayerLeakyRectifier---");
    System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    System.out.println(activation);
    System.out.println("blame:");
    System.out.println(blame);
  }
}
