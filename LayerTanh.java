import java.util.Random;


public class LayerTanh extends Layer {

  int getNumberWeights() { return 0; }

  LayerTanh(int outputs) {
    super(outputs, outputs);
  }

  // NOTE: this layer has no weights, so the weights vector is unused
  void activate(Vec weights, Vec x) {
    for(int i = 0; i < outputs; ++i) {
      activation.set(i, Math.tanh(x.get(i)));
    }
  }

  // NOTE: this layer contains no weights, so the weights parameter is unused
  Vec backProp(Vec weights, Vec prevBlame) {
    if(activation.size() != blame.size())
      throw new IllegalArgumentException("derivative problem, vector size mismatch");

    Vec nextBlame = new Vec(prevBlame.size());

    blame.fill(0.0);
    blame.add(prevBlame);

    for(int i = 0; i < inputs; ++i) {
      double derivative = prevBlame.get(i) * (1.0 - (activation.get(i) * activation.get(i)));
      nextBlame.set(i, derivative);
    }

    return nextBlame;
  }

  void updateGradient(Vec x, Vec gradient) {
  } // Tanh contains no weights so this is empty

  void initWeights(Vec weights, Random random) {
  } // Tanh contains no weights so this is empty

  void debug() {
    System.out.println("---LayerTanh---");
    System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    System.out.println(activation);
    System.out.println("blame:");
    System.out.println(blame);
  }

}
