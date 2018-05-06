import java.util.Random;

/// WARNING:
/// I've written this class with specific functionality for time series
class LayerSine extends Layer {
  boolean time_series = true;

  Vec prevInput;

  int getNumberWeights() { return 0; }

  LayerSine(int outputs) {
    super(outputs, outputs);
  }

  // NOTE: this layer has no weights, so the weights vector is unused
  void activate(Vec weights, Vec x) {
    for(int i = 0; i < outputs; ++i) {
      prevInput = new Vec(x);

      if(time_series) {
        if(i == activation.size() - 1)
          activation.set(i, x.get(i)); // Identity unit
        else
          activation.set(i, Math.sin(x.get(i)));

      } else {
        activation.set(i, Math.sin(x.get(i)));
      }

    }
  }

  // NOTE: this layer contains no weights, so the weights parameter is unused
  Vec backProp(Vec weights, Vec prevBlame) {
    if(activation.size() != blame.size())
      throw new IllegalArgumentException("derivative problem, vector size mismatch");

    Vec nextBlame = new Vec(prevBlame.size());

    blame.fill(0.0);
    blame.add(prevBlame);

    double derivative;
    for(int i = 0; i < inputs; ++i) {
      if(time_series) {
        if(i == activation.size() - 1)
          derivative = prevBlame.get(i) * 1; // The identity unit
        else
          derivative = prevBlame.get(i) * Math.cos(prevInput.get(i));

      } else {
        derivative = prevBlame.get(i) * Math.cos(prevInput.get(i));
      }

      nextBlame.set(i, derivative);
    }

    return nextBlame;
  }

  void updateGradient(Vec x, Vec gradient) {
  } // Sine contains no weights so this is empty

  void initWeights(Vec weights, Random random) {
  } // Sine contains no weights so this is empty


  void debug() {
    System.out.println("---LayerSine---");
    System.out.println("Weights: " + getNumberWeights());
    System.out.println("activation: ");
    System.out.println(activation);
    System.out.println("blame:");
    System.out.println(blame);
  }
}
