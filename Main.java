// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.Random;
import java.lang.*;

class Main
{
	static void test(SupervisedLearner learner, String challenge) {
		// Load the training data
		String fn = "data/" + challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_lab.arff");

		// Train the model
		//learner.train(trainFeatures, trainLabels, Training.NONE);

		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_lab.arff");

		// Measure and report accuracy
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	public static void testCV(SupervisedLearner learner) {
		Matrix f = new Matrix();
		f.newColumns(1);
		double[] f1 = {0};
		double[] f2 = {0};
		double[] f3 = {0};
		f.takeRow(f1);
		f.takeRow(f2);
		f.takeRow(f3);

		Matrix l = new Matrix();
		l.newColumns(1);
		double[] l1 = {2};
		double[] l2 = {4};
		double[] l3 = {6};
		l.takeRow(l1);
		l.takeRow(l2);
		l.takeRow(l3);

		double rmse = learner.cross_validation(1, 3, f, l, learner);
		System.out.println("RMSE: " + rmse);
	}

	public static void testOLS() {
		LayerLinear ll = new LayerLinear(13, 1);
		Random random = new Random(123456);
		Vec weights = new Vec(14);

		for(int i = 0; i < 14; ++i) {
			weights.set(i, random.nextGaussian());
		}

		Matrix x = new Matrix();
		x.newColumns(13);
		for(int i = 0; i < 100; ++i) {
			double[] temp = new double[13];
			for(int j = 0; j < 13; ++j) {
				temp[j] = random.nextGaussian();
			}
			x.takeRow(temp);
		}

		Matrix y = new Matrix(100, 1);
		for(int i = 0; i < y.rows(); ++i) {
			ll.activate(weights, x.row(i));
			for(int j = 0; j < ll.activation.size(); ++j) {
				double temp = ll.activation.get(j) + random.nextGaussian();
				y.row(i).set(j, temp);
			}
		}

		for(int i = 0; i < weights.size(); ++i) {
    	System.out.println(weights.get(i));
		}

		Vec olsWeights = new Vec(14);
		ll.ordinary_least_squares(x,y,olsWeights);

		System.out.println("-----------------------------");

		for(int i = 0; i < olsWeights.size(); ++i) {
			System.out.println(olsWeights.get(i));
		}
	}


	public static void testLayer() {
		double[] x = {0, 1, 2};
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		System.out.println(ll.activation.toString());
	}


	public static void opticalCharacterRecognition() {
		Random random = new Random(123456); // used for shuffling data


		/// Load training and testing data
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF("data/train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF("data/train_lab.arff");

		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF("data/test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF("data/test_lab.arff");

		/// Normalize our training/testing data by dividing by 256.0
		/// There are 256 possible values for any given entry
		trainFeatures.scale((1 / 256.0));
		testFeatures.scale((1 / 256.0));

		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainFeatures.rows()];
		int[] testIndices = new int[testFeatures.rows()];

		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }
		for(int i = 0; i < testIndices.length; ++i) { testIndices[i] = i; }

		/// Assemble and initialize a neural net
		NeuralNet nn = new NeuralNet(random);

		nn.layers.add(new LayerLinear(784, 80));
		nn.layers.add(new LayerTanh(80));

		nn.layers.add(new LayerLinear(80, 30));
		nn.layers.add(new LayerTanh(30));

		nn.layers.add(new LayerLinear(30, 10));
		nn.layers.add(new LayerTanh(10));

		nn.initWeights();


		/// Training and testing
		int mis = 10000;
		int epoch = 0;
		while(mis > 350) {
			//if(true)break;
			System.out.println("==============================");
			System.out.println("TRAINING EPOCH #" + epoch + '\n');

			mis = nn.countMisclassifications(testFeatures, testLabels);
			System.out.println("Misclassifications: " + mis);

			for(int i = 0; i < trainFeatures.rows(); ++i) {
				Vec in, target;

				// Train the network on a single input
				in = trainFeatures.row(i);

				target = new Vec(10);
				target.vals[(int) trainLabels.row(i).get(0)] = 1;

				//nn.refineWeights(in, target, nn.weights, 0.0175, Training.STOCHASTIC);
			}

			// Shuffle training and testing indices
			for(int i = 0; i < trainingIndices.length * 0.5; ++i) {
				int randomIndex = random.nextInt(trainingIndices.length);
				int temp = trainingIndices[i];
				trainingIndices[i] = trainingIndices[randomIndex];
				trainingIndices[randomIndex] = temp;

			}

			for(int i = 0; i < testIndices.length * 0.5; ++i) {
				int randomIndex = random.nextInt(testIndices.length);
				int temp = testIndices[i];
				testIndices[i] = testIndices[randomIndex];
				testIndices[randomIndex] = temp;
			}

			++epoch;
		}
	}

	public static void testBackProp() {
		double[] x = {0, 1, 2};
		Vec xx = new Vec(x);
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		double[] yhat = {9, 6};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		ll.blame = new Vec(yhat);
		ll.backProp(new Vec(m), new Vec(x));
		System.out.println(xx);
	}

	public static void testGradient() {
		double[] x = {0, 1, 2};
		Vec xx = new Vec(x);
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		Vec mm = new Vec(m);
		Vec g = new Vec(mm.size());
		g.fill(0.0);
		double[] yhat = {9, 6};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		ll.blame = new Vec(yhat);
		ll.updateGradient(xx, g);
		//System.out.println(xx);
		System.out.println(g);
	}

	public static void testNomCat() {
		Random random = new Random(123456);

		/// Load data
		Matrix data = new Matrix();
		data.loadARFF("data/hypothyroid.arff");

		// Matrix data = new Matrix(5, derp.cols());
		// data.copyBlock(0, 0, derp, 0, 0, 5, data.cols());

		/// Create a new filter to preprocess our data
		Filter f = new Filter(random);

		/// Partition the features from the labels
		Matrix features = new Matrix();
		Matrix labels = new Matrix();
		f.splitLabels(data, features, labels);

		/// PREPROCESSING
		// We need a set of preprocessors for both features and labels

		// Train the preprocessors for the training data
		f.train(features, labels, null, 0, 0.0);

		/// Partition the data into training and testing blocks
		/// With respective feature and labels blocks
		double splitRatio = 0.75;
		Matrix trainingFeatures = new Matrix();
		Matrix trainingLabels = new Matrix();
		Matrix testingFeatures = new Matrix();
		Matrix testingLabels = new Matrix();
		f.splitData(features, labels, trainingFeatures, trainingLabels,
			testingFeatures, testingLabels, 5, 0);


		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainingFeatures.rows()];
		int[] testIndices = new int[testingFeatures.rows()];

		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }
		for(int i = 0; i < testIndices.length; ++i) { testIndices[i] = i; }


		/// I want some intelligent way of getting the input and outputs
		f.nn.layers.add(new LayerLinear(trainingFeatures.cols(), 100));
		f.nn.layers.add(new LayerTanh(100));

		f.nn.layers.add(new LayerLinear(100, 4));
		f.nn.layers.add(new LayerTanh(4));

		f.nn.initWeights();

		int mis = testingLabels.rows();
		int epoch = 0;

		double testSSE = 0;
		double trainSSE = 0;

		double previous = 0;
		double tolerance = 0.0000009;


		System.out.println("batch,seconds,testRMSE,trainRMSE");
		int batch = 1;
		int batch_size = 10;
		double startTime = (double)System.nanoTime();
		while(true) {

			testSSE += f.sum_squared_error(testingFeatures, testingLabels);
			double testMSE = testSSE / testingFeatures.rows();
			double testRMSE = Math.sqrt(testMSE);

			trainSSE += f.sum_squared_error(trainingFeatures, trainingLabels);
			double trainMSE = trainSSE / trainingFeatures.rows();
			double trainRMSE = Math.sqrt(trainMSE);

			f.trainNeuralNet(trainingFeatures, trainingLabels, trainingIndices, batch_size, 0.0);
			// double mse = sse / batch;
			// double rmse = Math.sqrt(mse);

			double seconds = ((double)System.nanoTime() - startTime) / 1e9;
			System.out.println(batch + "," + seconds + "," + testRMSE + "," + trainRMSE);

			batch = batch + 1;

			// mis = f.countMisclassifications(testingFeatures, testingLabels);
			// System.out.println("mis: " + mis);

			double convergence = Math.abs(1 - (previous / testSSE));
			previous = testSSE;
			testSSE = 0;
			trainSSE = 0;
			if(convergence < tolerance) break;

		}

	}

	public static void debugSpew() {
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);
		nn.layers.add(new LayerConv(new int[]{4, 4}, new int[]{3, 3, 2},
			new int[]{4, 4, 2}));
		nn.layers.add(new LayerLeakyRectifier(4 * 4 * 2));
		nn.layers.add(new LayerMaxPooling2D(4, 4, 2));

		double[] w = {
			0,							// bias #1
			0.1,						// bias #2

			0.01,0.02,0.03, // filter #1
			0.04,0.05,0.06,
			0.07,0.08,0.09,

			0.11,0.12,0.13, // filter #2
			0.14,0.15,0.16,
			0.17,0.18,0.19
		};
		nn.weights = new Vec(w);
		nn.gradient = new Vec(nn.weights.size());
		nn.gradient.fill(0.0);

		double[] in = {
			0,0.1,0.2,0.3,
			0.4,0.5,0.6,0.7,
			0.8,0.9,1,1.1,
			1.2,1.3,1.4,1.5
		};
		Vec input = new Vec(in);

		double[] t = {
			0.7,0.6,
			0.5,0.4,

			0.3,0.2,
			0.1,0
		};
		Vec target = new Vec(t);

		// Forward Prop
		nn.predict(input);
		System.out.println("activation 0:\n" + nn.layers.get(0).activation);
		nn.predict(input);
		System.out.println("activation 0:\n" + nn.layers.get(0).activation);
		// System.out.println("activation 1:\n" + nn.layers.get(1).activation);
		// System.out.println("activation 2:\n" + nn.layers.get(2).activation);
		//
		// // error
		// System.out.println("output Blame: ");
		// for(int i = 0; i < target.size(); ++i) {
		// 	System.out.print((target.get(i) - nn.layers.get(2).activation.get(i)) + ",");
		// }
		// System.out.println("");
		//
		// // backProp
		// nn.backProp(target);
		// System.out.println("blame 2: " + nn.layers.get(2).blame);
		// System.out.println("blame 1: " + nn.layers.get(1).blame);
		// System.out.println("blame 0: " + nn.layers.get(0).blame);
		//
		// nn.updateGradient(input);
		// System.out.println("gradient: " + nn.gradient);
		//
		// nn.cd_gradient = new Vec(nn.gradient.size());
		// nn.central_difference(input, target);
		// System.out.println("cd: " + nn.cd_gradient);
		//
		// int count = 0;
		// for(int i = 0; i < nn.gradient.size(); ++i) {
		// 	double difference = (nn.cd_gradient.get(i) - nn.gradient.get(i)) / nn.cd_gradient.get(i);
		// 	if(difference > 0.005)
		// 		++count;
		// }
		//
		// System.out.println("Difference exceeds tolerance " + count
		// 	+ " times out of " + nn.gradient.size() + " elements");

	}

	public static void debugSpew2() {
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		nn.layers.add(new LayerConv(new int[]{4, 4}, new int[]{3, 3},
			new int[]{4, 4}));
		nn.layers.add(new LayerConv(new int[]{4, 4}, new int[]{3, 3, 2},
			new int[]{4, 4, 2}));
		nn.layers.add(new LayerLeakyRectifier(4 * 4 * 2));
		nn.layers.add(new LayerMaxPooling2D(4, 4, 2));

		double[] w = {
			0,							// bias #1

			0.01,0.02,0.03, // filter #1
			0.04,0.05,0.06,
			0.07,0.08,0.09,

			0.1,						// bias #2
			0.20,						// bias #3

			0.11,0.12,0.13, // filter #2
			0.14,0.15,0.16,
			0.17,0.18,0.19,

			0.21,0.22,0.23, // filter #3
			0.24,0.25,0.26,
			0.27,0.28,0.29
		};
		nn.weights = new Vec(w);
		nn.gradient = new Vec(nn.weights.size());
		nn.gradient.fill(0.0);

		double[] in = {
			0,0.1,0.2,0.3,
			0.4,0.5,0.6,0.7,
			0.8,0.9,1,1.1,
			1.2,1.3,1.4,1.5
		};
		Vec input = new Vec(in);

		double[] t = {
			0.7,0.6,
			0.5,0.4,

			0.3,0.2,
			0.1,0
		};
		Vec target = new Vec(t);

		nn.predict(input);
		System.out.println("activation 0:\n" + nn.layers.get(0).activation);
		System.out.println("activation 1:\n" + nn.layers.get(1).activation);
		System.out.println("activation 2:\n" + nn.layers.get(2).activation);
		System.out.println("activation 3:\n" + nn.layers.get(3).activation);

		// error
		System.out.println("output Blame: ");
		for(int i = 0; i < target.size(); ++i) {
			System.out.print((target.get(i) - nn.layers.get(3).activation.get(i)) + ",");
		}
		System.out.println("");

		// backProp
		nn.backProp(target);
		System.out.println("blame 2: " + nn.layers.get(2).blame);
		System.out.println("blame 1: " + nn.layers.get(1).blame);
		System.out.println("blame 0: " + nn.layers.get(0).blame);

		nn.updateGradient(input);
		System.out.println("gradient: " + nn.gradient);

		nn.refineWeights(0.01);
		System.out.println("weights: " + nn.weights);

		//Vec cd = nn.centralDifference(input);
		//System.out.println("cd: " + cd);
	}

	public static void asgn4() {
		/// Instantiate net
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		/// Build topology
		nn.layers.add(new LayerConv(new int[]{8, 8}, new int[]{5, 5, 4}, new int[]{8, 8, 4}));
		nn.layers.add(new LayerLeakyRectifier(8 * 8 * 4));
		nn.layers.add(new LayerMaxPooling2D(8, 8, 4));
		nn.layers.add(new LayerConv(new int[]{4, 4, 4}, new int[]{3, 3, 4, 6}, new int[]{4, 4, 1, 6}));
		nn.layers.add(new LayerLeakyRectifier(4 * 4 * 6));
		nn.layers.add(new LayerMaxPooling2D(4, 4, 1 * 6));
		nn.layers.add(new LayerLinear(2 * 2 * 6, 3));
		nn.initWeights();

		/// Test data
		int inSize = nn.layers.get(0).inputs;
		Vec in = new Vec(inSize);
		for(int i = 0; i < in.size(); ++i) {
			in.set(i, i / 100.0);
		}

		int size = nn.layers.size();
		int outSize = nn.layers.get(size-1).outputs;
		Vec target = new Vec(outSize);
		for(int i = 0; i < target.size(); ++i) {
			target.set(i, i / 10.0);
		}

		nn.finite_difference(in, target);
	}

	public static void timeseries() {
		/// Instantiate net
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		/// Build topology
		nn.layers.add(new LayerLinear(1, 101));
		nn.layers.add(new LayerSine(101));
		nn.layers.add(new LayerLinear(101, 1));
		nn.initWeights();

		/// Initilizize first layer weights
		int numWeights = nn.layers.get(0).getNumberWeights();
		// Strip the weights
		Vec layerOne = new Vec(nn.weights, 0, numWeights); // Strip the weights
		// Separate the bias and populate it
		Vec bias = new Vec(layerOne, 0, nn.layers.get(0).outputs);
		for(int i = 0; i < bias.size(); ++i) {
			if(i < 50)
				bias.set(i, Math.PI);
			else
				bias.set(i, Math.PI / 2);
		}

		// Separate M and populate bias
		Vec m = new Vec(layerOne, bias.size(), layerOne.size()-bias.size());
		for(int i = 0; i < m.size() - 1; ++i) {
			if(i < 50)
				m.set(i, (i+1) * 2 * Math.PI);
			else
				m.set(i, (i+1) * 2 * Math.PI);
		}
		m.set(m.size()-1, 0.01);

		/// Build the training features matrix
		Matrix trainingFeatures = new Matrix(256, 1);
		for(int i = 0; i < trainingFeatures.rows(); ++i) {
			trainingFeatures.row(i).set(0, i / 256.0);
		}

		/// Build the testing features matrix
		Matrix testingFeatures = new Matrix(100, 1);
		for(int i = 0; i < testingFeatures.rows(); ++i) {
			// testingFeatures.row(i).set(0, (256.0 + i) / 256.0);
			testingFeatures.row(i).set(0, (i) / 256.0);
		}

		/// Load label data from file
		Matrix data = new Matrix();
		data.loadARFF("data/unemployment.arff");

		/// split into training labels matrix
		Matrix trainingLabels = new Matrix(256, 1);
		for(int i = 0; i < trainingLabels.rows(); ++i) {
			double val = data.row(i).get(0);
			trainingLabels.row(i).set(0, val);
		}

		/// Split into testing labels matrix
		Matrix testingLabels = new Matrix (100, 1);
		for(int i = 0; i < testingLabels.rows(); ++i) {
			double val = data.row(256 + i).get(0);
			testingLabels.row(i).set(0, val);
		}

		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainingFeatures.rows()];
		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }

		/// Train the net
		for(int i = 0; i < 1; ++i) {
			nn.train(trainingFeatures, trainingLabels, trainingIndices, 1, 0.0);
		}

		/// produce a matrix of the predicted results
		Vec predictions = new Vec(testingFeatures.rows());
		for(int i = 0; i < testingFeatures.rows(); ++i) {
			Vec pred = new Vec(nn.predict(testingFeatures.row(i)));
			predictions.set(i, pred.get(0));
		}

		for(int i = 0; i < predictions.size(); ++i) {
			System.out.println(predictions.get(i));
		}
	}


	public static void tsDebugSimple() {
		/// Instantiate net
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		/// Build topology
		nn.layers.add(new LayerLinear(1, 5));
		nn.layers.add(new LayerSine(5));
		nn.layers.add(new LayerLinear(5, 1));

		double[] w = {
			3.1415926535898,3.1415926535898,1.5707963267949,1.5707963267949,
			0,6.2831853071796,12.566370614359,6.2831853071796,12.566370614359,0,
			0.01,0.01,0.01,0.01,0.01,0.01
		};

		// double[] w = {
		// 	3.1415926535898,3.1412587951824,1.5707963267949,1.5707963267949,0,6.2831853071796,
		// 	12.566370614359,6.2831853071796,12.566370614359,0,0.043385840734641,0.01,0.01,0.043385840734641,
		// 	0.043385840734641,0.11488471198587
		// };
		nn.weights = new Vec(w);
		nn.gradient = new Vec(nn.weights.size());

		double[] in = {0.0};
		Vec input = new Vec(in);

		double[] t = {3.4};
		Vec target = new Vec(t);

		nn.predict(input);
		nn.backProp(target);
		nn.updateGradient(input);

		System.out.println("activation 0:\n" + nn.layers.get(0).activation);
		System.out.println("activation 1:\n" + nn.layers.get(1).activation);
		System.out.println("activation 2:\n" + nn.layers.get(2).activation);

		System.out.println(nn.gradient);
	}

	public static void inference() {
		/// Instantiate net
		Random r = new Random(123456);
		NeuralNet nn = new NeuralNet(r);

		/// Build topology
		nn.layers.add(new LayerLinear(4, 12));
		nn.layers.add(new LayerTanh(12));
		nn.layers.add(new LayerLinear(12, 12));
		nn.layers.add(new LayerTanh(12));
		nn.layers.add(new LayerLinear(12, 3));
		nn.layers.add(new LayerTanh(3));

		/// Load data
		Matrix x_observations = new Matrix();
		x_observations.loadARFF("data/observations.arff");
		x_observations.scale(1/256.0); // normalize data


	}

	public static void infer_test() {
		/// Instantiate net (Observation function /decoder)
		Random r = new Random(12345);
		NeuralNet nn1 = new NeuralNet(r);

		/// Build topology for observation function
		nn1.layers.add(new LayerLinear(4, 12));
		nn1.layers.add(new LayerTanh(12));
		nn1.layers.add(new LayerLinear(12, 12));
		nn1.layers.add(new LayerTanh(12));
		nn1.layers.add(new LayerLinear(12, 3));
		nn1.layers.add(new LayerTanh(3));
		nn1.initWeights();
		// Make the weights a bit smaller
		nn1.weights.scale(0.1);
		// for(int i = 0; i < nn1.weights.size(); ++i) {
		// 	nn1.weights.set(i, Math.abs(nn1.weights.get(i)));
		// }

		/// Instantiate transition function
		NeuralNet nn2 = new NeuralNet(r);

		/// Build topology for transition function
		nn2.layers.add(new LayerLinear(6, 6));
		nn2.layers.add(new LayerTanh(6));
		nn2.layers.add(new LayerLinear(6, 2));
		nn2.initWeights();
		nn2.weights.scale(0.1);

		/// Load observations from file
		Matrix x_observations = new Matrix();
		x_observations.loadARFF("data/observations.arff");
		x_observations.scale(1/255.0); // normalize data

		/// Load actions from file
		Matrix actions = new Matrix();
		actions.loadARFF("data/actions.arff");

		/// Empty matrix to hold predicted states
		int k = 2; // # of degrees of freedom in the system
		Matrix states = new Matrix(x_observations.rows(), k);
		states.fill(0.0);

		/// passing (Data from file, states from (1) -> (n))
		nn1.train_with_images(x_observations, states);

		Matrix verify = new Matrix(999, k);
		verify.copyBlock(0, 0, states, 0, 0, 999, k);

		/// The transition function requires both states + actions
		// States from (0) -> (n-1)

		Matrix v = new Matrix(actions.rows()-1, states.cols() + actions.cols());
		// copy over states (0) -> (n-1)
		v.copyBlock(0, 0, states, 0, 0, states.rows()-1, states.cols());
		// copy over actions (0) -> (n-1)
		v.copyBlock(0, 2, actions, 0, 0, actions.rows()-1, actions.cols());

		/// Convert nominal action values to continuous values
		NomCat nomcat = new NomCat();
		nomcat.train(v);
		Matrix v_andActions = nomcat.outputTemplate();
		for(int i = 0; i < v.rows(); ++i) {
			double[] in = v.row(i).vals;
			double[] out = new double[v_andActions.cols()];
			nomcat.transform(in, out);
			v_andActions.takeRow(out);
		}

		/// Labels are the predicted state, ie (1) -> (n)
		Matrix v_t = new Matrix(states.rows() - 1, states.cols());
		v_t.copyBlock(0, 0, states, 1, 0, states.rows()-1, states.cols());

		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[v_andActions.rows()];
		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }

		for(int i = 0; i < 10; ++i) {
			nn2.train(v_andActions, v_t, trainingIndices, 1, 0.0);
		}

		System.out.println("------------------------");
		System.out.println(verify);
		System.out.println("------------------------");


		// String filename1 = new String("img/framesdfsdf.png");
		// Vec ooo = new Vec(2);
		// System.out.println(states.row(0));
		// ooo.set(0, states.row(0).get(0));
		// ooo.set(1, states.row(0).get(1));
		// nn1.make_image(filename1, ooo);

		Vec out = new Vec(2);
		out.set(0, v.row(0).get(0));
		out.set(1, v.row(0).get(1));
		String filename = new String("img/frame" + 0 + ".png");
		nn1.make_image(filename, out);

		/// First generate starting state
		Vec state_in = new Vec(6);
		state_in.set(0, v_t.row(0).get(0));
		state_in.set(1, v_t.row(0).get(1));
		state_in.set(2, 1.0);
		state_in.set(3, 0.0);
		state_in.set(4, 0.0);
		state_in.set(5, 0.0);
		out = nn2.predict(state_in);


		for(int i = 1; i < 6; ++i) {
			filename = new String("img/frame" + i + ".png");
			nn1.make_image(filename, out);

			/// FInd the difference between the current state and the next,
			// add it to the current state to predict the next
			Vec state_in_w = new Vec(state_in, 0, 2);
			Vec difference = new Vec(out);
			difference.addScaled(-1.0, state_in_w);
			state_in_w.add(difference);

			//state_in.set(0, out.get(0));
			//state_in.set(1, out.get(1));
			state_in.set(2, 1.0);


			out = nn2.predict(state_in);
			System.out.println("state_in: " + state_in);
		}

		state_in.set(2, 0.0);
		state_in.set(4, 1.0);
		out = nn2.predict(state_in);

		for(int i = 6; i < 11; ++i) {
			filename = new String("img/frame" + i + ".png");
			nn1.make_image(filename, out);

			/// FInd the difference between the current state and the next,
			// add it to the current state to predict the next
			Vec state_in_w = new Vec(state_in, 0, 2);
			Vec difference = new Vec(out);
			difference.addScaled(-1.0, state_in_w);
			state_in_w.add(difference);

			//state_in.set(0, out.get(0));
			//state_in.set(1, out.get(1));
			state_in.set(4, 1.0);


			out = nn2.predict(state_in);
			System.out.println("state_in: " + state_in);
		}

		// /// Produce a left vector
		// Vec goleft = new Vec(6);
		// goleft.set(2, 1.0);
		// goleft.set(3, 0.0);
		// goleft.set(4, 0.0);
		// goleft.set(5, 0.0);
		//
		// Vec in = new Vec(4);
		//
		// /// Go left five times
		// for(int i = 0; i < 5; ++i) {
		// 	goleft.set(0, states.row(i+1).get(0));
		// 	goleft.set(1, states.row(i+1).get(1));
		// 	Vec out = nn2.predict(goleft);
		//
		//
		// 	in.set(2, out.get(0));
		// 	in.set(3, out.get(1));
		// 	System.out.println(in);
		//
		// 	/// produce an image
		// 	for(int j = 0; j < ib.height; ++j) {
		// 		for(int l = 0; l < ib.width; ++l) {
		// 			in.set(0, (double)l/ib.width);
		// 			in.set(1, (double)j/ib.height);
		//
		// 			Vec color = nn1.predict(in);
		// 			color.scale(255.0);
		//
		// 			ib.WritePixelBuffer(l, j, color);
		// 		}
		// 	}
		//
		// 	ib.outputToPNG("img/frame" + (i+1) + ".png");
		// }
		//
		// /// Produce a left vector
		// Vec goup = new Vec(6);
		// goup.set(2, 0.0);
		// goup.set(3, 0.0);
		// goup.set(4, 1.0);
		// goup.set(5, 0.0);
		//
		// /// Go up five times
		// for(int i = 0; i < 5; ++i) {
		// 	goup.set(0, states.row(i+6).get(0));
		// 	goup.set(1, states.row(i+6).get(1));
		// 	Vec out = nn2.predict(goup);
		//
		//
		// 	in.set(2, out.get(0));
		// 	in.set(3, out.get(1));
		//
		// 	/// produce an image
		// 	for(int j = 0; j < ib.height; ++j) {
		// 		for(int l = 0; l < ib.width; ++l) {
		// 			in.set(0, (double)l/ib.width);
		// 			in.set(1, (double)j/ib.height);
		//
		// 			Vec color = nn1.predict(in);
		// 			color.scale(255.0);
		//
		// 			ib.WritePixelBuffer(l, j, color);
		// 		}
		// 	}
		//
		// 	ib.outputToPNG("img/frame" + (i+6) + ".png");
		// }

		// Pre


		// for(int i = 0; i < v_t_1_andActions; ++i) {
		// 	/// Train by sections so we can obtain the prediction
		// 	nn2.predict(v_t_1_andActions.row(trainingIndices[i]));
		// 	nn2.backProp(v_t.row(trainingIndices[i]));
		// 	nn2.updateGradient(v_t_1_andActions.row(trainingIndices[i]));
		// 	refineWeights(nn2.learning_rate);
		// 	nn2.learning_rate -= 0.00001;
		// 	nn2.gradient.fill(0.0);
		//
		// }

	}

	public static void testNN() {
		/// Instantiate net (Observation function /decoder)
		Random r = new Random(123);
		NeuralNet nn1 = new NeuralNet(r);

		/// Build topology for observation function
		nn1.layers.add(new LayerLinear(4, 12));
		nn1.layers.add(new LayerTanh(12));
		nn1.layers.add(new LayerLinear(12, 12));
		nn1.layers.add(new LayerTanh(12));
		nn1.layers.add(new LayerLinear(12, 3));
		nn1.layers.add(new LayerTanh(3));

		double[] w  = {
			0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,

			0, 0.003, 0.006, 0.009,
			0.007, 0.01, 0.013, 0.016,
			0.014, 0.017, 0.02, 0.023,
			0.021, 0.024, 0.027, 0.03,
			0.028, 0.031, 0.034, 0.037,
			0.035, 0.038, 0.041, 0.044,
			0.042, 0.045, 0.048, 0.051,
			0.049, 0.052, 0.055, 0.058,
			0.056, 0.059, 0.062, 0.065,
			0.063, 0.066, 0.069, 0.072,
			0.07, 0.073, 0.076, 0.079,
			0.077, 0.08, 0.083, 0.086,

			0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,

			0, 0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021, 0.024, 0.027, 0.03, 0.033,
			0.007, 0.01, 0.013, 0.016, 0.019, 0.022, 0.025, 0.028, 0.031, 0.034, 0.037, 0.04,
			0.014, 0.017, 0.02, 0.023, 0.026, 0.029, 0.032, 0.035, 0.038, 0.041, 0.044, 0.047,
			0.021, 0.024, 0.027, 0.03, 0.033, 0.036, 0.039, 0.042, 0.045, 0.048, 0.051, 0.054,
			0.028, 0.031, 0.034, 0.037, 0.04, 0.043, 0.046, 0.049, 0.052, 0.055, 0.058, 0.061,
			0.035, 0.038, 0.041, 0.044, 0.047, 0.05, 0.053, 0.056, 0.059, 0.062, 0.065, 0.068,
			0.042, 0.045, 0.048, 0.051, 0.054, 0.057, 0.06, 0.063, 0.066, 0.069, 0.072, 0.075,
			0.049, 0.052, 0.055, 0.058, 0.061, 0.064, 0.067, 0.07, 0.073, 0.076, 0.079, 0.082,
			0.056, 0.059, 0.062, 0.065, 0.068, 0.071, 0.074, 0.077, 0.08, 0.083, 0.086, 0.089,
			0.063, 0.066, 0.069, 0.072, 0.075, 0.078, 0.081, 0.084, 0.087, 0.09, 0.093, 0.096,
			0.07, 0.073, 0.076, 0.079, 0.082, 0.085, 0.088, 0.091, 0.094, 0.097, 0.1, 0.103,
			0.077, 0.08, 0.083, 0.086, 0.089, 0.092, 0.095, 0.098, 0.101, 0.104, 0.107, 0.11,

			0,0.001,0.002,

			0, 0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021, 0.024, 0.027, 0.03, 0.033,
			0.007, 0.01, 0.013, 0.016, 0.019, 0.022, 0.025, 0.028, 0.031, 0.034, 0.037, 0.04,
			0.014, 0.017, 0.02, 0.023, 0.026, 0.029, 0.032, 0.035, 0.038, 0.041, 0.044, 0.047
		};
		nn1.weights = new Vec(w);
		nn1.gradient = new Vec(nn1.weights.size());

		/// Load observations from file
		Matrix x_observations = new Matrix();
		x_observations.loadARFF("data/observations.arff");
		x_observations.scale(1/255.0); // normalize data

		/// Empty matrix to hold predicted states
		int k = 2; // # of degrees of freedom in the system
		Matrix states = new Matrix(x_observations.rows(), k);
		states.fill(0.0);

		/// passing (Data from file, states from (1) -> (n))
		nn1.train_with_images(x_observations, states);
	}

	public static void work() {
		Random random = new Random(123456);

		/// Load data
		Matrix data = new Matrix();
		data.loadARFF("data/labeled_data_noextras-random.arff");

		/// Create a new filter to preprocess our data
		Filter f = new Filter(random);

		/// Partition the features from the labels
		Matrix features = new Matrix();
		Matrix labels = new Matrix();
		f.splitLabels(data, features, labels);

		System.out.println(features.rows() + " " + features.cols());
		System.out.println(labels.rows() + " " + labels.cols());

		/// PREPROCESSING
		// We need a set of preprocessors for both features and labels

		// Train the preprocessors for the training data
		f.train(features, labels, null, 0, 0.0);

		System.out.println(features.rows() + " " + features.cols());
		System.out.println(labels.rows() + " " + labels.cols());

		/// Partition the data into training and testing blocks
		/// With respective feature and labels blocks
		double splitRatio = 0.75;
		Matrix trainingFeatures = new Matrix();
		Matrix trainingLabels = new Matrix();
		Matrix testingFeatures = new Matrix();
		Matrix testingLabels = new Matrix();
		f.splitData(features, labels, trainingFeatures, trainingLabels,
			testingFeatures, testingLabels, 5, 0);


		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainingFeatures.rows()];
		int[] testIndices = new int[testingFeatures.rows()];

		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }
		for(int i = 0; i < testIndices.length; ++i) { testIndices[i] = i; }


		/// I want some intelligent way of getting the input and outputs
		f.nn.layers.add(new LayerLinear(trainingFeatures.cols(), 80));
		f.nn.layers.add(new LayerTanh(80));

		f.nn.layers.add(new LayerLinear(80, 100));
		f.nn.layers.add(new LayerTanh(100));

		f.nn.layers.add(new LayerLinear(100, 20));
		f.nn.layers.add(new LayerSine(20));

		f.nn.layers.add(new LayerLinear(20, 1));
		f.nn.layers.add(new LayerTanh(1));

		f.nn.initWeights();

		int mis = testingLabels.rows();
		int epoch = 0;

		double testSSE = 0;
		double trainSSE = 0;

		double previous = 0;
		double tolerance = 0.0000009;


		System.out.println("batch,seconds,testRMSE,trainRMSE");
		int batch = 1;
		int batch_size = 1;
		double startTime = (double)System.nanoTime();

		double[] testpattern = {6,63,702,0,0,1,0,7,0,1};
		Vec vvv = new Vec(testpattern);
		while(true) {

			testSSE += f.sum_squared_error(testingFeatures, testingLabels);
			double testMSE = testSSE / testingFeatures.rows();
			double testRMSE = Math.sqrt(testMSE);

			trainSSE += f.sum_squared_error(trainingFeatures, trainingLabels);
			double trainMSE = trainSSE / trainingFeatures.rows();
			double trainRMSE = Math.sqrt(trainMSE);

			f.trainNeuralNet(trainingFeatures, trainingLabels, trainingIndices, batch_size, 0.0);
			// double mse = sse / batch;
			// double rmse = Math.sqrt(mse);

			double seconds = ((double)System.nanoTime() - startTime) / 1e9;
			System.out.println(batch + "," + seconds + "," + testRMSE + "," + trainRMSE);

			batch = batch + 1;

			// mis = f.countMisclassifications(testingFeatures, testingLabels);
			// System.out.println("mis: " + mis);

			double convergence = Math.abs(1 - (previous / testSSE));
			previous = testSSE;
			testSSE = 0;
			trainSSE = 0;
			System.out.println("testPattern" + f.nn.predict(vvv));
			// 0,188,7309,1,0,0,0,0,1,0,'authentic'
			if(convergence < tolerance) break;

		}

	}

	public static void main(String[] args) {
		work();
		//testNN();
		//infer_test();
	}
}
