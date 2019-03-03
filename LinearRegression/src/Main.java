import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class Main
{
	/*
	static void test(NeuralNet learner, String challenge)
	{
		// Load the training data
		String fn = challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_features.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_labels.arff");
		Vec w = new Vec(trainFeatures.cols()*trainLabels.cols());
		// Train the model, get weights (m and b)
		w=learner.layer1.ordinary_least_squares(trainFeatures, trainLabels, w);
		//System.out.println("Big Vector w has values: ");
		//w.print();
		// using generated w to compute Y 
		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_features.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_labels.arff");
		Vec realLabels = new Vec(testLabels.rows());
		realLabels = realLabels.convertVec(testLabels);

		// Matrix testY is used to store generated label values
		Matrix testY = new Matrix(testLabels.rows(),testLabels.cols()); // 506 instance, so corresponding 5Y.  Y = MX + b
		// Given matrix testFeatures and store each instance of X into a Vector[]
		// Initialize
		Vec[] arrayX = new Vec[testFeatures.rows()]; 
		for(int i = 0; i <testFeatures.rows();i++)
		{
			arrayX[i] = new Vec(testFeatures.cols());
		}
		// load values from Matrix testFeatures to arrayX[i]	
		for(int i = 0; i < testFeatures.rows();i++)
		{
			for(int j =0; j< testFeatures.cols();j++)
			{
				arrayX[i].set(j, testFeatures.getvalue(i, j));
			}
		}
		
		for(int i = 0; i < testFeatures.rows();i++)
		{
			System.out.println("arrayX[" + i + "] ");
			arrayX[i].print();
			
		}
		
		// calculate activate
		// Initialize 
		Matrix[] arrayY = new Matrix[testFeatures.rows()]; 
		for(int i = 0; i < testFeatures.rows();i++)
		{
			arrayY[i] = new Matrix(1,1);
		}
	
		for(int i = 0; i < testFeatures.rows();i++)
		{
			learner.layer1.activate(w, arrayX[i]);
			arrayY[i] = learner.layer1.activation;
		}
		
	
		
		for(int i =0; i < testFeatures.rows();i++)
		{
			System.out.println("Y[ " + i +" ]" );
			arrayY[i].print();
		}
		
		int index =0;
		Vec testy = new Vec(testY.rows());
		for(int i = 0; i< arrayY.length;i++)
			testy.set(index++, arrayY[i].getvalue(0, 0));
		
		//System.out.println("Vector testy is: ");
		//testy.print();
		//System.out.println("realLabels has size of " + realLabels.size()+ "values:" );
		//realLabels.print();
		//System.out.println("Predicted Y has size of " + testy.size()+ "values: ");
		//testy.print();
		double result = 0.0;
		result = learner.layer1.rootSquareError(realLabels, testy);
		System.out.println("RMSE(Root-mean-square-error) is: " + result);
	}
	*/
	/*
	public static void Backpro() // (SupervisedLearner learner)
	{
		
		// Load data
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF("train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF("train_lab.arff");
		Random rand = new Random();
		// Normalize each element to a range of [0,1]
		for(int i =0; i < trainFeatures.rows();i++)
		{
			for(int j = 0; j < trainFeatures.cols();j++)
			{
				trainFeatures.setMatrix(i, j, trainFeatures.getvalue(i, j)/256.0);
			}
		}
		// Onehot conversion
		Matrix onehotTrainLabel = new Matrix(trainLabels.rows(),10);
		for(int i =0; i < trainLabels.rows();i++)
		{
			int num = (int)trainLabels.getvalue(i, 0);

			for(int k = 0; k<9; k++)
			{
				if (k == num)
					onehotTrainLabel.setMatrix(i, k, 1);
				else
					onehotTrainLabel.setMatrix(i, k, 0);
			}
		}
		
		// Load Testing data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF("test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF("test_lab.arff");
		// Normalize each element to a range of [0,1]
		for(int i =0; i < testFeatures.rows();i++)
		{
			for(int j = 0; j < testFeatures.cols();j++)
			{
				testFeatures.setMatrix(i, j, testFeatures.getvalue(i, j)/256.0);
			}
		}
		// Onehot conversion
		Matrix onehotTestLabel = new Matrix(testLabels.rows(),10);
		for(int i =0; i < testLabels.rows();i++)
		{
			int num = (int)testLabels.getvalue(i, 0);
			for(int k = 0; k<9; k++)
			{
				if (k == num)
					onehotTestLabel.setMatrix(i, k, 1);
				else
					onehotTestLabel.setMatrix(i, k, 0);
			}
		}

		// Add 6 layers
		NeuralNet nn = new NeuralNet();

		nn.layers.add(new LayerLinear(784, 80));
		nn.layers.add(new LayerTanh(80));

		nn.layers.add(new LayerLinear(80, 30));
		nn.layers.add(new LayerTanh(30));

		nn.layers.add(new LayerLinear(30, 10));
		nn.layers.add(new LayerTanh(10));

		nn.initWeights();// set up random weight in each layer
		int epoch =0;
		int num = 10000;
		System.out.println("Let's start");

			System.out.println("Training Epoch [ " + epoch + " ] ============================>");
			num = nn.countMisclassifications(testFeatures, onehotTestLabel);
			System.out.println(num);
			
			ArrayList<Integer> shuffleIndex = new ArrayList<Integer>();
			for(int i = 0; i < trainFeatures.rows();i++)
			{
				shuffleIndex.add(i);
			}
			Collections.shuffle(shuffleIndex,rand);
			

			for(int i =0; i < trainFeatures.rows();i++)
			{
				int index = shuffleIndex.get(i);
				 Vec feat = trainFeatures.row(index);// feat is a vector containing features in a specific row
				 Vec lab = onehotTrainLabel.row(index); // lab is a vector containing labels in that corresponding specific row
				 // Forward Propagation
				 
				 // linear-layer 1:
				 nn.layers.get(0).activate(nn.weights.get(0), feat);
				 // Non-linear layer1:
				 nn.layers.get(1).activate(null, nn.layers.get(0).activation);
				 // Linear layer2:
				 nn.layers.get(2).activate(nn.weights.get(2), nn.layers.get(1).activation);
				 // Non-linear layer2:
				 nn.layers.get(3).activate(null, nn.layers.get(2).activation);
				 // Linear layer3:
				 nn.layers.get(4).activate(nn.weights.get(4), nn.layers.get(3).activation);
				 // Non-linear layer3:
				 nn.layers.get(5).activate(null, nn.layers.get(4).activation);
				 
				 // Calculate blame:
				 // Currently at last layer which is a non-linear, get Last layer'blame: target - prediction
				 nn.layers.get(5).blame = lab.subtract(lab,  nn.layers.get(5).activation);
				 //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
				 // blame on last linear layer get(4)
				 nn.layers.get(4).blame = nn.layers.get(5).backprop(null, nn.layers.get(5).blame);
				 // blame on 2nd non-linear layer get(3)
				 System.out.println("Print 2nd non-linear layer's weight size" + nn.weights.get(4).len);
				 System.out.println("Print 2nd non-linear layer's blame size" + nn.layers.get(4).blame.len);
				 
				 nn.layers.get(3).blame = nn.layers.get(4).backprop(nn.weights.get(4), nn.layers.get(4).blame);
				 // blame on 2nd linear layer get(2)
				 nn.layers.get(2).blame = nn.layers.get(3).backprop(null, nn.layers.get(3).blame);
				 // blame on 1st non-linear layer get(1)
				 nn.layers.get(1).blame = nn.layers.get(2).backprop(nn.weights.get(2), nn.layers.get(2).blame);
				 // blame on 1st linear layer get(0)
				 nn.layers.get(0).blame = nn.layers.get(1).backprop(null, nn.layers.get(1).blame);
		
				 // Compute update gradient
				 // 1st linear layer:
				 nn.layers.get(0).updateGradient(feat, null);
				 // non-linear layer update gradient is do nothing, so I just pass it
				 // 2nd linear layer update gradient:
				 nn.layers.get(2).updateGradient(nn.layers.get(1).activation, null);
				 // 3rd linear layer update gradient:
				 nn.layers.get(4).updateGradient(nn.layers.get(3).activation, null);
				 
				 // Last step update all weights in each layer
				 // 1st linear layer
				 nn.layers.get(0).weight.addScaled(0.01, nn.layers.get(0).gradient);
				 // Nothing to do in non-linear layer
				 // 2nd linear layer
				 nn.layers.get(2).weight.addScaled(0.01, nn.layers.get(2).gradient);
				 // 3rd linear layer
				 nn.layers.get(4).weight.addScaled(0.01, nn.layers.get(4).gradient);
				 
				 // set gradients to zero
				 nn.layers.get(0).gradient.fill(0.0);
				 nn.layers.get(0).gb.fill(0.0);
				 nn.layers.get(0).gM.fill(0.0);
				 nn.layers.get(2).gradient.fill(0.0);
				 nn.layers.get(2).gb.fill(0.0);
				 nn.layers.get(2).gM.fill(0.0);
				 nn.layers.get(4).gradient.fill(0.0);
				 nn.layers.get(4).gb.fill(0.0);
				 nn.layers.get(4).gM.fill(0.0); 
	
			
			
			epoch++;
		}
		*/
		/*
		for(int i = 0; i <nn.layers.size();i++)//6
		{
			nn.weights.add(nn.layers.get(i).weight);
			nn.activations.add(nn.layers.get(i).activation);
			nn.blames.add(nn.layers.get(i).blame);
			nn.gradients.add(nn.layers.get(i).gradient);

		}
		for(int i = 0; i <1 ;i++)
		{
			// Training
			nn.train(trainFeatures, onehotTrainLabel);
			System.out.println(nn.activations.size());
			
			for(int j = 0; j <6;j++)
			{
				nn.layers.get(j).gb.fill(0.0);
				nn.layers.get(j).gM.fill(0.0);
				
			}
			nn.refineWeights(null, null, null, 0.03);
			int num = nn.countMisclassifications(testFeatures, onehotTestLabel);
			System.out.println(num);
		}
		
		//System.out.println("In the first layer weights: ");
		//nn.layers.get(0).weight.print();
		
		for(int i = 0; i <nn.layers.size();i++)//6
		{
			nn.weights.add(nn.layers.get(i).weight);
			nn.activations.add(nn.layers.get(i).activation);
			nn.blames.add(nn.layers.get(i).blame);
			nn.gradients.add(nn.layers.get(i).gradient);
		}
		//System.out.println("In NeuralNet the last layer has activation: ");
		//nn.activations.get(0).print();
		//System.out.println("Last layer activation:" );
		//nn.layers.get(5).activation.print();
		//System.out.println("Last layer blame:" );
		//nn.layers.get(5).blame.print();
		*/
	//}
	/*
	public static void backproTest() 	
	{
		/*
		 * Copy from Web
		I made a simple 3-layer neural network with the following topology:
			0) [LayerLinear: 1->2, Weights=4]
			1) [LayerTanh: 2->2, Weights=0]
			2) [LayerLinear: 2->1, Weights=3]

			This neural network has a total of 7 weights, which I initialized to:
			[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3]
		
		// initialize
		NeuralNet nn= new NeuralNet();
		nn.layers.add(new LayerLinear(1,2));
		nn.layers.add(new LayerTanh(2));
		nn.layers.add(new LayerLinear(2,1));
		double[] x = {0.3};
		//Vec xx = new Vec(x);
		double[] w = {0.3, 0.4, 0.1, 0.2};
		double []w_2 = {0.2, 0.3, 0.1};
		double[]t = {0.7};// target
		Vec target = new Vec(t);
		// Activate in each layer
		// Linear layer1:
		for(int i =0; i < 3;i++)
		{
			nn.layers.get(0).activate(new Vec(w), new Vec(x));
			System.out.println("1st linearLayer activation:");
			System.out.println(nn.layers.get(0).activation);
			
			// Non-linear layer1:
			nn.layers.get(1).activate(null, nn.layers.get(0).activation);
			System.out.println("1st LayerTanh activation:");
			System.out.println(nn.layers.get(1).activation);
			// Linear layer2:
			
			nn.layers.get(2).activate(new Vec(w_2), nn.layers.get(1).activation);
			System.out.println("2nd LayerLinear activation:");
			System.out.println(nn.layers.get(2).activation);
			// Calculate blame:

			nn.layers.get(2).blame = target.subtract(target, nn.layers.get(2).activation);
			System.out.println("Arrive at the last layer, the value of blame is: ");
			System.out.println(nn.layers.get(2).blame);
			System.out.println("Let's do backpropagation");
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			// Calculate blame on non-linear layer
			nn.layers.get(1).blame = nn.layers.get(2).backprop(new Vec(w_2), nn.layers.get(2).blame);
			System.out.println("1st tanh layer blame: \n" + nn.layers.get(1).blame);
			// Calculate preBlame on 1st tanh layer
			nn.layers.get(0).blame = nn.layers.get(1).backprop(null, nn.layers.get(1).blame);
			System.out.println("On the tanh layer, the previous blame is: \n" + nn.layers.get(0).blame);
			
			// Compute update gradient
			// 1st linear layer:
			nn.layers.get(0).updateGradient(new Vec(x), null);
			System.out.println("1st linear layer weight gradient: ");
			nn.layers.get(0).gM.print();
			System.out.println("1st linear layer bias gradient: ");
			nn.layers.get(0).gb.print();
			// non-linear layer update gradient is do nothing, so I just pass it
			// 2nd linear layer update gradient:
			nn.layers.get(2).updateGradient(nn.layers.get(1).activation, null);
			System.out.println("2nd linear layer weight gradient: ");
			nn.layers.get(2).gM.print();
			System.out.println("2nd linear layer bias gradient: ");
			nn.layers.get(2).gb.print();
			
			// Last step update all weights in each layer
			// 1st linear layer
			nn.layers.get(0).weight.addScaled(0.1, nn.layers.get(0).gradient);
			System.out.println("1st linear layer==> updated weight is");
			nn.layers.get(0).weight.print();
			// Nothing to do in non-linear layer
			// 2nd linear layer
			nn.layers.get(2).weight.addScaled(0.1, nn.layers.get(2).gradient);
			System.out.println("2nd linear layer==> updated weight is");
			nn.layers.get(2).weight.print();
			// set gradients to zero
			nn.layers.get(0).gradient.fill(0.0);
			nn.layers.get(0).gb.fill(0.0);
			nn.layers.get(0).gM.fill(0.0);
			nn.layers.get(2).gradient.fill(0.0);
			nn.layers.get(2).gb.fill(0.0);
			nn.layers.get(2).gM.fill(0.0);
		}
	}
*/

	static void testBackpro()
	{
		Random rand = new Random(); 

		// Load data
		// training data
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF("train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF("train_lab.arff");
		// testing data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF("test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF("test_lab.arff");

		// Normalize each element to a range of [0,1]
		for(int i =0; i < trainFeatures.rows();i++)
		{
			for(int j = 0; j < trainFeatures.cols();j++)
			{
				trainFeatures.setMatrix(i, j, trainFeatures.getvalue(i, j)/256.0);
			}
		}
		for(int i =0; i < testFeatures.rows();i++)
		{
			for(int j = 0; j < testFeatures.cols();j++)
			{
				testFeatures.setMatrix(i, j, testFeatures.getvalue(i, j)/256.0);
			}
		}
		// Onehot conversion
		Matrix onehotTestLabel = new Matrix(testLabels.rows(),10);
		for(int i =0; i < testLabels.rows();i++)
		{
			int num = (int)testLabels.getvalue(i, 0);
			for(int k = 0; k<9; k++)
			{
				if (k == num)
					onehotTestLabel.setMatrix(i, k, 1);
				else
					onehotTestLabel.setMatrix(i, k, 0);
			}
		}
		// Add 6 layers
		NeuralNet nn = new NeuralNet();

		nn.layers.add(new LayerLinear(784, 160));
		nn.layers.add(new LayerTanh(160));

		nn.layers.add(new LayerLinear(160, 60));
		nn.layers.add(new LayerTanh(60));

		nn.layers.add(new LayerLinear(60, 10));
		nn.layers.add(new LayerTanh(10));

		nn.initWeights(rand);// set up random weight in each layer

		ArrayList<Integer> shuffleIndex = new ArrayList<Integer>();
		for(int i = 0; i < trainFeatures.rows();i++)
		{
			shuffleIndex.add(i);
		}
		Collections.shuffle(shuffleIndex,rand);
		
		/// Training and testing
		int mis = 10000;
		int epoch = 0;
		System.out.println("Let's start");
		while(mis > 350) {
			System.out.println("Training Epoch [ " + (epoch+1) + " ] ============================>");
			// display Missclassification
			mis = nn.countMisclassifications(testFeatures, testLabels);
			System.out.println("Misclassifications: " + mis);
			// for each instance in training data
			for(int i = 0; i < trainFeatures.rows(); ++i) {
				Vec target;
				// onehot representation
				target = new Vec(10);
				target.vals[(int) trainLabels.row(i).get(0)] = 1;

				Vec feat = trainFeatures.row(i);// feat is a vector containing features in a specific row
				// training
				nn.refineWeights(feat, target, nn.weights, 0.03);
			}

			++epoch;
		}
	}
	public static void testLinear(SupervisedLearner learner)
	{
		// Load data
		//String fn = "data/";
		Matrix houseFeatures = new Matrix();
		houseFeatures.loadARFF("housing_features.arff");
		//houseFeatures.print();
		Matrix houseLabels = new Matrix();
		houseLabels.loadARFF("housing_labels.arff");
		Vec houseLabelsVector = new Vec(houseLabels.rows());
		houseLabelsVector.convertVec(houseLabels);
		// Cross-validate the model: 5 iterations with 10 folds
		learner.crossValidation(5, 10, houseFeatures, houseLabels);	
	}
	/*
	public static void testLearner(NeuralNet learner)
	{
		test(learner, "1");
		test(learner, "2");
		test(learner, "3");
		test(learner, "4");
		test(learner, "5");
	}
	*/
	
	
	
	
	 
	public static void main(String[] args)
	{

		System.out.println("After approximately 40 epoches, the misclassifications is below 350");
		testBackpro();
	}
}
