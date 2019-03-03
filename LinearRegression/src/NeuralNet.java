import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;

class NeuralNet extends SupervisedLearner{
	// NeuralNet consisted with several layers including linear and non-linear layers
	public ArrayList<Layer> layers;
	protected Vec weights;
	protected Vec gradient;
	Random rand = new Random();
	public double rate; // learning rate
	public int iterations=10000;
	public int binSize=1;
	
	NeuralNet()
	{
		this.layers = new ArrayList<Layer>();	
	}
	
	 void backProp( Vec target) 
	 {
		Vec blame = new Vec(target.size());
		// blame = target - current layer activation
		blame.add(target);
		blame.addScaled(-1, layers.get(layers.size()-1).activation);
		
		for(int i = layers.size()-1; i >= 0; i--) {
			// set up backprop in each layer
			Layer layer = layers.get(i);
		    blame = layer.backprop(weights, blame);
		}
	  }

	 void updateGradient(Vec x) 
	 {
	    int position = 0;
	    for(int i = 0; i < layers.size(); i++) {
	      Layer layer = layers.get(i);
	      // for each layer, use the corresponding gradient
	      //public Vec(Vec v, int begin, int length)
	      Vec g = new Vec(gradient, position, layer.NumWeights());
	      
	      layer.updateGradient(x, g);
	      // input Vec is the activation from current layer
	      x = new Vec(layer.activation);
	      position += layer.NumWeights();
	    }
	  }
	
	@Override
	String name() {
		return "NeuralNet";
	}
	
	@Override
	void train(Matrix features, Matrix labels) {
		
		int numBin = features.rows() / binSize;
		
		if(this.iterations < numBin)
		{
			this.iterations = this.iterations;
		}
		else
		{
			this.iterations = numBin;
		}
		// shuffle the data
		ArrayList<Integer> shuffleIndex = new ArrayList<Integer>();
		for(int i = 0; i < features.rows();i++)
		{
			shuffleIndex.add(i);
		}
		Collections.shuffle(shuffleIndex,rand);
		// compute weights and gradients
		for(int i = 0; i < 1; i++)//this.iterations=10000
		{
			 for(int j = 0; j < this.binSize;j++)
			 {
				 int index = shuffleIndex.get(i * binSize + j);
				 Vec feat = features.row(index);// feat is a vector containing features in a specific row
				 Vec lab = labels.row(index); // lab is a vector containing labels in that specific row
				 //System.out.println("feat size:"+ feat.size());
				 //feat.print();
				 //forwardProp(feat);
				 //backProp(weights,lab);
				 //updateGradient(feat);
			 }
		}

	}
	// Calculate the activation of the last layer
	@Override
	public Vec predict(Vec x) 
	{
		int position = 0;
		// Calculate the activation from last layer in each 
		for(int i = 0; i < layers.size(); i++) 
		{
			Layer layer = layers.get(i);
			Vec w = new Vec(weights,position,layer.NumWeights());
			layer.activate(w, x);
			position += layer.NumWeights();
			x = layer.activation;
	    }
	    return (layers.get(layers.size()-1).activation);
	}
	/*
	// forward backprop
	public void forwardProp(Vec x) // x is a vector with 784 elements
	{
		// No problem with weights and x 
		// in first layer weights.size =  62800 and x.size = 784 
		Vec result = new Vec(x.len);

		layers.get(0).activate(weights.get(0),x);
		activations.add(layers.get(0).activation);
		for(int i =1; i < size; i++)
		{
			layers.get(i).activate(weights.get(i), layers.get(i-1).activation);
			//layers.get(i).activation.print();
			activations.add(layers.get(i).activation);
			//result.print();
		}
		//layers.get(1).activation.print();
		//System.out.println("1 non-linear layer activation: "+ layers.get(1).activation.len);
		//System.out.println("ForwardProp: Activation has size of " + a.len);
		
	}
	*/
	public void initWeights(Random r)
	{
		int size = 0;
	    for(int i = 0; i < layers.size(); i++)
	    {
		    Layer layer = layers.get(i);
		    size += layer.NumWeights();
	    }
	    gradient = new Vec(size);
	    weights = new Vec(size);

		int position = 0;
	    for(int i = 0; i < layers.size(); i++) 
	    {
	        Layer l = layers.get(i);
	        Vec w = new Vec(weights,position,l.NumWeights());
	        l.initWeights(w, r);
	        position +=  l.NumWeights();
	    }
	}
	
	public void refineWeights(Vec x, Vec y, Vec weights, double rate)
	{
		gradient.fill(0.0);
	    predict(x);
	    backProp(y);
	    updateGradient(x);
	    this.weights.addScaled(rate, gradient);
	}
	
	/*void testOLS()
	{
		/*Assume M is a matrix of 5*13=65
		x should have 13 rows
		b should have 5 rows 
		The size of weights is 5*13+ 5=70
		since big vector weights containing all M and b
		
		Matrix testX = new Matrix(5,13);// 5 instances, each instance has 13 features
		Vec testWeights = new Vec(13*1+1); // assume output labels have 1 value
		
		Vec x = new Vec(testX.rows()*testX.cols());
		Matrix testY = new Matrix(5,1); // 5 instance, so corresponding 5Y.  Y = MX + b
		NeuralNet nn = new NeuralNet(13,1);
		
		// step 1: Generate some random weights
		testWeights = testWeights.fillrand(testWeights);
		//System.out.println("testWeights size: " + testWeights.size());
		// step 2: Generate a random feature matrix, X.
		testX = testX.fillMatrix(testX);
		x = x.convertVec(testX);
		
		// step 3: compute a corresponding label matrix, Y.
		// Given matrix X and store each instance of X into a Vector[]
		// Initialize
		Vec[] arrayX = new Vec[testX.rows()]; 
		for(int i = 0; i < testX.rows();i++)
		{
			arrayX[i] = new Vec(testX.cols());
		}
		// load values from Matrix testX to arrayX[i]	
		for(int i = 0; i < testX.rows();i++)
		{
			for(int j =0; j< testX.cols();j++)
			{
				arrayX[i].set(j, testX.getvalue(i, j));
			}
		}
		
		// calculate Y
		// Initialize 
		
		Vec[] arrayY = new Vec[testX.rows()]; 
		for(int i = 0; i < testX.rows();i++)
		{
			arrayY[i] = new Vec(1);
		}

		
		for(int i = 0; i < testX.rows();i++)
		{
			nn.layer1.activate(testWeights, arrayX[i]);
			//arrayY[i] = nn.layer1.activation;
			//activate(testWeights, arrayX[i]);
		}

		// store arraY[i] into a double y[]
		double [] y = new double[arrayY.length];
		for(int i =0; i< arrayY.length;i++)
		{
			y[i] = arrayY[i].get(0);
		}
		// convert double y[] into a Matrix testY
		for(int i =0; i< arrayY.length;i++)
		{
			testY.setMatrix(i, 0, y[i]);
		}

		// step 6
		Vec newWeights = new Vec(testWeights.size());
		newWeights= nn.layer1.ordinary_least_squares(testX, testY, newWeights);
		System.out.println("New weight has size of " + newWeights.size()+ " and values:  ");
		newWeights.print();
		System.out.println("Original weight has size of " + testWeights.size()+ " and values:  ");
		testWeights.print();
		double result = 0.0;
		result = layer1.rootSquareError(newWeights, testWeights);
		System.out.println("Root-squre-error of original weights and generated weights is: " + result);
	}
*/

	
}
