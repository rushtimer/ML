import java.util.Random;

class LayerLinear extends Layer 
{		
	public LayerLinear(int inputs, int outputs) 
	{
		super(inputs, outputs);
	}
	
	void activate(Vec weights, Vec x) 
	{
		Vec m = new Vec (weights, outputs, outputs*inputs);
		//System.out.println("size of vector m is " + m.size());
		Vec b = new Vec (weights, 0, outputs);
		// M: a matrix with "outputs" row and "inputs" columns
		Matrix M = new Matrix(outputs,inputs);//1*13 matrix M
		//System.out.println("size of matrix M is " + (M.rows()*M.cols()));
		// convertX is used to multiply two matrixes
		Matrix convertX = new Matrix(inputs,1);
		//compute  activation = M * x 
		Matrix Mx = new Matrix(outputs,1);
		// convert vector b to matrix
		Matrix convertB= new Matrix(outputs,1);
		//Matrix activation = new Matrix(outputs,1);
		// Load values into matrix M
		int index = 0;
		for(int i =0; i < outputs; i++)
		{
			for(int j = 0; j < inputs; j++)
			{
				M.setMatrix(i, j, m.get(index++));
			}
		}
		// Set up values into Matrix convertX
		index=0;
		for(int i = 0; i < inputs; i++)
		{
			for(int j = 0; j <1 ;j++)
			{
				convertX.setMatrix(i, j, x.get(index++));
			}
		}
		index = 0;
		// Convert Matrixes product Mx to Vector ConvertMx
		//Mx = M.matrixProduct(M, convertX, outputs, inputs);
		Mx = M.times(convertX);
		// Set up values into Matrix B
		for(int i = 0; i < outputs; i++)
		{
			for(int j= 0; j < 1;j++)
			{
				convertB.setMatrix(i, j, b.get(index++));
			}
		}
		index = 0; 
		// finally activation = Mx + b
		activation = activation.convertVec(Mx.plus(convertB));

	}
	
	// OLS function
	public Vec ordinary_least_squares (Matrix X, Matrix Y, Vec weights)
	{
		// initialize Vec centroid X and Y
		Vec x_cen = new Vec(X.cols());
		Vec y_cen = new Vec(Y.cols());
		// calculate the Vector x_cen
		for(int i = 0; i < X.cols(); i++)
		{
			double sumX = 0.0;
			for(int j = 0; j< X.rows();j++)
			{
				sumX = sumX + X.getvalue(j, i);
			}
			x_cen.set(i, sumX/X.rows());
		}
		//System.out.println("Vector x_cen is ");
		//x_cen.print();
		// calculate the Vector y_cen
		for(int i = 0; i < 1; i++)
		{
			double sumY = 0.0;
			for(int j = 0; j< Y.rows();j++)
			{
				sumY = sumY + Y.getvalue(j,i);
			}
			y_cen.set(i, sumY/Y.rows());
		}
		//System.out.println("Vector y_cen is: ");
		//y_cen.print();
		
		// calculate origin centroid data
		Matrix YY = new Matrix(Y.rows(),Y.cols());
		Matrix XX = new Matrix(X.rows(),X.cols());
		
		YY = YY.matrixMinusVec(Y, y_cen);
		//System.out.println("origin_centroid data of y: \n");
		//YY.print();
		XX = XX.matrixMinusVec(X, x_cen);
		//System.out.println("origin_centroid data of X: \n");
		//XX.print();
		Matrix nsum = new Matrix(Y.cols(),X.cols());// nsum is a 1 x 13 matrix
		Matrix dsum = new Matrix(X.cols(),X.cols()); // nsum is a 13 x 13 matrix
		//System.out.println("YY size:" + YY.rows()+ " x " + YY.cols());
		//System.out.println("YY_transpose size:" + YY.transpose().rows()+ " x " + YY.transpose().cols());
		//System.out.println("XX size:" + XX.rows()+ " x " + XX.cols());
		//System.out.println("YY transpose: " + YY.transpose());
		nsum = YY.transpose().times(XX);
		//System.out.println("numerator: ");
		//nsum.print();
		dsum = XX.transpose().times(XX);
		//System.out.println("denominator: ");
		//dsum.print();
		// Calculate Matrix X
		Matrix matrixM = new Matrix(nsum.rows(),dsum.cols());
		matrixM = nsum.times(dsum.pseudoInverse());
		//System.out.println("Matrix M is:  ");
		//matrixM.print();
		// Vector m is used for getting weights
		Vec m = new Vec(matrixM.rows()*matrixM.cols());
		m = m.convertVec(matrixM);
		//System.out.println("Vector M is:  ");
		//m.print();
		
		Vec b = new Vec(1);
		b = y_cen.computeB(y_cen, matrixM, x_cen);	
		// store values into Vector weights 		
		weights = m.concatenate(m, b);	
		//System.out.println("Big Vector weights is:  ");
		//weights.print();	
		return weights;
	}
	
	double rootSquareError(Vec a, Vec b)
	{
		double RSE = 0.0;
		if(a.size() != b.size())
			throw new RuntimeException("Error: matrix dimensions don't match.");
		double sum = 0.0;
		for(int i =0; i < a.size();i++)
		{
			sum = sum + Math.pow((a.get(i)-b.get(i)), 2);
		}
		RSE = 10*Math.sqrt(sum/a.size());
		return RSE;
	}
	
	//prevBlame = MT * blame.
	@Override
	Vec backprop(Vec weights, Vec prevBlame) 
	{
		// Vector weights is consisted of two parts: M and b
		// Step1: convert weights to Matrix M
		blame.fill(0.0);
	    blame.add(prevBlame);

	    int index = outputs; 
	    Matrix M = new Matrix(outputs, inputs);

	    /// Turn our section of weights into a Matrix
	    for(int i = 0; i < M.rows(); ++i)
	    {
	    	for(int j = 0; j < M.cols(); ++j)
	    	{
	    		M.row(i).set(j, weights.get(index));
	    		index++;
	        }
	    }
	    
	    Matrix Mt = M.transpose();
	    Vec nextBlame = new Vec(inputs);
	    nextBlame.fill(0.0);
	    //Step 2:nextBlame = MT * blame
	    for(int i = 0; i < nextBlame.size(); ++i)
	    {
	    	nextBlame.set(i, prevBlame.dotProduct(Mt.row(i)));
	    }
	    return nextBlame;

	}

	@Override
	void updateGradient(Vec x, Vec gradient) {
	    // gb += blame
	    Vec gb = new Vec(gradient, 0, outputs);
	    gb.add(blame);

	    // gM += blame * x
	    int position = outputs;
	    for(int i = 0; i < outputs; ++i) 
	    {
	    	Vec gM = new Vec(gradient, position, inputs);
	    	gM.addScaled( gb.get(i), x);
	    	position += inputs;
	    }
	}

	int NumWeights()
	{
		return(outputs + (inputs * outputs));
	}
	
	void initWeights(Vec weights, Random random) 
	{
	    for(int i = outputs; i < weights.size(); ++i) 
	    {
	    	weights.set(i, (Math.max(0.03,(1.0/inputs)) * random.nextGaussian()));
	    }
	}

}
