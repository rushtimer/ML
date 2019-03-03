// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
abstract class SupervisedLearner 
{
	/// Return the name of this learner
	abstract String name();

	/// Train this supervised learner
	abstract void train(Matrix features, Matrix labels);

	/// Make a prediction
	abstract Vec predict(Vec in);
	/*
	/// Measures the misclassifications with the provided test data
	int countMisclassifications(Matrix features, Matrix labels)
	{
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		int mis = 0;
		for(int i = 0; i < features.rows(); i++)
		{
			Vec feat = features.row(i);
			Vec pred = predict(feat);
			Vec lab = labels.row(i);
			for(int j = 0; j < lab.size(); j++)
			{
				if(pred.get(j) != lab.get(j))
					mis++;
			}
		}
		return mis;
	}
	*/
	/// Measures the misclassifications with the provided test data
	int countMisclassifications(Matrix features, Matrix labels) {
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		int mis = 0;
		for(int i = 0; i < features.rows(); i++) {
			Vec feat = features.row(i);
			Vec pred = predict(feat);
			
			int num = (int)labels.getvalue(i, 0);
			Vec lab = convert(num);
				
			if(justify(pred, lab))
				mis++;
		}
		return mis;
	}


	
	Vec convert(int label) 
	{
		double[] value = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		value[label] = 1;
		return new Vec(value);
	}
	
	boolean justify(Vec pred, Vec lab) {
		pred.oneHot();
		for(int i = 0; i < pred.size(); ++i) 
		{
			if(pred.get(i) != lab.get(i))
				return true;
		}
		return false;
	}

	// Measures sum-squared-error
	double sumSquaredError(Matrix features, Matrix labels)
	{
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		double sse = 0;
		for(int i = 0; i < features.rows(); i++)
		{
			Vec feat = features.row(i);
			Vec pred = predict(feat);
			Vec lab = labels.row(i);
			for(int j = 0; j < feat.size(); j++)
			{
				sse = sse + (feat.get(j) - pred.get(j))*(feat.get(j) - pred.get(j));
			}
		}
		return sse;
	}
	// performs 5 repetitions of 10-fold cross-validation
	// performs m-repetitions of n-fold cross-validation.
	
	void crossValidation(int iterations,int fold, Matrix features, Matrix labels)
	{
		Random rand = new Random();
		double[][] myArray = new double[506][14];
		// detect 
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		// number of each Matrix size:
		int num = features.rows()/fold; //50
		int lastNum = features.rows() - (fold-1)*num;// lastNum = 56
		// merge two arff files together
		Matrix bigfile = new Matrix(features.rows(),features.cols()+labels.cols());
		bigfile.copyBlock(0, 0, features, 0, 0, features.rows(), features.cols());
		bigfile.copyBlock(0, 13, labels, 0, 0, labels.rows(), labels.cols());
		//bigfile.print();
		// convert Matrix bigfile to a Vecter[]
		Vec[] bigArray = new Vec [bigfile.rows()];
		for(int i = 0; i < bigfile.rows();i++)
		{
			bigArray[i] = new Vec(bigfile.cols());
		} 
		for(int i = 0; i< bigfile.rows();i++)
		{
			for(int j = 0; j < bigfile.cols();j++)
			{
				bigArray[i].set(j, bigfile.getvalue(i, j));
			}
		}
		shuffleArray(bigArray);
		// create new shuffled bigfile
		Matrix shuffledBigfile = new Matrix(features.rows(),features.cols()+labels.cols());
		shuffledBigfile = shuffledBigfile.arrayVecToMatrix(bigArray, bigfile);
		// create new shuffled features and labels
		Matrix shuffledfeatures = new Matrix(features.rows(),features.cols());
		Matrix shuffledlabels = new Matrix(labels.rows(),labels.cols());
		shuffledfeatures.copyBlock(0, 0, shuffledBigfile, 0, 0, features.rows(), features.cols());
		shuffledlabels.copyBlock(0, 0, shuffledBigfile, 0, 13, labels.rows(), labels.cols());
		
		// Initialize: divide Matrix features and labels into 10 parts
		Matrix [] MatrixFeatures = new Matrix[fold]; // store houseFeatures into Matrix array which has size of 10
		Matrix [] MatrixLabels = new Matrix [fold]; // store houseLabels into Matrix array which has size of 10
		
		for(int i = 0; i < fold-1;i++) // 0-9, 10 times
		{
			MatrixFeatures[i] = new Matrix(num,shuffledfeatures.cols());// num=50
			MatrixLabels  [i] = new Matrix(num,shuffledlabels.cols()); 
		}
		MatrixFeatures[fold-1] = new Matrix(lastNum,shuffledfeatures.cols());//56
		MatrixLabels  [fold-1] = new Matrix(lastNum,shuffledlabels.cols());
		
        for(int i = 0; i < fold-2;i++)//fold-1?
        {
        	MatrixFeatures[i].copyBlock(0, 0, shuffledfeatures, 0+num*i, 0, num, shuffledfeatures.cols());
        	MatrixLabels[i].copyBlock(0, 0, shuffledlabels, 0+num*i, 0, num, shuffledlabels.cols());
        }
        // last one
        MatrixFeatures[fold-1].copyBlock(0, 0, shuffledfeatures, shuffledfeatures.rows()-(fold-1)*num, 0, lastNum, shuffledfeatures.cols());
    	MatrixLabels[fold-1].copyBlock(0, 0, shuffledlabels, shuffledlabels.rows() - (fold-1)*num, 0, lastNum, shuffledlabels.cols());
    	// Now housing_features and housing_labels are seperated into MatrixFeatures[] and MatrixLabels[], each of them has 10 parts
    	// Then we need to generate training data set and testing data set
    	// Initialize 
    	Matrix train_features[] = new Matrix[fold];
    	Matrix train_labels[] = new Matrix [fold];
    	Matrix test_features[] = new Matrix[fold];
    	Matrix test_labels[] = new Matrix[fold];
    	
    	for(int i = 0; i <fold;i++)
    	{
    		train_features[i] = new Matrix(shuffledfeatures.rows()- MatrixFeatures[fold-1].rows(),shuffledfeatures.cols());//450 x 13	
    		train_labels  [i] = new Matrix(shuffledlabels.rows()- MatrixFeatures[fold-1].rows(),  shuffledlabels.cols());//450 x 1
    		test_features [i] = new Matrix(num, shuffledfeatures.cols()); // 50 x 13
    		test_labels   [i] = new Matrix(num,shuffledlabels.cols());// 50 x 1
    		
    	}
    	
    	
    	
    	for(int i = 0; i < iterations;i++)
    	{
    		int pickNum = rand.nextInt(10);
    		// training data
    		for(int j = 0; j < fold -1; j++)
    		{
    			train_features[i].copyBlock(0+num*j, 0, MatrixFeatures[j], 0, 0, num, MatrixFeatures[j].cols());
        		train_labels[i].copyBlock(0+num*j, 0, MatrixLabels[j], 0, 0, num, MatrixLabels[j].cols());
    		}
    		// testing data
        	test_features[i].copyBlock(0, 0, MatrixFeatures[pickNum], 0, 0, num, MatrixFeatures[pickNum].cols());
        	test_labels[i].copyBlock(0, 0, MatrixLabels[pickNum], 0, 0, num, MatrixLabels[pickNum].cols());
        	train_features[i].saveARFF((i+1)+ "_train_features.arff");
        	train_labels[i].saveARFF((i+1)+ "_train_labels.arff");
        	test_features[i].saveARFF((i+1)+ "_test_features.arff");
        	test_labels[i].saveARFF((i+1)+ "_test_labels.arff");
    	}
    	
	}
	
	public static void shuffleArray(Vec[] bigArray) {
        int n = bigArray.length;
        Random r = new Random();
        r.nextInt();
        for (int i = 0; i < n; i++) {
            int change = i + r.nextInt(n - i);
            swap(bigArray, i, change);
        }
    }
	 private static void swap(Vec[] a, int i, int change) {
		 Vec helper = a[i];
	        a[i] = a[change];
	        a[change] = helper;
	  }
	 


}
