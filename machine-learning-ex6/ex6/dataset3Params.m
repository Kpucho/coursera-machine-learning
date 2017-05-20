function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
best = 1;
bestC = -1;
betsS = -1;

for i = 0:5
  curS = sigma * 2 ^ i;
  
  for j = 0:5
    
    curC = C * 2 ^ j;
    
    model = svmTrain( X, y, curC, @(x1,x2) gaussianKernel(x1,x2, curS ) );
    predictions = svmPredict(model, Xval);
    newBest = mean(double(predictions ~= yval));
    
    if (best > newBest)
      best = newBest;
      bestC = curC;
      bestS = curS;
    end
    
   end
end
      
C = bestC;
sigma = bestS;





% =========================================================================

end
