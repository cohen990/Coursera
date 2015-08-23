function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


theta1 = theta(1);
x1 = X(:,1);
thetaToEnd = theta(2:end);
xToEnd = X(:,2:end);

sigmoided = sigmoid(X*theta);
positiveTerm = y.*log(sigmoided);
negativeTerm = (1-y).*log(1-sigmoided);
lambdaTerm = lambda/(2*m)*sum(thetaToEnd.^2);
J = 1/m* sum(-positiveTerm - negativeTerm) + lambdaTerm;

grad(1) = 1/m * (sigmoided - y)'*x1;

lambdaTerm = (lambda/m)*thetaToEnd;
grad(2:end) = (1/m * (sigmoided - y)'*xToEnd)' + lambdaTerm;


% =============================================================

end
