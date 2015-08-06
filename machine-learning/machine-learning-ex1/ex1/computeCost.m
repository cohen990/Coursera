function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

fprintf("\nsize(X) = %d %d\n", size(X)(1), size(X)(2))
fprintf("\nsize(theta) = %d %d\n", size(theta)(1), size(theta)(2))

J = 1/(2 * m) * sum(X*theta - y)

% =========================================================================

end
