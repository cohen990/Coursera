function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


sizeX = size(X);
fprintf('size(X): %d %d\n', sizeX(1), sizeX(2)); 
sizeTheta = size(theta);
fprintf('size(theta): %d %d\n', sizeTheta(1), sizeTheta(2)); 
fprintf('alpha: %d\n', alpha);
    
for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    fprintf('alpha: %d\n', alpha);
    fprintf('before step theta: %d %d\n', theta(1), theta(2));


    firstThing = (X*theta)-y;
    THING = firstThing'*X;
    fprintf('THING %s\n', THING);
    theta = theta - alpha/m * THING';

    fprintf('after step theta: %d %d\n', theta(1), theta(2));



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    

    
    fprintf('J: %d\n', J_history(iter));
end

end
