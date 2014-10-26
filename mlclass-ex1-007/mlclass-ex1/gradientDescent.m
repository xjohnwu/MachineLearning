function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

%    delta1 = 0;
%    for i = 1:m
%      delta1 += (theta(1) + theta(2) * X(i,2) - y(i));
%    end 
%    
%    delta2 = 0;
%    for i = 1:m
%      delta2 += (theta(1) + theta(2) * X(i,2) - y(i))*X(i,2);
%    end 
%    
%    temp1 = theta(1)-alpha * delta1 / m;
%    temp2 = theta(2)-alpha * delta2 / m;
%    theta = [temp1; temp2];

    theta = theta - alpha/m * X' * (X * theta - y);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
