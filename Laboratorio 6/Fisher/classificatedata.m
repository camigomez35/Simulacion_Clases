function value = classificatedata(X, Y)
    classes = max(Y);
    for i = 0:classes
       indexes = (Y == i);
       value{i+1} = X(indexes, :);
    end