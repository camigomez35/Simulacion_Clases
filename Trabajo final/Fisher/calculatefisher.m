function fishers = calculatefisher(dataForClasses)
    classes = size(dataForClasses, 2);
    featuresAmount = size(dataForClasses{1, 1}, 2);

    values = zeros(classes, featuresAmount);
    
    means = mean(dataForClasses{1, 1});
    stds = std(dataForClasses{1, 1});
    
    for index = 2:classes
       means = [means; mean(dataForClasses{1, index})];
       stds = [stds; std(dataForClasses{1, index})];
    end

    for i = 1:featuresAmount
        sums = 0;
        for j = 1:classes
            for k = 1:classes
                if (j ~= k)                  
                    sums = sums + (power((means(j, i) - means(k, i)), 2.0) /...
                        (power(stds(j, i), 2.0) + power(stds(k, i), 2.0)));
                end
           end

        values(j, i) = sums;
        end
    end

    fishers = sum(values);
    fishers = fishers / max(fishers);