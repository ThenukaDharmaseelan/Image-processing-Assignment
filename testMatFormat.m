 function testMatFormat(data)
     arguments
        data (4,4) cell
     end
     % tests that the format of the answer cell array is correct.
    cols = {'white','red','green','blue','yellow'};
     for p=1:16
         if ~matches(data{p},cols)
            warning('Fail: Unknown colour %s',data{p})
         end
     end
    fprintf('Passed\n')


% function ttestMatFormat(data)
%     arguments
%         data (4,4) cell
%     end
% 
%     % Define the set of allowed colors
%     cols = {'white','red','green','blue','yellow'};
% 
%     % Check each element of the cell array
%     for p = 1:numel(data)
%         if ~ismember(find_color(data{p}), cols)
%             warning('Fail: Unknown colour %s', data{p});
%         end
%     end
% 
%     % Indicate that the test has passed
%     fprintf('Passed\n');
% end





