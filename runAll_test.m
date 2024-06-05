% find all file png files. Your png images and mat files must be in the
% 'images' sub directory.
D=dir('/MATLAB Drive/New Folder/images/*.png');

score = [];

%load and process each file in turn.
for ind=1:length(D)

    %name of png file
    filename = fullfile(D(ind).folder,D(ind).name);

    %name of answer file .mat

    [folder, baseFileName, ~] = fileparts(filename);
    mat_filename = fullfile(folder, sprintf('%s.mat',baseFileName));
  
    %test result
    
    %call the actual findColours function - this is the function that the 
    % student needs to write
    %res = findColours(filename);
     
    
    %call the actual findColours function - this is the function that the 
    % student needs to write
    try
        res = Assignment_findColours(filename);
    catch ME
    % display the error message
    disp(['Error occurred: ', ME.message]);
    end
    % check the answers.
    mm = check_answer(res,mat_filename);

    score=[score,mm];

end
%print out the score.
str=repmat('%.2f ', 1, length(score));
fprintf('Score is: ');
fprintf(str,score);
fprintf('\nMean score %f\n',mean(score));   

