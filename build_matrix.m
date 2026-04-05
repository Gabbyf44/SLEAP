%% ========================================================
%  BUILD_MATRIX.M
%  Loads all experiments and builds the feature matrix.
%  Run this once (or when you add new experiments/features).
%  Saves: feature_matrix.mat
%% ========================================================

root_dir = 'path/to/your/data';  % ← CHANGE THIS

%% --------------------------------------------------------
%  Load all experiments
%% --------------------------------------------------------

folders = dir(root_dir);
folders = folders([folders.isdir] & ~startsWith({folders.name}, '.'));

all_features  = [];   % (total_frames × n_features)
all_labels    = [];   % (total_frames × 1)  0=isolated, 1=familiar
feature_names = {};

for f = 1:numel(folders)
    folder_name = folders(f).name;
    folder_path = fullfile(root_dir, folder_name);
    perframe_path = fullfile(folder_path, 'perframe');

    % Label from folder name
    if contains(folder_name, 'Familiar', 'IgnoreCase', true)
        label = 1;
    elseif contains(folder_name, 'Isolated', 'IgnoreCase', true)
        label = 0;
    else
        continue
    end

    % Find all .mat feature files
    mat_files = dir(fullfile(perframe_path, '*.mat'));
    if isempty(mat_files)
        fprintf('Skipping (no .mat files): %s\n', folder_name);
        continue
    end

    % Save feature names once from first valid folder
    if isempty(feature_names)
        feature_names = cellfun(@(x) strrep(x, '.mat', ''), ...
                        {mat_files.name}, 'UniformOutput', false);
        n_features = numel(feature_names);
        fprintf('Found %d features: %s\n\n', n_features, strjoin(feature_names, ', '));
    end

    % Load all feature files
    feature_cell = cell(1, n_features);
    for i = 1:n_features
        s = load(fullfile(perframe_path, mat_files(i).name));
        feature_cell{i} = s.data;   % {1 x n_flies}
    end

    n_flies = numel(feature_cell{1});

    % Stack all frames from all flies in this experiment
    for fly = 1:n_flies
        n_frames = numel(feature_cell{1}{fly});

        fly_matrix = NaN(n_frames, n_features);
        for i = 1:n_features
            fly_matrix(:, i) = feature_cell{i}{fly}(:);
        end

        all_features = [all_features; fly_matrix];
        all_labels   = [all_labels;   repmat(label, n_frames, 1)];
    end

    fprintf('Loaded: %s  (%d flies, label=%d)\n', folder_name, n_flies, label);
end

%% --------------------------------------------------------
%  Clean
%% --------------------------------------------------------

% Remove frames where more than half features are NaN
valid_rows = sum(isnan(all_features), 2) < n_features * 0.5;
all_features = all_features(valid_rows, :);
all_labels   = all_labels(valid_rows);

% Fill remaining NaNs with column mean
for i = 1:n_features
    col = all_features(:, i);
    col(isnan(col)) = nanmean(col);
    all_features(:, i) = col;
end

%% --------------------------------------------------------
%  Save
%% --------------------------------------------------------

save_path = fullfile(root_dir, 'feature_matrix.mat');
save(save_path, 'all_features', 'all_labels', 'feature_names');

fprintf('\n✓ Matrix: %d frames × %d features\n', size(all_features));
fprintf('✓ %d familiar frames | %d isolated frames\n', ...
    sum(all_labels==1), sum(all_labels==0));
fprintf('✓ Saved to: %s\n', save_path);
