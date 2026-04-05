%% ========================================================
%  RUN_SVM.M
%  Loads the feature matrix and runs SVM analysis.
%  Run build_matrix.m first to generate feature_matrix.mat
%% ========================================================

root_dir = 'path/to/your/data';  % ← CHANGE THIS (same as build_matrix.m)

%% --------------------------------------------------------
%  Load matrix
%% --------------------------------------------------------

load(fullfile(root_dir, 'feature_matrix.mat'));
% → loads: all_features, all_labels, feature_names

n_features = numel(feature_names);
fprintf('Loaded: %d frames × %d features\n', size(all_features));
fprintf('Familiar: %d frames | Isolated: %d frames\n\n', ...
    sum(all_labels==1), sum(all_labels==0));

%% --------------------------------------------------------
%  Normalize
%% --------------------------------------------------------

X = normalize(all_features, 1);   % z-score per feature column
y = all_labels;

%% --------------------------------------------------------
%  Statistical test — which features differ between groups?
%% --------------------------------------------------------

p_values = NaN(1, n_features);
for i = 1:n_features
    [~, p_values(i)] = ranksum(X(y==1, i), X(y==0, i));
end

% FDR correction (controls for testing many features at once)
p_fdr   = mafdr(p_values', 'BHFDR', true)';
sig_idx = find(p_fdr < 0.05);

fprintf('--- Significant features (FDR < 0.05): %d/%d ---\n', ...
    numel(sig_idx), n_features);
for i = 1:numel(sig_idx)
    fprintf('  %-25s  p=%.2e  p_fdr=%.4f\n', ...
        feature_names{sig_idx(i)}, p_values(sig_idx(i)), p_fdr(sig_idx(i)));
end

%% --------------------------------------------------------
%  SVM with 5-fold cross validation
%% --------------------------------------------------------

cv  = cvpartition(y, 'KFold', 5, 'Stratify', true);
mdl = fitcsvm(X, y, ...
    'KernelFunction', 'linear', ...
    'Standardize',   false, ...
    'CrossVal',      'on', ...
    'CVPartition',   cv);

acc         = (1 - kfoldLoss(mdl)) * 100;
predictions = kfoldPredict(mdl);

fprintf('\n✓ SVM accuracy: %.1f%%\n', acc);
fprintf('✓ Familiar accuracy:  %.1f%%\n', mean(predictions(y==1)==1)*100);
fprintf('✓ Isolated accuracy:  %.1f%%\n', mean(predictions(y==0)==0)*100);

%% --------------------------------------------------------
%  Feature importance from SVM weights
%% --------------------------------------------------------

mdl_full = fitcsvm(X, y, 'KernelFunction', 'linear', 'Standardize', false);
weights  = abs(mdl_full.Beta);
[sorted_w, sort_idx] = sort(weights, 'descend');

fprintf('\n--- Feature importance ranking ---\n');
for i = 1:n_features
    idx        = sort_idx(i);
    sig_marker = '';
    if ismember(idx, sig_idx), sig_marker = '  *'; end
    fprintf('  %2d. %-25s  weight=%.4f%s\n', ...
        i, feature_names{idx}, sorted_w(i), sig_marker);
end
fprintf('  (* = FDR significant)\n');

%% --------------------------------------------------------
%  Plots
%% --------------------------------------------------------

% --- Plot 1: Feature importance bar chart ---
figure('Name', 'Feature Importance');
bar(sorted_w, 'FaceColor', [0.6 0.6 0.6]);
hold on;
for i = 1:numel(sorted_w)
    if ismember(sort_idx(i), sig_idx)
        bar(i, sorted_w(i), 'FaceColor', [0.85 0.2 0.2]);
    end
end
xticks(1:n_features);
xticklabels(feature_names(sort_idx));
xtickangle(45);
ylabel('|SVM weight|');
title(sprintf('Feature Importance  (accuracy = %.1f%%)', acc));
legend({'Not significant', 'FDR < 0.05'}, 'Location', 'northeast');
grid on;

% --- Plot 2: Confusion matrix ---
figure('Name', 'Confusion Matrix');
confusionchart(y, predictions, ...
    'RowSummary',    'row-normalized', ...
    'ColumnSummary', 'column-normalized');
title(sprintf('Confusion Matrix  (accuracy = %.1f%%)', acc));

% --- Plot 3: Boxplots of top 6 features ---
figure('Name', 'Top Features');
top_n   = min(6, n_features);
top_idx = sort_idx(1:top_n);
for i = 1:top_n
    subplot(2, 3, i);
    feat_idx  = top_idx(i);
    data_fam  = all_features(y==1, feat_idx);
    data_iso  = all_features(y==0, feat_idx);
    boxplot([data_fam; data_iso], [ones(size(data_fam)); zeros(size(data_iso))], ...
        'Labels', {'Familiar', 'Isolated'});
    ylabel('value');
    if ismember(feat_idx, sig_idx)
        title([feature_names{feat_idx} '  *'], 'Interpreter', 'none', 'Color', 'r');
    else
        title(feature_names{feat_idx}, 'Interpreter', 'none');
    end
end
sgtitle('Top 6 Features by SVM Weight  (* = FDR significant)');

%% --------------------------------------------------------
%  Save results
%% --------------------------------------------------------

save_path = fullfile(root_dir, 'svm_results.mat');
save(save_path, ...
    'feature_names', 'p_values', 'p_fdr', 'sig_idx', ...
    'weights', 'sort_idx', 'predictions', 'acc');

fprintf('\n✓ Results saved to: %s\n', save_path);
