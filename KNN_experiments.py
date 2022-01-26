import random
import subprocess

from KNN import KNNClassifier
from utils import *

target_attribute = 'Outcome'


def run_knn(k, x_train, y_train, x_test, y_test, formatted_print=True):
    neigh = KNNClassifier(k=k)
    neigh.train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    acc = accuracy(y_test, y_pred)
    print(f'{acc * 100:.2f}%' if formatted_print else acc)


def get_top_b_features(x, y, b=5, k=51):
    """
    :param k: Number of nearest neighbors.
    :param x: array-like of shape (n_samples, n_features).
    :param y: array-like of shape (n_samples,).
    :param b: number of features to be selected.
    :return: indices of top 'b' features as the result of selection/dimensionality reduction on sample
            sets using sklearn.feature_selection module
    """
    # TODO: Implement get_top_b_features function
    #   - Note: The brute force approach which examines all subsets of size `b` will not be accepted.

    assert 0 < b < x.shape[1], f'm should be 0 < b <= n_features = {x.shape[1]}; got b={b}.'
    top_b_features_indices = []

    # ====== YOUR CODE: ======
    num_of_folds = 5
    features_remain = [i for i in range(x.shape[1])]
    best_acc = 0.0
    model = KNNClassifier(k=k)
    kf = KFold(n_splits=num_of_folds, shuffle=True, random_state=ID)
    while len(features_remain) > 0:
        tot_features_acc = []
        for index in features_remain:
            top_b_features_indices.append(index)
            curr_features_acc = []
            for train, test in kf.split(x):
                x_train_fold = np.copy(x[train])
                y_train_fold = np.copy(y[train])
                x_test_fold = np.copy(x[test])
                y_test_fold = np.copy(y[test])
                x_train_feats = x_train_fold[:, top_b_features_indices]
                x_test_feats = x_test_fold[:, top_b_features_indices]
                model.train(x_train=x_train_feats, y_train=y_train_fold)
                y_pred = model.predict(x_test=x_test_feats)
                acc = accuracy(y_test_fold, y_pred)
                curr_features_acc.append(acc)
            avg_feats_acc = np.mean(np.array(curr_features_acc))
            tot_features_acc.append(avg_feats_acc)
            top_b_features_indices.pop()
        best_next_feature = np.argmax(np.array(tot_features_acc))
        curr_best_acc = tot_features_acc[best_next_feature]
        if curr_best_acc > best_acc:
            best_acc = curr_best_acc
        else:
            break
        top_b_features_indices.append(features_remain[best_next_feature])
        features_remain.pop(best_next_feature)
        top_b_features_indices.sort()
    # ========================
    return top_b_features_indices


def run_cross_validation():
    """
    cross validation experiment, k_choices = [1, 5, 11, 21, 31, 51, 131, 201]
    """
    file_path = str(pathlib.Path(__file__).parent.absolute().joinpath("KNN_CV.pyc"))
    subprocess.run(['python', file_path])


def exp_print(to_print):
    print(to_print + ' ' * (30 - len(to_print)), end='')


# ========================================================================
if __name__ == '__main__':
    """
       Usages helper:
       (*) cross validation experiment
            To run the cross validation experiment over the K,Threshold hyper-parameters
            uncomment below code and run it
    """
    run_cross_validation()

    # # ========================================================================

    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')

    best_k = 51
    b = 2

    # # ========================================================================

    print("-" * 10 + f'k  = {best_k}' + "-" * 10)
    exp_print('KNN in raw data: ')
    run_knn(best_k, x_train, y_train, x_test, y_test)

    top_m = get_top_b_features(x_train, y_train, b=b, k=best_k)
    x_train_new = x_train[:, top_m]
    x_test_test = x_test[:, top_m]
    exp_print(f'KNN in selected feature data: ')
    run_knn(best_k, x_train_new, y_train, x_test_test, y_test)
