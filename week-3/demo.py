import numpy as np


def create_training_data():
    data = [[' Sunny ', ' Hot ', ' High ', ' Weak ', ' no '],
            [' Sunny ', ' Hot ', ' High ', ' Strong ', ' no '],
            [' Overcast ', ' Hot ', ' High ', ' Weak ', ' yes '],
            [' Rain ', ' Mild ', ' High ', ' Weak ', ' yes '],
            [' Rain ', ' Cool ', ' Normal ', ' Weak ', ' yes '],
            [' Rain ', ' Cool ', ' Normal ', ' Strong ', ' no '],
            [' Overcast ', ' Cool ', ' Normal ', ' Strong ', ' yes '],
            [' Overcast ', ' Mild ', ' High ', ' Weak ', ' no '],
            [' Sunny ', ' Cool ', ' Normal ', ' Weak ', ' yes '],
            [' Rain ', ' Mild ', ' Normal ', ' Weak ', ' yes ']]
    return np.array(data)


train_data = create_training_data()
print(train_data)


def compute_prior_probability(data):
    y_unique = [' no ', ' yes ']
    prior_probability = np.zeros(len(y_unique))
    final_column = list(data[:, -1])
    for i in range(len(y_unique)):
        prior_probability[i] = final_column.count(
            y_unique[i]) / len(final_column)
    return prior_probability


print("################14################")
prior_probablity = compute_prior_probability(train_data)
print("P(play tennis = 'No')", prior_probablity[0])
print("P(play tennis = 'Yes')", prior_probablity[1])


def get_index_from_value(feature_name, list_features):
    return np.where(list_features == feature_name)[0][0]


def compute_conditional_probability(data):
    y_unique = [' no ', ' yes ']
    conditional_probability = []
    list_x_name = []
    for i in range(0, data.shape[1] - 1):
        x_unique = np.unique(data[:, i])
        list_x_name.append(x_unique)
        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        for j in range(len(x_unique)):
            label_indicies = {k for k in range(
                len(data[:, -1])) if data[k, -1] == ' no '}
            target_indices = {k for k in range(
                len(data[:, i])) if data[k, i] == x_unique[j] and k in label_indicies}
            x_conditional_probability[0, j] = target_indices.intersection(
                label_indicies).__len__() / label_indicies.__len__()

        for j in range(len(x_unique)):
            label_indicies = {k for k in range(
                len(data[:, -1])) if data[k, -1] == ' yes '}
            target_indices = {k for k in range(
                len(data[:, i])) if data[k, i] == x_unique[j] and k in label_indicies}
            x_conditional_probability[1, j] = target_indices.intersection(
                label_indicies).__len__() / label_indicies.__len__()

        conditional_probability.append(x_conditional_probability)

    return conditional_probability, list_x_name


print("################15################")
_, list_x_name = compute_conditional_probability(train_data)
print('x1 = ', list_x_name[0])
print('x2 = ', list_x_name[1])
print('x3 = ', list_x_name[2])
print('x4 = ', list_x_name[3])


print("################16################")
outlook = list_x_name[0]
i1 = get_index_from_value(' Overcast ', outlook)
i2 = get_index_from_value(' Rain ', outlook)
i3 = get_index_from_value(' Sunny ', outlook)
print(i1, i2, i3)


print("################17################")
conditional_probailities, list_x_name = compute_conditional_probability(
    train_data)
# Compute P (" Outlook "=" Sunny "| Play Tennis "=" Yes ")
x1 = get_index_from_value(' Sunny ', list_x_name[0])
print("P(Outlook = Sunny | Play Tennis = Yes) = ",
      np.round(conditional_probailities[0][1, x1], 2))


print("################18################")
# Compute P (" Outlook "=" Sunny "| Play Tennis "=" No ")
x1 = get_index_from_value(' Sunny ', list_x_name[0])
print("P(Outlook = Sunny | Play Tennis = No) = ",
      np.round(conditional_probailities[0][0, x1], 2))


def train_naive_bayes(train_data):
    prior_probablity = compute_prior_probability(train_data)
    conditional_probailities, list_x_name = compute_conditional_probability(
        train_data)
    return prior_probablity, conditional_probailities, list_x_name


def prediction_play_tennis(X, list_x_name, prior_probablity, conditional_probailities):
    x1 = get_index_from_value(X[0], list_x_name[0])
    x2 = get_index_from_value(X[1], list_x_name[1])
    x3 = get_index_from_value(X[2], list_x_name[2])
    x4 = get_index_from_value(X[3], list_x_name[3])

    p0 = 0
    p1 = 0
    for i in range(len(prior_probablity)):
        p0 += prior_probablity[0] * conditional_probailities[i][0, x1] * conditional_probailities[i][0,
                                                                                                     x2] * conditional_probailities[i][0, x3] * conditional_probailities[i][0, x4]
        p1 += prior_probablity[1] * conditional_probailities[i][1, x1] * conditional_probailities[i][1,
                                                                                                     x2] * conditional_probailities[i][1, x3] * conditional_probailities[i][1, x4]
    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


X = [' Sunny ', ' Cool ', ' High ', ' Strong ']
prior_probablity, conditional_probailities, list_x_name = train_naive_bayes(
    train_data)
pred = prediction_play_tennis(
    X, list_x_name, prior_probablity, conditional_probailities)


print("################19################")
if (pred):
    print("Ad should go!")
else:
    print("Ad should not go!")
