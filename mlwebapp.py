import numpy as np
# import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from matplotlib.colors import ListedColormap
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# * Intializations
st.title("ML with Streamlit")

st.write("""
# explore models with Streamlit
 - which model to use?
 - which model gives best results?
 - which model represents data better?
""")


dataset_name = st.sidebar.selectbox(
    "Select a dataset", ["Iris", "Breast Cancer", "Wine"])
model_name = st.sidebar.selectbox("Select a model", [
    "Logistic Regression", "Decision Tree", "Random Forest", "Linear SVC", "KNN"])

# * identifying the Datasets


def get_data(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()

    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()

    elif dataset_name == "Wine":
        data = datasets.load_wine()

    x = data.data
    y = data.target
    features = data.feature_names
    return x, y, features

# * identifying the models


def Identify_model(model_name):
    params = dict()
    if model_name == "KNN":
        k = st.sidebar.slider("K", 1, 15)
        params["k"] = k
        model = KNeighborsClassifier(n_neighbors=params["k"])

    elif model_name == "Logistic Regression":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
        model = LogisticRegression(C=params["C"])

    elif model_name == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20)
        params["max_depth"] = max_depth
        model = DecisionTreeClassifier(max_depth=params["max_depth"])

    elif model_name == "Random Forest":
        n_estimators = st.sidebar.slider("N Estimators", 1, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, 20)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"])

    elif model_name == "Linear SVC":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
        model = SVC(C=params["C"])

    return model


# * Training the model
x, y, features = get_data(dataset_name)
features_k1 = features
features_k2 = features
st.write("shape dataset: ", x.shape)
st.write("number of classes: ", len(np.unique(y)))

# # Feature_1 = st.selectbox(r"choose features one",
# #                          (features_k1), key=1)


# # Feature_2 = st.selectbox(r"choose features one",
# #                          (features_k2), key=2)

# # indicated_list = [Feature_1, Feature_2]
# # st.write(indicated_list)
# # st.write(type(indicated_list))

# Feature_2 = st.selectbox(r"choose feature two",
#                           (features_2))
# # * reducing the dimension to 2d
# x = pd.DataFrame(x, columns=features)
# # st.write(x[[Feature_1,Feature_2]].head())
# # st.write(x.loc[2].head())


# x_projected = x[[Feature_1, Feature_2]]
x_projected = PCA(2).fit_transform(x)


model = Identify_model(model_name)
xtrain, xtest, ytrain, ytest = train_test_split(
    x_projected, y, train_size=0.8, random_state=0)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)

acc = accuracy_score(ytest, y_pred)

st.write(f"model is : {model_name}")
st.write(f"Score is : {acc}")

fig1 = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Componant 1")
plt.ylabel("Principal Componant 2")

plt.colorbar()
st.pyplot(fig1)
st.set_option('deprecation.showPyplotGlobalUse', False)


# * ploting the descion boundaries
# * plotting classified data

#########################################################################################################
#########################################################################################################
# import warnings

# def plot_decision_boundary2D(clf, X: np.ndarray, y: np.ndarray, classes: list, colormap: np.ndarray,
#                              step: float = 0.1, prob_dot_scale: int = 40, prob_dot_scale_power: int = 3,
#                              true_dot_size: int = 50, pad: float = 1.0,
#                              prob_values: list = [0.4, 0.6, 0.8, 1.0]) -> None:

#     # Handling X data dimension issues. If X doesn't have enough dimensions, throw error. Too many, use first two dimensions.
#     # X_dim = X.shape[1]
#     # if X_dim < 2:
#     #     raise Exception("Error: Not enough dimensions in input data. Data must be at least 2-dimensional.")
#     # elif X_dim > 2:
#     #     warnings.warn(f"Warning: input data was {X_dim} dimensional. Expected 2. Using first 2 dimensions provided.")

#     # Change colormap to a numpy array if it isn't already (necessary to prevent scalar error)
#     # if not isinstance(colormap, np.ndarray):
#     #     colormap = np.array(colormap)

#     numClasses = np.amax(y) + 1
#     #['light yellow' , 'white ' ,'','']
#     color_list_light = ['#c4caff', '#c4ffff', '#AAFFAA', '#AAAAFF']
#     color_list_bold = ['#c44bff', '#4fc9ff', '#00CC00', '#0000CC']
#     cmap_light = ListedColormap(color_list_light[0:numClasses])
#     cmap_bold = ListedColormap(color_list_bold[0:numClasses])
#     # create the x0, x1 feature. This is only a 2D plot after all.
#     x0 = X[X.columns[0]]
#     x1 = X[X.columns[1]]

#     # create 1D arrays representing the range of probability data points
#     x0_min, x0_max = np.round(x0.min())-pad, np.round(x0.max()+pad)
#     x1_min, x1_max = np.round(x1.min())-pad, np.round(x1.max()+pad)
#     x0_axis_range = np.arange(x0_min,x0_max, step)
#     x1_axis_range = np.arange(x1_min,x1_max, step)

#     # create meshgrid between the two axis ranges
#     xx0, xx1 = np.meshgrid(x0_axis_range, x1_axis_range)

#     # put the xx in the same dimensional format as the original X
#     xx = np.reshape(np.stack((xx0.ravel(),xx1.ravel()),axis=1),(-1,2))

#     yy_hat = clf.predict(xx) # prediction of all the little dots
#     yy_prob = clf.predict_proba(xx) # probability of each dot being
#                                     # the predicted color
#     yy_size = np.max(yy_prob, axis=1)

#     # make figure
#     plt.style.use('seaborn-whitegrid') # set style because it looks nice
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6), dpi=150)

#     # plot all the little dots
#     ax.scatter(xx[:,0], xx[:,1], c=colormap[yy_hat], alpha=0.4, s=prob_dot_scale*yy_size**prob_dot_scale_power, linewidths=0,)

#     # plot the contours
#     ax.contour(x0_axis_range, x1_axis_range,
#                np.reshape(yy_hat,(xx0.shape[0],-1)),
#                levels=3, linewidths=1,
#                colors=[colormap[0],colormap[1], colormap[1], colormap[2],])

#     # plot the original x values.
#     ax.scatter(x0, x1, c=colormap[y], s=true_dot_size, zorder=3, linewidths=0.7, edgecolor='k')

#     # create legends - Not sure if these serve a purpose but I left them in just in case
#     x_min, x_max = ax.get_xlim()
#     y_min, y_max = ax.get_ylim()

#     ax.set_ylabel(r"$x_1$")
#     ax.set_xlabel(r"$x_0$")

#     # set the aspect ratio to 1, for looks
#     ax.set_aspect(1)

#     # create class legend
#     legend_class = []
#     for class_id, color in zip(classes, colormap):
#         legend_class.append(Line2D([0], [0], marker='o', label=class_id,ls='None',
#                                    markerfacecolor=color, markersize=np.sqrt(true_dot_size),
#                                    markeredgecolor='k', markeredgewidth=0.7))

#     # iterate over each of the probabilities to create prob legend
#     legend_prob = []
#     for prob in prob_values:
#         legend_prob.append(Line2D([0], [0], marker='o', label=prob, ls='None', alpha=0.8,
#                                   markerfacecolor='grey',
#                                   markersize=np.sqrt(prob_dot_scale*prob**prob_dot_scale_power),
#                                   markeredgecolor='k', markeredgewidth=0))


#     legend1 = ax.legend(handles=legend_class, loc='center',
#                         bbox_to_anchor=(1.05, 0.35),
#                         frameon=False, title='class')

#     legend2 = ax.legend(handles=legend_prob, loc='center',
#                         bbox_to_anchor=(1.05, 0.65),
#                         frameon=False, title='prob', )

#     ax.add_artist(legend1) # add legend back after it disappears

#     ax.set_yticks(np.arange(x1_min,x1_max, 1)) # I don't like the decimals
#     ax.grid(False) # remove gridlines (inherited from 'seaborn-whitegrid' style)

#     # only use integers for axis tick labels
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))

#     # remove first ticks from axis labels, for looks
#     ax.set_xticks(ax.get_xticks()[1:-1])
#     ax.set_yticks(np.arange(x1_min,x1_max, 1)[1:])

#     plt.show()

#####################################################################################################################
#####################################################################################################################

# def plot_class_regions_for_classifier(clf, X, y, title=None, plot_decision_regions=True, st=st):

#     numClasses = np.amax(y) + 1
#     #['light yellow' , 'white ' ,'','']
#     color_list_light = ['#c4caff', '#c4ffff', '#AAFFAA', '#AAAAFF']
#     color_list_bold = ['#c44bff', '#4fc9ff', '#00CC00', '#0000CC']
#     cmap_light = ListedColormap(color_list_light[0:numClasses])
#     cmap_bold = ListedColormap(color_list_bold[0:numClasses])

#     h = 0.05
#     k = 0.5
#     x_plot_adjust = 0.1
#     y_plot_adjust = 0.1
#     plot_symbol_size = 50

#     x_min = X[X.columns[0]].min()
#     x_max = X[X.columns[0]].max()
#     y_min = X[X.columns[1]].min()
#     y_max = X[X.columns[1]].max()
#     x2, y2 = np.meshgrid(np.arange(x_min-k, x_max+k, h),
#                          np.arange(y_min-k, y_max+k, h))
#     P = clf.predict(np.c_[x2.ravel(), y2.ravel()])
#     P = P.reshape(x2.shape)
#     fig = plt.figure()

#     if plot_decision_regions:
#         plt.contourf(x2, y2, P, cmap=cmap_light, alpha=0.8)

#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
#                 s=plot_symbol_size, edgecolor='black')
#     plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
#     plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)

#     if (title is not None):
#         plt.title(title)
#     # plt.show()
#     st.pyplot(fig)


# plot_class_regions_for_classifier(model, x_projected, y)
