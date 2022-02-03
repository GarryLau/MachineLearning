#if 1
#include "auxiliary.hpp"

#define USE_LESS_EIGEN               // 打开该宏尽可能的少用eigen库,更接近PCA的推导过程的实现
#define FEAT_REMAIN_RATIO  0.5       // 特征维度保留比例
#define EQUALIZATION_WITH_SKLEARN    // 使计算结果与sklearn一致

int main()
{
    /* Step0,读取鸢尾花数据  */
    std::vector<std::vector<double> > iris;
    int ret = GetData(iris);
    if (0 != ret)
    {
        std::cout << "GetData() failed." << std::endl;
        system("pause");
        return ret;
    }

    /* Step1,读取鸢尾花数据的标签  */
    std::vector<int> labels;
    ret = GetLabels(labels);
    if (0 != ret)
    {
        std::cout << "GetData() failed." << std::endl;
        system("pause");
        return ret;
    }

    /* Step2,将鸢尾花数据中心化  */
#ifdef USE_LESS_EIGEN
    // 计算原始数据各维度均值
    std::vector<double> mean(iris[0].size());  // 特征各维度均值
    std::vector<double> sum(iris[0].size());
    for (auto sample : iris)
    {
        for (int i = 0; i < sample.size(); ++i)
        {
            sum[i] += sample[i];
        }
    }
    for (int i = 0; i < sum.size(); ++i)
    {
        mean[i] = sum[i] / iris.size();
    }

    // 计算中心化后的数据
    Eigen::MatrixXd m(iris.size(), iris[0].size());  // 存储中心化后的数据
    for (int i = 0; i < iris.size(); ++i)
    {
        for (int j = 0; j < iris[0].size(); ++j)
        {
            m(i, j) = iris[i][j] - mean[j];
            //std::cout << m(i, j) << std::endl;
        }
    }
    Eigen::MatrixXd zero_mean_m = m;
#else
    Eigen::MatrixXd m(iris.size(), iris[0].size());  // 存储中心化后的数据
    for (int i = 0; i < iris.size(); ++i)
    {
        for (int j = 0; j < iris[0].size(); ++j)
        {
            m(i, j) = iris[i][j];
        }
    }
    Eigen::MatrixXd mean_vec = m.colwise().mean();
    Eigen::RowVectorXd mean_vec_row(Eigen::RowVectorXd::Map(mean_vec.data(), m.cols()));
    Eigen::MatrixXd zero_mean_m = m;
    zero_mean_m.rowwise() -= mean_vec_row;
#endif

    /* Step3,计算中心化后数据zero_mean_m的协方差矩阵 */
#ifdef USE_LESS_EIGEN
    Eigen::MatrixXd zero_mean_transpose = zero_mean_m.transpose();
    Eigen::MatrixXd covariance = zero_mean_transpose * zero_mean_m / (zero_mean_m.rows() - 1);
    std::cout << covariance << std::endl;
#else
    Eigen::MatrixXd covariance = (zero_mean_m.adjoint()*zero_mean_m) / double(zero_mean_m.rows() - 1);
    std::cout << covariance << std::endl;
#endif

    /* Step4,计算协方差矩阵的特征值及特征向量 */
    Eigen::EigenSolver<Eigen::MatrixXd>eigen_solver(covariance);
    if (eigen_solver.info() != Eigen::Success)
    {
        std::cout << "Solver error." << std::endl;
    }
    std::cout << "特征值：\n" << eigen_solver.eigenvalues().real() << std::endl;
    std::cout << "特征向量：\n" << eigen_solver.eigenvectors().real() << std::endl;

    /* Step5,将特征值从大到小排序,并同步排序特征向量 */
    std::vector<std::pair<double, std::vector<double>>>eigen_values_vectors;
    for (int i = 0; i < covariance.rows(); ++i)
    {
        std::pair<double, std::vector<double>> eigen_value_vector;
        eigen_value_vector.first = eigen_solver.eigenvalues().real()[i];

        for (int j = 0; j < covariance.cols(); ++j)
        {
            // 注意eigen中特征向量是按列排布,因此下面取特征向量的索引是(j, i)
            eigen_value_vector.second.push_back(eigen_solver.eigenvectors()(j, i).real()); 
        }
        eigen_values_vectors.push_back(eigen_value_vector);
    }
    struct CEigenBig {
        bool operator() (std::pair<double, std::vector<double>> i, std::pair<double, std::vector<double>> j) { return (i.first > j.first); }
    }eigen_big;
    std::sort(eigen_values_vectors.begin(), eigen_values_vectors.end(), eigen_big); // 将特征值从大到小排序,并同步排序特征向量

    int remain = eigen_values_vectors.size() * FEAT_REMAIN_RATIO;     // 计算保留的特征维度数
    Eigen::MatrixXd components(eigen_values_vectors.size(), remain);  // components是保留维度的特征向量
    for (int j = 0; j < components.cols(); ++j)
    {
        for (int i = 0; i < components.rows(); ++i)
        {
            // 该语句块可使最终降维后数据与sklearn中的完全一致,
            // 由于实现细节的差异使得特征向量的正负号有差异导致降维后数据存在正负号的差异,
            // 对于鸢尾花数据sklearn的实现与此处的实现其第二大特征值对应的特征向量存在正负号的差异,
            // 因此想要降维后的数据第二各维度的符号一致可以将第二大特征值取反
#ifdef EQUALIZATION_WITH_SKLEARN
            if (1 == j)
            {
                components(i, j) = -eigen_values_vectors[j].second[i];
            }
            else
            {
                components(i, j) = eigen_values_vectors[j].second[i];
            }
#else
            components(i, j) = eigen_values_vectors[j].second[i];
#endif
        }
    }
    std::cout << components << std::endl;

    /* Step6,将中心化的数据乘以(保留维度的)特征向量得到降维后的数据 */
    Eigen::MatrixXd  pca_m = zero_mean_m * components;
    std::cout << "降维后数据：\n" << pca_m << std::endl;

    /* Step7,保存降维后数据 */
    ret = SaveDecompositionData(pca_m);
    if (0 != ret)
    {
        std::cout << "GetData() failed." << std::endl;
        system("pause");
        return ret;
    }

    /* Step8,可视化降维后数据 */
    Visialization(pca_m, labels);

    system("pause");
    return 0;
}
#endif

#if 0
#include <iostream>
#include "Eigen\Dense"

int main()
{
    Eigen::Matrix2d m;
    m << 3, 2, 1, 4;
    std::cout << "m=\n" << m << std::endl;
    Eigen::EigenSolver<Eigen::Matrix2d>eigen_solver(m);
    if (eigen_solver.info() != Eigen::Success)
    {
        std::cout << "eigen solver failed." << std::endl;
        return -1;
    }
    std::cout << "特征值：\n" << eigen_solver.eigenvalues().real() << std::endl;
    std::cout << "特征向量：\n" << eigen_solver.eigenvectors().real() << std::endl;
}
#endif
