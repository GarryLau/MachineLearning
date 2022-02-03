#if 1
#include "auxiliary.hpp"

#define USE_LESS_EIGEN               // �򿪸ú꾡���ܵ�����eigen��,���ӽ�PCA���Ƶ����̵�ʵ��
#define FEAT_REMAIN_RATIO  0.5       // ����ά�ȱ�������
#define EQUALIZATION_WITH_SKLEARN    // ʹ��������sklearnһ��

int main()
{
    /* Step0,��ȡ�β������  */
    std::vector<std::vector<double> > iris;
    int ret = GetData(iris);
    if (0 != ret)
    {
        std::cout << "GetData() failed." << std::endl;
        system("pause");
        return ret;
    }

    /* Step1,��ȡ�β�����ݵı�ǩ  */
    std::vector<int> labels;
    ret = GetLabels(labels);
    if (0 != ret)
    {
        std::cout << "GetData() failed." << std::endl;
        system("pause");
        return ret;
    }

    /* Step2,���β���������Ļ�  */
#ifdef USE_LESS_EIGEN
    // ����ԭʼ���ݸ�ά�Ⱦ�ֵ
    std::vector<double> mean(iris[0].size());  // ������ά�Ⱦ�ֵ
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

    // �������Ļ��������
    Eigen::MatrixXd m(iris.size(), iris[0].size());  // �洢���Ļ��������
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
    Eigen::MatrixXd m(iris.size(), iris[0].size());  // �洢���Ļ��������
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

    /* Step3,�������Ļ�������zero_mean_m��Э������� */
#ifdef USE_LESS_EIGEN
    Eigen::MatrixXd zero_mean_transpose = zero_mean_m.transpose();
    Eigen::MatrixXd covariance = zero_mean_transpose * zero_mean_m / (zero_mean_m.rows() - 1);
    std::cout << covariance << std::endl;
#else
    Eigen::MatrixXd covariance = (zero_mean_m.adjoint()*zero_mean_m) / double(zero_mean_m.rows() - 1);
    std::cout << covariance << std::endl;
#endif

    /* Step4,����Э������������ֵ���������� */
    Eigen::EigenSolver<Eigen::MatrixXd>eigen_solver(covariance);
    if (eigen_solver.info() != Eigen::Success)
    {
        std::cout << "Solver error." << std::endl;
    }
    std::cout << "����ֵ��\n" << eigen_solver.eigenvalues().real() << std::endl;
    std::cout << "����������\n" << eigen_solver.eigenvectors().real() << std::endl;

    /* Step5,������ֵ�Ӵ�С����,��ͬ�������������� */
    std::vector<std::pair<double, std::vector<double>>>eigen_values_vectors;
    for (int i = 0; i < covariance.rows(); ++i)
    {
        std::pair<double, std::vector<double>> eigen_value_vector;
        eigen_value_vector.first = eigen_solver.eigenvalues().real()[i];

        for (int j = 0; j < covariance.cols(); ++j)
        {
            // ע��eigen�����������ǰ����Ų�,�������ȡ����������������(j, i)
            eigen_value_vector.second.push_back(eigen_solver.eigenvectors()(j, i).real()); 
        }
        eigen_values_vectors.push_back(eigen_value_vector);
    }
    struct CEigenBig {
        bool operator() (std::pair<double, std::vector<double>> i, std::pair<double, std::vector<double>> j) { return (i.first > j.first); }
    }eigen_big;
    std::sort(eigen_values_vectors.begin(), eigen_values_vectors.end(), eigen_big); // ������ֵ�Ӵ�С����,��ͬ��������������

    int remain = eigen_values_vectors.size() * FEAT_REMAIN_RATIO;     // ���㱣��������ά����
    Eigen::MatrixXd components(eigen_values_vectors.size(), remain);  // components�Ǳ���ά�ȵ���������
    for (int j = 0; j < components.cols(); ++j)
    {
        for (int i = 0; i < components.rows(); ++i)
        {
            // �������ʹ���ս�ά��������sklearn�е���ȫһ��,
            // ����ʵ��ϸ�ڵĲ���ʹ�������������������в��쵼�½�ά�����ݴ��������ŵĲ���,
            // �����β������sklearn��ʵ����˴���ʵ����ڶ�������ֵ��Ӧ�������������������ŵĲ���,
            // �����Ҫ��ά������ݵڶ���ά�ȵķ���һ�¿��Խ��ڶ�������ֵȡ��
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

    /* Step6,�����Ļ������ݳ���(����ά�ȵ�)���������õ���ά������� */
    Eigen::MatrixXd  pca_m = zero_mean_m * components;
    std::cout << "��ά�����ݣ�\n" << pca_m << std::endl;

    /* Step7,���潵ά������ */
    ret = SaveDecompositionData(pca_m);
    if (0 != ret)
    {
        std::cout << "GetData() failed." << std::endl;
        system("pause");
        return ret;
    }

    /* Step8,���ӻ���ά������ */
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
    std::cout << "����ֵ��\n" << eigen_solver.eigenvalues().real() << std::endl;
    std::cout << "����������\n" << eigen_solver.eigenvectors().real() << std::endl;
}
#endif
