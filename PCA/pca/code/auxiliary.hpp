#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/* 解析鸢尾花数据集 */
std::vector<double> ParseStrLine(const std::string& line) {
    std::vector<double> one_sample;
    auto found = line.find_first_of(",");
    double val = 0.0;
    std::string substr = line;
    while (found != std::string::npos)
    {
        val = atof(std::string(substr.begin(), substr.begin() + found).c_str());
        one_sample.push_back(val);
        substr = std::string(substr.begin() + found + 1, substr.end());

        found = substr.find_first_of(",");
    }
    val = atof(substr.c_str());
    one_sample.push_back(val);

    return one_sample;
}

/* 读取鸢尾花数据集 */
int GetData(std::vector<std::vector<double> > &iris)
{
    std::ifstream data_txt("../data/iris.txt");
    if (!data_txt.is_open())
    {
        std::cout << "Read data failed." << std::endl;
        system("pause");
        return -1;
    }

    // iris中存储原始数据
    std::string line("");
    while (std::getline(data_txt, line))
    {
        iris.push_back(ParseStrLine(line));
    }

    return 0;
}

/* 读取鸢尾花数据集标签 */
int GetLabels(std::vector<int> &labels)
{
    std::ifstream label_txt("../data/iris_label.txt");
    if (!label_txt.is_open())
    {
        std::cout << "Read data failed." << std::endl;
        system("pause");
        return -1;
    }

    // iris中原始数据的标签
    std::string line("");
    std::getline(label_txt, line);
    auto found = line.find_first_of(",");
    std::string substr = line;
    int val = 0;
    while (found != std::string::npos)
    {
        val = atoi(std::string(substr.begin(), substr.begin() + found).c_str());
        labels.push_back(val);
        substr = std::string(substr.begin() + found + 1, substr.end());
        found = substr.find_first_of(",");
    }
    val = atoi(substr.c_str());
    labels.push_back(val);

    return 0;
}

/* 保存PCA降维后的数据 */
int SaveDecompositionData(const Eigen::MatrixXd  &pca_m)
{
    std::ofstream pca_txt("../data/pca_iris.txt");
    if (!pca_txt.is_open())
    {
        std::cout << "Read data failed." << std::endl;
        system("pause");
        return -1;
    }
    char content[30] = { '\0' };
    for (int i = 0; i < pca_m.rows(); ++i)
    {
        sprintf_s(content, 30, "%.12f", pca_m(i, 0));
        pca_txt << content << ", ";
        sprintf_s(content, 30, "%.12f", pca_m(i, 1));
        pca_txt << content << std::endl;
    }
    pca_txt.close();

    return 0;
}

/* 可视化PCA降维后的数据 */
void Visialization(const Eigen::MatrixXd  &pca_m, std::vector<int> &labels)
{
    cv::Mat visual(220, 220, CV_8UC3);
    cv::Scalar color[] = { cv::Scalar(255,0,0),cv::Scalar(0,255,0), cv::Scalar(0,0,255) };
    for (int i = 0; i < pca_m.rows(); ++i)
    {
        cv::Point center(pca_m(i, 0) * 30 + 100, pca_m(i, 1) * 30 + 100);
        visual.at<cv::Vec3b>(center.x, center.y) = cv::Vec3b(color[labels[i]][0], color[labels[i]][1], color[labels[i]][2]);
    }
    cv::imshow("visual", visual);
    cv::waitKey(0);
}
