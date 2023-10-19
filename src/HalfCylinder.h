#ifndef HALFCYLINDER_H
#define HALFCYLINDER_H

#include <Eigen/Core>
#include <vector>

void makeHalfCylinder(double radius, double height, double triangleArea,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);


void getBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<int>& bdryVertices);

#endif